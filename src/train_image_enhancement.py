import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import models # VGGロード用
import cv2
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import random # データ拡張のシード設定のため

from dataset import RelativeImagePairDataset

# Perceptual Loss (VGG Loss) の定義
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[2, 7, 16, 25, 34]): # VGG19のrelu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        super(PerceptualLoss, self).__init__()
        # VGG19をロードし、特徴抽出器のみを使用
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # VGG19はImageNetで学習済みなので、入力がRGBであることを想定
        # また、入力画像の平均と標準偏差で正規化が必要
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # VGGの特徴抽出層を構築
        self.feature_extractor = nn.ModuleList()
        current_layer = 0
        for i, layer in enumerate(vgg19): # type: ignore
            if isinstance(layer, nn.Conv2d):
                self.feature_extractor.append(layer)
                current_layer += 1
            elif isinstance(layer, nn.ReLU):
                self.feature_extractor.append(layer)
                current_layer += 1
            elif isinstance(layer, nn.MaxPool2d):
                self.feature_extractor.append(layer)
            
            if current_layer in feature_layers:
                # 指定された層までを特徴抽出器とする
                # 勾配計算を無効化し、評価モードにする
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                self.feature_extractor.eval()
                break # 目的の層に達したらループを抜ける
        
        self.feature_layers = feature_layers
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_image, target_image):
        # VGGの入力要件に合わせて画像を正規化
        pred_image = self.normalize(pred_image)
        target_image = self.normalize(target_image)

        # 各特徴層での損失を計算
        loss = 0
        x_pred = pred_image
        x_target = target_image
        
        feature_idx = 0
        for layer in self.feature_extractor:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if feature_idx in self.feature_layers: # 適切な層で損失を計算
                 loss += self.mse_loss(x_pred, x_target)
            
            if isinstance(layer, (nn.Conv2d, nn.ReLU)): # ConvやReLU層の後に特徴抽出層のインデックスをカウント
                feature_idx += 1
        
        return loss

# --- Squeeze-and-Excitation Block (Attention Mechanism) ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Residual Block with optional SEBlock ---
# _conv_block の代わりとして独立したクラスで定義
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se_block):
        super(ResBlock, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE Block
        self.se_block = SEBlock(out_channels) if use_se_block else nn.Identity()

        # Shortcut for residual connection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.final_relu = nn.ReLU(inplace=True) # Residual Add後にReLU

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se_block(out) # Apply SEBlock
        
        out += identity # Residual Add
        out = self.final_relu(out) # Apply final ReLU
        return out

# --- UNetモデルの定義 (修正版) ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=None, use_se_block=False):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.features = features
        self.use_se_block = use_se_block

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_in_channels = in_channels
        # Down part of UNet
        for feature in self.features:
            self.downs.append(ResBlock(current_in_channels, feature, use_se_block))
            current_in_channels = feature

        # Bottleneck
        self.bottleneck = ResBlock(self.features[-1], self.features[-1] * 2, use_se_block)

        # Up part of UNet
        # featuresを逆順に回し、ボトルネックの出力からfeatures[-1]への最初のアップサンプリングも含める
        for i in range(len(self.features) - 1, -1, -1):
            feature = self.features[i]
            # ConvTranspose2d の出力チャネルは、結合するスキップコネクションのチャネル数と同じ、つまり feature
            # 結合後のResBlockの入力は feature * 2 (スキップとConvTranspose2dの出力の結合)
            # 結合後のResBlockの出力は feature
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2 # Upsampling input: current_x_channels, output: feature
                )
            )
            self.ups.append(ResBlock(feature * 2, feature, use_se_block)) # ResBlock input: (skip + upsampled), output: feature


        # Final Convolution with Sigmoid
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.features[0], out_channels, kernel_size=1),
            nn.Sigmoid() # 出力を [0, 1] に制限
        )

    def forward(self, x):
        skip_connections = []

        # Down path
        for down_block in self.downs:
            x = down_block(x) # ResBlockを直接呼び出し
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x) # BottleneckもResBlock

        skip_connections = skip_connections[::-1] # Reverse skip connections

        # Up path
        for i in range(len(self.ups) // 2):
            trans_conv = self.ups[i * 2] # ConvTranspose2d
            res_block = self.ups[i * 2 + 1] # ResBlock

            x = trans_conv(x)
            
            skip_connection = skip_connections[i]

            # サイズが合わない場合の調整 (クロップ)
            if x.shape != skip_connection.shape:
                _, _, H_x, W_x = x.shape
                _, _, H_skip, W_skip = skip_connection.shape
                
                # スキップコネクションの方が大きい場合はクロップ
                if H_skip > H_x or W_skip > W_x:
                    diff_H = H_skip - H_x
                    diff_W = W_skip - W_x
                    skip_connection = skip_connection[:, :, diff_H // 2 : H_skip - diff_H // 2,
                                                         diff_W // 2 : W_skip - diff_W // 2]
                # アップサンプリングされた方が大きい場合はクロップ (通常は起こらないはずだが念のため)
                elif H_x > H_skip or W_x > W_skip:
                    diff_H = H_x - H_skip
                    diff_W = W_x - W_skip
                    x = x[:, :, diff_H // 2 : H_x - diff_H // 2,
                          diff_W // 2 : W_x - diff_W // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = res_block(concat_skip) # ResBlockを直接呼び出し

        return self.final_conv(x)


# --- メインの学習関数 ---
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 特徴量リストを文字列から変換
    features_list = [int(f) for f in args.features.split(',')]

    # モデルのインスタンス化
    model = UNet(in_channels=3, out_channels=3, features=features_list, use_se_block=args.use_se_block).to(device)

    # データセットの準備
    # root_dir には、unet_datasets/ のパスを指定します
    # ここでは、degraded_dir を root_dir として利用し、その下の動画ファイル名を走査します
    full_dataset = RelativeImagePairDataset(root_dir=args.dataset_dir, # <-- ここを変更
                                             image_height=args.image_height, 
                                             image_width=args.image_width)

    # データの分割
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 最適化手法と損失関数の定義
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # L1 Loss と Perceptual Loss を組み合わせる
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device) # <-- Perceptual Loss をインスタンス化

    best_val_psnr = 0.0 # 最良のPSNRを記録
    
    # 学習ループ
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (degraded_images, clean_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")):
            degraded_images = degraded_images.to(device)
            clean_images = clean_images.to(device)

            optimizer.zero_grad()
            outputs = model(degraded_images)

            # 損失計算: L1 Loss と Perceptual Loss の組み合わせ
            loss_l1 = l1_loss(outputs, clean_images)
            loss_perceptual = perceptual_loss(outputs, clean_images)
            
            # 損失の重み付け (ハイパーパラメータ)
            # 例えば、L1 Loss を重視しつつ、Perceptual Loss で見た目の品質をガイドする
            total_loss = loss_l1 + args.lp_weight * loss_perceptual # <-- ここで損失を組み合わせる

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # バリデーション
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for batch_idx, (degraded_images, clean_images) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)")):
                degraded_images = degraded_images.to(device)
                clean_images = clean_images.to(device)

                outputs = model(degraded_images)

                # PSNRとSSIMの計算（既存のコードと同じ）
                for i in range(outputs.shape[0]):
                    output_np = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    clean_np = clean_images[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # PSNRとSSIMは0-1範囲で計算されることを前提
                    val_psnr += psnr_metric(output_np, clean_np, data_range=1)
                    val_ssim += ssim_metric(output_np, clean_np, data_range=1, channel_axis=-1) # type: ignore

                    # print(f"Output NP dtype: {output_np.dtype}, Min: {output_np.min()}, Max: {output_np.max()}")
                    # print(f"Clean NP dtype: {clean_np.dtype}, Min: {clean_np.min()}, Max: {clean_np.max()}")
        
        val_psnr /= len(val_loader.dataset) # type: ignore
        val_ssim /= len(val_loader.dataset) # type: ignore

        print(f"Epoch {epoch+1} 終了 - Train Loss: {avg_train_loss:.4f}, Val PSNR: {val_psnr:.2f} dB, Val SSIM: {val_ssim:.4f}")

        # モデルの保存 (PSNRが改善した場合)
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            model_save_path = f"{args.model_save_dir}/best_unet_model_psnr_{best_val_psnr:.2f}.pth"
            os.makedirs(args.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"ベストモデルを保存しました: {model_save_path}")


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の強化版学習スクリプト。")
    parser.add_argument("dataset_dir", type=str,
                        help="データセットディレクトリ（動画ごとのファイル名のサブディレクトリを含む）")
    parser.add_argument("model_save_dir", type=str,
                        help="モデルの保存先ディレクトリ")
    parser.add_argument("--image_height", type=int, default=544,
                        help="モデルへの入力画像高さ。")
    parser.add_argument("--image_width", type=int, default=720,
                        help="モデルへの入力画像幅。")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("--epochs", type=int, default=100,
                        help="学習エポック数。")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="初期学習率。")
    parser.add_argument("--lp_weight", type=float, default=0.1,
                        help="lp_weight")
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="学習データセットの割合。残りは検証データセット。")
    parser.add_argument("--num_workers", type=int, default=(os.cpu_count() or 2) // 2,
                        help="データローダーが使用するワーカースレッド数。")
    parser.add_argument("--no_cuda", action="store_true",
                        help="CUDA (GPU) を使用しない場合、このフラグを設定。")
    parser.add_argument("--features", type=str, default="64,128,256",
                        help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("--use_se_block", action="store_true",
                        help="Squeeze-and-Excitation (SE) Blockを使用する場合、このフラグを設定。")

    args = parser.parse_args()

    train_model(args)
