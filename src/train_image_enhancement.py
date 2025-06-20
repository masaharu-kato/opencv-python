import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import models # VGGロード用
import os
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_optimizer

from dataset import RelativeImagePairDataset

# RGB to HSV conversion (PyTorch version)
# Input: Tensor of shape (N, 3, H, W) in RGB format, range [0, 1]
# Output: Tensor of shape (N, 3, H, W) in HSV format, H:[0,1], S:[0,1], V:[0,1]
def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (* , 3, H, W). Got {image.shape}")

    # Ensure the image is in [0, 1] range for conversion
    # Assume image is already in [0,1] as it comes from ToTensor()
    
    _R: torch.Tensor = image[..., 0, :, :]
    _G: torch.Tensor = image[..., 1, :, :]
    _B: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(dim=-3)[0]
    minc: torch.Tensor = image.min(dim=-3)[0]

    H: torch.Tensor = torch.zeros_like(maxc)
    
    delta: torch.Tensor = maxc - minc
    
    # Check if delta is zero to avoid division by zero later
    # This also handles gray pixels where H is undefined but typically set to 0
    mask_maxc_eq_minc = (delta == 0)

    # Hue calculation
    # For R
    mask_r = (_R == maxc) & ~mask_maxc_eq_minc
    H[mask_r] = (_G[mask_r] - _B[mask_r]) / delta[mask_r]

    # For G
    mask_g = (_G == maxc) & ~mask_maxc_eq_minc
    H[mask_g] = 2.0 + (_B[mask_g] - _R[mask_g]) / delta[mask_g]

    # For B
    mask_b = (_B == maxc) & ~mask_maxc_eq_minc
    H[mask_b] = 4.0 + (_R[mask_b] - _G[mask_b]) / delta[mask_b]

    H = H / 6.0  # Normalize H to [0, 1]
    H[H < 0] += 1.0 # Handle negative Hue values

    # Saturation calculation
    S: torch.Tensor = torch.zeros_like(maxc)
    # Check if maxc is zero to avoid division by zero
    mask_maxc_nz = (maxc != 0)
    S[mask_maxc_nz] = delta[mask_maxc_nz] / maxc[mask_maxc_nz]

    # Value calculation (V is just maxc)
    V: torch.Tensor = maxc

    return torch.stack([H, S, V], dim=-3)

# ----------------------------------------------------
# OR, for Lab conversion (more complex, typically use kornia)
# If you want Lab, it's recommended to use kornia library:
# pip install kornia
# import kornia.color as kc
# 
# def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
#     # Ensure image is float and in [0, 1] range
#     return kc.rgb_to_lab(image)
# ----------------------------------------------------

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
                # inplace=False に変更する (勾配計算に影響を与えないため)
                self.feature_extractor.append(nn.ReLU(inplace=False))
                current_layer += 1
            elif isinstance(layer, nn.MaxPool2d):
                self.feature_extractor.append(layer)
            
            # feature_layers に達したらループを抜ける
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
        # loss をテンソルで初期化する
        loss = torch.tensor(0.0, device=pred_image.device) # <<<< ここを修正

        x_pred = pred_image
        x_target = target_image
        
        feature_idx = 0
        for i, layer in enumerate(self.feature_extractor): # layer を直接使う
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            # ConvやReLU層の後に特徴抽出層のインデックスをカウント
            # MaxPoolは特徴量マップを生成しないのでカウントしない
            if isinstance(layer, (nn.Conv2d, nn.ReLU)):
                feature_idx += 1
                
            if feature_idx in self.feature_layers: # 適切な層で損失を計算
                loss += self.mse_loss(x_pred, x_target)
        
        return loss # これでテンソルが返される

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
    full_dataset = RelativeImagePairDataset(args.dataset_dir, (args.image_width, args.image_height), use_cj=args.use_cj)

    # データの分割
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 最適化手法と損失関数の定義
    optimizer = pytorch_optimizer.RAdam(model.parameters(), lr=args.learning_rate)

    # スケジューラーの定義
    # T_0: 最初の周期の長さ（エポック数）
    # T_mult: 各周期の長さを次の周期でどれだけ伸ばすか（1にすると周期長は一定）
    # eta_min: 学習率の最小値
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6) # 例としてT_0=10 (10エポックで学習率をリセット)

        
    # L1 Loss と Perceptual Loss を組み合わせる
    l1_loss = nn.L1Loss(reduction='none')
    perceptual_loss = PerceptualLoss().to(device) # <-- Perceptual Loss をインスタンス化

    best_avg_val_l1_loss = 0.0
    best_val_ssim = -1.0
    
    # 学習ループ
    for epoch in range(args.epochs):
        
        model.train()
        running_l1_loss = 0.0
        running_perceptual_loss = 0.0
        running_hsv_s_loss = 0.0
        running_total_loss = 0.0

        for batch_idx, (input_tensor, clean_tensor, mask_tensor) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")):
            input_tensor = input_tensor.to(device)
            clean_tensor = clean_tensor.to(device)
            mask_tensor = mask_tensor.to(device) # マスクもデバイスへ

            optimizer.zero_grad()
            output_tensor = model(input_tensor)

            # L1 Loss の計算 with Mask
            # l1_loss_per_pixel の形状は (N, C, H, W)
            l1_loss_per_pixel = l1_loss(output_tensor, clean_tensor)
            
            # 各ピクセルの全チャンネルのL1誤差を合計 (または平均) して、マスクの形状に合わせる
            # 例: (N, 3, H, W) -> (N, 1, H, W)
            if l1_loss_per_pixel.dim() == 4: # Assuming N C H W
                # 各ピクセルのRGBチャンネルの平均誤差を取る
                l1_loss_per_pixel_mean_channels = torch.mean(l1_loss_per_pixel, dim=1, keepdim=True) 
            else:
                l1_loss_per_pixel_mean_channels = l1_loss_per_pixel # もし最初から (N, 1, H, W) ならそのまま

            # マスクを適用
            # mask_tensor は (N, 1, H, W) を想定。l1_loss_per_pixel_mean_channels も (N, 1, H, W)
            masked_l1_loss = l1_loss_per_pixel_mean_channels * mask_tensor 
            
            # 有効なマスクピクセル数で割る (ゼロ除算防止)
            # torch.sum(mask_tensor) はバッチ内の全マスクピクセルの合計 (0 or 1 の値)
            num_valid_pixels = torch.sum(mask_tensor) 
            if num_valid_pixels > 0:
                loss_l1 = torch.sum(masked_l1_loss) / num_valid_pixels
            else:
                loss_l1 = torch.tensor(0.0).to(device) # 全てマスクされている場合は損失0

            total_loss = loss_l1
            running_l1_loss += loss_l1.item()

            # Perceptual Loss の計算 (マスクは適用しない、画像全体で計算)
            # モデルの出力とクリーン画像を直接渡す
            if args.lp_weight:
                loss_perceptual = perceptual_loss(output_tensor, clean_tensor)
                total_loss += args.lp_weight * loss_perceptual
                running_perceptual_loss += loss_perceptual.item()

            # --- ここから HSV 彩度損失の追加 ---
            if args.hsv_s_weight:
                clean_hsv = rgb_to_hsv(clean_tensor)
                output_hsv = rgb_to_hsv(output_tensor)
                
                hsv_s_loss_per_pixel = l1_loss(output_hsv[:, 1:2, :, :], clean_hsv[:, 1:2, :, :])
                masked_hsv_s_loss = hsv_s_loss_per_pixel * mask_tensor
                
                if num_valid_pixels > 0:
                    loss_hsv_s = torch.sum(masked_hsv_s_loss) / num_valid_pixels
                else:
                    loss_hsv_s = torch.tensor(0.0).to(device)
                # --- HSV 彩度損失の追加ここまで ---
                total_loss += args.hsv_s_weight * loss_hsv_s
                running_hsv_s_loss += loss_hsv_s.item()
            
            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()

            # スケジューラーを更新 (Optimizer.step() の直後が一般的)
            scheduler.step(args.epochs + batch_idx / len(train_loader)) # 現在のエポック進捗を渡す

        avg_l1_loss = running_l1_loss / len(train_loader)
        avg_perceptual_loss = running_perceptual_loss / len(train_loader)
        avg_val_hsv_s_loss = running_hsv_s_loss / len(val_loader)
        avg_total_loss = running_total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Train Loss: L1={avg_l1_loss:.4f}, Perceptual={avg_perceptual_loss:.4f}, HSV_S={avg_val_hsv_s_loss:4f}, Total={avg_total_loss:.4f}")


        # バリデーション
        model.eval()

        val_running_l1_loss = 0.0
        val_running_perceptual_loss = 0.0
        val_running_hsv_s_loss = 0.0
        val_running_total_loss = 0.0
        val_ssim_scores = [] # SSIMスコアを格納するリスト

        with torch.no_grad():
            for batch_idx, (input_tensor, clean_tensor, mask_tensor) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)")):
                input_tensor = input_tensor.to(device)
                clean_tensor = clean_tensor.to(device)
                mask_tensor = mask_tensor.to(device) # マスクもデバイスへ

                output_tensor = model(input_tensor)

                # L1 Loss の計算 with Mask (訓練時と同じロジック)
                l1_loss_per_pixel = l1_loss(output_tensor, clean_tensor)
                if l1_loss_per_pixel.dim() == 4:
                    l1_loss_per_pixel_mean_channels = torch.mean(l1_loss_per_pixel, dim=1, keepdim=True) 
                else:
                    l1_loss_per_pixel_mean_channels = l1_loss_per_pixel

                masked_l1_loss = l1_loss_per_pixel_mean_channels * mask_tensor 
                
                num_valid_pixels = torch.sum(mask_tensor)
                if num_valid_pixels > 0:
                    loss_l1 = torch.sum(masked_l1_loss) / num_valid_pixels
                else:
                    loss_l1 = torch.tensor(0.0).to(device)

                total_loss = loss_l1
                val_running_l1_loss += loss_l1.item()

                # Perceptual Loss の計算 (訓練時と同じロジック)
                if args.lp_weight:
                    loss_perceptual = perceptual_loss(output_tensor, clean_tensor)
                    total_loss += args.lp_weight * loss_perceptual
                    val_running_perceptual_loss += loss_perceptual.item()


                # HSV 彩度損失
                if args.hsv_s_weight:
                    clean_hsv = rgb_to_hsv(clean_tensor)
                    output_hsv = rgb_to_hsv(output_tensor)
                    
                    hsv_s_loss_per_pixel = l1_loss(output_hsv[:, 1:2, :, :], clean_hsv[:, 1:2, :, :])
                    masked_hsv_s_loss = hsv_s_loss_per_pixel * mask_tensor
                    
                    if num_valid_pixels > 0:
                        loss_hsv_s = torch.sum(masked_hsv_s_loss) / num_valid_pixels
                    else:
                        loss_hsv_s = torch.tensor(0.0).to(device)
                
                    total_loss += args.hsv_s_weight * loss_hsv_s
                    val_running_hsv_s_loss += loss_hsv_s.item()

                val_running_total_loss += total_loss.item()
                
                # --- SSIM 計算 ---
                # PyTorchテンソル (N, C, H, W) から NumPy配列 (H, W, C) に変換し、0-255スケールに戻す
                for i in range(input_tensor.shape[0]): # バッチ内の各画像に対して
                    output_img = (output_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    clean_img = (clean_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # RGBAのAチャンネルはSSIM計算では使わないので、RGB部分を渡す
                    # mask_alpha = (mask_tensor[i].detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
                    
                    # channel_axis=2 は (H, W, C) 形式の場合に指定
                    # data_range は画像の最大値-最小値
                    # win_size はSSIM計算のウィンドウサイズ。通常は7x7や11x11
                    current_ssim = ssim_metric(output_img, clean_img, data_range=255, channel_axis=2, win_size=11)
                    val_ssim_scores.append(current_ssim)


        avg_val_l1_loss = val_running_l1_loss / len(val_loader)
        avg_val_perceptual_loss = val_running_perceptual_loss / len(val_loader)
        avg_val_hsv_s_loss = val_running_hsv_s_loss / len(val_loader) 
        avg_val_total_loss = val_running_total_loss / len(val_loader)
        avg_val_ssim = np.mean(val_ssim_scores) if val_ssim_scores else 0.0

        print(f"Epoch [{epoch+1}/{args.epochs}] Validation Loss: L1={avg_val_l1_loss:.4f}, Perceptual={avg_val_perceptual_loss:.4f}, HSV_S={avg_val_hsv_s_loss:4f}, Total={avg_val_total_loss:.4f}, SSIM={avg_val_ssim:.4f}")

        # Save model if L1 Loss and/or SSIM are improved
        f_save = False
        if avg_val_l1_loss < best_avg_val_l1_loss:
            best_avg_val_l1_loss = avg_val_l1_loss
            f_save = True

        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            f_save = True

        if f_save:
            model_save_path = f"{args.model_save_dir}/best_unet_model_l1loss_{avg_val_l1_loss:.2f}_ssim_{avg_val_ssim:.2f}.pth"
            os.makedirs(args.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")


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
    parser.add_argument("--lp_weight", type=float, default=1.0,
                        help="lp_weight")
    parser.add_argument("--hsv_s_weight", type=float, default=1.0,
                        help="hsv_s_weight")
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
    parser.add_argument("--use_cj", action="store_true",
                        help="use ColorJitter on the dataset")

    args = parser.parse_args()

    train_model(args)
