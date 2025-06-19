import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import os
import time
import argparse
import random
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# --- 1. データセットクラスの定義 ---
class ImageEnhancementDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, transform=None):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.transform = transform
        
        self.image_filenames = []
        # 劣化画像のファイル名を基にペアを構築
        for filename in os.listdir(degraded_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.image_filenames.append(filename)
        
        # 劣化画像と対応する綺麗な画像のパスをマッピング
        # 例: degraded_image_name = "clean_img_001_deg_low_contrast_DRC_00.png"
        #     clean_image_name   = "clean_img_001.png"
        # ファイル名の命名規則に依存するので注意
        self.degraded_to_clean_map = {}
        for deg_name in self.image_filenames:
            # 劣化画像のファイル名から元の綺麗な画像名を推測
            # 例: 'img_abc_deg_low_contrast_DRC_01.png' -> 'img_abc.png'
            parts = deg_name.split('_deg_')
            if len(parts) > 1:
                clean_base_name = parts[0]
                clean_ext = os.path.splitext(deg_name)[1]
                clean_name = clean_base_name + clean_ext
                self.degraded_to_clean_map[deg_name] = clean_name
            else:
                # 命名規則が合わない場合はスキップ（またはエラーハンドリング）
                print(f"Warning: Unexpected filename format for degraded image: {deg_name}. Skipping.")
                continue


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        degraded_filename = self.image_filenames[idx]
        clean_filename = self.degraded_to_clean_map.get(degraded_filename)

        if clean_filename is None:
            raise FileNotFoundError(f"Clean image not found for degraded image: {degraded_filename}")

        degraded_path = os.path.join(self.degraded_dir, degraded_filename)
        clean_path = os.path.join(self.clean_dir, clean_filename)

        # OpenCVで画像を読み込み (BGR形式)
        degraded_image = cv2.imread(degraded_path)
        clean_image = cv2.imread(clean_path)

        if degraded_image is None:
            raise FileNotFoundError(f"Degraded image not found at {degraded_path}")
        if clean_image is None:
            raise FileNotFoundError(f"Clean image not found at {clean_path}")

        # RGBに変換 (PyTorchはRGBを想定することが多い)
        degraded_image = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2RGB)
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        
        # NumPy配列からPIL Imageに変換してtransformsを適用
        # もしくは、NumPy配列のままtransformsを適用するカスタムTransformを実装
        # ここではPIL Image経由でtransformsを使う
        from PIL import Image
        degraded_image = Image.fromarray(degraded_image)
        clean_image = Image.fromarray(clean_image)

        if self.transform:
            # DegradedとCleanに同じランダム変換（クロップ、フリップなど）を適用するため
            # seedを設定してtransformの順序を揃える
            seed = np.random.randint(2147483647) # 乱数シードを生成
            
            random.seed(seed)
            torch.manual_seed(seed)
            degraded_image = self.transform(degraded_image)
            
            random.seed(seed)
            torch.manual_seed(seed)
            clean_image = self.transform(clean_image)

        return degraded_image, clean_image

# --- 修正後のUNetモデルの定義（より一般的なアプローチ） ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(self._conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        # ボトルネックの入力はfeatures[-1] (256)、出力はfeatures[-1]*2 (512)
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        # Up part of UNet
        # features[:-1]は[32, 64, 128]
        # reversed(features[:-1])は[128, 64, 32]
        for feature in reversed(features): # 修正：features全体を逆順にする (ボトルネックのfeatureも含む)
            # 最初のアップサンプリング層の入力はボトルネックの出力（features[-1] * 2）
            # それ以降は、前の層のチャネル（feature * 2）
            if feature == features[-1]: # ボトルネックからのアップサンプリング
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature * 2, feature, kernel_size=2, stride=2 # 512 -> 256
                    )
                )
                self.ups.append(self._conv_block(feature * 2, feature)) # (256 + 256) -> 256
            else: # 通常のアップサンプリング
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature * 2, feature, kernel_size=2, stride=2 # 例: 256 -> 128, 128 -> 64, 64 -> 32
                    )
                )
                self.ups.append(self._conv_block(feature * 2, feature)) # (128 + 128) -> 128 など

        # Final Convolution
        self.final_conv = nn.Sequential(nn.Conv2d(features[0], out_channels, kernel_size=1), nn.Sigmoid()) # Sigmoidを追加


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse skip connections

        # 変更点：idxの範囲を適切に修正
        # self.ups は ConvTranspose と _conv_block のペアが features の数だけ含まれる
        # したがって、len(self.ups) は featuresの数 * 2
        for idx in range(len(self.ups) // 2): # featureの数だけ繰り返す (各idxはConvTransposeのインデックス)
            # ConvTranspose層
            x = self.ups[idx * 2](x) 
            
            # 対応するスキップコネクション
            skip_connection = skip_connections[idx]

            # サイズが合わない場合の調整 (クロップ)
            if x.shape != skip_connection.shape:
                _, _, H_x, W_x = x.shape
                _, _, H_skip, W_skip = skip_connection.shape
                
                # スキップコネクションの方が大きい場合、中心を基準にクロップ
                if H_skip > H_x or W_skip > W_x:
                    diff_H = H_skip - H_x
                    diff_W = W_skip - W_x
                    skip_connection = skip_connection[:, :, diff_H // 2 : H_skip - diff_H // 2,
                                                         diff_W // 2 : W_skip - diff_W // 2]
                # xの方が大きい場合、スキップコネクションをパディングするか、xをクロップ
                # 今回はxをクロップする (ConvTransposeでよく発生)
                elif H_x > H_skip or W_x > W_skip:
                    diff_H = H_x - H_skip
                    diff_W = W_x - W_skip
                    x = x[:, :, diff_H // 2 : H_x - diff_H // 2,
                          diff_W // 2 : W_x - diff_W // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx * 2 + 1](concat_skip) # Conv block

        return self.final_conv(x)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

# --- 3. メインの学習関数 ---
def train_model(args):
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用デバイス: {device}")

    # TensorBoardロガー設定
    log_dir = os.path.join("runs", f"exp_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoardログディレクトリ: {log_dir}")

    # データ変換とデータ拡張
    # データ拡張は、訓練データセットに対してのみ適用する
    common_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), # 画像サイズを統一
        transforms.ToTensor(), # PIL ImageをTensorに変換し、0-1に正規化
    ])
    
    # 訓練用に追加のデータ拡張
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(), # ランダム水平反転
        transforms.RandomVerticalFlip(),   # ランダム垂直反転
        # transforms.ColorJitter(brightness=0.1, contrast=0.1), # 明るさ・コントラストの微調整 (劣化と競合する可能性があるので注意)
        common_transforms
    ])

    dataset = ImageEnhancementDataset(args.degraded_dir, args.clean_dir, transform=train_transforms)
    
    # データセットの分割
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 検証セットはデータ拡張なし
    val_dataset.dataset.transform = common_transforms

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # モデルの初期化
    model = UNet(in_channels=3, out_channels=3, features=[32, 64, 128, 256]).to(device) # 軽量なfeatures設定
    # 以前 [64,128,256,512] を提案しましたが、より軽量化するため [32, 64, 128, 256] から始めます。
    # 32,64,128,256 -> MaxPool3回 -> 256 Bottleneck 512

    # 損失関数と最適化手法
    criterion = nn.L1Loss() # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 学習率スケジューラー (オプション)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_val_psnr = 0.0 # 最も良い検証PSNRを記録

    # 学習ループ
    print("\n--- 学習開始 ---")
    for epoch in range(1, args.epochs + 1):
        model.train() # モデルを訓練モードに設定
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} (Train)", leave=False)
        for batch_idx, (degraded_imgs, clean_imgs) in enumerate(train_loop):
            degraded_imgs = degraded_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # 順伝播
            outputs = model(degraded_imgs)
            loss = criterion(outputs, clean_imgs)

            # 逆伝播と最適化
            optimizer.zero_grad() # 勾配をゼロクリア
            loss.backward()       # 逆伝播
            optimizer.step()      # パラメータ更新

            running_loss += loss.item() * degraded_imgs.size(0) # バッチサイズでスケーリング
            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_dataset)
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)

        # 検証フェーズ
        model.eval() # モデルを評価モードに設定 (DropoutやBatchNormが評価モードになる)
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} (Val)", leave=False)
        with torch.no_grad(): # 勾配計算を無効化 (メモリと速度の最適化)
            for degraded_imgs, clean_imgs in val_loop:
                degraded_imgs = degraded_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                outputs = model(degraded_imgs)
                
                # PSNRとSSIM計算のためにTensorをNumPy配列に変換
                # 0-1範囲から0-255範囲に戻し、uint8に変換
                outputs_np = (outputs.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                clean_imgs_np = (clean_imgs.cpu().numpy() * 255).astype(np.uint8)

                # バッチ内の各画像に対して計算
                for i in range(outputs_np.shape[0]):
                    # HWC形式に変換 (skimage.metricsはHWCを想定)
                    pred_image = outputs_np[i].transpose(1, 2, 0)
                    true_image = clean_imgs_np[i].transpose(1, 2, 0)

                    val_psnr_sum += psnr_metric(true_image, pred_image, data_range=255)
                    # SSIMはグレースケールで計算されることが多いが、カラー画像にも適用可能
                    # multichannel=True for color images, channel_axis for skimage v0.19+
                    val_ssim_sum += ssim_metric(true_image, pred_image, data_range=255, channel_axis=-1)

        epoch_val_psnr = val_psnr_sum / len(val_dataset)
        epoch_val_ssim = val_ssim_sum / len(val_dataset)

        writer.add_scalar('Metrics/PSNR', epoch_val_psnr, epoch)
        writer.add_scalar('Metrics/SSIM', epoch_val_ssim, epoch)
        
        print(f"Epoch {epoch} 終了 - Train Loss: {epoch_train_loss:.4f}, Val PSNR: {epoch_val_psnr:.2f} dB, Val SSIM: {epoch_val_ssim:.4f}")

        # 最良モデルの保存
        if epoch_val_psnr > best_val_psnr:
            best_val_psnr = epoch_val_psnr
            model_save_path = os.path.join("models", f"best_unet_model_psnr_{best_val_psnr:.2f}.pth")
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"--> 最良モデルを保存しました: {model_save_path}")

        # scheduler.step(epoch_val_psnr) # 学習率スケジューラーを使用する場合

    print("\n--- 学習完了 ---")
    writer.close()

# --- 4. コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の学習スクリプト。")
    parser.add_argument("degraded_dir", type=str,
                        help="劣化画像が保存されているディレクトリのパス。")
    parser.add_argument("clean_dir", type=str,
                        help="対応する綺麗な画像が保存されているディレクトリのパス。")
    parser.add_argument("--image_size", type=int, default=256,
                        help="モデルへの入力画像サイズ (正方形)。")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("--epochs", type=int, default=100,
                        help="学習エポック数。")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="初期学習率。")
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="学習データセットの割合。残りは検証データセット。")
    parser.add_argument("--num_workers", type=int, default=(os.cpu_count() or 2) // 2, # CPUコア数の半分を推奨
                        help="データローダーが使用するワーカースレッド数。")
    parser.add_argument("--no_cuda", action="store_true",
                        help="CUDA (GPU) を使用しない場合、このフラグを設定。")

    args = parser.parse_args()

    train_model(args)