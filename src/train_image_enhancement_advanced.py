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
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import random # データ拡張のシード設定のため

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

# --- データセットクラスの定義 (変更なし) ---
class ImageEnhancementDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, transform=None):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.transform = transform
        
        self.image_filenames = []
        for filename in os.listdir(degraded_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.image_filenames.append(filename)
        
        self.degraded_to_clean_map = {}
        for deg_name in self.image_filenames:
            parts = deg_name.split('_deg_')
            if len(parts) > 1:
                clean_base_name = parts[0]
                clean_ext = os.path.splitext(deg_name)[1]
                clean_name = clean_base_name + clean_ext
                self.degraded_to_clean_map[deg_name] = clean_name
            else:
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

        degraded_image = cv2.imread(degraded_path)
        clean_image = cv2.imread(clean_path)

        if degraded_image is None:
            raise FileNotFoundError(f"Degraded image not found at {degraded_path}")
        if clean_image is None:
            raise FileNotFoundError(f"Clean image not found at {clean_path}")

        degraded_image = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2RGB)
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        degraded_image = Image.fromarray(degraded_image)
        clean_image = Image.fromarray(clean_image)

        if self.transform:
            seed = np.random.randint(2147483647)
            
            random.seed(seed)
            torch.manual_seed(seed)
            degraded_image = self.transform(degraded_image)
            
            random.seed(seed)
            torch.manual_seed(seed)
            clean_image = self.transform(clean_image)

        return degraded_image, clean_image

# --- メインの学習関数 ---
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用デバイス: {device}")

    log_dir = os.path.join("runs_advanced", f"exp_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoardログディレクトリ: {log_dir}")

    target_height = args.image_height
    target_width = args.image_width
    
    if target_height <= 0 or target_width <= 0:
        raise ValueError("image_height and image_width must be positive integers.")

    common_transforms = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
    ])
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        common_transforms
    ])

    dataset = ImageEnhancementDataset(args.degraded_dir, args.clean_dir, transform=train_transforms)
    
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_dataset.dataset.transform = common_transforms

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    features_list = [int(f) for f in args.features.split(',')]

    model = UNet(in_channels=3, out_channels=3, features=features_list, use_se_block=args.use_se_block).to(device)
    print(f"モデルのfeatures: {features_list}")
    print(f"SEBlock使用: {args.use_se_block}")
    
    # オプション: モデルの構造を表示して確認
    # from torchsummary import summary
    # summary(model, (3, target_height, target_width)) # Requires 'torchsummary' package

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_psnr = 0.0
    models_save_dir = "models_advanced"
    os.makedirs(models_save_dir, exist_ok=True)


    print("\n--- 学習開始 ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} (Train)", leave=False)
        for batch_idx, (degraded_imgs, clean_imgs) in enumerate(train_loop):
            degraded_imgs = degraded_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            outputs = model(degraded_imgs)
            loss = criterion(outputs, clean_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * degraded_imgs.size(0)
            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_dataset)
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)

        model.eval()
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} (Val)", leave=False)
        with torch.no_grad():
            for degraded_imgs, clean_imgs in val_loop:
                degraded_imgs = degraded_imgs.to(device)
                clean_imgs = clean_imgs.to(device)

                outputs = model(degraded_imgs)
                
                outputs_np = (outputs.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                clean_imgs_np = (clean_imgs.cpu().numpy() * 255).astype(np.uint8)

                for i in range(outputs_np.shape[0]):
                    pred_image = outputs_np[i].transpose(1, 2, 0)
                    true_image = clean_imgs_np[i].transpose(1, 2, 0)

                    val_psnr_sum += psnr_metric(true_image, pred_image, data_range=255)
                    val_ssim_sum += ssim_metric(true_image, pred_image, data_range=255, channel_axis=-1)

        epoch_val_psnr = val_psnr_sum / len(val_dataset)
        epoch_val_ssim = val_ssim_sum / len(val_dataset)

        writer.add_scalar('Metrics/PSNR', epoch_val_psnr, epoch)
        writer.add_scalar('Metrics/SSIM', epoch_val_ssim, epoch)
        
        print(f"Epoch {epoch} 終了 - Train Loss: {epoch_train_loss:.4f}, Val PSNR: {epoch_val_psnr:.2f} dB, Val SSIM: {epoch_val_ssim:.4f}")

        if epoch_val_psnr > best_val_psnr:
            best_val_psnr = epoch_val_psnr
            model_save_path = os.path.join(models_save_dir, f"best_unet_model_psnr_{best_val_psnr:.2f}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"--> 最良モデルを保存しました: {model_save_path}")

    print("\n--- 学習完了 ---")
    writer.close()

# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の強化版学習スクリプト。")
    parser.add_argument("degraded_dir", type=str,
                        help="劣化画像が保存されているディレクトリのパス。")
    parser.add_argument("clean_dir", type=str,
                        help="対応する綺麗な画像が保存されているディレクトリのパス。")
    parser.add_argument("--image_height", type=int, default=256,
                        help="モデルへの入力画像高さ。")
    parser.add_argument("--image_width", type=int, default=256,
                        help="モデルへの入力画像幅。")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("--epochs", type=int, default=100,
                        help="学習エポック数。")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="初期学習率。")
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="学習データセットの割合。残りは検証データセット。")
    parser.add_argument("--num_workers", type=int, default=(os.cpu_count() or 2) // 2,
                        help="データローダーが使用するワーカースレッド数。")
    parser.add_argument("--no_cuda", action="store_true",
                        help="CUDA (GPU) を使用しない場合、このフラグを設定。")
    parser.add_argument("--features", type=str, default="64,128,256,512",
                        help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("--use_se_block", action="store_true",
                        help="Squeeze-and-Excitation (SE) Blockを使用する場合、このフラグを設定。")

    args = parser.parse_args()

    train_model(args)