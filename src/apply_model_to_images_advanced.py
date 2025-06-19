import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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

# --- UNetモデルの定義 (train_image_enhancement_advanced_v2.py と同じものをコピー) ---
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
        for i in range(len(self.features) - 1, -1, -1):
            feature = self.features[i]
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ResBlock(feature * 2, feature, use_se_block))


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
                
                if H_skip > H_x or W_skip > W_x:
                    diff_H = H_skip - H_x
                    diff_W = W_skip - W_x
                    skip_connection = skip_connection[:, :, diff_H // 2 : H_skip - diff_H // 2,
                                                         diff_W // 2 : W_skip - diff_W // 2]
                elif H_x > H_skip or W_x > W_skip:
                    diff_H = H_x - H_skip
                    diff_W = W_x - W_skip
                    x = x[:, :, diff_H // 2 : H_x - diff_H // 2,
                          diff_W // 2 : W_x - diff_W // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = res_block(concat_skip) # ResBlockを直接呼び出し

        return self.final_conv(x)

# --- メインの推論関数 ---
def apply_model(model_path, input_dir, output_dir, image_height, image_width, features, use_se_block):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    features_list = [int(f) for f in features.split(',')]
    model = UNet(in_channels=3, out_channels=3, features=features_list, use_se_block=use_se_block).to(device)
    
    if not os.path.exists(model_path):
        print(f"エラー: 指定されたモデルファイルが見つかりません: {model_path}")
        return
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"モデルの重みをロードしました: {model_path}")
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return

    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")

    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
    ])

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    
    if not image_files:
        print(f"エラー: {input_dir} に画像ファイルが見つかりません。")
        return

    print(f"\n--- モデル適用開始 ---")
    print(f"処理対象の画像数: {len(image_files)}枚")
    print(f"結果画像を {output_dir} に保存します。\n")

    for img_name in tqdm(image_files, desc="画像処理中"):
        img_path = os.path.join(input_dir, img_name)
        
        input_image_bgr = cv2.imread(img_path)
        if input_image_bgr is None:
            print(f"警告: {img_path} を読み込めませんでした。スキップします。")
            continue
        
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        
        input_image_pil = Image.fromarray(input_image_rgb)
        input_tensor = transform(input_image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_image_np = (output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        output_image_bgr = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)

        base_name, ext = os.path.splitext(img_name)
        output_filename = f"{base_name}_enhanced.png"
        output_full_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_full_path, output_image_bgr)
    
    print(f"\n--- モデル適用完了 ---")
    print(f"生成された画質改善画像の総数: {len(os.listdir(output_dir))}枚")


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みモデルを画像に適用し、画質改善を行うスクリプト。")
    parser.add_argument("model_path", type=str,
                        help="ロードする学習済みモデル (.pth) ファイルのパス。")
    parser.add_argument("input_dir", type=str,
                        help="モデルを適用する入力画像が保存されているディレクトリのパス。")
    parser.add_argument("--output_dir", type=str, default="res/enhanced_images_advanced",
                        help="画質改善された画像を保存するディレクトリのパス。")
    parser.add_argument("--image_height", type=int, default=256,
                        help="モデルへの入力画像高さ (学習時と同じ高さ)。")
    parser.add_argument("--image_width", type=int, default=256,
                        help="モデルへの入力画像幅 (学習時と同じ幅)。")
    parser.add_argument("--features", type=str, default="64,128,256,512",
                        help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("--use_se_block", action="store_true",
                        help="Squeeze-and-Excitation (SE) Blockを使用した場合、このフラグを設定。")
    
    args = parser.parse_args()
    
    apply_model(args.model_path, args.input_dir, args.output_dir, 
                args.image_height, args.image_width, args.features, args.use_se_block)