import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- UNetモデルの定義 (train_image_enhancement.py と同じものをコピー) ---
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
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        # Up part of UNet
        for feature in reversed(features):
            if feature == features[-1]: # ボトルネックからのアップサンプリング
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature * 2, feature, kernel_size=2, stride=2
                    )
                )
                self.ups.append(self._conv_block(feature * 2, feature))
            else: # 通常のアップサンプリング
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature * 2, feature, kernel_size=2, stride=2
                    )
                )
                self.ups.append(self._conv_block(feature * 2, feature))

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

        for idx in range(len(self.ups) // 2):
            x = self.ups[idx * 2](x) 
            
            skip_connection = skip_connections[idx]

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
            x = self.ups[idx * 2 + 1](concat_skip)

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

# --- メインの推論関数 ---
def apply_model(model_path, input_dir, output_dir, image_size=256):
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデルの初期化
    # 学習時と同じ features 設定を使用すること
    model = UNet(in_channels=3, out_channels=3, features=[32, 64, 128, 256]).to(device)
    
    # 学習済み重みのロード
    if not os.path.exists(model_path):
        print(f"エラー: 指定されたモデルファイルが見つかりません: {model_path}")
        return
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"モデルの重みをロードしました: {model_path}")
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return

    model.eval() # モデルを評価モードに設定 (DropoutやBatchNormが評価モードになる)

    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")

    # 画像変換 (学習時の検証データセットと同じ変換)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), # PIL ImageをTensorに変換し、0-1に正規化
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
        
        # OpenCVで画像を読み込み (BGR形式)
        input_image_bgr = cv2.imread(img_path)
        if input_image_bgr is None:
            print(f"警告: {img_path} を読み込めませんでした。スキップします。")
            continue
        
        # RGBに変換
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        
        # PIL Imageに変換してtransformを適用
        input_image_pil = Image.fromarray(input_image_rgb)
        input_tensor = transform(input_image_pil).unsqueeze(0).to(device) # バッチ次元を追加

        with torch.no_grad(): # 推論時は勾配計算不要
            output_tensor = model(input_tensor)

        # 出力テンソルを画像に変換
        # 0-1の範囲から0-255に戻し、CPUに移動、NumPy配列に変換 (CxHxW -> HxWxC)
        output_image_np = (output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # RGBからBGRに戻す (OpenCVで保存するため)
        output_image_bgr = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)

        # 結果を保存
        base_name, ext = os.path.splitext(img_name)
        output_filename = f"{base_name}_enhanced.png" # PNGで統一
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
    parser.add_argument("--output_dir", type=str, default="res/enhanced_images",
                        help="画質改善された画像を保存するディレクトリのパス。")
    parser.add_argument("--image_size", type=int, default=256,
                        help="モデルへの入力画像サイズ (学習時と同じサイズ)。")
    
    args = parser.parse_args()
    
    apply_model(args.model_path, args.input_dir, args.output_dir, args.image_size)