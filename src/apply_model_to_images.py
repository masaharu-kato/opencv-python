import torch
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# モデルの定義を models/unet_model.py からインポート
# sys.path に models ディレクトリを追加して、モジュールとして認識させる
import sys
# 現在のスクリプトの親ディレクトリ（src）をpathに追加し、そこからmodelsをimportできるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from models.unet import UNet # models/unet_model.py から UNet クラスをインポート

# --- メインの推論関数 ---
def apply_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    features_list = [int(f) for f in args.features.split(',')]
    
    # UNetモデルのインスタンス化
    model = UNet(in_channels=3, out_channels=3, features=features_list, use_se_block=args.use_se_block, use_cbam=args.use_cbam).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"エラー: 指定されたモデルファイルが見つかりません: {args.model_path}")
        return
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"モデルの重みをロードしました: {args.model_path}")
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return

    model.eval() # 推論モードに設定

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"出力ディレクトリを作成しました: {args.output_dir}")

    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        transforms.ToTensor(),
    ])

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    
    if not image_files:
        print(f"エラー: {args.input_dir} に画像ファイルが見つかりません。")
        return

    print(f"\n--- モデル適用開始 ---")
    print(f"処理対象の画像数: {len(image_files)}枚")
    print(f"結果画像を {args.output_dir} に保存します。\n")

    for img_name in tqdm(image_files, desc="画像処理中"):
        img_path = os.path.join(args.input_dir, img_name)
        
        input_image_bgr = cv2.imread(img_path)
        if input_image_bgr is None:
            print(f"警告: {img_path} を読み込めませんでした。スキップします。")
            continue
        
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        
        input_image_pil = Image.fromarray(input_image_rgb)
        input_tensor = transform(input_image_pil).unsqueeze(0).to(device) # type: ignore

        with torch.no_grad(): # 勾配計算を無効化
            output_tensor = model(input_tensor)

        # 出力テンソルを画像に変換し、保存
        # モデルの最終出力がSigmoidで[0, 1]に正規化されていることを前提
        output_image_np = (output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        output_image_bgr = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)

        base_name, ext = os.path.splitext(img_name)
        # 既存のファイルに上書きしないように、output_dir内のファイル名に_enhanced.pngを付加
        output_filename = f"{base_name}_enhanced.png"
        output_full_path = os.path.join(args.output_dir, output_filename)
        cv2.imwrite(output_full_path, output_image_bgr)
    
    print(f"\n--- モデル適用完了 ---")
    print(f"生成された画質改善画像の総数: {len(os.listdir(args.output_dir))}枚") # 保存されたファイル数をカウント


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みモデルを画像に適用し、画質改善を行うスクリプト。")
    parser.add_argument("model_path", type=str,
                        help="ロードする学習済みモデル (.pth) ファイルのパス。")
    parser.add_argument("input_dir", type=str,
                        help="モデルを適用する入力画像が保存されているディレクトリのパス。")
    parser.add_argument("output_dir", type=str,
                        help="画質改善された画像を保存するディレクトリのパス。")
    parser.add_argument("--image_height", type=int, required=True,
                        help="モデルへの入力画像高さ (学習時と同じ高さ)。")
    parser.add_argument("--image_width", type=int, required=True,
                        help="モデルへの入力画像幅 (学習時と同じ幅)。")
    parser.add_argument("--features", type=str, required=True,
                        help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("--use_se_block", action="store_true",
                        help="Squeeze-and-Excitation (SE) Blockを使用した場合、このフラグを設定。")
    parser.add_argument("--use_cbam", action="store_true",
                        help="use_cbam")
    
    args = parser.parse_args()
    
    apply_model(args)