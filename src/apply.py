import logging
from pathlib import Path
import torch
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.unet import UNet

# --- メインの推論関数 ---
def apply_model(model_path: Path, input_dir: Path, output_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    
    try:
        model, cp = UNet.load(model_path, device)

    except Exception as e:
        raise RuntimeError(f"An error has occured while model loading: {e}") from e
        

    model.eval() # Evaluation mode

    output_dir.mkdir(exist_ok=True)

    # transform = transforms.Compose([
    #     transforms.Resize((image_height, image_width)),
    #     transforms.ToTensor(),
    # ])

    image_files = sorted(input_dir.glob("*.png"))
    
    if not image_files:
        logging.info(f"No images found in {input_dir}")
        return

    logging.info(f"\n--- Model application start ---")
    logging.info(f"Detected {len(image_files)} images.")

    # モデルが要求する入力サイズの倍数 (U-Netの層数に応じて変更)
    # 例: 3層U-Netなら 2^3=8
    # コマンドライン引数に追加するか、args.image_width/heightから逆算するなど
    model_multiple = 2 ** len(cp['features']) # features_listの数で層数を判断 (近似)
                                          # あるいは args.model_multiple を追加

    for img_path in tqdm(image_files, desc="Processing"):
        
        input_image_bgr = cv2.imread(str(img_path))
        if input_image_bgr is None:
            logging.info(f"Warning: Failed to load {img_path}")
            continue
        
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        
        input_image_pil = Image.fromarray(input_image_rgb)
        
        # --- ここからパディング処理の追加 ---
        # original_width, original_height = input_image_pil.size
        # padded_input_pil, padding_coords = pad_to_multiple(input_image_pil, model_multiple, fill_value=(0,0,0))
        padded_input_pil = input_image_pil.resize(cp['in_size'])
        
        # Convert to a tensor
        input_tensor = transforms.ToTensor()(padded_input_pil).unsqueeze(0).to(device)

        with torch.no_grad(): # 勾配計算を無効化
            output_tensor = model(input_tensor)

        # モデルの出力が [-1, 1] の場合、[0, 1] に変換
        # もしGeneratorの最終層がSigmoidなら不要ですが、Tanhなら必要です
        if cp.get('model_output_range') == "minus1_to_1":
            output_tensor = (output_tensor + 1.0) / 2.0
        
        # 出力テンソルを画像に変換し、パディングをトリミング
        output_image_np = (output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)) # まだ0-1のfloat
        
        # パディング部分をトリミング
        # p_left, p_top, p_right, p_bottom = padding_coords
        
        # output_image_np は HWC (Height, Width, Channels)
        # トリミング範囲: [top:bottom, left:right]
        # trimmed_output_np = output_image_np[p_top:original_height + p_top, p_left:original_width + p_left, :]
        trimmed_output_np = output_image_np
        
        # 0-1 から 0-255 に変換し、UINT8型に
        trimmed_output_np = (trimmed_output_np * 255).astype(np.uint8)

        output_image_bgr = cv2.cvtColor(trimmed_output_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_dir / f"{img_path.stem}_enhanced.png"), output_image_bgr)
    
    logging.info(f"\n--- Model application completed ---")
    logging.info(f"Generated {len(os.listdir(output_dir))} images")


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みモデルを画像に適用し、画質改善を行うスクリプト。")
    parser.add_argument("model_path", type=str,
                        help="ロードする学習済みモデル (.pth) ファイルのパス。")
    parser.add_argument("input_dir", type=str,
                        help="モデルを適用する入力画像が保存されているディレクトリのパス。")
    parser.add_argument("output_dir", type=str,
                        help="画質改善された画像を保存するディレクトリのパス。")
    
    args = parser.parse_args()
    apply_model(Path(args.model_path), Path(args.input_dir), Path(args.output_dir))
