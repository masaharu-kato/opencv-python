import logging
from pathlib import Path
import torch
import cv2
from cv2.typing import MatLike
import numpy as np
import os
import argparse
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.unet import UNet


class ModelApplyer:
    """A class to apply a pre-trained model to images for enhancement."""

    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_path}")
        
        try:
            model, cp = UNet.load(model_path, self.device, {})
            model.load_state_dict(model.state_dict())

        except Exception as e:
            raise RuntimeError(f"An error has occured while model loading: {e}") from e
        
        model.eval() # Evaluation mode
        self.model = model
        self.cp = cp

        # transform = transforms.Compose([
        #     transforms.Resize((image_height, image_width)),
        #     transforms.ToTensor(),
        # ])

    def apply(self, input_image_bgr: MatLike) -> MatLike:
        
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        input_image_pil = Image.fromarray(input_image_rgb)
        input_tensor = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad(): # 勾配計算を無効化
            output_tensor = self.model(input_tensor)

        if self.model.opts.model_output_range == "minus1_to_1":
            output_tensor = (output_tensor + 1.0) / 2.0
        
        output_image_np = (output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)) # 0-1 range, HWC format
        output_np = (output_image_np * 255).astype(np.uint8)
        output_image_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        return output_image_bgr


def apply_model_to_images(model_path: Path, input_dir: Path, output_dir: Path, image_size: tuple[int, int] | None = None):
    """Apply the pre-trained model to all images in the input directory and save the enhanced images to the output directory."""

    applyer = ModelApplyer(model_path)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Applying model {model_path} to images in {input_dir} ...")

    for img_path in tqdm([f for f in sorted(input_dir.glob("*.png")) if f.is_file()], desc="Processing images"):

        input_image_bgr = cv2.imread(str(img_path))
        if input_image_bgr is None:
            logging.warning(f"Failed to load {img_path}")
            continue
        if image_size:
            input_image_bgr = cv2.resize(input_image_bgr, image_size)
        
        output_image_bgr = applyer.apply(input_image_bgr)
        
        cv2.imwrite(str(output_dir / f"{img_path.stem}_e.png"), output_image_bgr)

    logging.info(f"\n--- Model application completed ---")
    logging.info(f"Generated {len(os.listdir(output_dir))} images")


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みモデルを画像に適用し、画質改善を行うスクリプト。")
    parser.add_argument("model_path", type=str, help="Model file (*.pth) to load.")
    parser.add_argument("input_dir", type=str, help="Input directory containing images to enhance.")
    parser.add_argument("output_dir", type=str, help="Output directory to save enhanced images.")
    parser.add_argument("-imgw", "--image_width", type=int, help="Input/output image width (default: same as model input width).")
    parser.add_argument("-imgh", "--image_height", type=int, help="Input/output image height (default: same as model input height).")
    
    args = parser.parse_args()
    apply_model_to_images(
        Path(args.model_path),
        Path(args.input_dir),
        Path(args.output_dir),
        (args.image_width, args.image_height) if args.image_height and args.image_width else None
    )
