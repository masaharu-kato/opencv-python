import logging
from dataclasses import dataclass
from pathlib import Path
import torch.utils.data
from torchvision import transforms
from PIL import Image

from models.unet import ModelOptions

@dataclass
class DatasetOptions:
    dataset_dir: Path
    load_all_to_ram: bool = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opts: DatasetOptions, model_opts: ModelOptions):
        self.opts = opts
        self.model_opts = model_opts
        self.relpath_pairs: list[tuple[str, str]] = []
        self.img_pairs: list[tuple[Image.Image, Image.Image]] | None = None

        self._make_pairs(opts.dataset_dir)

        if opts.load_all_to_ram:
            self._load_all_image_pairs()

        # データ拡張（ランダムクロップ、フリップなど）
        # 学習時に適用することで、モデルの汎化能力を高めます
        self.transform = transforms.Compose([
            # *([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] if use_cj else []),
            transforms.Resize(model_opts.input_size()), # Resize
            transforms.ToTensor(), # PIL Image to Tensor (0-1 range)
        ])

    def __len__(self) -> int:
        return len(self.relpath_pairs)

    def __getitem__(self, idx):

        if self.img_pairs is not None:
            bimg, gimg = self.img_pairs[idx]
        else:
            bimg, gimg = self._load_image_pair(idx)

        gimg_rgb = gimg.convert("RGB")
        # アルファチャンネルを抽出
        gimg_alpha = gimg.getchannel("A")
        
        btensor = self.transform(bimg)
        gtensor = self.transform(gimg_rgb)
        gmasktensor = self.transform(gimg_alpha)

        return btensor, gtensor, gmasktensor, idx
    
    def _load_image_pair(self, idx: int) -> tuple[Image.Image, Image.Image]:
        brelpath, grelpath = self.relpath_pairs[idx]
        bpath, gpath = self.root_dir / brelpath, self.root_dir / grelpath

        bimg = Image.open(bpath).convert("RGB")
        gimg= Image.open(gpath).convert("RGBA")
        return bimg, gimg
    
    def _load_all_image_pairs(self):
        if self.img_pairs is None:
            self.img_pairs = []
            for brelpath, grelpath in self.relpath_pairs:
                bpath, gpath = self.root_dir / brelpath, self.root_dir / grelpath
                bimg = Image.open(bpath).convert("RGB")
                gimg = Image.open(gpath).convert("RGBA")
                self.img_pairs.append((bimg, gimg))
            logging.info(f"Loaded all {len(self.img_pairs)} image pairs into memory.")

    def _release_image_pairs(self):
        if self.img_pairs is not None:
            self.img_pairs = None

    def _make_pairs(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        logging.info(f"Loading dataset from: {root_dir}")

        vdirs = sorted(d for d in self.root_dir.iterdir() if d.is_dir())
        
        for vname in vdirs:

            bad: Path | None = None

            for fpath in sorted(vname.glob('*.png')):
                assert '_g.' in fpath.name if bad is not None else '_b.' in fpath.name
                
                if bad is None:
                    bad = fpath
                else:
                    self.relpath_pairs.append((str(bad.relative_to(root_dir)), str(fpath.relative_to(root_dir))))
                    bad = None

        if not self.relpath_pairs:
            raise RuntimeError(f"No image pairs found in {root_dir}. Please check the directory structure.")

        logging.info(f"Detected {len(self.relpath_pairs)} image pairs.")


