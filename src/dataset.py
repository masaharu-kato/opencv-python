# dataset.py (修正版)
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RelativeImagePairDataset(Dataset):
    def __init__(self, rootdir: Path | str, imgsize: tuple[int, int], use_cj: bool):
        self.rootdir = Path(rootdir)
        self.imgsize = imgsize
        self.pairs: list[tuple[Path, Path]] = [] # (degraded_path, clean_path) のタプルを格納

        # データ拡張（ランダムクロップ、フリップなど）
        # 学習時に適用することで、モデルの汎化能力を高めます
        self.transform = transforms.Compose([
            *([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] if use_cj else []),
            transforms.Resize(imgsize), # Resize
            transforms.ToTensor(), # PIL Image to Tensor (0-1 range)
        ])

        self._load_pairs()

    def _load_pairs(self):
        print(f"Loading dataset from: {self.rootdir}")

        # root_dir の直下に動画名ディレクトリがある場合を想定
        # unet_datasets/<動画ファイル名>/frame_xxx.png
        vdirs = sorted(d for d in self.rootdir.iterdir() if d.is_dir())
        
        for vname in vdirs:

            bad: Path | None = None

            for fpath in sorted(vname.glob('*.png')):
                assert '_g.' in fpath.name if bad is not None else '_b.' in fpath.name
                
                if bad is None:
                    bad = fpath
                else:
                    self.pairs.append((bad, fpath))
                    bad = None

        assert self.pairs

        print(f"Loaded {len(self.pairs)} image pairs.")


    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        bpath, gpath = self.pairs[idx]

        bimg = Image.open(bpath).convert("RGB")
        gimg_rgba = Image.open(gpath).convert("RGBA")
        gimg = gimg_rgba.convert("RGB")
        # アルファチャンネルを抽出
        gimg_alpha = gimg_rgba.getchannel("A")
        
        btensor = self.transform(bimg)
        gtensor = self.transform(gimg)
        gmasktensor = self.transform(gimg_alpha)

        return btensor, gtensor, gmasktensor
