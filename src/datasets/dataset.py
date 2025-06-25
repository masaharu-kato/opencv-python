import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import IO
import torch.utils.data
from torchvision import transforms
from PIL import Image

from datasets.path_pair_groups import PathPairGroups
from models.unet import ModelOptions

@dataclass
class DatasetOptions:
    """Options for the dataset."""
    load_all_to_ram: bool = True

class ImageCache:
    """A simple cache for images to avoid reloading them multiple times."""
    def __init__(self):
        self.cache = {}

    def get(self, path: Path) -> Image.Image:
        if path not in self.cache:
            img = Image.open(path).convert("RGBA")
            self.cache[path] = img
        return self.cache[path]

cache = ImageCache()


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, ppair_groups: PathPairGroups, opts: DatasetOptions, model_opts: ModelOptions):
        self.opts = opts
        self.model_opts = model_opts
        self.ppair_groups = ppair_groups
        self.path_pairs = list(itertools.chain.from_iterable(self.ppair_groups))

        # データ拡張（ランダムクロップ、フリップなど）
        # 学習時に適用することで、モデルの汎化能力を高めます
        self.transform = transforms.Compose([
            # *([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] if use_cj else []),
            transforms.Resize(model_opts.input_size()), # Resize
            transforms.ToTensor(), # PIL Image to Tensor (0-1 range)
        ])

    def split(self, ratio: float) -> tuple['ImagePairDataset', 'ImagePairDataset']:
        """Splits the dataset into two datasets based on the given ratio."""
        return self._split(self.ppair_groups, ratio)
    
    def random_split(self, ratio: float) -> tuple['ImagePairDataset', 'ImagePairDataset']:
        """Randomly splits the dataset into two datasets based on the given ratio."""
        groups = self.ppair_groups.copy_shuffled()  # Shuffle the groups before splitting
        return self._split(groups, ratio)

    def _split(self, groups: PathPairGroups, ratio: float) -> tuple['ImagePairDataset', 'ImagePairDataset']:
        """Splits into two datasets baased on the image groups"""
        if len(groups) == 0:
            raise ValueError("Empty dataset, cannot split.")
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between 0 and 1.")
        if len(groups) == 1:
            if ratio == 0:
                return self.clone(self.ppair_groups), self.clone(PathPairGroups())  # Return the original dataset and an empty one
            elif ratio == 1:
                return self.clone(PathPairGroups()), self.clone(self.ppair_groups)  # Return an empty dataset and the original one
            raise ValueError("Cannot split a single group dataset without 0 or 1 ratio.")
        
        split_index = min(max(1, int(len(groups) * ratio)), len(groups) - 1)  # Ensure at least one group in each split
        group1, group2 = groups.split(split_index)
        return self.clone(group1), self.clone(group2)

    
    def clone(self, groups: PathPairGroups) -> 'ImagePairDataset':
        """Creates a new dataset with the same options but different path groups."""
        return ImagePairDataset(groups, self.opts, self.model_opts)

    def __len__(self) -> int:
        return len(self.path_pairs)

    def __getitem__(self, idx: int):

        ppair= self.path_pairs[idx]
        bimg, gimg = cache.get(ppair.bpath), cache.get(ppair.gpath)

        bimg_rgb = bimg.convert("RGB")
        gimg_rgb = gimg.convert("RGB")
        # アルファチャンネルを抽出
        gimg_alpha = gimg.getchannel("A")
        
        btensor = self.transform(bimg_rgb)
        gtensor = self.transform(gimg_rgb)
        gmasktensor = self.transform(gimg_alpha)

        return btensor, gtensor, gmasktensor, idx

    
    # def _load_image_pair(self, idx: int) -> tuple[Image.Image, Image.Image]:
    #     bpath, gpath = self.path_pairs[idx]

    #     bimg = Image.open(bpath).convert("RGB")
    #     gimg= Image.open(gpath).convert("RGBA")
    #     return bimg, gimg
    
    # def _load_all_image_pairs(self):
    #     if self.img_pairs is None:
    #         self.img_pairs = []
    #         for bpath, gpath in self.path_pairs:
    #             bimg = Image.open(bpath).convert("RGB")
    #             gimg = Image.open(gpath).convert("RGBA")
    #             self.img_pairs.append((bimg, gimg))
    #         logging.info(f"Loaded all {len(self.img_pairs)} image pairs into memory.")

    # def _release_image_pairs(self):
    #     if self.img_pairs is not None:
    #         self.img_pairs = None
