import itertools
from dataclasses import dataclass
from pathlib import Path
import torch.utils.data
from torchvision import transforms
from PIL import Image

from datasets.path_pair_groups import PathPairGroups
from models.unet import ModelOptions

@dataclass
class DatasetOptions:
    """Options for the dataset."""
    # load_all_to_ram: bool = True

class ImageCache:
    """A simple cache for images to avoid reloading them multiple times."""
    def __init__(self):
        self.cache: dict[tuple[Path, tuple[int, int]], tuple[Image.Image, Image.Image]] = {} # ((path, (width, height)) -> Image)

    def get(self, path: Path, imgsize: tuple[int, int]) -> tuple[Image.Image, Image.Image]: # (RGB, A)
        if (path, imgsize) not in self.cache:
            img = Image.open(path).convert("RGBA").resize(imgsize)
            img.load()  # Load the image data into memory
            self.cache[path, imgsize] = (img.convert("RGB"), img.getchannel("A"))  # Store RGB and alpha channel separately
        return self.cache[path, imgsize]

cache = ImageCache()


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, ppair_groups: PathPairGroups, opts: DatasetOptions, model_opts: ModelOptions):
        self.opts = opts
        self.model_opts = model_opts
        self.ppair_groups = ppair_groups
        self.path_pairs = list(itertools.chain.from_iterable(self.ppair_groups))

        self.transform = transforms.Compose([
            # transforms.Resize(self.input_size), # Already resized in cache
            transforms.ToTensor(), # PIL Image to Tensor (0-1 range)
        ])

    @property
    def input_size(self) -> tuple[int, int]:
        """Returns the image size as (height, width)."""
        return self.model_opts.input_size
    
    def clone(self, groups: PathPairGroups) -> 'ImagePairDataset':
        """Creates a new dataset with the same options but different path groups."""
        return ImagePairDataset(groups, self.opts, self.model_opts)

    def __len__(self) -> int:
        return len(self.path_pairs)

    def __getitem__(self, idx: int):

        ppair= self.path_pairs[idx]
        (bimg_rgb, _), (gimg_rgb, gimg_a) = cache.get(ppair.bpath, self.input_size), cache.get(ppair.gpath, self.input_size)

        btensor = self.transform(bimg_rgb)
        gtensor = self.transform(gimg_rgb)
        gmasktensor = self.transform(gimg_a)

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
