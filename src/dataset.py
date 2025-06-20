# dataset.py (修正版)
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tf_functional
from PIL import Image
import os
import random
import re # 正規表現モジュールをインポート

class RelativeImagePairDataset(Dataset):
    def __init__(self, root_dir, image_height, image_width):
        self.root_dir = root_dir
        self.image_height = image_height
        self.image_width = image_width
        self.pairs = [] # (degraded_path, clean_path) のタプルを格納

        # データ拡張（ランダムクロップ、フリップなど）
        # 学習時に適用することで、モデルの汎化能力を高めます
        self.transform = transforms.Compose([
            transforms.ToTensor(), # PIL Image to Tensor (0-1 range)
        ])

        self._load_pairs()

    def _load_pairs(self):
        print(f"Loading dataset from: {self.root_dir}")
        
        # root_dir の直下に動画名ディレクトリがある場合を想定
        # unet_datasets/<動画ファイル名>/frame_xxx.png
        video_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for video_dir_name in sorted(video_dirs):
            video_path = os.path.join(self.root_dir, video_dir_name)
            
            # 各動画ディレクトリ内のフレームを走査
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])
            
            # フレーム番号を抽出するための正規表現
            # frame_001234.png -> 1234
            # frame_001234g.png -> 1234
            frame_num_pattern = re.compile(r'frame_(\d+)(g)?\.png')

            # 各組の「良いフレーム」を特定するための辞書
            # キー: 組の代表フレーム番号（例えば、組内で最小のフレーム番号）
            # 値: その組で最も良いとされるフレームのパス (frame_XXXXg.png)
            current_good_frame_path = None
            current_group_start_frame_num = -1
            prev_frame_num = -1

            for i, frame_name in enumerate(frame_files):
                match = frame_num_pattern.match(frame_name)
                if not match:
                    continue
                
                frame_num = int(match.group(1))
                is_good_frame = (match.group(2) == 'g') # 'g' が付いているか

                full_frame_path = os.path.join(video_path, frame_name)

                if i == 0: # 最初のフレームの処理
                    current_group_start_frame_num = frame_num
                    if is_good_frame:
                        current_good_frame_path = full_frame_path
                    else:
                        current_good_frame_path = None # まだ良いフレームが見つかっていない
                else:
                    # フレーム番号が連続しない場合、新しい組の始まりと見なす
                    # もしくは、現在処理中の組の"good"フレームを特定できた場合
                    # （ここでは、組内の最初の"g"フレームをその組のgoodフレームと仮定）
                    if frame_num != (prev_frame_num + 1):
                        # 新しい組が始まる
                        current_group_start_frame_num = frame_num
                        current_good_frame_path = None # 新しい組なのでリセット
                        
                    if is_good_frame:
                        # このフレームが現在の組の"good"フレーム
                        # 同じ組に複数の'g'があった場合の挙動を定義する必要あり
                        # 例：最初に発見した'g'をその組のgoodフレームとする
                        if current_good_frame_path is None:
                            current_good_frame_path = full_frame_path
                
                # 'g'がついていないフレームは、現在の組の"good"フレームとペアにする
                if not is_good_frame and current_good_frame_path is not None:
                    self.pairs.append((full_frame_path, current_good_frame_path))
                
                prev_frame_num = frame_num # 次のフレーム比較のために現在のフレーム番号を保存

        if not self.pairs:
            print("警告: ペアとなる画像が見つかりませんでした。データセットのパスと命名規則を確認してください。")
            print(f"期待される形式: <フレーム番号>.png と <フレーム番号>g.png")

        print(f"Loaded {len(self.pairs)} image pairs.")


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        degraded_path, clean_path = self.pairs[idx]

        degraded_image = Image.open(degraded_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        # # ランダムクロップとランダムフリップを適用 (以前のコードと同じロジック)
        # # 画像サイズが540x720なので、クロップは不要でそのままリサイズとして扱えるか、
        # # もしくは、元画像がそれより大きい場合のみクロップを適用
        # # ここでは、image_height/widthが最終的な出力サイズとして固定されていると仮定
        if degraded_image.size != (self.image_width, self.image_height):
            # リサイズが必要な場合
            degraded_image = tf_functional.resize(degraded_image, (self.image_height, self.image_width)) # type: ignore
            clean_image = tf_functional.resize(clean_image, (self.image_height, self.image_width)) # type: ignore

        # # RandomHorizontalFlip and RandomVerticalFlip
        # if random.random() > 0.5:
        #     degraded_image = tf_functional.hflip(degraded_image)
        #     clean_image = tf_functional.hflip(clean_image)
        # if random.random() > 0.5:
        #     degraded_image = tf_functional.vflip(degraded_image)
        #     clean_image = tf_functional.vflip(clean_image)
        
        degraded_tensor = self.transform(degraded_image)
        clean_tensor = self.transform(clean_image)

        return degraded_tensor, clean_tensor
