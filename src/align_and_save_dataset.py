from pathlib import Path
import cv2
import numpy as np
import argparse
import re
from tqdm import tqdm

def align_images_with_optical_flow_rgba(img_a_path, img_g_path):
    """
    OpenCVのFarnebackオプティカルフローを使って画像をアラインし、RGBA形式で返す。
    img_g (良い画像) を img_a (悪い画像) にアラインする。
    """
    img_a_color = cv2.imread(img_a_path)
    img_g_color = cv2.imread(img_g_path)

    if img_a_color is None or img_g_color is None:
        print(f"Error: Could not read images {img_a_path} or {img_g_path}. Skipping.")
        return None, None
    

    # アルファチャンネルを追加してRGBAに変換
    # img_g_color のみをワープするので、img_g_rgba を作成
    alpha_channel_g = np.full(img_g_color.shape[:2], 255, dtype=np.uint8)
    img_g_rgba = cv2.merge([img_g_color[:,:,0], img_g_color[:,:,1], img_g_color[:,:,2], alpha_channel_g])

    # オプティカルフローはグレースケール画像で計算
    img_a_gray = cv2.cvtColor(img_a_color, cv2.COLOR_BGR2GRAY)
    img_g_gray = cv2.cvtColor(img_g_color, cv2.COLOR_BGR2GRAY)
    
    _tgs = max(int(min(img_a_gray.shape) * 0.015) // 2 * 2, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(_tgs, _tgs)) 

    img_a_gray = clahe.apply(img_a_gray)
    img_g_gray = clahe.apply(img_g_gray)

    # winsizeをimg_a_grayの短辺のwinsizerateパーセントで最も近い奇数に設定
    winsizerate = 0.05
    winsize = max(int(min(img_a_gray.shape) * winsizerate) // 2 * 2 + 1, 3)

    # Farneback法でオプティカルフローを計算
    flow = cv2.calcOpticalFlowFarneback(
        prev=img_a_gray,
        next=img_g_gray, 
        flow=None, # type: ignore
        pyr_scale=0.5, # small to capture dynamic movement
        levels=5, # increase to capture dynamic movement
        winsize=winsize, # increase to smooth, decrease to detail/local
        iterations=5, # increases result precision
        poly_n=5, # 5 or 7 (for complexity of movement)
        poly_sigma=1.1, # 1.1 if poly_n=5, 1.5 if poly_n=7
        flags=0 # cv2.OPTFLOW_FARNEBACK_GAUSSIAN # set cv2.OPTFLOW_FARNEBACK_GAUSSIAN to increase quality
    )

    h, w = img_a_gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + flow[:,:,0]).astype(np.float32)
    map_y = (y_coords + flow[:,:,1]).astype(np.float32)

    # cv2.remap を使って img_g_rgba を img_a_color にアライン
    aligned_g_rgba = cv2.remap(src=img_g_rgba, 
                               map1=map_x, 
                               map2=map_y, 
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0)) # 完全に透明な黒

    # img_a_color もRGBAに変換して返す (アルファチャンネルは全て不透明)
    alpha_channel_a = np.full(img_a_color.shape[:2], 255, dtype=np.uint8)
    img_a_rgba = cv2.merge([img_a_color[:,:,0], img_a_color[:,:,1], img_a_color[:,:,2], alpha_channel_a])

    return img_a_rgba, aligned_g_rgba


def process_dataset(indir: Path, outdir: Path):
    outdir.mkdir(exist_ok=True)
    
    # regex for frame files: frame_<frame_number>g.png or frame_<frame_number>.png
    namepat = re.compile(r'frame_(\d+)(g)?\.png')

    vdirs = sorted(d for d in indir.iterdir() if d.is_dir())
    vfpaths: list[dict[int, Path]] = [] # list(video)[frame index -> frame path]

    # 各動画ディレクトリを走査
    vfgroups: list[list[list[int]]] = [] # list(video)[list(pair)[tuple[igood, ibad1, ibad2, ...]]]
    haserror = False
    for vdir in vdirs:
        vfgroups.append([])
        vfpaths.append({})

        print(f"Checking good/bad frame pairs on {vdir}")

        lasti = -2
        for fpath in sorted(vdir.iterdir()):
            if not (m := namepat.match(fpath.name)):
                continue

            i = int(m.group(1))
            isgood = m.group(2) == 'g'

            vfpaths[-1][i] = fpath

            if lasti + 1 != i:
                vfgroups[-1].append([-1])
            lasti = i

            if isgood:
                if vfgroups[-1][-1][0] != -1:
                    print(f"Dataset Error: Multiple good frames on frame {i}")
                    haserror = True
                vfgroups[-1][-1][0] = i
            else:
                vfgroups[-1][-1].append(i)

        for ig, *ibs in vfgroups[-1]:
            if ig == -1:
                print(f"Dataset Error: No good frame for bad frame(s) {', '.join(map(str, ibs))}")
            if not len(ibs):
                print(f"Dataset Error: No bad frames for good frame {ig}")

        if haserror:
            raise RuntimeError("Dataset has error(s).")


    for vdir, fpaths, fgroups in zip(vdirs, vfpaths, vfgroups):
        # 各グループに対してアラインメントと保存を実行
        
        (outvdir := outdir / vdir.name).mkdir(exist_ok=True)

        print(f"Processing image pairs on {vdir.name} to {outvdir}")

        # すべての (ig, ib) ペアをリスト化
        pairs = [(ig, ib) for ig, *ibs in fgroups for ib in ibs]

        for ig, ib in tqdm(pairs, desc=f"Aligning {vdir.name}"):
                
                # アラインメントを実行 (img_a_rgba は img_b_rgba に相当)
                img_b_rgba, aligned_g_rgba = align_images_with_optical_flow_rgba(fpaths[ib], fpaths[ig])

                if img_b_rgba is None or aligned_g_rgba is None:
                    raise RuntimeError(f"  Skipping pair due to error: ib:{ib}, ig:{ig}")

                cv2.imwrite(str(outvdir / f"ig{ig:06d}_ib{ib:06d}_g.png"), aligned_g_rgba)
                cv2.imwrite(str(outvdir / f"ig{ig:06d}_ib{ib:06d}_b.png"), img_b_rgba)
                
        print(f"Finished processing video: {vdir.name}.")
    print("Dataset processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and save image dataset in RGBA format.")
    parser.add_argument("input_dir", type=str, help="Path to the input dataset directory (e.g., /mnt/d/res/unet_datasets)")
    parser.add_argument("output_dir", type=str, help="Path to the output processed dataset directory (e.g., processed_datasets)")
    args = parser.parse_args()

    process_dataset(Path(args.input_dir), Path(args.output_dir))
