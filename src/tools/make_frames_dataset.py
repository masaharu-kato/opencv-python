from pathlib import Path
from attr import dataclass
import cv2
from cv2.typing import MatLike
import numpy as np
import argparse
from tqdm import tqdm

from dump_frame_pairs import load_frame_pairs

def align_images_with_optical_flow_rgba(img_bad_bgr: MatLike, img_good_bgr: MatLike):
    """
    OpenCVのFarnebackオプティカルフローを使って画像をアラインし, RGBA形式で返す. 
    img_g_color (良い画像) を img_a_color (悪い画像) にアラインする.

    """
        # アルファチャンネルを追加してRGBAに変換
    # img_g_color のみをワープするので、img_g_rgba を作成
    img_good_alpha = np.full(img_good_bgr.shape[:2], 255, dtype=np.uint8)
    img_good_rgba = cv2.merge([img_good_bgr[:,:,0], img_good_bgr[:,:,1], img_good_bgr[:,:,2], img_good_alpha])

    # オプティカルフローはグレースケール画像で計算
    img_bad_gray = cv2.cvtColor(img_bad_bgr, cv2.COLOR_BGR2GRAY)
    img_good_gray = cv2.cvtColor(img_good_bgr, cv2.COLOR_BGR2GRAY)
    
    gridsize = max(int(min(img_bad_gray.shape) * 0.015) // 2 * 2, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize)) 

    img_bad_gray = clahe.apply(img_bad_gray)
    img_good_gray = clahe.apply(img_good_gray)

    # winsizeをimg_a_grayの短辺のwinsizerateパーセントで最も近い奇数に設定
    winsizerate = 0.05
    winsize = max(int(min(img_bad_gray.shape) * winsizerate) // 2 * 2 + 1, 3)

    # Farneback法でオプティカルフローを計算
    flow = cv2.calcOpticalFlowFarneback(
        prev=img_bad_gray,
        next=img_good_gray, 
        flow=None, # type: ignore
        pyr_scale=0.5, # small to capture dynamic movement
        levels=5, # increase to capture dynamic movement
        winsize=winsize, # increase to smooth, decrease to detail/local
        iterations=5, # increases result precision
        poly_n=5, # 5 or 7 (for complexity of movement)
        poly_sigma=1.1, # 1.1 if poly_n=5, 1.5 if poly_n=7
        flags=0 # cv2.OPTFLOW_FARNEBACK_GAUSSIAN # set cv2.OPTFLOW_FARNEBACK_GAUSSIAN to increase quality
    )

    h, w = img_bad_gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + flow[:,:,0]).astype(np.float32)
    map_y = (y_coords + flow[:,:,1]).astype(np.float32)

    # cv2.remap を使って img_g_rgba を img_a_color にアライン
    aligned_good_rgba = cv2.remap(src=img_good_rgba, 
                               map1=map_x, 
                               map2=map_y, 
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0)) # 完全に透明な黒

    # img_a_color もRGBAに変換して返す (アルファチャンネルは全て不透明)
    bad_alpha = np.full(img_bad_bgr.shape[:2], 255, dtype=np.uint8)
    img_bad_rgba = cv2.merge([img_bad_bgr[:,:,0], img_bad_bgr[:,:,1], img_bad_bgr[:,:,2], bad_alpha])

    return img_bad_rgba, aligned_good_rgba


@dataclass
class _FrameImagePair:
    ig: int
    ib: int
    good_frame: MatLike | None = None
    bad_frame: MatLike | None = None

def process_dataset(frame_pairs_path: Path, video_path: Path, output_dir: Path, image_width: int | None = None, image_height: int | None = None):

    frame_index_pairs = load_frame_pairs(frame_pairs_path)
    frame_image_pairs: list[_FrameImagePair | None] = [_FrameImagePair(ig, ib, None, None) for ig, ib in frame_index_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_image_pair(pair: _FrameImagePair):

        if pair.bad_frame is None or pair.good_frame is None:
            raise RuntimeError("Bad/Good frame is not set.")

        img_b_rgba, aligned_g_rgba = align_images_with_optical_flow_rgba(pair.bad_frame, pair.good_frame)

        if img_b_rgba is None or aligned_g_rgba is None:
            raise RuntimeError(f"Failed to process alignment.")

        cv2.imwrite(str(output_dir / f"ig{pair.ig:06d}_ib{pair.ib:06d}_g.png"), aligned_g_rgba)
        cv2.imwrite(str(output_dir / f"ig{pair.ig:06d}_ib{pair.ib:06d}_b.png"), img_b_rgba)

        pair.good_frame = None
        pair.bad_frame = None


    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_image_width = image_width if image_width else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_image_height = image_height if image_height else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    iframe_ref_ipairs: list[set[int]] = [set() for _ in range(total_frames)]
    for ipair, pair in enumerate(frame_image_pairs):
        assert pair is not None
        if pair.ig < 0 or pair.ig >= total_frames or pair.ib < 0 or pair.ib >= total_frames:
            raise RuntimeError(f"Invalid frame index on pair ({pair.ig}, {pair.ib})")
        iframe_ref_ipairs[pair.ig].add(ipair)
        iframe_ref_ipairs[pair.ib].add(ipair)

    last_iframe = -1
    for iframe, ref_ipairs in enumerate(tqdm(iframe_ref_ipairs)):
        
        if not ref_ipairs:
            continue

        if last_iframe + 1 != iframe:
            cap.set(cv2.CAP_PROP_POS_FRAMES, iframe)

        ret, frame = cap.read()
        last_iframe = iframe

        if not ret:
            raise RuntimeError(f"Failed to load frame {iframe}")
        for ipair in ref_ipairs:
            pair = frame_image_pairs[ipair]
            if pair is not None:
                if pair.ib == iframe and pair.bad_frame is None:
                    pair.bad_frame = cv2.resize(frame, (out_image_width, out_image_height))
                if pair.ig == iframe and pair.good_frame is None:
                    pair.good_frame = cv2.resize(frame, (out_image_width, out_image_height))
                if pair.bad_frame is not None and pair.good_frame is not None:
                    write_image_pair(pair)
                    frame_image_pairs[ipair] = None
                
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load video and align and save frame images dataset in RGBA format.")
    parser.add_argument("frame_pairs_path", type=str, help="Path to the frame pairs text file")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("output_dir", type=str, help="Path to the output processed dataset directory")
    parser.add_argument("-imgw", "--image-width", type=int)
    parser.add_argument("-imgh", "--image-height", type=int)
    args = parser.parse_args()

    process_dataset(Path(args.frame_pairs_path), Path(args.video_path), Path(args.output_dir),
                    int(args.image_width) if args.image_width else None, int(args.image_height) if args.image_height else None)
