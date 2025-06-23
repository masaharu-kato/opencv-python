import argparse
import os
from dataclasses import dataclass
import cv2
import numpy as np
import pickle
from tqdm import tqdm

@dataclass
class ImageQualityMetrics:
    # laplacian_var = float('nan')
    hist_std_dev = float('nan')
    hist_entropy = float('nan')
    # black_clip_ratio = float('nan')
    # white_clip_ratio = float('nan')


# --- 画像品質指標の計算関数 ---
def calculate_image_quality_metrics(img_bgr) -> ImageQualityMetrics:
    """
    画像ファイルパスまたはOpenCVフレームから品質指標を計算する。
    """

    qmets = ImageQualityMetrics()

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # # ラプラシアン分散 (シャープネス/ブレ)
    # # ゼロ除算防止のため、画像が全て同じ値の場合の例外処理を追加
    # laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    # qmets.laplacian_var = laplacian.var() if laplacian.size > 0 else 0.0

    # ヒストグラム計算
    hist, bins = np.histogram(img_gray.flatten(), 256, (0, 256))

    # ヒストグラムの標準偏差
    # histが全て0の場合の例外処理
    qmets.hist_std_dev = float(np.std(hist)) if np.sum(hist) > 0 else 0.0

    # ヒストグラムのエントロピー
    hist_prob = hist / (np.sum(hist) + 1e-10) # 0除算防止
    # log2(0)を避ける
    qmets.hist_entropy = -np.sum(hist_prob * np.log2(hist_prob + 1e-10)) if np.sum(hist) > 0 else 0.0

    # # 白飛び/黒潰れピクセル比率
    # total_pixels = img_gray.size + 1e-10 # 0除算防止
    # black_pixels = np.sum(img_gray < 10)
    # white_pixels = np.sum(img_gray > 245) # 245-255を白飛びとする
    
    # qmets.black_clip_ratio = black_pixels / total_pixels
    # qmets.white_clip_ratio = white_pixels / total_pixels

    return qmets



def calculate_video_image_quality_metrics(video_path: str, pickle_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_metrics: list[ImageQualityMetrics] = []
    
    for i in tqdm(range(total_frames), desc="Processing frames..."):
        ret, frame = cap.read()
        if not ret:
            break
        all_metrics.append(calculate_image_quality_metrics(frame))
        
    cap.release()

    with open(pickle_path, mode='wb') as fw:
        pickle.dump(all_metrics, fw)

    print(f"Generated: {pickle_path}")


# --- メインの処理関数 ---
def detect_and_extract_quality_changes(video_path: str, output_dir: str | None,
                                    #    lap_var_threshold, # ラプラシアン分散の変化閾値
                                       hist_std_dev_threshold, # ヒストグラム標準偏差の変化閾値
                                       hist_entropy_threshold, # ヒストグラムエントロピーの変化閾値
                                    #    black_clip_ratio_threshold,
                                    #    white_clip_ratio_threshold,
                                      ):
    """
    動画から品質指標の変化点を検出し、その前後のフレームを保存する。
    """

    pickle_path = os.path.splitext(video_path)[0] + ".pickle"
    if not os.path.exists(pickle_path):
        calculate_video_image_quality_metrics(video_path, pickle_path)
    
    with open(pickle_path, mode='rb') as f:
        all_metrics: list[ImageQualityMetrics] = pickle.load(f)
    
    print(f"Loaded: {pickle_path}")
    
    output_dir = output_dir if output_dir else os.path.splitext(video_path)[0] + "_qcs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 差分を計算（絶対値）
    # np.diff は要素間の差を計算
    # 例: [10, 20, 15] -> [10, -5]
    # np.abs で絶対値を取る
    # len(diff) は元のリストの長さ-1 になることに注意
    # laplacian_var_diff = np.abs(np.diff(np.array([m.laplacian_var for m in all_metrics])))
    hist_std_dev_diff = np.abs(np.diff(np.array([m.hist_std_dev for m in all_metrics])))
    hist_entropy_diff = np.abs(np.diff(np.array([m.hist_entropy for m in all_metrics])))
    # black_clip_ratio_diff = np.abs(np.diff(np.array([m.black_clip_ratio for m in all_metrics])))
    # white_clip_ratio_diff = np.abs(np.diff(np.array([m.white_clip_ratio for m in all_metrics])))
                                           
    # 各指標の変化点検出, 重複を排除
    all_change_indices: list[int] = np.unique(np.concatenate((
        # np.where(laplacian_var_diff > lap_var_threshold)[0],
        np.where(hist_std_dev_diff > hist_std_dev_threshold)[0],
        np.where(hist_entropy_diff > hist_entropy_threshold)[0],
        # np.where(black_clip_ratio_diff > black_clip_ratio_threshold)[0],
        # np.where(white_clip_ratio_diff > white_clip_ratio_threshold)[0],
    ))).tolist()

    is_target: dict[int, bool] = {}

    for i in all_change_indices:
        is_target[i] = True
        is_target[i+1] = True


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(total_frames), desc="Extracting Frames..."):
        ret, frame = cap.read()
        if not ret:
            break
        if is_target.get(i):
            frame_filename = os.path.join(output_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_filename, frame)

    cap.release()

    print("Completed.")


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="動画の品質変化点を検出し、前後フレームを抽出するスクリプト。")
    parser.add_argument("video_path", type=str,
                        help="解析する動画ファイルへのパス。")
    parser.add_argument("--output_dir", type=str,
                        help="抽出されたフレームを保存するディレクトリのパス。")
    # parser.add_argument("--lap_var_threshold", type=float, default=99999.0,
    #                     help="ラプラシアン分散の差分の閾値。これより大きい変化で検出。")
    parser.add_argument("--hist_std_dev_threshold", type=float, default=2000.0,
                        help="ヒストグラム標準偏差の差分の閾値。これより大きい変化で検出。")
    parser.add_argument("--hist_entropy_threshold", type=float, default=1.0,
                        help="ヒストグラムエントロピーの差分の閾値。これより大きい変化で検出。")
    # parser.add_argument("--black_clip_ratio_threshold", type=float, default=1.1,
    #                     help="黒潰れ比率の差分の閾値。これより大きい変化で検出。")
    # parser.add_argument("--white_clip_ratio_threshold", type=float, default=1.1,
    #                     help="白飛び比率の差分の閾値。これより大きい変化で検出。")
    
    args = parser.parse_args()
    
    detect_and_extract_quality_changes(
        args.video_path,
        args.output_dir,
        # args.lap_var_threshold,
        args.hist_std_dev_threshold,
        args.hist_entropy_threshold,
        # args.black_clip_ratio_threshold,
        # args.white_clip_ratio_threshold
    )
