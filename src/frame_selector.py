import cv2
import os
import time
import argparse
from cv2.typing import NumPyArrayNumeric
import numpy as np

# --- グローバル変数とコールバック関数 ---
# シークバーの現在位置（フレーム番号）を保持するグローバル変数
window_name = "Video Frame Selector"
inext = 0
last_trackbar_updated = time.time()

def on_trackbar(frame_pos):
    global inext
    inext = frame_pos

def updateInext():
    global last_trackbar_updated
    cv2.setTrackbarPos("Position", window_name, inext)
    last_trackbar_updated = time.time()


def calculate_image_quality_metrics(img_bgr):

    # グレースケール変換 (ラプラシアン分散用)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- 1. ブレ/ピントの評価: ラプラシアン分散 ---
    # ラプラシアンフィルター適用
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # --- 2. ダイナミックレンジの評価 ---
    # 輝度チャンネル（Y）またはグレースケールで評価
    # ヒストグラム計算 (輝度値を0-255に正規化)
    hist, bins = np.histogram(img_gray.flatten(), 256, (0, 256))

    # ヒストグラムの標準偏差
    hist_std_dev = np.std(hist)

    # ヒストグラムのエントロピー (輝度分布の均一性)
    # 値が小さいほど輝度分布が偏っている
    hist_prob = hist / hist.sum() + 1e-10 # 0除算防止
    hist_entropy = -np.sum(hist_prob * np.log2(hist_prob))

    # 白飛び/黒潰れピクセル比率
    # 例: 0-10 を黒潰れ、245-255 を白飛びとする
    black_pixels = np.sum(img_gray < 10)
    white_pixels = np.sum(img_gray > 245)
    total_pixels = img_gray.size
    
    black_clip_ratio = black_pixels / total_pixels
    white_clip_ratio = white_pixels / total_pixels
    total_clip_ratio = black_clip_ratio + white_clip_ratio

    return {
        "laplacian_variance": laplacian_var,
        "hist_std_dev": hist_std_dev,
        "hist_entropy": hist_entropy,
        "black_clip_ratio": black_clip_ratio,
        "white_clip_ratio": white_clip_ratio,
        "total_clip_ratio": total_clip_ratio
    }

# --- メイン関数 ---
def select_good_frames(video_path, output_dir="selected_frames"):
    """
    動画を再生し、様々な操作でフレームを選択・保存するツール

    Parameters:
    video_path (str): 入力動画ファイルのパス
    output_dir (str): 選択したフレームを保存するディレクトリ
    """
    global inext, ishown, last_trackbar_updated

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"エラー: {video_path} を開けません。動画のパスを確認してください。")
        return

    # 動画の情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0: # fpsが取得できない場合の対策
        fps = 30 
        print(f"警告: FPS情報が取得できませんでした。デフォルト値 {fps} を使用します。")
    
    # 再生速度の初期値
    play_speed_multiplier = 1.0
    delay_ms = int(1000 / (fps * play_speed_multiplier))

    # ウィンドウの設定
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height) # ウィンドウサイズを動画のサイズに合わせる

    # シークバーの作成
    cv2.createTrackbar("Position", window_name, 0, total_frames - 1, on_trackbar)
    trackber_update_interval = 0.25
    last_trackbar_updated = time.time()

    # --- 状態変数 ---
    playing = True
    ishown = None
    
    print("\n--- 操作ガイド ---")
    print(f"  動画を再生します: {video_path}")
    print(f"  総フレーム数: {total_frames}, FPS: {fps:.2f}")
    print(f"  's'    : 現在のフレームを保存")
    print(f"  'space': 一時停止/再生")
    print(f"  '←'    : 1フレーム戻る")
    print(f"  '→'    : 1フレーム進む")
    print(f"  'PgUp' : 再生速度を上げる (0.1倍ずつ)")
    print(f"  'PgDn' : 再生速度を下げる (0.1倍ずつ)")
    print(f"  'Esc'    : 終了")
    print("------------------\n")

    frame: cv2.Mat | NumPyArrayNumeric | None = None
    pos_str = ''

    isaved: set[float] = set()

    while True:
        inext = min(max(inext, 0), total_frames - 1)
        # print(f"\rCurrent frame: {ishown}, Next frame: {inext}, Playing: {playing}", end="", flush=True)
        if ishown is None or inext != (ishown + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, inext)
        
        if inext != ishown:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {inext}. Ending playback.")
                break
            
            ishown = inext
            if time.time() - last_trackbar_updated > trackber_update_interval:
                updateInext()

            # 画面に現在の時間やフレーム情報などを表示
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            pos_str = f"{time.strftime("%H:%M:%S", time.gmtime(pos_ms / 1000))}.{int(pos_ms * 1000 % 1000):03d}"

            qmetrics = calculate_image_quality_metrics(frame)

            # フレーム情報を表示
            display_frame = frame.copy()
            text_info = f"Time: {pos_str} | Frame: {ishown}/{total_frames} | Speed: {play_speed_multiplier:.1f}x"
            cv2.putText(display_frame, text_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 画質評価指標を表示
            y0, dy = 70, 35
            for i, (k, v) in enumerate(qmetrics.items()):
                text = f"{k}: {v:.4f}"
                cv2.putText(display_frame, text, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (64, 128, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

        # キー入力処理
        key = cv2.waitKey(delay_ms) & 0xFF

        # print("Key:", key)

        if key == 27: # Escキー: 終了
            break

        if ishown is None or frame is None or not pos_str:
            continue

        if key == ord('s'): # or key == 82 or key == 83:  # 's'/'←'/'→': Save current frame
            if ishown not in isaved:
                out_name = os.path.join(output_dir, f"{pos_str.replace(':', '_').replace('.', '_')}_{ishown:06d}.png")
                cv2.imwrite(out_name, frame)
                isaved.add(ishown)
                print(f"Saved: {out_name} (total: {len(isaved)} frames)")
            else:
                print(f"Frame {ishown} is already saved.")
            if not playing:
                inext = ishown + 1
                print("Set inext to ", inext, "ishown=", ishown)
                updateInext()
        
        if key == ord('p') or key == ord(' '):  # Space: Pause/Resume
            playing = not playing
            updateInext()

        # if key == 81 or key == 82: # '←'/'↑'
        #     inext = max(ishown - 1, 0)
        #     update_trackbar()
        #     playing = False

        # if key == 83 or key == 84: # '→'/'↓'
        #     inext = min(ishown + 1, total_frames - 1)
        #     update_trackbar()
        #     playing = False
        
        if key == 82: # 
            play_speed_multiplier = play_speed_multiplier * 2
            delay_ms = max(int(1000 / (fps * play_speed_multiplier)), 1)
            print(f"Speed: {play_speed_multiplier:.1f}x")
        
        if key == 84: # 
            play_speed_multiplier = play_speed_multiplier * 0.5
            delay_ms = max(int(1000 / (fps * play_speed_multiplier)), 1)
            print(f"Speed: {play_speed_multiplier:.1f}x")

        if playing:
            inext = ishown + 1

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()
    print(f"--- Total {len(isaved)} frames are saved. ---")

# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="動画を再生し、手動で「良い」フレームを選択・保存するツール")
    parser.add_argument("video_path", type=str, help="処理する入力動画ファイルのパス")
    parser.add_argument("output_dir", type=str, help="選択したフレームを保存するディレクトリのパス")
    
    args = parser.parse_args()
    
    select_good_frames(args.video_path, args.output_dir)
