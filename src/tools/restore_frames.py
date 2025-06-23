import cv2
import os
import sys
import re
from tqdm import tqdm

def read_frame_numbers(frame_file):
    with open(frame_file, 'r') as f:
        return sorted(((int(match.group(1)), line.strip()) for line in f if (match := re.search(r'frame_(\d+)', line))), key=lambda v: v[0])

def save_frames(video_path: str, frame_numbers: list[tuple[int, str]], output_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for iframe, filename in tqdm(frame_numbers):
        if iframe < 0 or iframe >= total_frames:
            print(f"Frame {iframe} is out of range (0-{total_frames-1})")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, frame)
            # print(f"Saved {out_path}")
        else:
            print(f"Failed to read frame {iframe}")

    cap.release()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python restore_frames.py <video_path> <frame_numbers.txt> <output_dir>")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_file = sys.argv[2]
    output_dir = sys.argv[3]

    frame_numbers = read_frame_numbers(frame_file)
    save_frames(video_path, frame_numbers, output_dir)
