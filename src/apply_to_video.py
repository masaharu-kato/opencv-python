import argparse
import logging
import cv2
from pathlib import Path
from tqdm import tqdm

from apply import ModelApplyer

def apply_model_to_video(model_path: Path, input_video_path: Path, output_video_path: Path, *, frame_size: tuple[int, int] | None = None, limit: int | None = None, fps: float | None = None):
    """Applies a pre-trained model to each frame of a video and saves the enhanced video."""
    
    model_applyer = ModelApplyer(model_path)
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type:ignore

    fps = input_fps if fps is None else fps
    frame_limit = min(frame_count, limit) if limit is not None else frame_count
    frame_interval = max(round(input_fps / fps), 1)
    
    out = None

    # Process each frame
    try:
        for i in tqdm(range(frame_limit), desc="Processing video frames"):
            ret, frame_bgr = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame {i}. Ending video processing.")
                break

            if (i % frame_interval) != 0:
                continue

            if frame_size is not None:
                # Resize frame if frame_size is specified
                frame_bgr = cv2.resize(frame_bgr, frame_size)

            # Apply the model to the frame
            enhanced_frame_bgr = model_applyer.apply(frame_bgr)

            # cv2.imshow("Enhanced Frame", enhanced_frame_bgr)  # Optional: Show the enhanced frame in a window (for debugging)
            # cv2.waitKey(0)  # Wait for a key press to proceed to the next frame (for debugging)

            if out is None:
                outh, outw = enhanced_frame_bgr.shape[:2]
                out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (outw, outh))

            # Write the enhanced frame to the output video
            out.write(enhanced_frame_bgr)
        
    finally:
        # Release resources
        cap.release()
        if out is not None:
            out.release()

    logging.info(f"Enhanced video saved to {output_video_path}")


def main():
    argp = argparse.ArgumentParser(description="Apply a pre-trained model to each frame of a video.", add_help=False)
    argp.add_argument("model_path", type=Path, help="Path to the pre-trained model file (*.pth).")
    argp.add_argument("input_video_path", type=Path, help="Path to the input video file.")
    argp.add_argument("output_video_path", type=Path, help="Path to save the enhanced video file.")
    argp.add_argument("-w", "--width", type=int, help="frame width (default: same as video).")
    argp.add_argument("-h", "--height", type=int, help="frame height (default: same as video).")
    argp.add_argument("-l", "--limit", type=int, help="Limit the number of frames to process (default: all frames).")
    argp.add_argument("-fps", "--fps", type=float, help="Frames per second for the output video (default: same as input video).")
    
    args = argp.parse_args()
    
    apply_model_to_video(args.model_path, args.input_video_path, args.output_video_path,
                         frame_size=(args.width, args.height) if args.width and args.height else None, fps=args.fps, limit=args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
