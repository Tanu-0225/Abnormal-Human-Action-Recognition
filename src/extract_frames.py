"""Step 1: extract frames from every video in data/videos/<class>/.

Frames from each clip are kept in their own folder, so the split step can
keep all frames of one clip on the same side of the train/val/test split.

    python src/extract_frames.py [--every N] [--size 224]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def extract_video(video_path: Path, out_dir: Path, every: int, size: int) -> int:
    """Save every `every`-th frame of `video_path` into `out_dir`. Returns the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}", file=sys.stderr)
        return 0

    saved = index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % every == 0:
            if size:
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"frame_{index:05d}.jpg"), frame)
            saved += 1
        index += 1
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=config.VIDEO_DIR)
    parser.add_argument("--out", type=Path, default=config.FRAMES_DIR)
    parser.add_argument("--every", type=int, default=1, help="keep every Nth frame (default: all)")
    parser.add_argument("--size", type=int, default=config.IMG_SIZE, help="resize to SIZE x SIZE; 0 keeps original")
    args = parser.parse_args()

    total = 0
    for cls in config.CLASSES:
        class_dir = args.videos / cls
        if not class_dir.is_dir():
            sys.exit(f"Missing folder: {class_dir}  (expected one folder per class: {config.CLASSES})")
        videos = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in config.VIDEO_EXTENSIONS)
        if not videos:
            sys.exit(f"No videos found in {class_dir}")
        for video in videos:
            n = extract_video(video, args.out / cls / video.stem, args.every, args.size)
            print(f"{cls:12s} {video.name:30s} {n:5d} frames")
            total += n
    print(f"Done: {total} frames written to {args.out}")


if __name__ == "__main__":
    main()
