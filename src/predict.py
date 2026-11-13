"""Run the trained model on images, videos, or a live webcam feed.

    python src/predict.py image.jpg folder/ clip.mp4 [--out predictions/]
    python src/predict.py --webcam [--save demo.mp4] [--gif demo.gif]

Images and videos are written to --out with the predicted action and
confidence drawn on them. For videos and the webcam, predictions are
averaged over the last --smooth frames to stop the label flickering.
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import keras
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

NAMES = [config.CLASS_NAMES[c] for c in config.CLASSES]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def predict_probs(model: keras.Model, frame_bgr: np.ndarray) -> np.ndarray:
    """Class probabilities for one OpenCV (BGR) frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # the model was trained on RGB
    x = cv2.resize(rgb, (config.IMG_SIZE, config.IMG_SIZE)).astype("float32")[None]
    return model.predict(x, verbose=0)[0]  # rescaling happens inside the model


def annotate(frame: np.ndarray, probs: np.ndarray) -> np.ndarray:
    i = int(probs.argmax())
    frame = frame.copy()
    for row, text in enumerate((f"Action: {NAMES[i]}", f"Confidence: {probs[i] * 100:.1f}%")):
        y = 28 + row * 28
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def save_gif(frames: list[np.ndarray], path: Path, fps: float, width: int) -> None:
    """Write BGR frames as an animated GIF (plays inline on GitHub)."""
    from PIL import Image

    step = max(1, round(fps / 10))  # ~10 fps keeps the file small
    images = []
    for f in frames[::step]:
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not images:
        return
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 * step / fps), loop=0, optimize=True)
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def run_stream(model, source, args, save: Path | None = None, gif: Path | None = None, show: bool = False) -> None:
    """Classify a video file or camera, frame by frame."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120:
        fps = 30.0
    recent = deque(maxlen=args.smooth)
    writer = None
    gif_frames = []
    if show:
        print("Press q to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        recent.append(predict_probs(model, frame))
        out = annotate(frame, np.mean(recent, axis=0))
        if save:
            if writer is None:
                save.parent.mkdir(parents=True, exist_ok=True)
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
        if gif:
            gif_frames.append(out)
        if show:
            cv2.imshow("Abnormal action recognition", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"  wrote {save}")
    if show:
        cv2.destroyAllWindows()
    if gif:
        save_gif(gif_frames, gif, fps, args.gif_width)


def run_files(model: keras.Model, paths: list[Path], args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for p in paths:
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]

    for path in files:
        ext = path.suffix.lower()
        if ext in config.VIDEO_EXTENSIONS:
            print(f"{path.name}:")
            gif = args.out / f"{path.stem}_pred.gif" if args.gif else None
            run_stream(model, str(path), args, save=args.out / f"{path.stem}_pred.mp4", gif=gif)
        elif ext in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  ! could not read {path}", file=sys.stderr)
                continue
            probs = predict_probs(model, frame)
            dest = args.out / f"{path.stem}_pred.jpg"
            cv2.imwrite(str(dest), annotate(frame, probs))
            i = int(probs.argmax())
            print(f"{path.name:35s} {NAMES[i]:12s} {probs[i] * 100:5.1f}%  -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, help="images, videos, or folders")
    parser.add_argument("--webcam", action="store_true", help="classify a live camera feed")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--save", type=Path, help="webcam: also record the annotated feed to this .mp4")
    parser.add_argument("--gif", nargs="?", const=True, type=Path,
                        help="also write an animated GIF (webcam: give a path; files: one GIF per video)")
    parser.add_argument("--gif-width", type=int, default=480)
    parser.add_argument("--smooth", type=int, default=10, help="average predictions over N frames (video/webcam)")
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR / "model.keras")
    parser.add_argument("--out", type=Path, default=config.ROOT / "predictions")
    args = parser.parse_args()
    if not args.webcam and not args.inputs:
        parser.error("give image/video paths or folders, or --webcam")
    if args.webcam and args.gif is True:
        parser.error("with --webcam, give the GIF a path: --gif demo.gif")

    model = keras.models.load_model(args.model)
    if args.webcam:
        run_stream(model, args.camera, args, save=args.save, gif=args.gif, show=True)
    else:
        run_files(model, args.inputs, args)


if __name__ == "__main__":
    main()
