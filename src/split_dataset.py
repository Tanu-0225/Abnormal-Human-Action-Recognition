"""Step 2: split extracted frames into train / val / test.

Two strategies:

  --by video  (default) every clip goes entirely into one split. The test set
              then contains only clips the model has never seen, which is the
              honest way to measure generalisation.
  --by frame  frames are shuffled individually, as in the original paper.
              Neighbouring, near-identical frames of the same clip end up in
              both train and test, so accuracy is optimistic.

    python src/split_dataset.py [--by video|frame]

The split is reproducible (fixed seed) and recorded in data/split/manifest.json.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SPLITS = ("train", "val", "test")


def allocate(items: list, ratios: tuple, rng: random.Random, min_each: int) -> dict:
    """Shuffle `items` and divide them by `ratios`, giving val/test at least `min_each`."""
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_test = max(min_each, round(n * ratios[2]))
    n_val = max(min_each, round(n * ratios[1]))
    if n - n_val - n_test < 1:
        raise ValueError(f"only {n} items; need at least {2 * min_each + 1} to fill train/val/test")
    return {
        "test": items[:n_test],
        "val": items[n_test:n_test + n_val],
        "train": items[n_test + n_val:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=config.FRAMES_DIR)
    parser.add_argument("--out", type=Path, default=config.SPLIT_DIR)
    parser.add_argument("--by", choices=["video", "frame"], default="video")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    if args.out.exists():
        if not (args.out / "manifest.json").exists():
            sys.exit(f"{args.out} exists but was not created by this script; refusing to overwrite it.")
        shutil.rmtree(args.out)

    rng = random.Random(args.seed)
    manifest = {"strategy": args.by, "seed": args.seed, "ratios": config.SPLIT_RATIOS, "classes": {}}

    for cls in config.CLASSES:
        clips = sorted(p for p in (args.frames / cls).iterdir() if p.is_dir())
        if not clips:
            sys.exit(f"No extracted clips in {args.frames / cls}; run extract_frames.py first")

        if args.by == "video":
            try:
                by_split = allocate(clips, config.SPLIT_RATIOS, rng, min_each=1)
            except ValueError as e:
                sys.exit(f"{cls}: {e} (clips)")
            files = {s: [f for clip in by_split[s] for f in sorted(clip.glob("*.jpg"))] for s in SPLITS}
            manifest["classes"][cls] = {s: [c.name for c in by_split[s]] for s in SPLITS}
        else:
            frames = sorted(f for clip in clips for f in clip.glob("*.jpg"))
            files = allocate(frames, config.SPLIT_RATIOS, rng, min_each=1)
            manifest["classes"][cls] = {s: len(files[s]) for s in SPLITS}

        for split in SPLITS:
            dest = args.out / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files[split]:
                # prefix with the clip name so files from different clips never collide
                shutil.copy2(f, dest / f"{f.parent.name}_{f.name}")
        counts = "  ".join(f"{s}={len(files[s])}" for s in SPLITS)
        print(f"{cls:12s} {counts}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Split ({args.by}-level) written to {args.out}")


if __name__ == "__main__":
    main()
