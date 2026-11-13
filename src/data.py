"""Dataset loading and augmentation shared by train.py and evaluate.py."""

import sys
from pathlib import Path

import keras
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_split(split: str, split_dir: Path = config.SPLIT_DIR, shuffle: bool = False) -> tf.data.Dataset:
    """Load data/split/<split>/ as (image, one-hot label) batches with raw 0-255 RGB pixels."""
    directory = split_dir / split
    if not directory.is_dir():
        sys.exit(f"Missing {directory}; run split_dataset.py first")
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
    )


def augmentation() -> keras.Sequential:
    """Training-time augmentation, equivalent to the paper's ImageDataGenerator settings
    (rotation 30 deg, shift 20 %, zoom 20 %, horizontal flip, nearest fill)."""
    return keras.Sequential(
        [
            keras.layers.RandomRotation(30 / 360, fill_mode="nearest"),
            keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
            keras.layers.RandomZoom(0.2, fill_mode="nearest"),
            keras.layers.RandomFlip("horizontal"),
        ],
        name="augmentation",
    )
