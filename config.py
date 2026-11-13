"""Shared settings for the whole pipeline. Every script imports from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data layout (the dataset itself is not distributed with this repo) ---
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"   # input:  data/videos/<class>/<clip>.mp4
FRAMES_DIR = DATA_DIR / "frames"  # output: data/frames/<class>/<clip>/frame_00000.jpg
SPLIT_DIR = DATA_DIR / "split"    # output: data/split/{train,val,test}/<class>/*.jpg

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Folder names under data/videos. Keep them sorted: Keras assigns label
# indices alphabetically, so this order is also the model's output order.
CLASSES = ["chest_pain", "cough", "headache", "stomachache"]
CLASS_NAMES = {
    "chest_pain": "Chest pain",
    "cough": "Cough",
    "headache": "Headache",
    "stomachache": "Stomachache",
}

# --- Hyperparameters used in the paper ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 35              # stage 1: frozen MobileNetV2 backbone
FINE_TUNE_EPOCHS = 17    # stage 2: EPOCHS // 2
FINE_TUNE_LAYERS = 50    # number of backbone layers unfrozen in stage 2
LEARNING_RATE = 1e-4     # stage 2 uses LEARNING_RATE / 10
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test
SEED = 42

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

assert CLASSES == sorted(CLASSES), "CLASSES must be in alphabetical order"
