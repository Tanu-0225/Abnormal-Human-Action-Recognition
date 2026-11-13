"""Step 4: evaluate a trained model on the test split and save all figures.

    python src/evaluate.py [--model models/model.keras] [--out results/latest]

Writes: metrics.json, classification_report.txt, confusion_matrix.png,
roc_curves.png, pr_curves.png, accuracy.png, loss.png, misclassified.png
"""

import argparse
import json
import sys
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")  # save files, no windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data import load_split  # noqa: E402

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "figure.dpi": 110})
NAMES = [config.CLASS_NAMES[c] for c in config.CLASSES]


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_history(history: dict, key: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history[key], marker="o", label=f"Train {key}")
    ax.plot(history[f"val_{key}"], marker="o", label=f"Validation {key}")
    if "fine_tune_start_epoch" in history:
        ax.axvline(history["fine_tune_start_epoch"] - 0.5, color="gray", ls="--", label="fine-tuning starts")
    ax.set(xlabel="Epoch", ylabel=key.capitalize(), title=f"Training & validation {key}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR / "model.keras")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "latest")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(args.model)
    test_ds = load_split("test")  # not shuffled, so predictions line up with file_paths
    files = test_ds.file_paths
    y_true = np.concatenate([y for _, y in test_ds]).argmax(axis=1)
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    acc = float((y_true == y_pred).mean())
    print(f"Test accuracy: {acc:.4f} on {len(y_true)} frames")
    report = classification_report(y_true, y_pred, labels=range(len(NAMES)), target_names=NAMES, digits=3, zero_division=0)
    print(report)
    (args.out / "classification_report.txt").write_text(report)

    split_manifest = config.SPLIT_DIR / "manifest.json"
    strategy = json.loads(split_manifest.read_text())["strategy"] if split_manifest.exists() else "unknown"
    (args.out / "metrics.json").write_text(
        json.dumps({"test_accuracy": acc, "test_frames": len(y_true), "split_strategy": strategy}, indent=2)
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(NAMES)))
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix")
    save(fig, args.out / "confusion_matrix.png")

    # ROC and precision-recall, one curve per class
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(NAMES):
            positives = y_true == i
            if positives.sum() == 0:
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(positives, probs[:, i])
                ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
            else:
                precision, recall, _ = precision_recall_curve(positives, probs[:, i])
                ax.plot(recall, precision, lw=2, label=name)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="gray")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve per class")
        else:
            ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve per class")
        ax.legend(loc="lower right" if kind == "roc" else "lower left")
        save(fig, args.out / f"{kind}_curves.png")

    # Training curves
    history_path = args.model.parent / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plot_history(history, "accuracy", args.out / "accuracy.png")
        plot_history(history, "loss", args.out / "loss.png")

    # Misclassified frames (up to 9), loaded from their actual file paths
    wrong = np.flatnonzero(y_true != y_pred)[:9]
    if len(wrong):
        cols = min(3, len(wrong))
        rows = int(np.ceil(len(wrong) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, wrong):
            ax.imshow(keras.utils.load_img(files[i]))
            ax.set_title(f"true: {NAMES[y_true[i]]}\npred: {NAMES[y_pred[i]]}", fontsize=11)
        save(fig, args.out / "misclassified.png")
    else:
        print("  no misclassified frames")


if __name__ == "__main__":
    main()
