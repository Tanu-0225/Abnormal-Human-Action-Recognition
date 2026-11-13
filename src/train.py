"""Step 3: train the MobileNetV2 classifier.

Stage 1 trains a new classification head on top of a frozen, ImageNet
pre-trained MobileNetV2. Stage 2 unfreezes the last 50 backbone layers and
fine-tunes with a 10x smaller learning rate.

    python src/train.py [--epochs 35] [--fine-tune-epochs 17]

Writes models/model.keras and models/history.json (both stages).
"""

import argparse
import json
import sys
from pathlib import Path

import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data import augmentation, load_split  # noqa: E402


def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    """Return (full model, backbone). The model takes raw 0-255 RGB images."""
    backbone = keras.applications.MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = augmentation()(inputs)                 # active only during training
    x = keras.layers.Rescaling(1.0 / 255)(x)   # same normalisation as the paper
    x = backbone(x, training=False)            # keep BatchNorm statistics frozen
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="abhar_mobilenetv2"), backbone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--out", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args()

    keras.utils.set_random_seed(config.SEED)
    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val")
    test_ds = load_split("test")

    model, backbone = build_model(len(config.CLASSES))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    print("\n=== Stage 1: training the classification head ===")
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[reduce_lr])

    print(f"\n=== Stage 2: fine-tuning the last {config.FINE_TUNE_LAYERS} backbone layers ===")
    backbone.trainable = True
    for layer in backbone.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(args.lr / 10), loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=[reduce_lr])

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}   test loss: {test_loss:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "model.keras")
    history = {k: [float(v) for v in h1.history[k] + h2.history.get(k, [])] for k in h1.history if k != "learning_rate"}
    history["fine_tune_start_epoch"] = len(h1.history["loss"])
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved model and history to {args.out}")


if __name__ == "__main__":
    main()
