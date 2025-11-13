import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from utils import load_slmnist_csv  # expects Kaggle-style CSVs


# ===========================
# Label remapping utilities
# ===========================
def make_label_lut(y_train, y_test):
    """
    Build a look-up table that maps the dataset's raw labels
    to contiguous indices [0..(n_classes-1)].

    This LUT is built from BOTH train and test labels so that
    the mapping is consistent across the whole dataset.
    """
    uniq = np.array(sorted(np.unique(np.concatenate([y_train, y_test]))))
    lut = {old: i for i, old in enumerate(uniq)}
    return lut, len(uniq)


def apply_lut(y, lut):
    return np.vectorize(lut.get)(y)


# ===========================
# CNN architecture
# ===========================
def build_cnn(input_shape=(28, 28, 1), n_classes=24):
    """
    Stronger CNN for Sign Language MNIST.
    - Data augmentation
    - 3 conv blocks with increasing filters
    - GlobalAveragePooling + dense head
    """
    inputs = keras.Input(shape=input_shape)

    # ---- Data augmentation (only applied during training) ----
    x = layers.RandomRotation(0.08)(inputs)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomTranslation(0.10, 0.10)(x)

    # ---- Conv block 1 ----
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # ---- Conv block 2 ----
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # ---- Conv block 3 ----
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # ---- Head ----
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="asl_cnn")
    return model


# ===========================
# Main training routine
# ===========================
def main(args):
    # 1) Load data via your existing utils function
    #    Expecting X, Xte in shape (N,28,28,1) and already normalized to [0,1].
    (X, y_raw), (Xte, yte_raw) = load_slmnist_csv(args.train_csv, args.test_csv)

    # 2) Build label LUT from BOTH train and test so mapping is consistent
    lut, n_classes = make_label_lut(y_raw, yte_raw)
    y = apply_lut(y_raw, lut)
    yte = apply_lut(yte_raw, lut)

    print(f"[INFO] Unique raw labels: {sorted(np.unique(y_raw))}")
    print(f"[INFO] Remapped to {n_classes} contiguous classes: {sorted(set(y))}")

    # 3) Train/validation split (stratified)
    Xtr, Xval, ytr, yval = train_test_split(
        X, y,
        test_size=0.10,
        stratify=y,
        random_state=42
    )

    # 4) Build model
    model = build_cnn(input_shape=X.shape[1:], n_classes=n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # 5) Callbacks: LR schedule, early stopping, best model checkpoint
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "asl_cnn_best.keras"

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-5,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 6) Train
    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=40,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    # 7) Evaluate on full test set
    test_loss, test_acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    # 8) Save final model (and you've already got best model as ckpt)
    final_model_path = out_dir / "asl_cnn.keras"
    model.save(final_model_path)
    print(f"✅ Saved final model to {final_model_path}")
    print(f"✅ Best (checkpoint) model at {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="path/to/sign_mnist_train.csv")
    p.add_argument("--test_csv", required=True, help="path/to/sign_mnist_test.csv")
    p.add_argument("--out_dir", default="artifacts")
    main(p.parse_args())
