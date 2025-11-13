# eval_cnn.py
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Drop label 24 if present (since your model has 24 outputs, NOT 25)
    if (df["label"] == 24).any():
        df = df[df["label"] != 24].reset_index(drop=True)

    y = df["label"].values.astype("int64")
    X = df.drop(columns=["label"]).values.astype("float32").reshape(-1, 28, 28, 1)

    # ⚠️ THIS MUST MATCH train_cnn.py
    X /= 255.0
    return X, y

def main():
    model_path = "artifacts/asl_cnn_best.keras"
    test_csv = "../sign_mnist_test.csv"

    assert Path(model_path).exists(), f"Missing model at {model_path}"
    assert Path(test_csv).exists(), f"Missing test CSV at {test_csv}"

    X_test, y_test = load_data(test_csv)

    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test loss: {loss:.4f}")
    print(f"Test acc : {acc:.4f}")

if __name__ == "__main__":
    main()
