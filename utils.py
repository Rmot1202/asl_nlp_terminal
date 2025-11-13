import numpy as np
import pandas as pd
from pathlib import Path

# Sign Language MNIST CSV loader (Kaggle style: label + 784 pixels)
def load_slmnist_csv(train_csv: str, test_csv: str):
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)
    X_train = (train.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype("float32")) / 255.0
    y_train = train.iloc[:, 0].values
    X_test  = (test.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype("float32")) / 255.0
    y_test  = test.iloc[:, 0].values
    return (X_train, y_train), (X_test, y_test)

# Map numeric labels (0..23) to ASL letters A..Y except J,Z
# Typical SL-MNIST mapping (Kaggle): 0->A, 1->B, ... skipping J(9) and Z(25)
LETTER_MAP = [chr(c) for c in range(ord('A'), ord('Z')+1) if chr(c) not in ('J','Z')]

def id_to_letter(idx: int) -> str:
    return LETTER_MAP[idx]
