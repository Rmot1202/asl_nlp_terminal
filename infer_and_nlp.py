#!/usr/bin/env python3
"""
infer_and_nlp.py

ASL → English (terminal):
- Load images (from .npy or directly from SL-MNIST CSV)
- Run CNN to get letter predictions
- Feed the letter sequence to GPT to "naturalize" the text

Typical usage (with separate picker):
  python infer_and_nlp.py \
    --model_path artifacts/asl_cnn_best.keras \
    --npy_path   picks_iloveu/selected_images.npy \
    --meta_json  picks_iloveu/selected_meta.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from dotenv import load_dotenv
from openai import OpenAI

# Load .env (OPENAI_API_KEY, OPENAI_MODEL, etc.)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------------------------------------------------------------------
# 24-class label <-> letter mapping (same as pick_by_letters.py)
# 0..23 → 24 letters A-Y (no J, no Z)
# -------------------------------------------------------------------
_VALID_LETTERS_24 = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ("J", "Z")]
assert len(_VALID_LETTERS_24) == 24

LABEL_TO_LETTER = {i: _VALID_LETTERS_24[i] for i in range(24)}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}
SUPPORTED_LETTERS = set(_VALID_LETTERS_24)


def _assert_supported_letters(letters: List[str]):
    bad = [L for L in letters if L.upper() not in SUPPORTED_LETTERS]
    if bad:
        raise ValueError(
            f"Unsupported letters for this SL-MNIST setup (J and Z excluded): {sorted(set(bad))}"
        )


# -------------------------------------------------------------------
# Helper: build LUT from raw CSV labels -> contiguous 0..23
# (Mirrors train_cnn.py and pick_by_letters.py)
# -------------------------------------------------------------------
def build_label_lut_from_df(df: pd.DataFrame):
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    y_raw = df["label"].values.astype("int64")
    uniq = np.array(sorted(np.unique(y_raw)))  # e.g. [0,1,2,3,4,5,6,7,8,10,...,24]
    n_classes = len(uniq)
    if n_classes != 24:
        print(f"[WARN] Expected 24 unique labels, found {n_classes}: {uniq}")
    raw_to_class = {old: i for i, old in enumerate(uniq)}  # old label -> 0..23
    return raw_to_class, y_raw


# -------------------------------------------------------------------
# Optional CSV-based selector (if you ever want to bypass pick_by_letters)
# -------------------------------------------------------------------
def select_images_from_csv(csv_path: str, letters: List[str], per_class: int = 1) -> Tuple[np.ndarray, List[str]]:
    """
    Load Sign Language MNIST CSV and return a batch of images for the requested letters.
    Uses same LUT style as train_cnn.py to keep mapping consistent.

    Returns:
      X: (N, 28, 28) float32 images in [0,1]
      y_letters: list of letters (length N) in the same order
    """
    letters = [L.upper() for L in letters]
    _assert_supported_letters(letters)

    df = pd.read_csv(csv_path)

    # Build LUT raw_label -> contiguous class index (0..23)
    raw_to_class, y_raw = build_label_lut_from_df(df)
    y_class = np.array([raw_to_class[int(lbl)] for lbl in y_raw])  # 0..23
    y_letters_full = np.array([LABEL_TO_LETTER[int(i)] for i in y_class])

    X_all = df.drop(columns=["label"]).values.astype("float32").reshape(-1, 28, 28)

    chosen_imgs, chosen_letters = [], []
    rng = np.random.default_rng(1202)

    for L in letters:
        idxs = np.where(y_letters_full == L)[0]
        if idxs.size == 0:
            raise ValueError(f"No samples for letter {L} in {csv_path}")
        if per_class > idxs.size:
            raise ValueError(f"Requested {per_class} samples for {L}, but dataset has {idxs.size}.")
        take = rng.choice(idxs, size=per_class, replace=False)
        for idx in take:
            chosen_imgs.append(X_all[idx])
            chosen_letters.append(L)

    X = np.stack(chosen_imgs, axis=0)  # (N, 28, 28)
    # Same normalization as training: raw 0..255 -> /255.0 -> 0..1
    X /= 255.0
    return X.astype("float32"), chosen_letters


# -------------------------------------------------------------------
# CNN inference helpers
# -------------------------------------------------------------------
def run_cnn(model_path: str, X: np.ndarray) -> Tuple[List[str], List[float]]:
    """
    X: (N, 28, 28) in [0,1]
    Returns predicted letters and confidences.
    Assumes CNN output dimension matches len(LABEL_TO_LETTER) == 24.
    """
    model = tf.keras.models.load_model(model_path)
    Xn = np.expand_dims(X, -1)  # (N, 28, 28, 1)

    # Keras model already outputs softmax probabilities
    probs = model.predict(Xn, verbose=0)  # shape (N, 24)

    top_idx = probs.argmax(axis=1)
    letters = [LABEL_TO_LETTER[int(i)] for i in top_idx]
    confs = probs.max(axis=1).tolist()
    return letters, confs


# -------------------------------------------------------------------
# GPT naturalization
# -------------------------------------------------------------------
PROMPT = """You are a careful text normalizer for ASL letter sequences.
Given a sequence of uppercase letters with no spaces (e.g., ILOVEYOU or HELLOMYNAMEISRAVEN),
1) insert spaces between words,
2) fix capitalization,
3) output a natural, fluent English sentence.
Rules:
- Do NOT add words that are not implied by the letters, except common punctuation.
- If the letters clearly spell a common phrase (e.g., ILOVEYOU), return it as that phrase.
- If letters don't form recognizable words, add minimal spacing (e.g., "A B C") and return as-is.
Return ONLY the final sentence.
"""


def gpt_naturalize(letter_sequence: str, model: str = None) -> str:
    """
    Call GPT to convert ASL letter sequence to a natural sentence.
    If OPENAI_API_KEY is missing, returns a simple spaced version of letters.
    """
    if not letter_sequence:
        return ""

    if not OPENAI_API_KEY:
        # Fallback: no key set, just space out letters
        return " ".join(list(letter_sequence))

    if model is None:
        model = DEFAULT_GPT_MODEL

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=60,
        messages=[
            {"role": "system", "content": "You convert ASL letter sequences into readable English."},
            {"role": "user",   "content": f"{PROMPT}\n\nLetters: {letter_sequence}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# -------------------------------------------------------------------
# CLI / Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pick SL-MNIST images by letter -> CNN -> GPT naturalization."
    )
    ap.add_argument("--model_path", required=True,
                    help="Path to trained Keras model (e.g., artifacts/asl_cnn_best.keras)")

    # MAIN MODE: read pre-made .npy (recommended with pick_by_letters.py)
    ap.add_argument("--npy_path",
                    help="Path to (N,28,28) .npy to run directly (bypasses CSV selection)")
    ap.add_argument("--meta_json",
                    help="Optional: selected_meta.json (for debugging/printing)")

    # OPTIONAL: combined mode (CSV + letters) if you ever want it
    ap.add_argument("--csv_path",
                    help="Path to sign_mnist_train.csv or sign_mnist_test.csv")
    ap.add_argument("--letters", nargs="+",
                    help="Letters to pick from CSV, e.g., I L O V E U")
    ap.add_argument("--per_class", type=int, default=1,
                    help="How many images per requested letter (CSV mode)")

    # Options
    ap.add_argument("--gpt_model", default=None,
                    help="GPT model for naturalization (default from OPENAI_MODEL or gpt-4o-mini)")
    ap.add_argument("--min_conf", type=float, default=0.0,
                    help="Drop predictions below this confidence (0..1)")
    ap.add_argument("--save_meta", default="",
                    help="Optional path to save a small JSON report")

    args = ap.parse_args()

    # ------------------------------
    # Build input images
    # ------------------------------
    requested_letters = None

    if args.npy_path and Path(args.npy_path).exists():
        X = np.load(args.npy_path).astype("float32")  # (N,28,28)
        # From pick_by_letters.py we already store X in [0,1]
        X = np.clip(X, 0.0, 1.0)
        src_info = {
            "mode": "npy",
            "path": str(Path(args.npy_path).resolve()),
        }

        # If meta_json provided, we can show requested letters
        if args.meta_json and Path(args.meta_json).exists():
            meta = json.loads(Path(args.meta_json).read_text())
            requested_letters = meta.get("requested", None)

    elif args.csv_path and args.letters:
        X, requested_letters = select_images_from_csv(args.csv_path, args.letters, args.per_class)
        src_info = {
            "mode": "csv",
            "path": str(Path(args.csv_path).resolve()),
            "requested_letters": args.letters,
            "per_class": args.per_class,
        }
    else:
        raise SystemExit("Provide either --npy_path OR (--csv_path AND --letters).")

    # ------------------------------
    # Run CNN
    # ------------------------------
    pred_letters, confs = run_cnn(args.model_path, X)

    # Filter by confidence if requested
    filtered = [(L, p) for (L, p) in zip(pred_letters, confs) if p >= args.min_conf]
    final_letters = [L for (L, _) in filtered]
    final_confs = [p for (_, p) in filtered]
    raw_sequence = "".join(final_letters)

    print("\n=== Inference (CNN) ===")
    for i, (L, p) in enumerate(zip(pred_letters, confs)):
        mark = "" if p >= args.min_conf else "  (dropped)"
        print(f" #{i:02d}: {L}  p≈{p:.3f}{mark}")

    if requested_letters:
        print("\nRequested letters:", [str(s).upper() for s in requested_letters])
    print("Kept sequence     :", " ".join(final_letters) if final_letters else "(none)")
    print("Raw sequence      :", raw_sequence if raw_sequence else "(empty)")

    # ------------------------------
    # GPT naturalization
    # ------------------------------
    natural = gpt_naturalize(raw_sequence, model=args.gpt_model)

    print("\n=== Naturalized (GPT) ===")
    print(natural if natural else "(no output)")

    # ------------------------------
    # Optional JSON report
    # ------------------------------
    if args.save_meta:
        report = {
            "source": src_info,
            "pred_letters": pred_letters,
            "pred_confidences": confs,
            "kept_letters": final_letters,
            "kept_confidences": final_confs,
            "raw_sequence": raw_sequence,
            "naturalized": natural,
            "gpt_model": args.gpt_model or DEFAULT_GPT_MODEL,
            "min_conf": args.min_conf,
            "requested_letters": requested_letters,
        }
        out_path = Path(args.save_meta)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
