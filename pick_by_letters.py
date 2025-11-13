#!/usr/bin/env python3
"""
pick_by_letters.py

Select images from a Sign Language MNIST-style CSV by letters
and save them for inference with infer_and_nlp.py.

Outputs (in --out_dir):
- selected_images.npy : shape (N, 28, 28), float32 in [0,1]
- selected_meta.json  : letters, indices, csv_path, seed, per_class, requested
- pngs/*.png          : optional previews if --save_pngs

Usage example (phrase mode):
  python pick_by_letters.py \
    --csv_path ../sign_mnist_test.csv \
    --phrase  "I LOVE U" \
    --per_class 1 \
    --out_dir  picks_iloveu \
    --save_pngs
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 24-class letter order: A–Y with J and Z removed
# This MUST match the mapping used in train_cnn.py + infer_and_nlp.py.
# -------------------------------------------------------------------
_VALID_LETTERS_24 = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ("J", "Z")]
assert len(_VALID_LETTERS_24) == 24, "Expected 24 letters for 24 classes."

LABEL_TO_LETTER = {i: _VALID_LETTERS_24[i] for i in range(24)}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}
SUPPORTED_LETTERS = set(_VALID_LETTERS_24)  # A..Y without J,Z


def normalize_letters(letters_or_phrase: List[str], allow_phrase: bool) -> List[str]:
    """
    Normalize input into an uppercase list of letters.
    If allow_phrase=True and a single string is given (e.g., "I LOVE U"),
    it will extract all alphabetic characters as letters.
    """
    if allow_phrase and len(letters_or_phrase) == 1:
        s = letters_or_phrase[0]
        letters = re.findall(r"[A-Za-z]", s)
    else:
        letters = []
        for token in letters_or_phrase:
            letters.extend(list(token))

    letters = [c.upper() for c in letters if c.isalpha()]

    bad = [c for c in letters if c not in SUPPORTED_LETTERS]
    if bad:
        raise ValueError(
            f"Unsupported letters for this SL-MNIST setup (J and Z excluded): {sorted(set(bad))}"
        )
    return letters


def build_label_lut_from_df(df: pd.DataFrame):
    """
    Build the SAME style LUT as train_cnn.py:
    raw CSV labels (e.g. [0,1,2,3,4,5,6,7,8,10,...,24]) -> contiguous [0..23].

    This keeps the 'class index' aligned with the CNN and LABEL_TO_LETTER.
    """
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    y_raw = df["label"].values.astype("int64")
    uniq = np.array(sorted(np.unique(y_raw)))  # e.g. [0,1,2,3,4,5,6,7,8,10,...,24]
    n_classes = len(uniq)
    if n_classes != 24:
        print(f"[WARN] Expected 24 unique labels, found {n_classes}: {uniq}")

    raw_to_class = {old: i for i, old in enumerate(uniq)}  # old label -> 0..23
    return raw_to_class, y_raw


def select_images(
    csv_path: str,
    letters: List[str],
    per_class: int,
    seed: int
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Selects images for the given letters from the CSV.

    Returns:
      imgs: (N, 28, 28) float32 in [0,1]
      chosen_letters: list[str] length N
      chosen_indices: list[int] length N (row indices in original CSV)
    """
    df = pd.read_csv(csv_path)

    # Build LUT from raw labels -> contiguous class indices (0..23),
    # exactly like train_cnn.py
    raw_to_class, y_raw = build_label_lut_from_df(df)
    y_class = np.array([raw_to_class[int(lbl)] for lbl in y_raw])  # 0..23

    # Map class indices -> letters using LABEL_TO_LETTER
    y_letters = np.array([LABEL_TO_LETTER[int(i)] for i in y_class])

    X = df.drop(columns=["label"]).values.astype("float32").reshape(-1, 28, 28)

    rng = np.random.default_rng(seed)
    chosen_imgs, chosen_letters, chosen_indices = [], [], []

    for L in letters:
        idxs = np.where(y_letters == L)[0]
        if idxs.size == 0:
            raise ValueError(f"No samples for letter {L} in {csv_path}")
        if per_class > idxs.size:
            raise ValueError(
                f"Requested per_class={per_class} for {L}, "
                f"but dataset has only {idxs.size} samples."
            )
        take = rng.choice(idxs, size=per_class, replace=False)
        for idx in take:
            chosen_imgs.append(X[idx])
            chosen_letters.append(L)
            chosen_indices.append(int(idx))

    arr = np.stack(chosen_imgs, axis=0)  # (N, 28, 28)

    # Match training: raw pixels 0..255 -> divide by 255.0 -> 0..1
    arr /= 255.0

    return arr.astype("float32"), chosen_letters, chosen_indices


def maybe_save_pngs(arr: np.ndarray, letters: List[str], png_dir: Path):
    """Optional: save PNG previews using imageio (if installed)."""
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[INFO] PNGs requested but imageio not installed. Run: pip install imageio")
        return

    png_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, L) in enumerate(zip(arr, letters)):
        im = (img * 255).astype("uint8")
        imageio.imwrite(str(png_dir / f"{i:03d}_{L}.png"), im)


def main():
    ap = argparse.ArgumentParser(
        description="Select SL-MNIST images by letters and save for inference."
    )
    ap.add_argument("--csv_path", required=True,
                    help="Path to sign_mnist_train.csv or sign_mnist_test.csv")
    ap.add_argument("--letters", nargs="+",
                    help="Letters to pick, e.g., I L O V E U")
    ap.add_argument("--phrase",
                    help='Alternative to --letters, e.g., --phrase "I LOVE U"')
    ap.add_argument("--per_class", type=int, default=1,
                    help="How many images per requested letter")
    ap.add_argument("--out_dir", default="picks_out",
                    help="Output directory")
    ap.add_argument("--seed", type=int, default=1202,
                    help="RNG seed for reproducible sampling")
    ap.add_argument("--save_pngs", action="store_true",
                    help="Also write PNG previews to out_dir/pngs")
    args = ap.parse_args()

    if not args.letters and not args.phrase:
        raise SystemExit('Provide either --letters ... or --phrase "I LOVE U"')

    letters = normalize_letters(
        args.letters if args.letters else [args.phrase],
        allow_phrase=bool(args.phrase),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr, chosen_letters, chosen_indices = select_images(
        csv_path=args.csv_path,
        letters=letters,
        per_class=args.per_class,
        seed=args.seed,
    )

    npy_path = out_dir / "selected_images.npy"
    np.save(npy_path, arr)

    meta = {
        "requested": letters,
        "letters": chosen_letters,
        "indices": chosen_indices,
        "csv_path": str(Path(args.csv_path).resolve()),
        "per_class": args.per_class,
        "seed": args.seed,
        "note": "Stack order matches 'requested' sequence, each repeated per_class.",
    }
    meta_path = out_dir / "selected_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    if args.save_pngs:
        maybe_save_pngs(arr, chosen_letters, out_dir / "pngs")

    print(f"[OK] Saved images -> {npy_path}  (shape={arr.shape})")
    print(f"[OK] Saved meta   -> {meta_path}")
    if args.save_pngs:
        print(f"[OK] Preview PNGs -> {out_dir/'pngs'}")


if __name__ == "__main__":
    main()

