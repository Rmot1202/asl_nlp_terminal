# AI-Based ASL Interpreter with NLP (Terminal-Only Prototype)

> CSCI 402 – AI & Data Mining • Final Project
> Author: **Raven Mott**
> Semester: Spring 2025

This repository contains a **terminal-based AI pipeline** that translates static American Sign Language (ASL) alphabet images into **English text**, then uses a lightweight **NLP step (via GPT)** to “naturalize” the output into fluent sentences.

The prototype focuses on:

* **Computer vision** using a CNN (Sign Language MNIST)
* **Letter sequence reconstruction** (ASL → letters → words)
* **NLP naturalization** (letters → readable English)

No GUI or mobile app is included here; everything runs from the command line to emphasize the **core AI pipeline**.

---
## Video Walkthrough


<img src='demo_AI.gif' title='Video Walkthrough' width='' alt='Video Walkthrough' />

<!-- Replace this with whatever GIF tool you used! -->
GIF created with ...  
[Kap](https://getkap.co/) for macOS

## 1. Project Overview

### Problem

Deaf and Hard-of-Hearing individuals frequently rely on ASL for communication, but hearing individuals often lack sign language skills or access to human interpreters. This project explores how an AI system can:

1. Recognize **ASL alphabet handshapes** from images
2. Convert them into a sequence of **letters and words**
3. Use **NLP** to reformat those sequences into more natural English sentences

Example:

> ASL images → `ITHANKU` → **“I thank you.”**

### Goals

* Build and train a **CNN** to classify ASL alphabet images
* Provide a **terminal script** that:

  * Loads a trained model
  * Runs inference on selected ASL images
  * Optionally calls **GPT** for sentence smoothing
* Demonstrate end-to-end translation on sample phrases using the **Sign Language MNIST** dataset

---

## 2. Repository Layout

Inside `2025 pj ai and datamining/`:

```text
.
├─ README.md                     # This file
├─ requirements.txt              # Python dependencies
├─ RUN.txt                       # Step-by-step run guide (detailed)
├─ train_cnn.py                  # Train CNN on Sign Language MNIST
├─ infer_and_nlp.py              # Inference + optional GPT naturalization
├─ pick_by_letters.py            # Utility: pick images for chosen letters/phrases
├─ utils.py                      # Shared helpers (data loading, mapping, etc.)
├─ artifacts/
│  └─ asl_cnn.keras              # Saved model (after training)
├─ picks_*/                      # Example letter picks from test set (created later)
└─ runs/
   └─ *_run.json                 # Optional run metadata and predictions
```

**Dataset expectation (not committed to Git):**

```text
2025 pj ai and datamining/
├─ sign_mnist_train.csv   # one level up OR in sibling folder
├─ sign_mnist_test.csv
└─ asl_nlp_terminal/      # this repo folder (optional naming)
```

> **Note:** Large files (virtual environments, dataset CSVs) should **not** be committed to Git. Use `.gitignore` to keep the repo lightweight.

---

## 3. Environment Setup

From the project root (this folder):

```bash
# 1) Create & activate virtual env (macOS / Linux)
python3 -m venv .venv-asl
source .venv-asl/bin/activate

# Windows (PowerShell)
# py -3 -m venv .venv-asl
# .\.venv-asl\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Key dependencies** (see `requirements.txt`):

* `numpy`, `pandas`, `scikit-learn`
* `opencv-python`
* `tensorflow==2.16.1`
* `pyspellchecker`
* `openai>=1.40.0`
* `python-dotenv>=1.0.1`

---

## 4. Training the CNN (ASL Alphabet Classifier)

The model is trained on **Sign Language MNIST** (28×28 grayscale images, 24 classes: A–Y excluding J and Z).

From inside the project folder:

```bash
python train_cnn.py \
  --train_csv ../sign_mnist_train.csv \
  --test_csv  ../sign_mnist_test.csv \
  --out_dir   artifacts
```

After training, you should see:

```text
artifacts/
└─ asl_cnn.keras
```

This file is the saved Keras model used for inference.

---

## 5. Selecting Test Images by Letters or Phrase

Before running full inference, use `pick_by_letters.py` to extract specific letters or phrases from the **test** split and save them as `.npy`:

### A) Phrase Mode (recommended)

```bash
python pick_by_letters.py \
  --csv_path ../sign_mnist_test.csv \
  --phrase  "I THANK YOU" \
  --per_class 1 \
  --out_dir  picks_ithankyou \
  --save_pngs
```

### B) Explicit Letters

```bash
python pick_by_letters.py \
  --csv_path ../sign_mnist_test.csv \
  --letters I T H A N K U \
  --per_class 1 \
  --out_dir  picks_ithankyou \
  --save_pngs
```

This creates:

```text
picks_ithankyou/
├─ selected_images.npy      # (N, 28, 28) images in [0,1]
└─ selected_meta.json       # letters, indices, reproducibility info
    # (plus pngs/ if --save_pngs was used)
```

> **Note:** The Sign Language MNIST dataset **does not include J and Z**, so the script will error if you request those.

---

## 6. Inference + NLP Naturalization

### 6.1 Optional: Configure GPT

If you want naturalized sentences (e.g., from `ITHANKU` → `I thank you.`), set:

```bash
export OPENAI_API_KEY="sk-..."    # macOS / Linux
# or on Windows (PowerShell):
# setx OPENAI_API_KEY "sk-..."
# $env:OPENAI_API_KEY="sk-..."
```

If `OPENAI_API_KEY` is **not** set, inference will still run; you just get **raw letter sequences**.

### 6.2 Run Inference

Example using the “I THANK YOU” picks:

```bash
python infer_and_nlp.py \
  --model_path artifacts/asl_cnn.keras \
  --npy_path   picks_ithankyou/selected_images.npy \
  --meta_json  picks_ithankyou/selected_meta.json \
  --gpt_model  gpt-4o-mini \
  --min_conf   0.0 \
  --save_meta  runs/ithankyou_run.json
```

You’ll see:

* Per-image predictions + confidences
* Raw letter sequence, e.g. `ITHANKU`
* **Naturalized output**, e.g. `I THANK YOU.`

Metadata (including letters, confidences, and GPT output) is saved in `runs/ithankyou_run.json`.

---

## 7. Example Result (From `ithankyou_run.json`)

Sample output (simplified):

```json
{
  "pred_letters": ["I", "T", "H", "A", "N", "K", "U"],
  "pred_confidences": [0.9999, 0.6866, 0.9467, 0.9999, 0.9955, 0.9277, 0.9991],
  "raw_sequence": "ITHANKU",
  "naturalized": "I THANK YOU.",
  "gpt_model": "gpt-4o-mini"
}
```

This demonstrates:

* **CNN** correctly classifies most letters
* **Sequence reconstruction** yields the correct phrase structure
* **NLP** produces a clear and readable English sentence

---

## 8. Troubleshooting

* **`ValueError: unsupported letters` in `pick_by_letters.py`**
  → You requested `J` or `Z`, which are not in Sign Language MNIST.

* **Very low confidences / wrong letters**

  * Try increasing `--per_class` (more image samples)
  * Retrain the model with more epochs or augmentation
  * Lower `--min_conf` if you’re filtering too aggressively

* **No GPT output**

  * Check that `OPENAI_API_KEY` is set in your environment
  * Make sure `openai` and `python-dotenv` are installed

---

## 9. For CSCI 402 Grading (Rubric Mapping)

* **Content & Understanding**

  * README + report clearly define the accessibility problem and relevance.
* **AI Methodology**

  * CNN architecture (computer vision) + GPT-based NLP naturalization.
* **Results & Analysis**

  * Quantitative metrics (accuracy) and qualitative phrase examples (e.g., “I THANK YOU”).

See the separate **project paper / slides** for more detailed analysis, results, and discussion.

---

## 10. License / Credits

* ASL data: based on the public **Sign Language MNIST** dataset.
* GPT model: uses the **OpenAI API** (requires your own API key).
* Code authored by **Raven Mott** for the **CSCI 402 – AI & Data Mining** course.
