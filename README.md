# Signature Verification with Siamese Neural Networks

A deep learning approach to handwritten signature verification using a Siamese CNN architecture. The model learns to distinguish genuine signatures from forgeries by comparing pairs of signature images.

---

## Overview

This project trains a Siamese Neural Network on the [CEDAR Signature Dataset](http://www.cedar.buffalo.edu/NIJ/data/) to perform binary classification on signature pairs:

- **Genuine pair** → signatures match (label: `1`)
- **Forged pair** → one signature is a forgery (label: `0`)

The shared CNN encodes each signature into a 4096-dimensional feature vector; the L1 distance between the two vectors is passed to a sigmoid output layer to produce a similarity score.

---

## Architecture

```
Input A (105×105×1) ──┐
                       ├── Shared Base CNN ──> Feature A ──┐
Input B (105×105×1) ──┘                                    ├── L1 Distance ──> Dense(1, sigmoid) ──> Score
                                             Feature B ──┘
```

**Base CNN layers:**
- Conv2D(64, 10×10) → MaxPool
- Conv2D(128, 7×7) → MaxPool
- Conv2D(128, 4×4) → MaxPool
- Conv2D(256, 4×4)
- Flatten → Dense(4096, sigmoid)

---

## Requirements

```
tensorflow
numpy
matplotlib
scikit-learn
seaborn
Pillow
unrar (system package)
```

Install Python dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn Pillow
```

---

## Dataset Setup

The notebook is designed to run on **Google Colab** and downloads the CEDAR dataset automatically:

```python
!wget -q "http://www.cedar.buffalo.edu/NIJ/data/signatures.rar"
!apt-get install -y unrar
!unrar x signatures.rar
```

After extraction, the expected directory structure is:

```
signatures/
├── full_org/    # Genuine signatures
└── full_forg/   # Forged signatures
```

Update the paths in the notebook if running locally:

```python
genuine_path = '/content/signatures/full_org'
forged_path  = '/content/signatures/full_forg'
```

---

## Usage

Run the notebook cells in order:

1. **Setup** — import libraries and verify TensorFlow version
2. **Download Dataset** — fetch and extract the CEDAR signatures
3. **Explore Data** — visualize sample genuine and forged signatures
4. **Preprocessing** — resize images to 105×105, convert to grayscale, normalize to [0, 1]
5. **Pair Generation** — create positive (genuine–genuine) and negative (genuine–forged) pairs
6. **Train/Test Split** — 80/20 split with `random_state=42`
7. **Build Model** — construct the Siamese network
8. **Train** — 20 epochs, batch size 16, Adam optimizer (lr=0.0001), binary cross-entropy loss
9. **Evaluate** — accuracy, loss, confusion matrix, and classification report
10. **Visualize Predictions** — side-by-side view of signature pairs with predicted vs. actual labels
11. **Save Model** — exports to `signature_verification_model.h5`

---

## Outputs

| File | Description |
|---|---|
| `samples.png` | Grid of sample genuine and forged signatures |
| `training_curves.png` | Accuracy and loss curves over epochs |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `predictions.png` | Sample test predictions with scores |
| `signature_verification_model.h5` | Saved Keras model |

---

## Notes

- The notebook is optimized for **Google Colab** (GPU recommended for faster training).
- Pair generation creates a sliding window of up to 3 positive pairs per genuine signature, and one negative pair per genuine–forged match.
- Threshold for classification is `0.5` — scores above this are classified as **Genuine**.
