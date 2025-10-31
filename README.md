# 🌱 AgroGenius — Soil Classification Challenges

This repository contains the solutions developed by **Team AgroGenius** for two soil classification challenges.
**Team Members:** Manisha Saini, Radhika Bhati, Deepanshu Aggarwal, Sayantan, and Sejal Kumari.

Each challenge directory includes our approach, methodologies, key challenges, solutions, and final outcomes.

---

## 🧭 Repository Structure

```
.
├── challenge-1/
│   ├── data/         # Dataset for Challenge 1
│   ├── docs/         # Documentation and architecture diagrams
│   ├── notebooks/    # Jupyter notebooks (e.g., challenge_1_solution.ipynb)
│   └── src/          # Source code and requirements.txt
│
└── challenge-2/
    ├── data/         # Dataset for Challenge 2
    ├── docs/         # Documentation and architecture diagrams
    ├── notebooks/    # Jupyter notebooks (e.g., challenge_2_solution.ipynb)
    └── src/          # Source code and requirements.txt
```

### Folder Details

* **data/** – Contains or is intended for datasets. For Kaggle notebooks, datasets should be added directly in the Kaggle environment.
* **docs/** – Supplementary documentation (e.g., `architecture.png` illustrating model architecture).
* **notebooks/** – Main Jupyter/Kaggle notebooks with model training and evaluation pipelines.
* **src/** – Source utility files and dependency lists (`requirements.txt`).

---

## ⚙️ General Setup

To clone and explore locally:

```bash
git clone <repository-url>
cd AgroGenius_Soil_classification_annam
```

---

## 🌾 Challenge 1 — Soil Type Classification

### Objective

Classify images into one of four soil types: **Alluvial**, **Red**, **Black**, and **Clay**.

### 🧩 How to Run

1. Navigate to `challenge-1/notebooks/`.
2. Open and upload the main notebook to your **Kaggle** account.
3. Add the dataset from the corresponding Kaggle competition.
4. Run all cells in the Kaggle environment.
5. Ensure dependencies listed in `challenge-1/src/requirements.txt` are available.
6. Refer to `challenge-1/docs/architecture.png` for the model architecture.

---

### 🔍 Approach

We built an end-to-end, class-imbalance–resilient pipeline:

* **Data Split:** Stratified 80/20 train-validation split from 1,222 labeled images.
* **Preprocessing:** RGB resizing (224×224), random flips, rotations, color jitter, and ImageNet normalization.
* **Class Imbalance Handling:**

  * Computed per-class weights (e.g., Alluvial: 528 vs. Clay: 199).
  * Used **Focal Loss (γ = 2.0)** and **WeightedRandomSampler** for balanced mini-batches.
* **Model Architecture:**

  * Pre-trained **Swin Transformer (tiny)** backbone from `timm`.
  * Replaced classifier head for 4-class output.
* **Training Strategy:**

  * Optimizer: **AdamW** with differential learning rates (1e-5 backbone, 1e-4 head).
  * Mixed precision training (autocast + GradScaler).
  * Scheduler: **ReduceLROnPlateau** (factor 0.5, patience 3).
  * Epochs: 50, with checkpoints saved on validation accuracy improvements.

---

### ⚠️ Challenges Faced

* **Severe Class Imbalance:** Bias toward Alluvial soil.
* **Validation Loss Spikes:** Disrupted smooth convergence.
* **Initial Underperformance:** Early models underfit minority classes.

---

### 💡 Solutions

* **Focal Loss** reduced emphasis on easy samples.
* **WeightedRandomSampler** balanced class representation.
* **Mixed-Precision Training** accelerated convergence and stabilized updates.
* **Adaptive LR Scheduling** (ReduceLROnPlateau) smoothed loss curves and improved stability.

---

### 📈 Results

| Metric              | Value                  |
| ------------------- | ---------------------- |
| Training Loss       | ↓ from 0.12 → 0.0025   |
| Validation Loss     | ↓ to 0.0154 (epoch 21) |
| Validation Accuracy | **98.8%**              |
| Weighted F1 Score   | **0.9878**             |
| Leaderboard Rank    | **#69**                |

✅ The combination of stratified sampling, weighted focal loss, adaptive learning rates, and the Swin Transformer backbone yielded near–state-of-the-art results.

---

## 🌍 Challenge 2 — Soil vs. Non-Soil Classification

### Objective

Distinguish between **soil** and **non-soil** images using an anomaly detection framework.

---

### 🧩 How to Run

1. Navigate to `challenge-2/notebooks/`.
2. Upload the notebook to **Kaggle**.
3. Add the dataset from the “Soil Classification Part 2” competition.
4. Run all cells in the Kaggle environment.
5. Ensure dependencies in `challenge-2/src/requirements.txt` are installed.
6. Refer to `challenge-2/docs/architecture.png` for the model design.

---

### 🔍 Approach

We framed the problem as **anomaly detection**:

> A well-trained model on "normal" (soil) images should detect deviations (non-soil).

#### Pipeline:

1. **Feature Extraction**

   * Used pre-trained **ResNet50** to generate 2048-dimensional feature vectors per image.

2. **Autoencoder Training**

   * Trained exclusively on soil image features.
   * Learned to compress and reconstruct soil representations accurately.
   * Reconstruction errors on unseen (non-soil) samples served as anomaly indicators.

3. **Thresholding**

   * Classification threshold derived from the **96th percentile** of reconstruction errors on training data.

---

### ⚠️ Challenges Faced

* Defining “normal” given soil variability.
* Managing 2048-dimensional feature vectors.
* Avoiding Autoencoder overfitting or identity mapping.
* Sensitivity to anomaly threshold selection.

---

### 💡 Solutions

* Leveraged **ResNet50** features for robust embeddings.
* Added **dropout** and **progressive compression** in Autoencoder architecture.
* **Standardized features** before training for stability.
* Empirical thresholding via **percentile-based tuning**.
* Iteratively optimized hyperparameters (bottleneck size, learning rate, epochs, dropout).

---

### 📈 Results

| Metric            | Result                                                                          |
| ----------------- | ------------------------------------------------------------------------------- |
| Approach          | ResNet50 + Autoencoder                                                          |
| Training Strategy | Soil-only reconstruction learning                                               |
| Leaderboard Rank | **#32**                                                                         |
| Key Insight       | Modeling "normal" soil patterns enables effective anomaly-based soil detection. |

✅ The Autoencoder effectively captured the intrinsic “soil-ness” of images, enabling accurate identification of non-soil anomalies.

---

## 📚 Summary

This project demonstrates how deep learning and transfer learning can be effectively combined for domain-specific visual classification and anomaly detection.
Both challenges showcase robust methodologies to handle **class imbalance**, **high-dimensional features**, and **adaptive learning** within real-world agricultural AI tasks.

---

## 🧠 Tech Stack

* **Python** (PyTorch, NumPy, OpenCV, scikit-learn)
* **Transfer Learning** (Swin Transformer, ResNet50)
* **Deep Learning Frameworks:** PyTorch / timm
* **Environment:** Kaggle (GPU-enabled)
* **Visualization:** Matplotlib, Seaborn

---

## 🏁 Acknowledgments

We extend our gratitude to the organizers of the Soil Classification Challenges and the Kaggle community for providing datasets and an open platform for experimentation.
