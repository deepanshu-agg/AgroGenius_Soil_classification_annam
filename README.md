# AgroGenius - Soil Classification Challenges

This repository contains the solutions developed by **Team AgroGenius** for two distinct soil classification challenges. Our team members include: Manisha Saini, Radhika Bhati, Deepanshu Aggarwal, Sayantan, and Sejal Kumari.

Below you will find details for each challenge, including our approach, the challenges we faced, how we overcame them, and our final results.

## Repository Structure

The repository is organized into two main folders, one for each challenge:
```
.
├── challenge-1/
│ ├── data/ # Contains or is intended for the dataset for Challenge 1
│ ├── docs/ # Documentation, including architecture.png for Challenge 1
│ ├── notebooks/ # Jupyter notebook for Challenge 1 (e.g., challenge_1_solution.ipynb)
│ └── src/ # Source code, including requirements.txt for Challenge 1
│
└── challenge-2/
├── data/ # Contains or is intended for the dataset for Challenge 2
├── docs/ # Documentation, including architecture.png for Challenge 2
├── notebooks/ # Jupyter notebook for Challenge 2 (e.g., challenge_2_solution.ipynb)
└── src/ # Source code, including requirements.txt for Challenge 2
```

*   **`data/`**: This folder is intended to store the datasets for each respective challenge. For running the Kaggle notebooks, you will typically need to add the data directly within the Kaggle environment from the competition's data source.
*   **`docs/`**: Contains supplementary documentation, most notably `architecture.png` which visually represents the model architecture used for that challenge.
*   **`notebooks/`**: Holds the primary Jupyter/Kaggle notebook containing the code, analysis, and execution pipeline for each challenge.
*   **`src/`**: Contains source utility files, if any, and the `requirements.txt` file specifying the Python dependencies for each challenge.

## General Setup and Prerequisites

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-url>
    ```
---

## Challenge 1: Soil-Type Classification

This challenge focused on classifying images into one of four soil types: Alluvial, Red, Black, and Clay.

### How to Run Challenge 1

1.  Navigate to the `challenge-1/notebooks/` directory.
2.  The primary solution is within a Jupyter notebook.
3.  This notebook is designed to be run on Kaggle.
    *   Upload the notebook to your Kaggle account.
    *   Add the dataset for this challenge from the specific Kaggle competition link.
    *   Run the cells in the Kaggle notebook environment.
4.  The required Python packages are listed in `challenge-1/src/requirements.txt`. Ensure these are available in your Kaggle environment or local setup if you adapt the notebook.
5.  The model architecture diagram can be found in `challenge-1/docs/architecture.png`.

### Challenge 1: Summary

**Approach of Solving the Problem:**
We implemented an end-to-end pipeline focusing on robustness. After ingesting `train_labels.csv` (1,222 images across Alluvial, Red, Black, Clay classes), we performed a stratified 80/20 train/validation split. Our custom `SoilDataset` handled image loading (RGB, 224x224 resize), training-time augmentations (random flips, rotations, color jitter), and ImageNet normalization. To address significant class imbalance (e.g., Alluvial: 528 vs. Clay: 199), we calculated per-class weights. These weights were used in a custom Focal Loss (γ=2.0) to prioritize harder, minority-class examples, and with a `WeightedRandomSampler` to ensure balanced mini-batches. Our model backbone was a pre-trained Swin Transformer "tiny" (from `timm`), with its head replaced by a linear layer for our four soil classes. We used AdamW optimizer with differential learning rates (1e-5 for backbone, 1e-4 for classifier head) and 0.01 weight decay. Training utilized mixed precision (`autocast`, `GradScaler`) for 50 epochs, logging losses, accuracies, and weighted F1 scores. A `ReduceLROnPlateau` scheduler (factor 0.5, patience 3) adjusted learning rates based on validation loss, and model checkpoints were saved on validation accuracy improvements.

**Challenges Faced:**
The primary hurdle was severe class imbalance, with Alluvial soil images being over 2.5 times more frequent than Clay soil, leading to model bias towards the majority class. We also encountered significant validation loss spikes during training, which risked disrupting convergence. Early iterations yielded sub-optimal accuracy and F1 scores, falling short of top-tier performance.

**How We Overcame the Challenges:**
Class imbalance was tackled using Focal Loss, which de-emphasized easy majority-class examples, and `WeightedRandomSampler` for balanced batch composition. Mixed-precision training (`autocast`, `GradScaler`) helped accelerate convergence and offered a slight regularizing effect. The `ReduceLROnPlateau` learning rate scheduler dynamically adjusted rates based on validation loss stagnation (factor 0.5, patience 3), smoothing the training curve. These combined strategies effectively mitigated loss spikes and improved performance on minority classes.

**Final Observation and Leaderboard Score:**
Our model demonstrated consistent convergence over 25 epochs: training loss dropped from ~0.12 to ~0.0025, and validation loss decreased from 0.0793 to a low of 0.0154 (epoch 21). Validation accuracy peaked at 98.8%, and the weighted F1 score reached 0.9878. The learning rate scheduler effectively managed validation loss spikes. Our best checkpoint (Val Loss ≈ 0.0154, Val Acc ≈ 0.9878, Val F1 ≈ 0.9878) was achieved at epoch 21.
**This model achieved Rank 96 on the competition leaderboard.** This success validates our approach of combining stratified sampling, weighted focal loss, adaptive learning rates, and a Swin Transformer backbone to effectively handle class imbalance and achieve near state-of-the-art performance.

---

## Challenge 2: Soil vs. Non-Soil Classification

This challenge aimed to distinguish between images containing soil and those that do not, framed as an anomaly detection problem.

### How to Run Challenge 2

1.  Navigate to the `challenge-2/notebooks/` directory.
2.  The primary solution is within a Jupyter notebook.
3.  This notebook is designed to be run on Kaggle.
    *   Upload the notebook to your Kaggle account.
    *   Add the dataset for this challenge from the "Soil Classification Part 2" Kaggle competition.
    *   Run the cells in the Kaggle notebook environment.
4.  The required Python packages are listed in `challenge-2/src/requirements.txt`. Ensure these are available in your Kaggle environment or local setup if you adapt the notebook.
5.  The model architecture diagram can be found in `challenge-2/docs/architecture.png`.

### Challenge 2: Summary

**Approach of Solving the Problem:**
Our core strategy was anomaly detection: if a model deeply understands "normal" soil images, deviations can be flagged as "non-soil." This involved a two-stage process:
1.  **Feature Extraction:** We utilized a pre-trained ResNet50 model to extract rich 2048-dimensional feature vectors from each image, converting them into compact numerical representations.
2.  **Autoencoder for Anomaly Detection:** We designed and trained an Autoencoder neural network *exclusively* on features from known soil images (training set). The Autoencoder learns to compress these soil features into a bottleneck and then reconstruct them accurately.
The underlying logic is that soil image features will be reconstructed with low error, while non-soil image features (unseen during training) will result in high reconstruction error. A threshold on this error, determined by a percentile of training set errors, was used for classification.

**Challenges Faced:**
1.  **Defining "Normalcy":** Soil exhibits natural variations. The Autoencoder needed to learn a robust "soil-ness" representation that was specific enough for anomaly detection yet general enough for these variations.
2.  **Feature Dimensionality:** ResNet50 features are high-dimensional (2048). Designing an effective Autoencoder architecture (layers, bottleneck size) to manage this was crucial.
3.  **Optimizing the Autoencoder:** Balancing the Autoencoder's complexity to learn meaningful patterns without overfitting or merely learning an identity function required tuning hyperparameters like bottleneck size, learning rate, epochs, and dropout.
4.  **Threshold Sensitivity:** The final classification heavily relied on the reconstruction error threshold; an improperly set threshold would lead to many misclassifications.

**How We Overcame the Challenges:**
1.  **Robust Design:** Strong ResNet50 features provided a good starting point. The Autoencoder architecture was iteratively refined with dropout layers for generalization and a structure for progressive compression/decompression.
2.  **Standardization & Focused Training:** Features were standardized before Autoencoder training. Training exclusively on soil image features sharpened its ability to model "soil."
3.  **Data-Driven Thresholding:** The anomaly threshold was based on the 96th percentile of reconstruction errors from the training (soil) data, making it empirically derived rather than arbitrary.
4.  **Iterative Experimentation:** We iteratively adjusted hyperparameters (learning rate, batch size, Autoencoder dimensions) by evaluating reconstruction errors on known soil images to find an optimal configuration.

**Final Observation and Leaderboard Score:**
Our anomaly detection approach using ResNet50 features and a specialized Autoencoder proved highly effective. The model successfully learned to distinguish soil patterns and identify deviations.
**This strategy resulted in a leaderboard score of 91%.** This validates our hypothesis that accurately modeling the "normal" class (soil) allows for effective identification of anomalies (non-soil) based on reconstruction fidelity. The Autoencoder became a proficient judge of "soil-like" features, key to its success.

---



