# Toxicity Detection in News Comments

This repository contains code and experiments for identifying and classifying toxic content in online news discussions. Leveraging the **Civil Comments (Jigsaw)** dataset, this project benchmarks traditional machine learning baselines against modern transformer-based architectures and topic-modeling approaches to improve moderation accuracy in imbalanced data scenarios.

## ⚡ Quickstart

```bash
# Clone the repository
git clone https://github.com/harshinik7/toxic-comments-project.git
cd toxic-comments-project

# Install dependencies
pip install -r requirements.txt
```

## 📖 Project Scope & Research Questions

Our research focuses on the challenges of real-world content moderation, specifically handling class imbalance and capturing subtle context.

*   **RQ1: Class Imbalance Mitigation** – How does severe class imbalance (8% toxic) affect classification performance, and which rebalancing strategies (SMOTE vs. Class-Weighting) most effectively improve the recall of toxic comments?
*   **RQ2: Representation Quality** – Do transformer-based text embeddings (DistilBERT) improve the detection of subtle toxicity (sarcasm, implied insults) compared to traditional Bag-of-Words/TF-IDF models?
*   **RQ3: Thematic Risk Analysis** – How do discussion topics relate to toxicity levels, and can features from Latent Dirichlet Allocation (LDA) improve classification accuracy?

## 🚀 Features

*   **Course Techniques:** TF-IDF vectorization, Logistic Regression, and SVM classifiers.
*   **Advanced Techniques:** Pretrained **DistilBERT** fine-tuning for contextual representations.
*   **Imbalance Handling:** Implementation of **SMOTE** and cost-sensitive learning (class-weighting).
*   **Unsupervised Learning:** Topic discovery using **LDA** to identify high-risk conversational themes.
*   **Robust Evaluation:** Precision-Recall curves and focused error analysis on borderline (0.3–0.7) toxicity scores.

## 📂 Repository Structure

```text
toxic-comments-project/
├── checkpoints/             # Project milestone notebooks
│   ├── checkpoint_1.ipynb   # EDA and data cleaning
│   └── checkpoint_2.ipynb   # Research questions & methodology
├── data/                    # Data access instructions and samples
├── assets/                  # Figures, EDA plots, and visualizations
├── main_notebook.ipynb      # Final curated project deliverable
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 📊 Methodology Summary

Our benchmarking approach isolates model performance by:
1.  **Establishing Baselines:** Using linear models on high-dimensional sparse features.
2.  **Evaluating Representations:** Comparing fixed vs. contextual embeddings.
3.  **Thematic Augmentation:** Measuring the uplift provided by adding unsupervised topic proportions to supervised features.

## ✨ Citation

If you use this work, please cite:
```text
@misc{harshini2026toxic,
  author = {Harshini},
  title = {Toxicity Detection in News Comments},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/harshinik7/toxic-comments-project}}
}
```
