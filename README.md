# Toxicity Detection in News Comments

Online platforms use machine learning to flag toxic content, but real-world data is often highly imbalanced. This project investigates how severe class imbalance (only ~8% toxic) affects classification performance and demonstrates how rebalancing strategies can double the recall of toxic comments compared to naive baselines.

👉 **Start here:** [`main_notebook.ipynb`](./main_notebook.ipynb)

🎥 **Project Video:** [PASTE YOUR VIDEO LINK HERE]

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

## 📊 Results Summary

Using **class-weighting** and **threshold tuning**, we successfully increased toxic comment recall from **0.346 to 0.619** (a nearly 2x improvement) while maintaining reasonable precision. This demonstrates that accuracy is a misleading metric for imbalanced moderation tasks, and that model calibration is critical for safety-critical applications.

## 🚀 Features

*   **Course Techniques:** TF-IDF vectorization, Logistic Regression, and SVM classifiers.
*   **Advanced Techniques:** Pretrained **DistilBERT** fine-tuning for contextual representations.
*   **Imbalance Handling:** Implementation of **SMOTE** and cost-sensitive learning (class-weighting).
*   **Unsupervised Learning:** Topic discovery using **LDA** to identify high-risk conversational themes.

## 📂 Repository Structure

```text
toxic-comments-project/
├── checkpoints/             # Project milestone notebooks
│   ├── checkpoint_1.ipynb   # Dataset selection and initial EDA
│   └── checkpoint_2.ipynb   # Research questions & methodology
├── data/                    # Data access instructions and samples
│   └── README.md            # Instructions for Kaggle dataset access
├── assets/                  # Figures, EDA plots, and visualizations
├── main_notebook.ipynb      # Final curated project deliverable
├── requirements.txt         # Full package list for reproduction
└── README.md                # This file
```

## 🛠️ Reproduction & Dependencies

This project was built using **Google Colab**. To reproduce our results:
1.  Ensure you have Python 3.10.12 installed.
2.  Install the required packages via `pip install -r requirements.txt`.
3.  Follow the notebook execution order: `checkpoint_1.ipynb` -> `checkpoint_2.ipynb` -> `main_notebook.ipynb`.

### Key Dependencies:
- **Python:** 3.10.12
- **Pandas:** 2.1.4
- **Scikit-learn:** 1.4.2
- **Imbalanced-learn:** 0.12.2
- **NLTK:** 3.8.1
- **Matplotlib:** 3.8.0

## ✨ Citation

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
