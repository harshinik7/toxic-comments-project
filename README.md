# Toxic Comments Project

A machine learning and NLP project on **toxic comment detection** using the Civil Comments dataset. This project investigates how severe class imbalance affects toxicity classification, and how rebalancing strategies and TF-IDF + Logistic Regression models can improve detection of toxic comments (which make up only ~8% of the data).

---

👉 **Start here:** [`main_notebook.ipynb`](./main_notebook.ipynb)

🎥 **Project video:** [PASTE YOUR VIDEO LINK HERE]

---

## Research Questions

1. How does severe class imbalance affect toxicity classification performance?
2. Which rebalancing strategies most effectively improve detection of toxic comments?
3. How do TF-IDF features compare across baseline and class-weighted Logistic Regression models?

---

## Repository Structure

```
toxic-comments-project/
├── README.md                   <- You are here
├── requirements.txt            <- Full dependency list exported from Colab
├── .gitignore
├── main_notebook.ipynb         <- Final curated deliverable notebook
├── checkpoints/
│   ├── checkpoint_1.ipynb      <- Checkpoint 1: dataset selection & initial EDA
│   └── checkpoint_2.ipynb      <- Checkpoint 2: research questions & methodology plan
├── data/
│   └── README.md               <- Data access instructions
└── assets/                     <- Figures and screenshots (optional)
```

---

## Project Overview

Online platforms use machine learning systems to flag toxic comments, but real moderation data is highly imbalanced. In the Civil Comments dataset, only about 8% of comments are toxic at a 0.5 threshold, meaning a naive model can achieve high accuracy while still missing most harmful content. This project studies that problem using TF-IDF text features and compares a naive baseline, standard Logistic Regression, and class-weighted Logistic Regression with threshold tuning.

---

## Data

This project uses the **Civil Comments** dataset (~1.8 million comments from online news discussions).

| Field | Description |
|---|---|
| `comment_text` | Raw comment text |
| `target` | Continuous toxicity score (0–1) |
| `severe_toxicity`, `insult`, `threat`, `identity_attack` | Additional toxicity labels |
| `created_date`, `article_id` | Metadata |
| `asian`, `female`, `muslim`, `black`, ... | Identity attribute annotations |

**Sources:**
- TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/civil_comments
- Kaggle Jigsaw Unintended Bias: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification

**Preprocessing steps:**
- Lowercasing and basic text cleaning
- Binary label creation: toxicity score ≥ 0.5 → toxic (1), else non-toxic (0)
- TF-IDF vectorization for model input
- Comment length feature engineering for EDA

See [`data/README.md`](./data/README.md) for instructions on accessing and placing the dataset.

---

## How to Reproduce

This project was developed in **Google Colab**.

1. Clone this repository:
   ```bash
   git clone https://github.com/harshinik7/toxic-comments-project.git
   cd toxic-comments-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from the sources listed above and place it as described in [`data/README.md`](./data/README.md).
4. Run notebooks in this order:
   - `checkpoints/checkpoint_1.ipynb` — dataset exploration
   - `checkpoints/checkpoint_2.ipynb` — methodology planning
   - `main_notebook.ipynb` — **final analysis and results**

---

## Key Dependencies

| Package | Version |
|---|---|
| Python | 3.10+ |
| pandas | 2.x |
| numpy | 1.x |
| matplotlib | 3.x |
| seaborn | 0.x |
| scikit-learn | 1.x |
| imbalanced-learn | 0.x |

> Update exact versions from your `requirements.txt` and `!python --version` output in Colab.

The full dependency list is in [`requirements.txt`](./requirements.txt).

---

## Results Summary

The Civil Comments dataset is highly imbalanced (~8% toxic, ~92% non-toxic), which makes plain accuracy misleading for moderation. Class-weighted Logistic Regression and threshold tuning significantly improve recall for the toxic class compared to a naive baseline, making the classifier much more useful for real-world content moderation.

Full analysis, visualizations, and model comparisons are in [`main_notebook.ipynb`](./main_notebook.ipynb).

---

## Checkpoints

- 📄 [`checkpoints/checkpoint_1.ipynb`](./checkpoints/checkpoint_1.ipynb) — Dataset selection, initial EDA, class imbalance exploration
- 📄 [`checkpoints/checkpoint_2.ipynb`](./checkpoints/checkpoint_2.ipynb) — Research questions, methodology planning, model strategy

---

## Notes

- Raw dataset files are **not committed** to this repo due to size. See [`data/README.md`](./data/README.md) for access instructions.
- All notebooks were developed and tested in **Google Colab**.
- To export your environment from Colab, run:
  ```python
  !pip freeze > requirements.txt
  from google.colab import files
  files.download('requirements.txt')
  ```
  Then commit the downloaded `requirements.txt` to the repo root.
