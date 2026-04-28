# Toxicity Detection in News Comments

Online platforms use machine learning to flag toxic content, but real-world data is often highly imbalanced. This project investigates how severe class imbalance (only ~8% toxic) affects classification performance and demonstrates how rebalancing strategies can double the recall of toxic comments compared to naive baselines.

👉 **Start here:** [`main_notebook.ipynb`](./main_notebook.ipynb)

🎥 **Project Video:** https://www.youtube.com/watch?v=-QZ2_sDofZU

## ⚡ Quickstart

```bash
# Clone the repository
git clone https://github.com/harshinik7/toxic-comments-project.git
cd toxic-comments-project

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
```

## 📖 Project Scope & Research Questions

Our research focuses on the challenges of real-world content moderation, specifically handling class imbalance and capturing subtle context.

*   **RQ1: Class Imbalance Mitigation** – How does severe class imbalance (8% toxic) affect classification performance, and which rebalancing strategies (SMOTE vs. Class-Weighting) most effectively improve the recall of toxic comments?
*   **RQ2: Representation Quality** – Do transformer-based text embeddings improve detection of subtle toxicity compared to traditional Bag-of-Words/TF-IDF models?
*   **RQ3: Thematic Risk Analysis** – How do discussion topics relate to toxicity levels?

📁 Data

We use a news comments dataset labeled for toxicity (toxic vs. non-toxic). The training data is expected at:

- `data/train.csv` – main labeled dataset used in `main_notebook.ipynb`

Preprocessing steps (tokenization, text cleaning, rebalancing) are implemented in the notebook and helper scripts under `src/`, so you can reproduce the pipeline end-to-end from the Colab environment.


## 📊 Results Summary

Using **class-weighting** and **threshold tuning**, we successfully increased toxic comment recall from **0.346 to 0.619** (a nearly 2x improvement).

| Model | Toxic Precision | Toxic Recall | Toxic F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Always Non-Toxic | 0.000 | 0.000 | 0.000 | 0.920 |
| LogReg (Baseline) | 0.796 | 0.346 | 0.482 | 0.941 |
| **LogReg (Balanced + Tuned)** | **0.502** | **0.619** | **0.555** | **0.920** |

## 🚀 Features & DRES Components

- **Running Experiments:** Use `scripts/evaluate.py` to train and evaluate models from the command line.
- **Testing:** Unit tests for data preprocessing logic are located in `tests/`.
- **Environment Management:** Configuration via `.env` file for data paths and model hyper-parameters.
- **Data Processing:** Clean, modular scripts in `src/` for reproducible data pipelines.

## 📂 Repository Structure

```text
toxic-comments-project/
├── assets/           # Figures and visualizations
├── checkpoints/      # Milestone notebooks
├── data/             # Data access instructions
├── scripts/          # Command-line experiment scripts
├── src/              # Modular source code
├── tests/            # Unit tests
├── .env.example      # Template for environment variables
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## 🛠️ Reproduction & Testing

### Running Tests
```bash
python -m unittest discover tests
```

### Running Experiments
```bash
python scripts/evaluate.py data/train.csv
```
### Colab Environment

This project was developed in Google Colab. The dependency file at the root of the repository (`requirements.txt`) was generated from the Colab session environment so the package versions used for development are documented here.

To regenerate the environment file from Colab, run:

```python
!pip freeze > requirements.txt
from google.colab import files
files.download("requirements.txt")
```

Then replace the repository’s `requirements.txt` with the downloaded file.

### Python Version

Project environment used in Colab:

```bash
Python 3.12.13
```

You can verify the version in Colab with:

```python
!python --version
```
### Key Dependencies & Versions

Some key packages used in this project:

- Python 3.12.13
- pandas 2.x
- scikit-learn 1.x
- numpy 1.x
- matplotlib / seaborn for visualizations
- (Add any NLP / model libraries you actually used, e.g., transformers, torch, etc.)

See `requirements.txt` for the full, Colab-generated environment.


## ✨ Citation
```text
@misc{harshini2026toxic,
  author = {Harshini},
  title = {Toxicity Detection in News Comments},
  year = {2026},
  url = {https://github.com/harshinik7/toxic-comments-project}
}
```
