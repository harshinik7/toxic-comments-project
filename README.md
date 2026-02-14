# toxic-comments-project

# Toxic Comment Classification

Course project for detecting toxic comments in online news discussions.

## Project overview

This project uses a large dataset of news article comments to predict how **toxic** a comment is.  
Each comment has a numeric toxicity score and additional labels such as severe toxicity, insult, and identity attack.

The first notebook (`01_exploratory_analysis.ipynb`) explores:
- Basic dataset structure (rows, columns, example comments)
- Comment length distribution
- Distribution of toxicity scores
- Class imbalance between toxic and non‑toxic comments
- Comment volume over time

## Data

The data consists of about 1.8 million comments with:
- Comment text and a toxicity score (`target`)
- Extra toxicity labels (e.g., `severe_toxicity`, `identity_attack`, `insult`, `threat`)
- Identity attributes (e.g., `asian`, `female`, `muslim`, `black`)
- Metadata such as `created_date`, `article_id`, and reaction counts (`likes`, `funny`, `wow`, etc.)

(Replace this section with your actual data source and link, if provided by the course.)

## Goals

- Understand patterns of toxic vs non‑toxic comments
- Build models to predict toxicity from raw text
- Explore how class imbalance and different features affect model performance

## Repository structure

- `01_exploratory_analysis.ipynb` – initial EDA on the comments dataset  
- `data/` – data files or data‑loading scripts (if allowed by the course)  
- `README.md` – project description (this file)

## Next steps

Planned work includes:
- Text preprocessing and feature engineering
- Training baseline models (e.g., logistic regression) for toxicity prediction
- Trying more advanced models and comparing performance
