# Data

This project uses the **Civil Comments** dataset (~1.8 million comments from online news article discussions).

Raw data files are **not committed** to this repository due to size constraints.

## Data Sources

- **TensorFlow Datasets (primary):**
  https://www.tensorflow.org/datasets/catalog/civil_comments

- **Kaggle - Jigsaw Unintended Bias in Toxicity Classification:**
  https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification

## How to Access the Data

### Option 1: Load via TensorFlow Datasets in Colab
```python
import tensorflow_datasets as tfds
dataset = tfds.load('civil_comments', split='train', shuffle_files=True)
```

### Option 2: Download from Kaggle
1. Accept the competition rules at the Kaggle link above.
2. Download `train.csv` and `test.csv`.
3. Place them in this `data/` folder:
   ```
   data/
   ├── train.csv
   └── test.csv
   ```
4. The notebooks will look for data files in this directory (or you can mount Google Drive in Colab).

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique comment ID |
| `comment_text` | string | Raw comment text |
| `target` | float | Toxicity score 0-1 (label threshold: >= 0.5 = toxic) |
| `severe_toxicity` | float | Severe toxicity score |
| `obscene` | float | Obscene content score |
| `identity_attack` | float | Identity attack score |
| `insult` | float | Insult score |
| `threat` | float | Threat score |
| `created_date` | datetime | When comment was posted |
| `article_id` | int | Article the comment belongs to |
| `asian`, `female`, `muslim`, `black`, ... | float | Identity attribute annotations |

## Class Imbalance Note

At a 0.5 threshold:
- **~92%** of comments are non-toxic (label = 0)
- **~8%** of comments are toxic (label = 1)

This severe class imbalance is a central focus of the project analysis.
