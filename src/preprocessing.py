import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path, sample_size=200000):
    """Loads the dataset and binarizes the target."""
    df = pd.read_csv(path)
    # Binary label: 1 if target >= 0.5, else 0
    df['is_toxic'] = (df['target'] >= 0.5).astype(int)
    if sample_size and sample_size < len(df):
        # Stratified sampling would be better but requires more imports
        df = df.sample(n=sample_size, random_state=42)
    return df

def get_vectorizer(max_features=5000):
    """Returns a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 1)
    )
