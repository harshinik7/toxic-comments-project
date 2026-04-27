import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocessing import load_data, get_vectorizer

def run_experiment(data_path):
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    X_text = df['comment_text'].astype(str)
    y = df['is_toxic'].values
    
    print("Vectorizing text...")
    vectorizer = get_vectorizer()
    X = vectorizer.fit_transform(X_text)
    
    print("Training balanced Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, y)
    
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.7).astype(int)
    
    print("
Results (Threshold = 0.7):")
    print(classification_report(y, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y, y_proba):.4f}")

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.csv'
    run_experiment(path)
