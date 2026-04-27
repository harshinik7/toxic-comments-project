import unittest
import pandas as pd
from src.preprocessing import get_vectorizer

class TestPreprocessing(unittest.TestCase):
    def test_binarization(self):
        df = pd.DataFrame({'target': [0.1, 0.5, 0.9, 0.0]})
        df['is_toxic'] = (df['target'] >= 0.5).astype(int)
        self.assertEqual(df['is_toxic'].iloc[0], 0)
        self.assertEqual(df['is_toxic'].iloc[1], 1)

    def test_vectorizer(self):
        vec = get_vectorizer(max_features=10)
        self.assertEqual(vec.max_features, 10)

if __name__ == '__main__':
    unittest.main()
