import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False

    def fit(self, data):
        self.model.fit(data)
        self.is_fitted = True

    def predict(self, data):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        # -1 for anomalies, 1 for normal
        return self.model.predict(data)

    def get_scores(self, data):
        return self.model.decision_function(data)

def generate_data(n_samples=1000):
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(n_samples, 2)
    X_train = np.r_[X + 2, X - 2]
    # Add some outliers
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2) 
    return np.r_[X_train, X_outliers]

if __name__ == "__main__":
    data = generate_data()
    detector = AnomalyDetector()
    detector.fit(data)
    predictions = detector.predict(data)
    print(f"Detected {np.sum(predictions == -1)} anomalies.")
