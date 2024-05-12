import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Optional, Tuple

class AnomalyDetector:
    """Unsupervised anomaly detection for streaming time-series data."""
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        self.model.fit(data)
        self.is_fitted = True
        print("Model fitted successfully.")

    def predict(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        # -1 for anomalies, 1 for normal
        return self.model.predict(data)

    def get_anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring.")
        return self.model.decision_function(data)

    def update_model(self, new_data: np.ndarray):
        """Simulate online learning by refitting with new data."""
        # IsolationForest doesn't support partial_fit, so we refit
        self.fit(new_data)

def generate_streaming_data(n_samples: int = 1000) -> np.ndarray:
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(n_samples, 2)
    X_train = np.r_[X + 2, X - 2]
    # Add some outliers
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    return np.r_[X_train, X_outliers]
