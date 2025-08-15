from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class MinutesModel:
    """Basic minutes prediction model using Ridge regression."""
    
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the minutes model."""
        if len(X) == 0:
            return
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected minutes."""
        if not self.fitted:
            return np.full(len(X), 60.0)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, 90)