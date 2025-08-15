from __future__ import annotations
import pandas as pd
class CardsModel:
    def predict(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        n=len(X); return pd.Series([0.05]*n, index=X.index), pd.Series([0.005]*n, index=X.index)
