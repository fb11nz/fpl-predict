from __future__ import annotations
import pandas as pd
from ..utils.io import read_parquet
from ..utils.cache import PROC
def load_model_features() -> pd.DataFrame:
    return read_parquet(PROC / "features.parquet")
