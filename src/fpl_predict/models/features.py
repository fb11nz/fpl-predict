from __future__ import annotations
import pandas as pd
from ..utils.io import read_parquet
from ..utils.cache import PROC
from ..utils.logging import get_logger

log = get_logger(__name__)

def load_model_features() -> pd.DataFrame:
    """Load features for model training. Prefers training_data.parquet with targets."""
    
    # Try to load complete training data first (features + targets)
    training_data_file = PROC / "training_data.parquet"
    if training_data_file.exists():
        log.info("Loading training data with actual match performance targets")
        return read_parquet(training_data_file)
    
    # Try to build training data if it doesn't exist
    log.info("Training data not found, building from match data...")
    try:
        from ..data.build_training_targets import build_training_dataset
        build_training_dataset()
        if training_data_file.exists():
            return read_parquet(training_data_file)
    except Exception as e:
        log.warning(f"Failed to build training data: {e}")
    
    # Fallback to basic features (no training targets)
    log.warning("Using basic features without training targets - model will use priors only")
    return read_parquet(PROC / "features.parquet")
