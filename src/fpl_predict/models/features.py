from __future__ import annotations
import pandas as pd
from ..utils.io import read_parquet
from ..utils.cache import PROC
from ..utils.logging import get_logger

log = get_logger(__name__)

def load_model_features(force_rebuild: bool = True) -> pd.DataFrame:
    """Load features for model training. Always rebuilds training data to ensure freshness.

    Args:
        force_rebuild: If True, always rebuild training data from latest match data.
                      This ensures models are trained on current player statistics.
    """

    training_data_file = PROC / "training_data.parquet"

    # ALWAYS rebuild training data when called during weekly updates
    # This fixes the bug where models were using stale data from months ago
    if force_rebuild or not training_data_file.exists():
        log.info("Building fresh training data from current match data...")
        try:
            from ..data.build_training_targets import build_training_dataset
            build_training_dataset()
            if training_data_file.exists():
                log.info("Successfully built training data with actual match performance targets")
                return read_parquet(training_data_file)
        except Exception as e:
            log.warning(f"Failed to build training data: {e}")

    # Only use existing file if force_rebuild is explicitly False
    if not force_rebuild and training_data_file.exists():
        log.info("Loading existing training data (force_rebuild=False)")
        return read_parquet(training_data_file)

    # Fallback to basic features (no training targets)
    log.warning("Using basic features without training targets - model will use priors only")
    return read_parquet(PROC / "features.parquet")
