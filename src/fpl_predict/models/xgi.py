from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import get_logger

log = get_logger(__name__)


class XGIModel:
    """
    Advanced goals and assists prediction using XGBoost with:
    - Hyperparameter optimization via Optuna
    - Time-series cross-validation
    - Uncertainty quantification
    - Position-specific models
    """
    
    def __init__(self, n_trials: int = 20, cv_splits: int = 3):
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.models_g = {}  # Position-specific goal models
        self.models_a = {}  # Position-specific assist models
        self.best_params_g = {}
        self.best_params_a = {}
        self.fitted = False
        self.feature_importance = {}
        
    def _optimize_hyperparams(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        task: str = "regression"
    ) -> dict:
        """Optimize XGBoost hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            }
            
            if task == "poisson":
                params['objective'] = 'count:poisson'
            else:
                params['objective'] = 'reg:squarederror'
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _create_position_features(self, X: pd.DataFrame, position: str) -> pd.DataFrame:
        """Add position-specific features."""
        X_pos = X.copy()
        
        # Position-specific feature engineering
        if position == "GKP":
            # Goalkeepers care less about attacking stats
            X_pos['defensive_form'] = X_pos.get('saves_l5', 0) * 2
            
        elif position == "DEF":
            # Defenders: balance of defensive and occasional attacking threat
            X_pos['attacking_threat'] = X_pos.get('prev_goals_per90', 0) * 1.5
            X_pos['set_piece_threat'] = X_pos.get('prev_xg_per90', 0) * 2
            
        elif position == "MID":
            # Midfielders: creativity and goal threat
            X_pos['creativity'] = X_pos.get('prev_xa_per90', 0) * X_pos.get('mins_l5', 0) / 90
            X_pos['goal_involvement'] = (
                X_pos.get('prev_goals_per90', 0) + X_pos.get('prev_assists_per90', 0)
            )
            
        elif position == "FWD":
            # Forwards: primarily goal threat
            X_pos['finishing_ability'] = X_pos.get('prev_xg_per90', 0) * 1.2
            X_pos['mins_momentum'] = X_pos.get('mins_l3', 0) / X_pos.get('mins_l5', 1).replace(0, 1)
        
        # Common derived features for all positions
        X_pos['form_trend'] = X_pos.get('form', 0) * X_pos.get('selected_by_percent', 0) / 100
        X_pos['fitness_score'] = X_pos.get('chance_next', 100) * X_pos.get('mins_l3', 0) / 9000
        X_pos['price_performance'] = X_pos.get('form', 0) / (X_pos.get('now_cost', 50) / 10)
        
        return X_pos
    
    def fit(self, X: pd.DataFrame, y_g: pd.Series, y_a: pd.Series, positions: pd.Series = None) -> None:
        """
        Fit position-specific models for goals and assists.
        
        Args:
            X: Feature matrix
            y_g: Goal targets
            y_a: Assist targets
            positions: Series indicating player positions
        """
        if len(X) == 0:
            return
        
        # If no positions provided, train a single model
        if positions is None:
            positions = pd.Series(['ALL'] * len(X), index=X.index)
        
        unique_positions = positions.unique()
        
        for pos in unique_positions:
            pos_mask = positions == pos
            if pos_mask.sum() < 10:  # Skip if too few samples
                continue
                
            X_pos = X[pos_mask]
            y_g_pos = y_g[pos_mask]
            y_a_pos = y_a[pos_mask]
            
            # Add position-specific features
            X_pos = self._create_position_features(X_pos, pos)
            
            # Optimize and train goal model
            if self.n_trials > 0 and len(X_pos) > 50:
                log.info(f"Optimizing hyperparameters for {pos} goals model...")
                self.best_params_g[pos] = self._optimize_hyperparams(X_pos, y_g_pos, task="poisson")
            else:
                self.best_params_g[pos] = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'objective': 'count:poisson'
                }
            
            self.models_g[pos] = xgb.XGBRegressor(
                **self.best_params_g[pos],
                random_state=42,
                verbosity=0
            )
            self.models_g[pos].fit(X_pos, y_g_pos)
            
            # Optimize and train assist model
            if self.n_trials > 0 and len(X_pos) > 50:
                log.info(f"Optimizing hyperparameters for {pos} assists model...")
                self.best_params_a[pos] = self._optimize_hyperparams(X_pos, y_a_pos, task="poisson")
            else:
                self.best_params_a[pos] = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'objective': 'count:poisson'
                }
            
            self.models_a[pos] = xgb.XGBRegressor(
                **self.best_params_a[pos],
                random_state=42,
                verbosity=0
            )
            self.models_a[pos].fit(X_pos, y_a_pos)
            
            # Store feature importance
            self.feature_importance[pos] = {
                'goals': pd.Series(
                    self.models_g[pos].feature_importances_,
                    index=X_pos.columns
                ).sort_values(ascending=False),
                'assists': pd.Series(
                    self.models_a[pos].feature_importances_,
                    index=X_pos.columns
                ).sort_values(ascending=False)
            }
        
        self.fitted = True
        log.info(f"Fitted models for positions: {list(self.models_g.keys())}")
    
    def predict_with_uncertainty(
        self, 
        X: pd.DataFrame, 
        positions: pd.Series = None,
        n_iterations: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Predict goals and assists with uncertainty quantification.
        
        Returns dict with 'goals' and 'assists' DataFrames containing:
        - mean: point prediction
        - std: standard deviation
        - lower: 5th percentile
        - upper: 95th percentile
        """
        if not self.fitted:
            n = len(X)
            return {
                'goals': pd.DataFrame({
                    'mean': [0.2] * n,
                    'std': [0.1] * n,
                    'lower': [0.1] * n,
                    'upper': [0.3] * n
                }, index=X.index),
                'assists': pd.DataFrame({
                    'mean': [0.1] * n,
                    'std': [0.05] * n,
                    'lower': [0.05] * n,
                    'upper': [.15] * n
                }, index=X.index)
            }
        
        if positions is None:
            positions = pd.Series(['ALL'] * len(X), index=X.index)
        
        # Initialize predictions
        preds_g = np.zeros((len(X), n_iterations))
        preds_a = np.zeros((len(X), n_iterations))
        
        for i, (idx, row) in enumerate(X.iterrows()):
            pos = positions.iloc[i] if i < len(positions) else 'ALL'
            
            # Use position-specific model if available
            if pos in self.models_g:
                X_pos = self._create_position_features(pd.DataFrame([row]), pos)
                
                # Generate multiple predictions with noise for uncertainty
                for j in range(n_iterations):
                    # Add small noise to features for uncertainty estimation
                    X_noisy = X_pos.copy()
                    noise_factor = 0.05
                    for col in X_noisy.columns:
                        if X_noisy[col].dtype in ['float64', 'int64']:
                            X_noisy[col] += np.random.normal(0, X_noisy[col].std() * noise_factor)
                    
                    preds_g[i, j] = max(0, self.models_g[pos].predict(X_noisy)[0])
                    preds_a[i, j] = max(0, self.models_a[pos].predict(X_noisy)[0])
            else:
                # Fallback to nearest position model or default
                fallback_pos = list(self.models_g.keys())[0] if self.models_g else None
                if fallback_pos:
                    X_pos = self._create_position_features(pd.DataFrame([row]), fallback_pos)
                    for j in range(n_iterations):
                        preds_g[i, j] = max(0, self.models_g[fallback_pos].predict(X_pos)[0])
                        preds_a[i, j] = max(0, self.models_a[fallback_pos].predict(X_pos)[0])
        
        # Calculate statistics
        results = {
            'goals': pd.DataFrame({
                'mean': preds_g.mean(axis=1),
                'std': preds_g.std(axis=1),
                'lower': np.percentile(preds_g, 5, axis=1),
                'upper': np.percentile(preds_g, 95, axis=1)
            }, index=X.index),
            'assists': pd.DataFrame({
                'mean': preds_a.mean(axis=1),
                'std': preds_a.std(axis=1),
                'lower': np.percentile(preds_a, 5, axis=1),
                'upper': np.percentile(preds_a, 95, axis=1)
            }, index=X.index)
        }
        
        return results
    
    def predict(self, X: pd.DataFrame, positions: pd.Series = None) -> Tuple[pd.Series, pd.Series]:
        """Simple prediction interface for backward compatibility."""
        preds = self.predict_with_uncertainty(X, positions, n_iterations=50)
        return preds['goals']['mean'], preds['assists']['mean']