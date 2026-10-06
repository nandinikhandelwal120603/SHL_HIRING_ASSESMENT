"""
Baseline model definitions and factory functions for Phase 2.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


class MeanPredictor(BaseEstimator, RegressorMixin):
    """
    Trivial reference baseline: predicts the mean target value of training fold.
    """
    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)


def get_duration_ridge_pipeline(alpha: float = 1.0):
    """
    Diagnostic baseline predicting score solely from audio duration.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha, random_state=42))
    ])


def get_audio_ridge_pipeline(alpha: float = 10.0):
    """
    Model A: Linear L2 Regularized Ridge Regression on standardized acoustic features.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha, random_state=42))
    ])


def get_audio_rf_model(n_estimators: int = 150, max_depth: int = 6, random_state: int = 42):
    """
    Model B: Random Forest Regressor on acoustic features.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )


def get_audio_extratrees_model(n_estimators: int = 150, max_depth: int = 6, random_state: int = 42):
    """
    Model B variant: Extra Trees Regressor on acoustic features.
    """
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )


def get_audio_lgbm_model(n_estimators: int = 100, learning_rate: float = 0.05, max_depth: int = 4, random_state: int = 42):
    """
    Model C: LightGBM Regressor on acoustic features.
    """
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1
    )
