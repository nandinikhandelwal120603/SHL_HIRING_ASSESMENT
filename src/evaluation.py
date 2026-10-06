"""
Validation framework and metric tracking for SHL Grammar Scoring Engine.
Implements continuous stratified 5-fold cross validation, RMSE, Pearson r,
and out-of-fold experiment logging.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error


def create_stratified_bins(y: pd.Series) -> np.ndarray:
    """
    Creates discrete stratification bins from continuous rubric scores.
    Handles rare classes ([1.0, 1.5] have only 4 total samples) by grouping
    them with 2.0 so every fold has representative low-score examples.
    """
    bins = np.zeros(len(y), dtype=int)
    for idx, val in enumerate(y):
        if val <= 0.2:
            bins[idx] = 0
        elif val <= 2.2:
            # Group rare 1.0 (1 sample), 1.5 (3 samples), and 2.0
            bins[idx] = 1
        elif val <= 2.7:
            bins[idx] = 2
        elif val <= 3.2:
            bins[idx] = 3
        elif val <= 3.7:
            bins[idx] = 4
        elif val <= 4.2:
            bins[idx] = 5
        elif val <= 4.7:
            bins[idx] = 6
        else:
            bins[idx] = 7
    return bins


def get_stratified_folds(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Assigns each sample to a fold [0, n_splits-1] using StratifiedKFold on target bins.
    """
    bins = create_stratified_bins(df["label"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.zeros(len(df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(df, bins)):
        folds[val_idx] = fold
    return folds


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Safely computes Pearson correlation, returning 0.0 if predictions have zero variance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.all(y_pred == y_pred[0]) or np.std(y_pred) < 1e-12:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(0.0 if np.isnan(r) else r)


def evaluate_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    model_factory,
    clip_range: tuple = (0.0, 5.0)
) -> dict:
    """
    Executes leak-free 5-fold cross validation.
    
    Parameters:
    - X: feature dataframe
    - y: ground-truth target array
    - folds: fold assignments [0..4]
    - model_factory: callable returning a new untrained model/pipeline
    - clip_range: tuple to clip predictions to valid score range [0, 5]
    
    Returns:
    - dict containing fold metrics, pooled OOF metrics, training RMSE, and OOF predictions.
    """
    n_splits = len(np.unique(folds))
    oof_preds = np.zeros(len(y), dtype=float)
    fold_train_rmses = []
    fold_val_rmses = []
    fold_val_pearsons = []
    fitted_models = []
    
    for fold in range(n_splits):
        train_mask = (folds != fold)
        val_mask = (folds == fold)
        
        X_train, y_train = X.iloc[train_mask], y[train_mask]
        X_val, y_val = X.iloc[val_mask], y[val_mask]
        
        model = model_factory()
        model.fit(X_train, y_train)
        fitted_models.append(model)
        
        # Training evaluation
        train_pred = model.predict(X_train)
        if clip_range:
            train_pred = np.clip(train_pred, clip_range[0], clip_range[1])
        tr_rmse = root_mean_squared_error(y_train, train_pred)
        fold_train_rmses.append(tr_rmse)
        
        # Validation evaluation
        val_pred = model.predict(X_val)
        if clip_range:
            val_pred = np.clip(val_pred, clip_range[0], clip_range[1])
        oof_preds[val_mask] = val_pred
        
        v_rmse = root_mean_squared_error(y_val, val_pred)
        v_pearson = safe_pearsonr(y_val, val_pred)
        
        fold_val_rmses.append(v_rmse)
        fold_val_pearsons.append(v_pearson)
        
    oof_rmse = root_mean_squared_error(y, oof_preds)
    oof_pearson = safe_pearsonr(y, oof_preds)
    mean_tr_rmse = float(np.mean(fold_train_rmses))
    
    return {
        "cv_rmse_mean": float(np.mean(fold_val_rmses)),
        "cv_rmse_std": float(np.std(fold_val_rmses)),
        "cv_pearson_mean": float(np.mean(fold_val_pearsons)),
        "cv_pearson_std": float(np.std(fold_val_pearsons)),
        "oof_rmse": float(oof_rmse),
        "oof_pearson": float(oof_pearson),
        "train_rmse": mean_tr_rmse,
        "oof_preds": oof_preds,
        "fitted_models": fitted_models
    }


class ExperimentLogger:
    """
    Logs and tracks experimental results across iterations into artifacts/experiment_log.csv.
    """
    def __init__(self, log_path: str = "artifacts/experiment_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        if os.path.exists(log_path):
            self.df = pd.read_csv(log_path)
        else:
            self.df = pd.DataFrame(columns=[
                "experiment_id", "features", "model",
                "cv_rmse_mean", "cv_rmse_std",
                "cv_pearson_mean", "cv_pearson_std",
                "oof_rmse", "oof_pearson", "train_rmse", "notes"
            ])
            
    def log(
        self,
        exp_id: str,
        features: str,
        model: str,
        results: dict,
        notes: str = ""
    ):
        row = {
            "experiment_id": exp_id,
            "features": features,
            "model": model,
            "cv_rmse_mean": round(results["cv_rmse_mean"], 4),
            "cv_rmse_std": round(results["cv_rmse_std"], 4),
            "cv_pearson_mean": round(results["cv_pearson_mean"], 4),
            "cv_pearson_std": round(results["cv_pearson_std"], 4),
            "oof_rmse": round(results["oof_rmse"], 4),
            "oof_pearson": round(results["oof_pearson"], 4),
            "train_rmse": round(results["train_rmse"], 4),
            "notes": notes
        }
        # Update or append
        if exp_id in self.df["experiment_id"].values:
            self.df.loc[self.df["experiment_id"] == exp_id] = row
        else:
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.log_path, index=False)
        return self.df
        
    def summary(self) -> pd.DataFrame:
        return self.df.copy()
