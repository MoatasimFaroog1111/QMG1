from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from qmg1.config import TrainingConfig
from qmg1.ml.model_factory import RegressorFactory, apply_training_lookback


@dataclass(frozen=True)
class HorizonMetrics:
    horizon_hours: int
    cv_splits: int
    rows_total: int
    rows_validation_total: int
    mae_usd_per_kg: float
    rmse_usd_per_kg: float
    smape_pct: float
    directional_accuracy_pct: float
    persistence_mae_usd_per_kg: float
    improvement_vs_persistence_pct: float
    residual_log_return_q10: float
    residual_log_return_q90: float


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    valid = denominator > 0
    if not valid.any():
        return 0.0
    return float(
        np.mean(2.0 * np.abs(y_pred[valid] - y_true[valid]) / denominator[valid])
        * 100.0
    )


def score_predictions(
    *,
    horizon_hours: int,
    rows_total: int,
    cv_splits: int,
    current_close: np.ndarray,
    actual_close: np.ndarray,
    predicted_close: np.ndarray,
    actual_log_return: np.ndarray,
    predicted_log_return: np.ndarray,
) -> HorizonMetrics:
    mae = float(mean_absolute_error(actual_close, predicted_close))
    rmse = float(math.sqrt(mean_squared_error(actual_close, predicted_close)))
    persistence_mae = float(mean_absolute_error(actual_close, current_close))

    actual_direction = np.sign(actual_close - current_close)
    predicted_direction = np.sign(predicted_close - current_close)
    directional_accuracy = float(
        np.mean(actual_direction == predicted_direction) * 100.0
    )
    improvement = (
        (persistence_mae - mae) / persistence_mae * 100.0
        if persistence_mae > 0
        else 0.0
    )
    residuals = actual_log_return - predicted_log_return

    return HorizonMetrics(
        horizon_hours=horizon_hours,
        cv_splits=cv_splits,
        rows_total=rows_total,
        rows_validation_total=len(actual_close),
        mae_usd_per_kg=mae,
        rmse_usd_per_kg=rmse,
        smape_pct=_smape(actual_close, predicted_close),
        directional_accuracy_pct=directional_accuracy,
        persistence_mae_usd_per_kg=persistence_mae,
        improvement_vs_persistence_pct=improvement,
        residual_log_return_q10=float(np.quantile(residuals, 0.10)),
        residual_log_return_q90=float(np.quantile(residuals, 0.90)),
    )


class WalkForwardEvaluator:
    """Walk-forward evaluation with target-time purge and optional recency window."""

    def __init__(self, config: TrainingConfig, model_factory: RegressorFactory) -> None:
        self.config = config
        self.model_factory = model_factory

    def evaluate(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        horizon_hours: int,
    ) -> HorizonMetrics:
        target_col = f"target_{horizon_hours}h"
        future_close_col = f"future_close_{horizon_hours}h"
        target_timestamp_col = f"target_timestamp_{horizon_hours}h"

        splitter = TimeSeriesSplit(n_splits=self.config.cv_splits)
        predicted_close_parts: list[np.ndarray] = []
        actual_close_parts: list[np.ndarray] = []
        current_close_parts: list[np.ndarray] = []
        actual_return_parts: list[np.ndarray] = []
        predicted_return_parts: list[np.ndarray] = []

        completed_folds = 0
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(frame), start=1):
            candidate_train = frame.iloc[train_idx]
            valid = frame.iloc[valid_idx]
            if candidate_train.empty or valid.empty:
                continue

            validation_start = valid.index[0]
            purged_train = candidate_train[
                candidate_train[target_timestamp_col] < validation_start
            ]
            train = apply_training_lookback(
                self.model_factory,
                purged_train,
                validation_start,
            )
            if train.empty:
                continue

            model = self.model_factory.create()
            model.fit(train[feature_columns], train[target_col])

            predicted_log_return = np.asarray(
                model.predict(valid[feature_columns]), dtype=float
            )
            current_close = valid["close"].to_numpy(dtype=float)
            actual_close = valid[future_close_col].to_numpy(dtype=float)
            actual_log_return = valid[target_col].to_numpy(dtype=float)
            predicted_close = current_close * np.exp(predicted_log_return)

            predicted_close_parts.append(predicted_close)
            actual_close_parts.append(actual_close)
            current_close_parts.append(current_close)
            actual_return_parts.append(actual_log_return)
            predicted_return_parts.append(predicted_log_return)
            completed_folds += 1

            print(
                f"  [CV {fold}/{self.config.cv_splits}] "
                f"model={self.model_factory.name} "
                f"train={len(train):,} validation={len(valid):,} "
                f"purged={len(candidate_train) - len(purged_train):,} "
                f"lookback_days={self.model_factory.lookback_days or 'all'}"
            )

        if not predicted_close_parts:
            raise ValueError("Walk-forward validation produced no usable folds")

        return score_predictions(
            horizon_hours=horizon_hours,
            rows_total=len(frame),
            cv_splits=completed_folds,
            current_close=np.concatenate(current_close_parts),
            actual_close=np.concatenate(actual_close_parts),
            predicted_close=np.concatenate(predicted_close_parts),
            actual_log_return=np.concatenate(actual_return_parts),
            predicted_log_return=np.concatenate(predicted_return_parts),
        )
