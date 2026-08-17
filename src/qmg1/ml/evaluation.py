from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from qmg1.config import TrainingConfig
from qmg1.ml.model_factory import RegressorFactory


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
    latest_fold_mae_usd_per_kg: float
    latest_fold_persistence_mae_usd_per_kg: float
    latest_fold_improvement_vs_persistence_pct: float
    latest_fold_directional_accuracy_pct: float
    fold_improvement_vs_persistence_pct: tuple[float, ...]
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


def _improvement_pct(model_mae: float, persistence_mae: float) -> float:
    if persistence_mae <= 0:
        return 0.0
    return (persistence_mae - model_mae) / persistence_mae * 100.0


class WalkForwardEvaluator:
    """Purged expanding/rolling walk-forward evaluation for one model candidate."""

    def __init__(
        self,
        config: TrainingConfig,
        model_factory: RegressorFactory,
        max_train_rows: int | None = None,
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.max_train_rows = max_train_rows

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
        residual_parts: list[np.ndarray] = []
        fold_improvements: list[float] = []
        latest_fold_mae = float("nan")
        latest_fold_persistence_mae = float("nan")
        latest_fold_directional_accuracy = float("nan")

        completed_folds = 0
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(frame), start=1):
            candidate_train = frame.iloc[train_idx]
            valid = frame.iloc[valid_idx]
            if candidate_train.empty or valid.empty:
                continue

            validation_start = valid.index[0]
            # Purge every training sample whose target uses a price from the
            # validation period or later. This is stricter and more exact than
            # assuming one dataframe row equals one elapsed hour.
            train = candidate_train[
                candidate_train[target_timestamp_col] < validation_start
            ]
            purged = len(candidate_train) - len(train)
            if train.empty:
                continue

            before_window = len(train)
            if self.max_train_rows is not None and len(train) > self.max_train_rows:
                train = train.iloc[-self.max_train_rows :]
            window_trimmed = before_window - len(train)

            model = self.model_factory.create()
            model.fit(train[feature_columns], train[target_col])

            predicted_log_return = np.asarray(
                model.predict(valid[feature_columns]), dtype=float
            )
            current_close = valid["close"].to_numpy(dtype=float)
            actual_close = valid[future_close_col].to_numpy(dtype=float)
            actual_log_return = valid[target_col].to_numpy(dtype=float)
            predicted_close = current_close * np.exp(predicted_log_return)

            fold_mae = float(mean_absolute_error(actual_close, predicted_close))
            fold_persistence_mae = float(
                mean_absolute_error(actual_close, current_close)
            )
            fold_improvement = _improvement_pct(fold_mae, fold_persistence_mae)
            actual_direction = np.sign(actual_close - current_close)
            predicted_direction = np.sign(predicted_close - current_close)
            fold_directional_accuracy = float(
                np.mean(actual_direction == predicted_direction) * 100.0
            )

            predicted_close_parts.append(predicted_close)
            actual_close_parts.append(actual_close)
            current_close_parts.append(current_close)
            residual_parts.append(actual_log_return - predicted_log_return)
            fold_improvements.append(fold_improvement)
            latest_fold_mae = fold_mae
            latest_fold_persistence_mae = fold_persistence_mae
            latest_fold_directional_accuracy = fold_directional_accuracy
            completed_folds += 1

            print(
                f"  [CV {fold}/{self.config.cv_splits}] "
                f"train={len(train):,} validation={len(valid):,} "
                f"purged={purged:,} window_trimmed={window_trimmed:,} "
                f"vs_persistence={fold_improvement:+.2f}%"
            )

        if not predicted_close_parts:
            raise ValueError("Walk-forward validation produced no usable folds")

        predicted_close = np.concatenate(predicted_close_parts)
        actual_close = np.concatenate(actual_close_parts)
        current_close = np.concatenate(current_close_parts)
        residuals = np.concatenate(residual_parts)

        mae = float(mean_absolute_error(actual_close, predicted_close))
        rmse = float(math.sqrt(mean_squared_error(actual_close, predicted_close)))
        persistence_mae = float(mean_absolute_error(actual_close, current_close))

        actual_direction = np.sign(actual_close - current_close)
        predicted_direction = np.sign(predicted_close - current_close)
        directional_accuracy = float(
            np.mean(actual_direction == predicted_direction) * 100.0
        )
        improvement = _improvement_pct(mae, persistence_mae)
        latest_improvement = _improvement_pct(
            latest_fold_mae,
            latest_fold_persistence_mae,
        )

        return HorizonMetrics(
            horizon_hours=horizon_hours,
            cv_splits=completed_folds,
            rows_total=len(frame),
            rows_validation_total=len(actual_close),
            mae_usd_per_kg=mae,
            rmse_usd_per_kg=rmse,
            smape_pct=_smape(actual_close, predicted_close),
            directional_accuracy_pct=directional_accuracy,
            persistence_mae_usd_per_kg=persistence_mae,
            improvement_vs_persistence_pct=improvement,
            latest_fold_mae_usd_per_kg=latest_fold_mae,
            latest_fold_persistence_mae_usd_per_kg=latest_fold_persistence_mae,
            latest_fold_improvement_vs_persistence_pct=latest_improvement,
            latest_fold_directional_accuracy_pct=latest_fold_directional_accuracy,
            fold_improvement_vs_persistence_pct=tuple(fold_improvements),
            residual_log_return_q10=float(np.quantile(residuals, 0.10)),
            residual_log_return_q90=float(np.quantile(residuals, 0.90)),
        )
