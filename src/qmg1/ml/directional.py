from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DirectionMetrics:
    accuracy_pct: float
    balanced_accuracy_pct: float
    majority_baseline_accuracy_pct: float
    improvement_vs_majority_pp: float
    positive_class_pct: float
    validation_rows: int
    cv_splits: int


@dataclass(frozen=True)
class DirectionDiagnosticReport:
    horizon_hours: int
    development: DirectionMetrics
    holdout: DirectionMetrics
    holdout_start_utc: str
    production_effect: str = "none"


class DirectionDiagnostics:
    """Leakage-safe directional diagnostics with no production effect."""

    @staticmethod
    def _classifier() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=750,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

    @staticmethod
    def _labels(log_returns: np.ndarray) -> np.ndarray:
        return (np.asarray(log_returns, dtype=float) > 0.0).astype(int)

    @classmethod
    def _score(
        cls,
        actual_log_returns: np.ndarray,
        predicted_labels: np.ndarray,
        *,
        cv_splits: int,
    ) -> DirectionMetrics:
        actual_labels = cls._labels(actual_log_returns)
        predicted = np.asarray(predicted_labels, dtype=int)
        if len(actual_labels) != len(predicted):
            raise ValueError("Directional score inputs have different lengths")
        if len(actual_labels) == 0:
            raise ValueError("Directional score requires validation rows")

        accuracy = float(np.mean(actual_labels == predicted) * 100.0)
        balanced = float(balanced_accuracy_score(actual_labels, predicted) * 100.0)
        positive_pct = float(np.mean(actual_labels == 1) * 100.0)
        majority = max(positive_pct, 100.0 - positive_pct)
        return DirectionMetrics(
            accuracy_pct=accuracy,
            balanced_accuracy_pct=balanced,
            majority_baseline_accuracy_pct=majority,
            improvement_vs_majority_pp=accuracy - majority,
            positive_class_pct=positive_pct,
            validation_rows=len(actual_labels),
            cv_splits=cv_splits,
        )

    @classmethod
    def walk_forward(
        cls,
        *,
        frame: pd.DataFrame,
        feature_columns: list[str],
        horizon_hours: int,
        cv_splits: int,
    ) -> DirectionMetrics:
        if cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")

        target_col = f"target_{horizon_hours}h"
        target_timestamp_col = f"target_timestamp_{horizon_hours}h"
        required = {target_col, target_timestamp_col, *feature_columns}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing directional diagnostic columns: {missing}")

        splitter = TimeSeriesSplit(n_splits=cv_splits)
        actual_parts: list[np.ndarray] = []
        predicted_parts: list[np.ndarray] = []
        completed_folds = 0

        for train_idx, valid_idx in splitter.split(frame):
            candidate_train = frame.iloc[train_idx]
            valid = frame.iloc[valid_idx]
            if candidate_train.empty or valid.empty:
                continue

            validation_start = valid.index[0]
            train = candidate_train[
                candidate_train[target_timestamp_col] < validation_start
            ]
            if train.empty:
                continue

            y_train = cls._labels(train[target_col].to_numpy(dtype=float))
            if np.unique(y_train).size < 2:
                continue

            model = cls._classifier()
            model.fit(train[feature_columns], y_train)
            predicted = np.asarray(model.predict(valid[feature_columns]), dtype=int)
            actual_parts.append(valid[target_col].to_numpy(dtype=float))
            predicted_parts.append(predicted)
            completed_folds += 1

        if not actual_parts:
            raise ValueError("Directional walk-forward produced no usable folds")

        return cls._score(
            np.concatenate(actual_parts),
            np.concatenate(predicted_parts),
            cv_splits=completed_folds,
        )

    @classmethod
    def evaluate_with_holdout(
        cls,
        *,
        frame: pd.DataFrame,
        feature_columns: list[str],
        horizon_hours: int,
        cv_splits: int,
        holdout_fraction: float = 0.20,
    ) -> DirectionDiagnosticReport:
        if not 0.10 <= holdout_fraction <= 0.40:
            raise ValueError("holdout_fraction must be between 0.10 and 0.40")

        split_at = int(len(frame) * (1.0 - holdout_fraction))
        development = frame.iloc[:split_at]
        holdout = frame.iloc[split_at:]
        if development.empty or holdout.empty:
            raise ValueError("Directional development/holdout split is empty")

        development_metrics = cls.walk_forward(
            frame=development,
            feature_columns=feature_columns,
            horizon_hours=horizon_hours,
            cv_splits=cv_splits,
        )

        target_col = f"target_{horizon_hours}h"
        target_timestamp_col = f"target_timestamp_{horizon_hours}h"
        holdout_start = holdout.index[0]
        purged_train = development[
            development[target_timestamp_col] < holdout_start
        ]
        if purged_train.empty:
            raise ValueError("No leakage-safe rows remain before directional holdout")

        y_train = cls._labels(purged_train[target_col].to_numpy(dtype=float))
        if np.unique(y_train).size < 2:
            raise ValueError("Directional holdout training has only one class")

        model = cls._classifier()
        model.fit(purged_train[feature_columns], y_train)
        predicted = np.asarray(model.predict(holdout[feature_columns]), dtype=int)
        holdout_metrics = cls._score(
            holdout[target_col].to_numpy(dtype=float),
            predicted,
            cv_splits=1,
        )

        return DirectionDiagnosticReport(
            horizon_hours=horizon_hours,
            development=development_metrics,
            holdout=holdout_metrics,
            holdout_start_utc=holdout_start.isoformat(),
        )
