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
class DirectionDiagnosticMetrics:
    applicable: bool
    accuracy_pct: float | None
    balanced_accuracy_pct: float | None
    majority_baseline_accuracy_pct: float | None
    improvement_vs_majority_pp: float | None
    validation_rows: int
    cv_splits: int


class DirectionDiagnostics:
    """Leakage-safe diagnostics for directional signal.

    These metrics are deliberately diagnostic-only. They do not participate in
    price-model promotion. A separate production decision must be made before a
    directional classifier can influence serving.
    """

    @staticmethod
    def score(
        *,
        actual_log_return: np.ndarray,
        predicted_log_return: np.ndarray,
        strategy_name: str,
    ) -> DirectionDiagnosticMetrics:
        if strategy_name == "persistence":
            return DirectionDiagnosticMetrics(
                applicable=False,
                accuracy_pct=None,
                balanced_accuracy_pct=None,
                majority_baseline_accuracy_pct=None,
                improvement_vs_majority_pp=None,
                validation_rows=len(actual_log_return),
                cv_splits=0,
            )

        actual = np.asarray(actual_log_return, dtype=float)
        predicted = np.asarray(predicted_log_return, dtype=float)
        actual_direction = (actual > 0.0).astype(int)
        predicted_direction = (predicted > 0.0).astype(int)
        accuracy = float(np.mean(actual_direction == predicted_direction) * 100.0)
        balanced = float(
            balanced_accuracy_score(actual_direction, predicted_direction) * 100.0
        )
        majority = float(
            max(np.mean(actual_direction == 0), np.mean(actual_direction == 1)) * 100.0
        )
        return DirectionDiagnosticMetrics(
            applicable=True,
            accuracy_pct=accuracy,
            balanced_accuracy_pct=balanced,
            majority_baseline_accuracy_pct=majority,
            improvement_vs_majority_pp=accuracy - majority,
            validation_rows=len(actual),
            cv_splits=1,
        )

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

    @classmethod
    def walk_forward_classifier(
        cls,
        *,
        frame: pd.DataFrame,
        feature_columns: list[str],
        horizon_hours: int,
        cv_splits: int,
    ) -> DirectionDiagnosticMetrics:
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

            y_train = (train[target_col].to_numpy(dtype=float) > 0.0).astype(int)
            if np.unique(y_train).size < 2:
                continue

            model = cls._classifier()
            model.fit(train[feature_columns], y_train)
            predicted = np.asarray(model.predict(valid[feature_columns]), dtype=int)
            actual = (valid[target_col].to_numpy(dtype=float) > 0.0).astype(int)
            actual_parts.append(actual)
            predicted_parts.append(predicted)
            completed_folds += 1

        if not actual_parts:
            raise ValueError("Directional walk-forward produced no usable folds")

        actual = np.concatenate(actual_parts)
        predicted = np.concatenate(predicted_parts)
        accuracy = float(np.mean(actual == predicted) * 100.0)
        balanced = float(balanced_accuracy_score(actual, predicted) * 100.0)
        majority = float(max(np.mean(actual == 0), np.mean(actual == 1)) * 100.0)

        return DirectionDiagnosticMetrics(
            applicable=True,
            accuracy_pct=accuracy,
            balanced_accuracy_pct=balanced,
            majority_baseline_accuracy_pct=majority,
            improvement_vs_majority_pp=accuracy - majority,
            validation_rows=len(actual),
            cv_splits=completed_folds,
        )
