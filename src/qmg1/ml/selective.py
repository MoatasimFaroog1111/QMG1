from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit


def _slice_rows(values, indices: np.ndarray):
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return values[indices]


def _price_mae(X, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate log-return predictions in target price units when possible."""
    if isinstance(X, pd.DataFrame) and "close" in X.columns:
        close = pd.to_numeric(X["close"], errors="coerce").to_numpy(dtype=float)
        finite = (
            np.isfinite(close)
            & (close > 0.0)
            & np.isfinite(y_true)
            & np.isfinite(y_pred)
        )
        if finite.any():
            actual = close[finite] * np.exp(y_true[finite])
            predicted = close[finite] * np.exp(y_pred[finite])
            return float(np.mean(np.abs(actual - predicted)))

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        return math.inf
    return float(np.mean(np.abs(y_true[finite] - y_pred[finite])))


class SelectiveShrinkageRegressor(BaseEstimator, RegressorMixin):
    """Abstain for weak in-sample signals and shrink active predictions."""

    def __init__(
        self,
        base_estimator: RegressorMixin,
        activation_quantile: float = 0.90,
        shrinkage: float = 0.50,
    ) -> None:
        self.base_estimator = base_estimator
        self.activation_quantile = activation_quantile
        self.shrinkage = shrinkage

    def fit(self, X, y):
        if not 0.0 < self.activation_quantile < 1.0:
            raise ValueError("activation_quantile must be between 0 and 1")
        if not 0.0 < self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in (0, 1]")

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)
        train_prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        finite = np.abs(train_prediction[np.isfinite(train_prediction)])
        self.activation_threshold_ = (
            float(np.quantile(finite, self.activation_quantile))
            if finite.size
            else float("inf")
        )
        return self

    def predict(self, X):
        if not hasattr(self, "estimator_"):
            raise RuntimeError("SelectiveShrinkageRegressor must be fitted first")
        raw = np.asarray(self.estimator_.predict(X), dtype=float)
        active = np.abs(raw) >= self.activation_threshold_
        return np.where(active, raw * self.shrinkage, 0.0)


class CrossFittedShrinkageRegressor(BaseEstimator, RegressorMixin):
    """Calibrate one global shrinkage factor from causal OOF predictions.

    The factor is selected only from expanding-window out-of-fold predictions
    inside the outer training fold. A zero factor is always a candidate, so the
    estimator conservatively collapses to Persistence when no learned signal
    improves price MAE on its own calibration history.
    """

    def __init__(
        self,
        base_estimator: RegressorMixin,
        calibration_splits: int = 3,
        shrinkages: Sequence[float] = (
            0.0,
            0.02,
            0.05,
            0.10,
            0.15,
            0.25,
            0.50,
            0.75,
            1.00,
        ),
    ) -> None:
        self.base_estimator = base_estimator
        self.calibration_splits = calibration_splits
        self.shrinkages = shrinkages

    def _validate_parameters(self, n_rows: int) -> None:
        if self.calibration_splits < 2:
            raise ValueError("calibration_splits must be >= 2")
        if n_rows <= self.calibration_splits + 1:
            raise ValueError("Not enough rows for cross-fitted calibration")
        if not self.shrinkages:
            raise ValueError("shrinkages must not be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.shrinkages):
            raise ValueError("shrinkages must be in [0, 1]")
        if 0.0 not in self.shrinkages:
            raise ValueError("shrinkages must include 0.0 as the persistence fallback")

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float)
        self._validate_parameters(len(y_array))

        oof_prediction = np.full(len(y_array), np.nan, dtype=float)
        splitter = TimeSeriesSplit(n_splits=self.calibration_splits)
        for train_index, validation_index in splitter.split(np.arange(len(y_array))):
            estimator = clone(self.base_estimator)
            estimator.fit(_slice_rows(X, train_index), _slice_rows(y, train_index))
            oof_prediction[validation_index] = np.asarray(
                estimator.predict(_slice_rows(X, validation_index)),
                dtype=float,
            )

        calibration_mask = np.isfinite(oof_prediction) & np.isfinite(y_array)
        if not calibration_mask.any():
            raise RuntimeError("Cross-fitted calibration produced no OOF predictions")

        calibration_indices = np.flatnonzero(calibration_mask)
        X_calibration = _slice_rows(X, calibration_indices)
        y_calibration = y_array[calibration_mask]
        raw_calibration = oof_prediction[calibration_mask]

        best_shrinkage = 0.0
        best_mae = _price_mae(
            X_calibration,
            y_calibration,
            np.zeros_like(y_calibration),
        )
        persistence_mae = best_mae

        for shrinkage in self.shrinkages:
            prediction = raw_calibration * float(shrinkage)
            mae = _price_mae(X_calibration, y_calibration, prediction)
            if mae < best_mae:
                best_mae = mae
                best_shrinkage = float(shrinkage)

        self.calibration_persistence_mae_ = float(persistence_mae)
        self.calibration_mae_ = float(best_mae)
        self.calibration_improvement_pct_ = (
            (persistence_mae - best_mae) / persistence_mae * 100.0
            if persistence_mae > 0.0
            else 0.0
        )
        self.selected_shrinkage_ = best_shrinkage

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, "estimator_"):
            raise RuntimeError("CrossFittedShrinkageRegressor must be fitted first")
        raw = np.asarray(self.estimator_.predict(X), dtype=float)
        return raw * self.selected_shrinkage_


class CrossFittedSelectiveRegressor(BaseEstimator, RegressorMixin):
    """Calibrate abstention and shrinkage from causal OOF training predictions.

    The outer validator supplies a chronological training fold. This estimator
    performs an additional expanding-window TimeSeriesSplit *inside that fold*.
    It chooses a signal quantile and shrinkage using only inner OOF predictions,
    optimizing MAE in USD/kg when the feature frame contains ``close``.
    Outer validation and final holdout rows are never used for calibration.
    """

    def __init__(
        self,
        base_estimator: RegressorMixin,
        calibration_splits: int = 3,
        activation_quantiles: Sequence[float] = (0.50, 0.70, 0.80, 0.90, 0.95),
        shrinkages: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
    ) -> None:
        self.base_estimator = base_estimator
        self.calibration_splits = calibration_splits
        self.activation_quantiles = activation_quantiles
        self.shrinkages = shrinkages

    def _validate_parameters(self, n_rows: int) -> None:
        if self.calibration_splits < 2:
            raise ValueError("calibration_splits must be >= 2")
        if n_rows <= self.calibration_splits + 1:
            raise ValueError("Not enough rows for cross-fitted calibration")
        if not self.activation_quantiles:
            raise ValueError("activation_quantiles must not be empty")
        if not self.shrinkages:
            raise ValueError("shrinkages must not be empty")
        if any(not 0.0 < value < 1.0 for value in self.activation_quantiles):
            raise ValueError("activation quantiles must be in (0, 1)")
        if any(not 0.0 < value <= 1.0 for value in self.shrinkages):
            raise ValueError("shrinkages must be in (0, 1]")

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float)
        self._validate_parameters(len(y_array))

        oof_prediction = np.full(len(y_array), np.nan, dtype=float)
        splitter = TimeSeriesSplit(n_splits=self.calibration_splits)
        for train_index, validation_index in splitter.split(np.arange(len(y_array))):
            estimator = clone(self.base_estimator)
            estimator.fit(
                _slice_rows(X, train_index),
                _slice_rows(y, train_index),
            )
            oof_prediction[validation_index] = np.asarray(
                estimator.predict(_slice_rows(X, validation_index)),
                dtype=float,
            )

        calibration_mask = np.isfinite(oof_prediction) & np.isfinite(y_array)
        if not calibration_mask.any():
            raise RuntimeError("Cross-fitted calibration produced no OOF predictions")

        calibration_indices = np.flatnonzero(calibration_mask)
        X_calibration = _slice_rows(X, calibration_indices)
        y_calibration = y_array[calibration_mask]
        raw_calibration = oof_prediction[calibration_mask]
        absolute_signal = np.abs(raw_calibration)

        persistence_prediction = np.zeros_like(y_calibration)
        persistence_mae = _price_mae(
            X_calibration,
            y_calibration,
            persistence_prediction,
        )

        best_mae = persistence_mae
        best_quantile: float | None = None
        best_shrinkage = 0.0

        for quantile in self.activation_quantiles:
            threshold = float(np.quantile(absolute_signal, quantile))
            active = absolute_signal >= threshold
            for shrinkage in self.shrinkages:
                prediction = np.where(active, raw_calibration * shrinkage, 0.0)
                mae = _price_mae(X_calibration, y_calibration, prediction)
                if mae < best_mae:
                    best_mae = mae
                    best_quantile = float(quantile)
                    best_shrinkage = float(shrinkage)

        self.calibration_persistence_mae_ = float(persistence_mae)
        self.calibration_mae_ = float(best_mae)
        self.calibration_improvement_pct_ = (
            (persistence_mae - best_mae) / persistence_mae * 100.0
            if persistence_mae > 0.0
            else 0.0
        )
        self.selected_activation_quantile_ = best_quantile
        self.selected_shrinkage_ = best_shrinkage

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)

        if best_quantile is None or best_shrinkage <= 0.0:
            self.activation_threshold_ = float("inf")
            return self

        full_train_prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        finite = np.abs(full_train_prediction[np.isfinite(full_train_prediction)])
        self.activation_threshold_ = (
            float(np.quantile(finite, best_quantile))
            if finite.size
            else float("inf")
        )
        return self

    def predict(self, X):
        if not hasattr(self, "estimator_"):
            raise RuntimeError("CrossFittedSelectiveRegressor must be fitted first")
        raw = np.asarray(self.estimator_.predict(X), dtype=float)
        active = np.abs(raw) >= self.activation_threshold_
        return np.where(active, raw * self.selected_shrinkage_, 0.0)
