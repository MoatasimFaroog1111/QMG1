from __future__ import annotations

import numpy as np
import pandas as pd

from qmg1.ml.selective import CrossFittedSelectiveRegressor, _slice_rows


class PurgedCrossFittedSelectiveRegressor(CrossFittedSelectiveRegressor):
    """Cross-fitted calibrator with conservative inner target-time purging.

    Outer QMG1 validation already purges rows using their exact persisted target
    timestamps. The nested calibration performed inside a training fold does not
    receive those target columns, so this component uses the feature DatetimeIndex
    plus a conservative ``purge_hours`` upper bound. For QMG1 this is configured as
    ``forecast_horizon + maximum target-forward tolerance``.

    This guarantees that an inner training row cannot have a label whose possible
    observation time reaches the inner validation start.
    """

    def __init__(
        self,
        base_estimator,
        calibration_splits: int = 3,
        activation_quantiles=(0.50, 0.70, 0.80, 0.90, 0.95),
        shrinkages=(0.10, 0.25, 0.50, 0.75, 1.00),
        purge_hours: int = 0,
    ) -> None:
        super().__init__(
            base_estimator=base_estimator,
            calibration_splits=calibration_splits,
            activation_quantiles=activation_quantiles,
            shrinkages=shrinkages,
        )
        self.purge_hours = purge_hours

    def _validate_parameters(self, n_rows: int) -> None:
        super()._validate_parameters(n_rows)
        if self.purge_hours < 0:
            raise ValueError("purge_hours must be >= 0")

    def _purged_train_indices(
        self,
        X,
        train_index: np.ndarray,
        validation_index: np.ndarray,
    ) -> np.ndarray:
        if self.purge_hours == 0:
            return train_index
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "Purged calibration requires a pandas DataFrame with DatetimeIndex"
            )
        if len(validation_index) == 0:
            return train_index[:0]

        validation_start = X.index[int(validation_index[0])]
        candidate_timestamps = X.index[train_index]
        possible_target_times = candidate_timestamps + pd.Timedelta(hours=self.purge_hours)
        safe = np.asarray(possible_target_times < validation_start, dtype=bool)
        return train_index[safe]

    def _run_oof_predictions(self, X, y_array: np.ndarray) -> np.ndarray:
        from sklearn.base import clone
        from sklearn.model_selection import TimeSeriesSplit

        oof_prediction = np.full(len(y_array), np.nan, dtype=float)
        splitter = TimeSeriesSplit(n_splits=self.calibration_splits)
        completed_folds = 0

        for train_index, validation_index in splitter.split(np.arange(len(y_array))):
            purged_train_index = self._purged_train_indices(
                X,
                train_index,
                validation_index,
            )
            if len(purged_train_index) == 0:
                continue

            estimator = clone(self.base_estimator)
            estimator.fit(
                _slice_rows(X, purged_train_index),
                _slice_rows(y_array, purged_train_index),
            )
            oof_prediction[validation_index] = np.asarray(
                estimator.predict(_slice_rows(X, validation_index)),
                dtype=float,
            )
            completed_folds += 1

        if completed_folds == 0:
            raise ValueError("Purged cross-fitted calibration produced no usable folds")

        self.completed_purged_calibration_folds_ = completed_folds
        self.inner_purge_hours_ = self.purge_hours
        return oof_prediction
