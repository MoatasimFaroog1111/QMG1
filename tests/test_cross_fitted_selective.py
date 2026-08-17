from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from qmg1.config import TrainingConfig
from qmg1.ml.model_factory import candidate_factories
from qmg1.ml.selective import CrossFittedSelectiveRegressor


class CausalProbeRegressor(RegressorMixin, BaseEstimator):
    fit_maxima: list[float] = []
    predict_minima: list[float] = []

    def fit(self, X, y):
        values = np.asarray(X)[:, 0].astype(float)
        self.level_ = float(np.mean(np.asarray(y, dtype=float)))
        type(self).fit_maxima.append(float(values.max()))
        return self

    def predict(self, X):
        values = np.asarray(X)[:, 0].astype(float)
        type(self).predict_minima.append(float(values.min()))
        return np.full(len(values), self.level_, dtype=float)


def test_cross_fitted_inner_predictions_are_strictly_forward() -> None:
    CausalProbeRegressor.fit_maxima.clear()
    CausalProbeRegressor.predict_minima.clear()

    X = np.arange(120.0).reshape(-1, 1)
    y = np.sin(np.arange(120.0) / 10.0) * 0.001
    model = CrossFittedSelectiveRegressor(
        base_estimator=CausalProbeRegressor(),
        calibration_splits=3,
    )
    model.fit(X, y)

    # The first three fit/predict pairs are the inner TimeSeriesSplit folds.
    for train_max, validation_min in zip(
        CausalProbeRegressor.fit_maxima[:3],
        CausalProbeRegressor.predict_minima[:3],
        strict=True,
    ):
        assert train_max < validation_min


def test_cross_fitted_calibration_optimizes_price_mae_from_oof_only() -> None:
    rows = 600
    signal = np.linspace(-1.0, 1.0, rows)
    X = pd.DataFrame(
        {
            "signal": signal,
            "close": np.full(rows, 1_500.0),
        }
    )
    # A stable causal relationship gives the OOF calibrator a real signal.
    y = pd.Series(signal * 0.01)

    model = CrossFittedSelectiveRegressor(
        base_estimator=Ridge(alpha=0.01),
        calibration_splits=3,
        activation_quantiles=(0.50, 0.80, 0.95),
        shrinkages=(0.25, 0.50, 1.00),
    ).fit(X, y)

    assert model.calibration_mae_ <= model.calibration_persistence_mae_
    assert model.calibration_improvement_pct_ >= 0.0
    assert model.selected_activation_quantile_ in {0.50, 0.80, 0.95}
    assert model.selected_shrinkage_ in {0.25, 0.50, 1.00}
    prediction = model.predict(X.tail(10))
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()


def test_routine_candidate_set_is_empirically_pruned() -> None:
    names = [factory.name for factory in candidate_factories(TrainingConfig())]
    assert names == [
        "median_return",
        "selective_hgb_q80_s25",
        "cross_fitted_selective_hgb",
    ]
    assert not any("ridge" in name for name in names)
    assert not any("lookback" in name for name in names)
