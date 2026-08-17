from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from qmg1.ml.model_factory import SelectiveShrinkageRegressor


def test_selective_regressor_abstains_on_weak_signals() -> None:
    X = np.arange(1.0, 101.0).reshape(-1, 1)
    y = 0.001 * X.ravel()

    model = SelectiveShrinkageRegressor(
        base_estimator=Ridge(alpha=1.0),
        activation_quantile=0.80,
        shrinkage=0.50,
    )
    model.fit(X, y)
    predictions = model.predict(X)

    nonzero = np.count_nonzero(predictions)
    assert 1 <= nonzero <= 25
    assert np.all(np.abs(predictions[np.nonzero(predictions)]) > 0.0)
    assert model.activation_threshold_ > 0.0


def test_selective_regressor_shrinks_active_predictions() -> None:
    X = np.arange(1.0, 101.0).reshape(-1, 1)
    y = 0.001 * X.ravel()

    base = Ridge(alpha=1.0).fit(X, y)
    raw = base.predict(X)

    model = SelectiveShrinkageRegressor(
        base_estimator=Ridge(alpha=1.0),
        activation_quantile=0.50,
        shrinkage=0.25,
    ).fit(X, y)
    selective = model.predict(X)

    active = np.abs(raw) >= model.activation_threshold_
    assert np.allclose(selective[active], raw[active] * 0.25)
    assert np.all(selective[~active] == 0.0)
