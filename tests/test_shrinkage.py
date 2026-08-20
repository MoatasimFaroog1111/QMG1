from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from qmg1.ml.selective import CrossFittedShrinkageRegressor


def test_cross_fitted_shrinkage_can_collapse_to_persistence() -> None:
    rows = 120
    frame = pd.DataFrame(
        {
            "close": np.full(rows, 1000.0),
            "noise": np.linspace(-1.0, 1.0, rows),
        }
    )
    target = np.zeros(rows, dtype=float)

    model = CrossFittedShrinkageRegressor(
        base_estimator=DummyRegressor(strategy="constant", constant=0.02),
        calibration_splits=3,
        shrinkages=(0.0, 0.25, 0.50, 1.0),
    )
    model.fit(frame, target)

    assert model.selected_shrinkage_ == 0.0
    assert np.allclose(model.predict(frame.iloc[-5:]), 0.0)
    assert model.calibration_improvement_pct_ == 0.0
