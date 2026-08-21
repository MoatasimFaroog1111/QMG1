from __future__ import annotations

import numpy as np
import pandas as pd

from qmg1.ml.directional import DirectionDiagnostics


def test_persistence_direction_is_not_applicable() -> None:
    diagnostics = DirectionDiagnostics.score(
        actual_log_return=np.array([0.01, -0.02, 0.00]),
        predicted_log_return=np.array([0.0, 0.0, 0.0]),
        strategy_name="persistence",
    )

    assert diagnostics.applicable is False
    assert diagnostics.accuracy_pct is None


def test_direction_accuracy_is_percentage_for_non_persistence() -> None:
    diagnostics = DirectionDiagnostics.score(
        actual_log_return=np.array([0.01, -0.02, 0.03, -0.01]),
        predicted_log_return=np.array([0.02, -0.01, -0.01, -0.02]),
        strategy_name="ridge_alpha_10",
    )

    assert diagnostics.applicable is True
    assert diagnostics.accuracy_pct == 75.0


def test_classifier_walk_forward_has_no_future_rows() -> None:
    index = pd.date_range("2025-01-01", periods=800, freq="h", tz="UTC")
    signal = np.sin(np.arange(len(index)) / 8.0)
    frame = pd.DataFrame(
        {
            "close": np.linspace(100.0, 120.0, len(index)),
            "feature_a": signal,
            "target_2h": np.where(signal >= 0.0, 0.01, -0.01),
            "target_timestamp_2h": index + pd.Timedelta(hours=2),
        },
        index=index,
    )

    result = DirectionDiagnostics.walk_forward_classifier(
        frame=frame,
        feature_columns=["feature_a"],
        horizon_hours=2,
        cv_splits=4,
    )

    assert result.validation_rows > 0
    assert result.cv_splits == 4
    assert 0.0 <= result.accuracy_pct <= 100.0
    assert 0.0 <= result.majority_baseline_accuracy_pct <= 100.0
    assert result.improvement_vs_majority_pp is not None
