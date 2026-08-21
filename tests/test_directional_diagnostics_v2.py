from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.ml.directional import DirectionDiagnostics  # noqa: E402


def _frame(rows: int = 1_200) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    signal = np.sin(np.arange(rows, dtype=float) / 11.0)
    return pd.DataFrame(
        {
            "feature_signal": signal,
            "target_2h": np.where(signal >= 0.0, 0.01, -0.01),
            "target_timestamp_2h": index + pd.Timedelta(hours=2),
        },
        index=index,
    )


def test_direction_diagnostic_uses_development_and_untouched_holdout() -> None:
    report = DirectionDiagnostics.evaluate_with_holdout(
        frame=_frame(),
        feature_columns=["feature_signal"],
        horizon_hours=2,
        cv_splits=4,
    )

    assert report.production_effect == "none"
    assert report.development.cv_splits == 4
    assert report.holdout.cv_splits == 1
    assert report.development.validation_rows > 0
    assert report.holdout.validation_rows > 0
    assert report.development.balanced_accuracy_pct > 90.0
    assert report.holdout.balanced_accuracy_pct > 90.0


def test_direction_metrics_are_percentages_and_baseline_is_explicit() -> None:
    report = DirectionDiagnostics.evaluate_with_holdout(
        frame=_frame(900),
        feature_columns=["feature_signal"],
        horizon_hours=2,
        cv_splits=3,
    )

    for metrics in (report.development, report.holdout):
        assert 0.0 <= metrics.accuracy_pct <= 100.0
        assert 0.0 <= metrics.balanced_accuracy_pct <= 100.0
        assert 50.0 <= metrics.majority_baseline_accuracy_pct <= 100.0
        assert -100.0 <= metrics.improvement_vs_majority_pp <= 50.0


def test_target_time_purge_can_remove_unsafe_training_rows() -> None:
    frame = _frame(300)
    # Move every development target beyond all validation windows. If the
    # evaluator ignored target-time purge it would still fit, which would be leakage.
    frame["target_timestamp_2h"] = frame.index[-1] + pd.Timedelta(days=10)

    try:
        DirectionDiagnostics.walk_forward(
            frame=frame,
            feature_columns=["feature_signal"],
            horizon_hours=2,
            cv_splits=3,
        )
    except ValueError as exc:
        assert "no usable folds" in str(exc).lower()
    else:
        raise AssertionError("Directional evaluator accepted future-leaking training rows")
