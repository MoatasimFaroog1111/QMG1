from __future__ import annotations

from qmg1.ml.directional import DirectionDiagnostics


def test_persistence_direction_metric_is_explicitly_not_applicable() -> None:
    metrics = DirectionDiagnostics.score(
        actual_log_return=[0.01, -0.01],
        predicted_log_return=[0.0, 0.0],
        strategy_name="persistence",
    )

    assert metrics.applicable is False
    assert metrics.accuracy_pct is None
    assert metrics.balanced_accuracy_pct is None
