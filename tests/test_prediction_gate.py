from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmg1.ml.prediction_gate import (  # noqa: E402
    OperationalPredictionGate,
    RejectedModelError,
)


def test_gate_accepts_only_when_overall_and_recent_both_beat_persistence() -> None:
    gate = OperationalPredictionGate()
    decision = gate.evaluate_metrics(
        {
            "improvement_vs_persistence_pct": 2.5,
            "latest_fold_improvement_vs_persistence_pct": 1.2,
        }
    )
    assert decision.accepted is True


def test_gate_rejects_recent_regime_failure_even_if_overall_is_positive() -> None:
    gate = OperationalPredictionGate()
    decision = gate.evaluate_metrics(
        {
            "improvement_vs_persistence_pct": 3.0,
            "latest_fold_improvement_vs_persistence_pct": -0.5,
        }
    )
    assert decision.accepted is False
    assert "most recent" in decision.reason


def test_gate_rejects_persistence_tie() -> None:
    gate = OperationalPredictionGate()
    decision = gate.evaluate_metrics(
        {
            "improvement_vs_persistence_pct": 0.0,
            "latest_fold_improvement_vs_persistence_pct": 0.0,
        }
    )
    assert decision.accepted is False


def test_gate_blocks_rejected_artifact() -> None:
    gate = OperationalPredictionGate()
    artifact = {
        "metal": "silver",
        "horizon_hours": 24,
        "metrics": {
            "improvement_vs_persistence_pct": -4.0,
            "latest_fold_improvement_vs_persistence_pct": -2.0,
        },
    }
    with pytest.raises(RejectedModelError, match="Operational forecast blocked"):
        gate.require_accepted(artifact)


def test_schema_v3_fallback_uses_overall_improvement_for_recent() -> None:
    gate = OperationalPredictionGate()
    decision = gate.evaluate_metrics({"improvement_vs_persistence_pct": -1.0})
    assert decision.accepted is False
    assert decision.recent_improvement_pct == -1.0
