from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from qmg1.config import TrainingConfig


class RejectedModelError(RuntimeError):
    """Raised when a persisted model has not earned operational use."""


@dataclass(frozen=True)
class PredictionGateDecision:
    accepted: bool
    overall_improvement_pct: float
    recent_improvement_pct: float
    min_overall_required_pct: float
    min_recent_required_pct: float
    reason: str


class OperationalPredictionGate:
    """Require OOS improvement over persistence before serving a forecast."""

    def __init__(
        self,
        min_overall_improvement_pct: float = 0.0,
        min_recent_improvement_pct: float = 0.0,
    ) -> None:
        self.min_overall_improvement_pct = min_overall_improvement_pct
        self.min_recent_improvement_pct = min_recent_improvement_pct

    @classmethod
    def from_training_config(cls, config: TrainingConfig) -> OperationalPredictionGate:
        return cls(
            min_overall_improvement_pct=config.min_operational_improvement_pct,
            min_recent_improvement_pct=config.min_recent_improvement_pct,
        )

    def evaluate_metrics(self, metrics: dict[str, Any]) -> PredictionGateDecision:
        overall = float(metrics.get("improvement_vs_persistence_pct", float("nan")))
        # Schema-v3 artifacts predate per-fold metrics. Falling back to the
        # overall result preserves backward compatibility while remaining safe:
        # a negative overall score is rejected exactly as before.
        recent = float(
            metrics.get("latest_fold_improvement_vs_persistence_pct", overall)
        )

        if not math.isfinite(overall) or not math.isfinite(recent):
            return PredictionGateDecision(
                accepted=False,
                overall_improvement_pct=overall,
                recent_improvement_pct=recent,
                min_overall_required_pct=self.min_overall_improvement_pct,
                min_recent_required_pct=self.min_recent_improvement_pct,
                reason="Validation improvement metrics are missing or non-finite.",
            )

        overall_ok = overall > self.min_overall_improvement_pct
        recent_ok = recent > self.min_recent_improvement_pct
        accepted = overall_ok and recent_ok

        if accepted:
            reason = (
                "Accepted: model beats persistence out of sample both overall "
                "and on the most recent validation fold."
            )
        elif not overall_ok and not recent_ok:
            reason = (
                "Rejected: model does not beat persistence either overall or "
                "on the most recent validation fold."
            )
        elif not overall_ok:
            reason = "Rejected: model does not beat persistence overall out of sample."
        else:
            reason = (
                "Rejected: model does not beat persistence on the most recent "
                "validation fold."
            )

        return PredictionGateDecision(
            accepted=accepted,
            overall_improvement_pct=overall,
            recent_improvement_pct=recent,
            min_overall_required_pct=self.min_overall_improvement_pct,
            min_recent_required_pct=self.min_recent_improvement_pct,
            reason=reason,
        )

    def evaluate_artifact(self, artifact: dict[str, Any]) -> PredictionGateDecision:
        metrics = artifact.get("metrics")
        if not isinstance(metrics, dict):
            return self.evaluate_metrics({})
        return self.evaluate_metrics(metrics)

    def require_accepted(self, artifact: dict[str, Any]) -> PredictionGateDecision:
        decision = self.evaluate_artifact(artifact)
        if not decision.accepted:
            metal = artifact.get("metal", "unknown")
            horizon = artifact.get("horizon_hours", "unknown")
            raise RejectedModelError(
                f"Operational forecast blocked for {metal} {horizon}h. {decision.reason} "
                f"overall={decision.overall_improvement_pct:.3f}% "
                f"recent={decision.recent_improvement_pct:.3f}%"
            )
        return decision
