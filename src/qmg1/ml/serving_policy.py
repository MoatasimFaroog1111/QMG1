from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServingDecision:
    """Resolved runtime semantics for a complete persisted training artifact."""

    serving_strategy: str
    governance_strategy: str
    selected_challenger: str
    model_metrics: Mapping[str, Any]
    feature_data_required: bool


class PersistedArtifactServingPolicy:
    """Separate audit promotion decisions from persisted-model inference decisions."""

    _FEATURE_INDEPENDENT_PREFIXES = ("median_return",)

    @classmethod
    def resolve(cls, artifact: Mapping[str, Any]) -> ServingDecision:
        model = artifact.get("model")
        if model is None:
            raise ValueError("Persisted trained artifact does not contain a fitted estimator")

        selection_value = artifact.get("selection")
        selection = (
            selection_value if isinstance(selection_value, Mapping) else {}
        )

        selected_challenger = str(
            artifact.get("serving_strategy")
            or artifact.get("selected_challenger")
            or selection.get("selected_model_name")
            or type(model).__name__
        )
        governance_strategy = str(
            artifact.get("governance_strategy")
            or selection.get("active_strategy")
            or artifact.get("active_strategy")
            or "unknown"
        )

        metrics_value = (
            artifact.get("model_metrics")
            or selection.get("challenger_holdout_metrics")
            or artifact.get("metrics")
        )
        if not isinstance(metrics_value, Mapping):
            raise ValueError("Persisted trained artifact does not contain model validation metrics")

        feature_data_required = not selected_challenger.startswith(
            cls._FEATURE_INDEPENDENT_PREFIXES
        )
        return ServingDecision(
            serving_strategy=selected_challenger,
            governance_strategy=governance_strategy,
            selected_challenger=selected_challenger,
            model_metrics=metrics_value,
            feature_data_required=feature_data_required,
        )
