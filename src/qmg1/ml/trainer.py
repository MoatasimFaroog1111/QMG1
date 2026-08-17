from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from qmg1.config import HORIZONS_HOURS, TrainingConfig
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.dataset import ForecastDatasetBuilder, PreparedDataset
from qmg1.ml.evaluation import HorizonMetrics
from qmg1.ml.model_selection import ForecastModelSelector
from qmg1.ml.prediction_gate import OperationalPredictionGate, PredictionGateDecision


@dataclass(frozen=True)
class HorizonTrainingOutcome:
    metrics: HorizonMetrics
    selected_model: str
    selected_max_train_rows: int | None
    operational_status: PredictionGateDecision


class ForecastTrainer:
    """Validate candidates, select conservatively, persist once, then stop."""

    def __init__(
        self,
        artifact_repository: ModelArtifactRepository,
        config: TrainingConfig | None = None,
        dataset_builder: ForecastDatasetBuilder | None = None,
        model_selector: ForecastModelSelector | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.dataset_builder = dataset_builder or ForecastDatasetBuilder()
        self.artifact_repository = artifact_repository
        self.model_selector = model_selector or ForecastModelSelector(self.config)
        self.prediction_gate = OperationalPredictionGate.from_training_config(self.config)

    def _train_prepared(
        self,
        prepared: PreparedDataset,
        metal: str,
    ) -> HorizonTrainingOutcome:
        horizon_hours = prepared.horizon_hours
        frame = prepared.frame
        features = prepared.feature_columns

        if len(frame) < self.config.min_rows:
            raise ValueError(
                f"Not enough hourly rows for {metal} {horizon_hours}h: "
                f"{len(frame):,} < {self.config.min_rows:,}"
            )

        selection = self.model_selector.select(frame, features, horizon_hours)
        selected = selection.selected
        metrics = selected.metrics
        final_training_frame = selected.candidate.final_training_frame(frame)

        final_model = selected.candidate.factory.create()
        target_col = f"target_{horizon_hours}h"
        final_model.fit(
            final_training_frame[features],
            final_training_frame[target_col],
        )

        metrics_dict = asdict(metrics)
        gate_decision = self.prediction_gate.evaluate_metrics(metrics_dict)
        candidate_summaries = [
            {
                "name": item.candidate.name,
                "max_train_rows": item.candidate.max_train_rows,
                "robust_improvement_pct": item.robust_improvement_pct,
                "metrics": asdict(item.metrics),
            }
            for item in selection.evaluations
        ]

        artifact = {
            "schema_version": 4,
            "metal": metal,
            "horizon_hours": horizon_hours,
            "feature_columns": features,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(final_training_frame),
            "available_training_rows": len(frame),
            "training_start_utc": final_training_frame.index[0].isoformat(),
            "training_end_utc": final_training_frame.index[-1].isoformat(),
            "training_config": asdict(self.config),
            "validation_method": "walk-forward with target-time purge and candidate selection",
            "selected_model": selected.candidate.name,
            "selected_max_train_rows": selected.candidate.max_train_rows,
            "model_selection": candidate_summaries,
            "operational_status": asdict(gate_decision),
            "metrics": metrics_dict,
            "model": final_model,
        }
        self.artifact_repository.save(metal, horizon_hours, artifact)

        status = "ACCEPTED" if gate_decision.accepted else "REJECTED"
        print(
            f"  [GATE] {metal} {horizon_hours}h {status}: "
            f"{gate_decision.reason}"
        )
        return HorizonTrainingOutcome(
            metrics=metrics,
            selected_model=selected.candidate.name,
            selected_max_train_rows=selected.candidate.max_train_rows,
            operational_status=gate_decision,
        )

    def train_one(
        self,
        csv_path: str,
        metal: str,
        horizon_hours: int,
    ) -> HorizonMetrics:
        prepared = self.dataset_builder.build(csv_path, horizon_hours)
        return self._train_prepared(prepared, metal).metrics

    def train_all(
        self,
        csv_path: str,
        metal: str,
        horizons: Sequence[int] = HORIZONS_HOURS,
    ) -> list[HorizonMetrics]:
        requested_horizons = tuple(horizons)
        if not requested_horizons:
            return []

        unknown = [horizon for horizon in requested_horizons if horizon not in HORIZONS_HOURS]
        if unknown:
            raise ValueError(f"Unsupported forecast horizons: {unknown}")

        # M1 loading, hourly resampling, and feature engineering are the most
        # expensive deterministic steps. Compute them once per metal, then
        # attach each requested horizon target to the same immutable base.
        base = self.dataset_builder.load_feature_base(csv_path)
        outcomes: list[HorizonTrainingOutcome] = []

        for horizon in requested_horizons:
            print(f"[TRAIN] {metal} horizon={horizon}h")
            prepared = self.dataset_builder.build_from_base(base, horizon)
            outcomes.append(self._train_prepared(prepared, metal))

        report_dir = self.artifact_repository.root / metal
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "metal": metal,
            "source_csv": csv_path,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "horizons_hours": list(requested_horizons),
            "validation_method": "walk-forward with target-time purge and candidate selection",
            "feature_base_reused_across_horizons": True,
            "selected_models": [
                {
                    "horizon_hours": outcome.metrics.horizon_hours,
                    "model": outcome.selected_model,
                    "max_train_rows": outcome.selected_max_train_rows,
                    "operational_status": asdict(outcome.operational_status),
                }
                for outcome in outcomes
            ],
            "metrics": [asdict(outcome.metrics) for outcome in outcomes],
        }
        (report_dir / f"{metal}_training_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return [outcome.metrics for outcome in outcomes]
