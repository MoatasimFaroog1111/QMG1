from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from qmg1.config import HORIZONS_HOURS, TrainingConfig
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.dataset import ForecastDatasetBuilder
from qmg1.ml.evaluation import HorizonMetrics, WalkForwardEvaluator
from qmg1.ml.model_factory import HistGradientBoostingFactory


class ForecastTrainer:
    """Train once, validate out-of-sample, then persist the final models."""

    def __init__(
        self,
        artifact_repository: ModelArtifactRepository,
        config: TrainingConfig | None = None,
        dataset_builder: ForecastDatasetBuilder | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.dataset_builder = dataset_builder or ForecastDatasetBuilder()
        self.artifact_repository = artifact_repository
        self.model_factory = HistGradientBoostingFactory(self.config)
        self.evaluator = WalkForwardEvaluator(self.config, self.model_factory)

    def train_one(
        self,
        csv_path: str,
        metal: str,
        horizon_hours: int,
    ) -> HorizonMetrics:
        prepared = self.dataset_builder.build(csv_path, horizon_hours)
        frame = prepared.frame
        features = prepared.feature_columns

        if len(frame) < self.config.min_rows:
            raise ValueError(
                f"Not enough hourly rows for {metal} {horizon_hours}h: "
                f"{len(frame):,} < {self.config.min_rows:,}"
            )

        metrics = self.evaluator.evaluate(frame, features, horizon_hours)

        final_model = self.model_factory.create()
        target_col = f"target_{horizon_hours}h"
        final_model.fit(frame[features], frame[target_col])

        artifact = {
            "schema_version": 3,
            "metal": metal,
            "horizon_hours": horizon_hours,
            "feature_columns": features,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(frame),
            "training_start_utc": frame.index[0].isoformat(),
            "training_end_utc": frame.index[-1].isoformat(),
            "training_config": asdict(self.config),
            "validation_method": "expanding walk-forward with target-time purge",
            "metrics": asdict(metrics),
            "model": final_model,
        }
        self.artifact_repository.save(metal, horizon_hours, artifact)
        return metrics

    def train_all(self, csv_path: str, metal: str) -> list[HorizonMetrics]:
        metrics: list[HorizonMetrics] = []
        for horizon in HORIZONS_HOURS:
            print(f"[TRAIN] {metal} horizon={horizon}h")
            metrics.append(self.train_one(csv_path, metal, horizon))

        report_dir = self.artifact_repository.root / metal
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "metal": metal,
            "source_csv": csv_path,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "horizons_hours": list(HORIZONS_HOURS),
            "validation_method": "expanding walk-forward with target-time purge",
            "metrics": [asdict(item) for item in metrics],
        }
        (report_dir / f"{metal}_training_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return metrics
