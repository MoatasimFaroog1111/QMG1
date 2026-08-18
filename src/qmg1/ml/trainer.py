from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone

from qmg1.config import HORIZONS_HOURS, TrainingConfig
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.dataset import FeatureBase, ForecastDatasetBuilder, PreparedDataset
from qmg1.ml.directional import DirectionDiagnostics
from qmg1.ml.evaluation import HorizonMetrics
from qmg1.ml.model_factory import apply_training_lookback
from qmg1.ml.selection import ChampionChallengerSelector


class ForecastTrainer:
    """Select on development data, verify on holdout, then persist once."""

    def __init__(
        self,
        artifact_repository: ModelArtifactRepository,
        config: TrainingConfig | None = None,
        dataset_builder: ForecastDatasetBuilder | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.dataset_builder = dataset_builder or ForecastDatasetBuilder()
        self.artifact_repository = artifact_repository
        self.selector = ChampionChallengerSelector(self.config)

    def _directional_diagnostic(
        self,
        prepared: PreparedDataset,
    ) -> dict[str, object]:
        frame = prepared.frame
        split_at = int(len(frame) * 0.80)
        development = frame.iloc[:split_at]
        metrics = DirectionDiagnostics.walk_forward_classifier(
            frame=development,
            feature_columns=prepared.feature_columns,
            horizon_hours=prepared.horizon_hours,
            cv_splits=self.config.cv_splits,
        )
        result = asdict(metrics)
        result["scope"] = "development_walk_forward_only"
        result["production_effect"] = "none"
        return result

    def _train_prepared(
        self,
        prepared: PreparedDataset,
        metal: str,
    ) -> HorizonMetrics:
        horizon_hours = prepared.horizon_hours
        frame = prepared.frame
        features = prepared.feature_columns

        if len(frame) < self.config.min_rows:
            raise ValueError(
                f"Not enough hourly rows for {metal} {horizon_hours}h: "
                f"{len(frame):,} < {self.config.min_rows:,}"
            )

        winning_factory, selection = self.selector.select(
            frame,
            features,
            horizon_hours,
        )
        directional_diagnostic = self._directional_diagnostic(prepared)

        target_col = f"target_{horizon_hours}h"
        final_reference = frame.index[-1]
        final_train = apply_training_lookback(
            winning_factory,
            frame,
            final_reference,
        )
        if final_train.empty:
            raise ValueError("Final challenger training window is empty")

        final_model = winning_factory.create()
        final_model.fit(final_train[features], final_train[target_col])

        active_metrics = selection.active_holdout_metrics
        artifact = {
            "schema_version": 7,
            "metal": metal,
            "horizon_hours": horizon_hours,
            "feature_columns": features,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(final_train),
            "training_start_utc": final_train.index[0].isoformat(),
            "training_end_utc": final_train.index[-1].isoformat(),
            "candidate_training_lookback_days": winning_factory.lookback_days,
            "training_config": asdict(self.config),
            "validation_method": (
                "development walk-forward selection + untouched 20% holdout "
                "with target-time purge, fixed recency challengers, and "
                "causal exogenous alignment"
            ),
            "exogenous_features": self.dataset_builder.exogenous_metadata(),
            "active_strategy": selection.active_strategy,
            "selected_challenger": selection.selected_model_name,
            "selection": selection.to_dict(),
            "metrics": asdict(active_metrics),
            "directional_diagnostic": directional_diagnostic,
            "model": final_model,
        }
        self.artifact_repository.save(metal, horizon_hours, artifact)
        return active_metrics

    def train_one(
        self,
        csv_path: str,
        metal: str,
        horizon_hours: int,
    ) -> HorizonMetrics:
        prepared = self.dataset_builder.build(csv_path, horizon_hours)
        return self._train_prepared(prepared, metal)

    def _train_all_from_base(
        self,
        base: FeatureBase,
        source_csv: str,
        metal: str,
        horizons: Sequence[int],
    ) -> list[HorizonMetrics]:
        requested_horizons = tuple(horizons)
        if not requested_horizons:
            return []

        unknown = [horizon for horizon in requested_horizons if horizon not in HORIZONS_HOURS]
        if unknown:
            raise ValueError(f"Unsupported forecast horizons: {unknown}")

        metrics: list[HorizonMetrics] = []
        horizon_status: list[dict[str, object]] = []
        for horizon in requested_horizons:
            print(f"[TRAIN] {metal} horizon={horizon}h")
            prepared = self.dataset_builder.build_from_base(base, horizon)
            horizon_metrics = self._train_prepared(prepared, metal)
            metrics.append(horizon_metrics)

            artifact = self.artifact_repository.load(metal, horizon)
            selection = artifact["selection"]
            horizon_status.append(
                {
                    "horizon_hours": horizon,
                    "active_strategy": artifact["active_strategy"],
                    "selected_challenger": artifact["selected_challenger"],
                    "candidate_training_lookback_days": artifact[
                        "candidate_training_lookback_days"
                    ],
                    "development_qualified": selection["development_qualified"],
                    "holdout_qualified": selection["holdout_qualified"],
                    "promotion_threshold_pct": selection[
                        "promotion_threshold_pct"
                    ],
                    "active_improvement_vs_persistence_pct": (
                        horizon_metrics.improvement_vs_persistence_pct
                    ),
                    "directional_diagnostic": artifact["directional_diagnostic"],
                }
            )

        report_dir = self.artifact_repository.root / metal
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "metal": metal,
            "source_csv": source_csv,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "horizons_hours": list(requested_horizons),
            "validation_method": (
                "development walk-forward champion/challenger selection + "
                "untouched 20% holdout + dual promotion gate"
            ),
            "directional_diagnostic_method": (
                "diagnostic-only balanced logistic classifier on development "
                "walk-forward folds with target-time purge; no production effect"
            ),
            "feature_base_reused_across_horizons": True,
            "exogenous_features": self.dataset_builder.exogenous_metadata(),
            "horizon_status": horizon_status,
            "metrics": [asdict(item) for item in metrics],
        }
        (report_dir / f"{metal}_training_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return metrics

    def train_all(
        self,
        csv_path: str,
        metal: str,
        horizons: Sequence[int] = HORIZONS_HOURS,
    ) -> list[HorizonMetrics]:
        base = self.dataset_builder.load_feature_base(csv_path)
        return self._train_all_from_base(base, csv_path, metal, horizons)

    def train_all_hourly(
        self,
        csv_path: str,
        metal: str,
        horizons: Sequence[int] = HORIZONS_HOURS,
    ) -> list[HorizonMetrics]:
        base = self.dataset_builder.load_hourly_feature_base(csv_path)
        return self._train_all_from_base(base, csv_path, metal, horizons)
