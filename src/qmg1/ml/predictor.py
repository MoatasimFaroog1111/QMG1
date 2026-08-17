from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from qmg1.features import build_features, load_m1_csv, resample_to_hourly
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.dataset import ForecastDatasetBuilder
from qmg1.ml.exogenous import (
    GOLD_SILVER_PROVIDER_NAME,
    SPX_PROVIDER_NAME,
    USD_INDEX_PROVIDER_NAME,
    WTI_PROVIDER_NAME,
    GoldSilverFeatureProvider,
    SpxFeatureProvider,
    UsdIndexFeatureProvider,
    WtiFeatureProvider,
)


class ForecastPredictor:
    """Load persisted champion strategy and predict without retraining."""

    def __init__(self, artifact_repository: ModelArtifactRepository) -> None:
        self.artifact_repository = artifact_repository

    @staticmethod
    def _dataset_builder_for_artifact(
        artifact: dict[str, object],
        exogenous_csv_paths: Mapping[str, str] | None,
    ) -> ForecastDatasetBuilder:
        providers = []
        paths = exogenous_csv_paths or {}
        for metadata in artifact.get("exogenous_features", []):
            if not isinstance(metadata, dict):
                raise ValueError("Invalid exogenous feature metadata in artifact")
            name = str(metadata.get("name", ""))
            if name == GOLD_SILVER_PROVIDER_NAME:
                gold_path = paths.get("gold")
                if not gold_path:
                    raise ValueError(
                        "This model requires a gold H1 dataset. "
                        "Pass exogenous_csv_paths={'gold': '<XAUUSD H1 csv>', ...}."
                    )
                providers.append(GoldSilverFeatureProvider.from_hourly_csv(gold_path))
            elif name == USD_INDEX_PROVIDER_NAME:
                udx_path = paths.get("udx")
                if not udx_path:
                    raise ValueError(
                        "This model requires a UDX H1 dataset. "
                        "Pass exogenous_csv_paths={'udx': '<UDXUSD H1 csv>', ...}."
                    )
                providers.append(UsdIndexFeatureProvider.from_hourly_csv(udx_path))
            elif name == SPX_PROVIDER_NAME:
                spx_path = paths.get("spx")
                if not spx_path:
                    raise ValueError(
                        "This model requires an SPX H1 dataset. "
                        "Pass exogenous_csv_paths={'spx': '<SPXUSD H1 csv>', ...}."
                    )
                providers.append(SpxFeatureProvider.from_hourly_csv(spx_path))
            elif name == WTI_PROVIDER_NAME:
                wti_path = paths.get("wti")
                if not wti_path:
                    raise ValueError(
                        "This model requires a WTI H1 dataset. "
                        "Pass exogenous_csv_paths={'wti': '<WTIUSD H1 csv>', ...}."
                    )
                providers.append(WtiFeatureProvider.from_hourly_csv(wti_path))
            else:
                raise ValueError(f"Unsupported exogenous feature provider: {name}")
        return ForecastDatasetBuilder(exogenous_providers=providers)

    def predict_latest(
        self,
        csv_path: str,
        metal: str,
        horizon_hours: int,
        exogenous_csv_paths: Mapping[str, str] | None = None,
    ) -> dict[str, float | str | int]:
        artifact = self.artifact_repository.load(metal, horizon_hours)
        active_strategy = str(artifact.get("active_strategy", "model"))
        selected_challenger = str(
            artifact.get("selected_challenger", active_strategy)
        )

        m1 = load_m1_csv(csv_path)
        hourly = resample_to_hourly(m1)

        if active_strategy == "persistence":
            ready_hourly = hourly.dropna(subset=["close"])
            if ready_hourly.empty:
                raise ValueError("No usable hourly close is available for prediction")
            timestamp = ready_hourly.index[-1]
            current_close = float(ready_hourly.iloc[-1]["close"])
            predicted_log_return = 0.0
        else:
            builder = self._dataset_builder_for_artifact(
                artifact,
                exogenous_csv_paths,
            )
            frame = build_features(hourly)
            for provider in builder.exogenous_providers:
                frame = provider.augment(frame, hourly)

            feature_columns: list[str] = artifact["feature_columns"]
            ready = frame.dropna(subset=feature_columns)
            if ready.empty:
                raise ValueError("No feature-complete row is available for prediction")

            latest = ready.iloc[[-1]]
            timestamp = ready.index[-1]
            current_close = float(latest.iloc[0]["close"])
            predicted_log_return = float(
                artifact["model"].predict(latest[feature_columns])[0]
            )

        predicted_close = current_close * math.exp(predicted_log_return)
        metrics = artifact["metrics"]
        low_log_return = predicted_log_return + float(
            metrics["residual_log_return_q10"]
        )
        high_log_return = predicted_log_return + float(
            metrics["residual_log_return_q90"]
        )
        low_close = current_close * math.exp(low_log_return)
        high_close = current_close * math.exp(high_log_return)
        target_timestamp = timestamp + pd.Timedelta(hours=horizon_hours)

        return {
            "metal": str(artifact["metal"]),
            "timestamp_utc": timestamp.isoformat(),
            "target_timestamp_utc": target_timestamp.isoformat(),
            "horizon_hours": int(artifact["horizon_hours"]),
            "active_strategy": active_strategy,
            "selected_challenger": selected_challenger,
            "current_usd_per_kg": current_close,
            "predicted_usd_per_kg": predicted_close,
            "prediction_interval_80_low_usd_per_kg": low_close,
            "prediction_interval_80_high_usd_per_kg": high_close,
            "predicted_change_pct": (predicted_close / current_close - 1.0) * 100.0,
            "validation_mae_usd_per_kg": float(metrics["mae_usd_per_kg"]),
            "validation_directional_accuracy_pct": float(
                metrics["directional_accuracy_pct"]
            ),
            "validation_improvement_vs_persistence_pct": float(
                metrics["improvement_vs_persistence_pct"]
            ),
            "interval_note": (
                "Empirical 10th-90th percentile of untouched holdout "
                "log-return residuals; not a guarantee."
            ),
        }
