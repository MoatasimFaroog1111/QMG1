from __future__ import annotations

import math

from qmg1.features import build_features, load_m1_csv, resample_to_hourly
from qmg1.ml.artifacts import ModelArtifactRepository


class ForecastPredictor:
    """Load persisted models and predict without retraining."""

    def __init__(self, artifact_repository: ModelArtifactRepository) -> None:
        self.artifact_repository = artifact_repository

    def predict_latest(
        self,
        csv_path: str,
        metal: str,
        horizon_hours: int,
    ) -> dict[str, float | str | int]:
        artifact = self.artifact_repository.load(metal, horizon_hours)

        m1 = load_m1_csv(csv_path)
        hourly = resample_to_hourly(m1)
        frame = build_features(hourly)

        feature_columns: list[str] = artifact["feature_columns"]
        ready = frame.dropna(subset=feature_columns)
        if ready.empty:
            raise ValueError("No feature-complete row is available for prediction")

        latest = ready.iloc[[-1]]
        current_close = float(latest.iloc[0]["close"])
        predicted_log_return = float(
            artifact["model"].predict(latest[feature_columns])[0]
        )
        predicted_close = current_close * math.exp(predicted_log_return)

        metrics = artifact["metrics"]
        low_log_return = predicted_log_return + float(metrics["residual_log_return_q10"])
        high_log_return = predicted_log_return + float(metrics["residual_log_return_q90"])
        low_close = current_close * math.exp(low_log_return)
        high_close = current_close * math.exp(high_log_return)

        timestamp = ready.index[-1]
        target_timestamp = timestamp + __import__("pandas").Timedelta(hours=horizon_hours)

        return {
            "metal": str(artifact["metal"]),
            "timestamp_utc": timestamp.isoformat(),
            "target_timestamp_utc": target_timestamp.isoformat(),
            "horizon_hours": int(artifact["horizon_hours"]),
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
            "interval_note": "Empirical 10th-90th percentile of out-of-sample log-return residuals; not a guarantee.",
        }
