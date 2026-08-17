from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .config import HORIZONS_HOURS, TrainingConfig
from .features import build_features, feature_columns, load_m1_csv, resample_to_hourly


@dataclass(frozen=True)
class HorizonMetrics:
    horizon_hours: int
    cv_splits: int
    rows_total: int
    rows_validation_total: int
    mae_usd_per_kg: float
    rmse_usd_per_kg: float
    smape_pct: float
    directional_accuracy_pct: float
    persistence_mae_usd_per_kg: float
    improvement_vs_persistence_pct: float


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 0
    if not valid.any():
        return 0.0
    return float(
        np.mean(2.0 * np.abs(y_pred[valid] - y_true[valid]) / denom[valid]) * 100.0
    )


def _build_model(cfg: TrainingConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        l2_regularization=cfg.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=cfg.random_state,
    )


def prepare_dataset(csv_path: str, horizon_hours: int) -> tuple[pd.DataFrame, list[str]]:
    m1 = load_m1_csv(csv_path)
    hourly = resample_to_hourly(m1)
    frame = build_features(hourly)

    frame[f"target_{horizon_hours}h"] = (
        np.log(frame["close"].shift(-horizon_hours)) - np.log(frame["close"])
    )
    frame[f"future_close_{horizon_hours}h"] = frame["close"].shift(-horizon_hours)

    cols = feature_columns(frame)
    required = cols + [
        "close",
        f"target_{horizon_hours}h",
        f"future_close_{horizon_hours}h",
    ]
    frame = frame.dropna(subset=required)
    return frame, cols


def evaluate_walk_forward(
    frame: pd.DataFrame,
    cols: list[str],
    horizon_hours: int,
    cfg: TrainingConfig,
) -> HorizonMetrics:
    splitter = TimeSeriesSplit(n_splits=cfg.cv_splits, gap=horizon_hours)

    predicted_close_parts: list[np.ndarray] = []
    actual_close_parts: list[np.ndarray] = []
    current_close_parts: list[np.ndarray] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(frame), start=1):
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        if train.empty or valid.empty:
            continue

        model = _build_model(cfg)
        model.fit(train[cols], train[f"target_{horizon_hours}h"])

        predicted_log_return = model.predict(valid[cols])
        current_close = valid["close"].to_numpy(dtype=float)
        actual_close = valid[f"future_close_{horizon_hours}h"].to_numpy(dtype=float)
        predicted_close = current_close * np.exp(predicted_log_return)

        print(
            f"  [CV {fold}/{cfg.cv_splits}] "
            f"train={len(train):,} validation={len(valid):,}"
        )

        predicted_close_parts.append(predicted_close)
        actual_close_parts.append(actual_close)
        current_close_parts.append(current_close)

    if not predicted_close_parts:
        raise ValueError("Walk-forward validation produced no folds")

    predicted_close = np.concatenate(predicted_close_parts)
    actual_close = np.concatenate(actual_close_parts)
    current_close = np.concatenate(current_close_parts)

    mae = float(mean_absolute_error(actual_close, predicted_close))
    rmse = float(math.sqrt(mean_squared_error(actual_close, predicted_close)))
    persistence_mae = float(mean_absolute_error(actual_close, current_close))

    actual_direction = np.sign(actual_close - current_close)
    predicted_direction = np.sign(predicted_close - current_close)
    directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100.0)
    improvement = (
        (persistence_mae - mae) / persistence_mae * 100.0
        if persistence_mae > 0
        else 0.0
    )

    return HorizonMetrics(
        horizon_hours=horizon_hours,
        cv_splits=cfg.cv_splits,
        rows_total=len(frame),
        rows_validation_total=len(actual_close),
        mae_usd_per_kg=mae,
        rmse_usd_per_kg=rmse,
        smape_pct=_smape(actual_close, predicted_close),
        directional_accuracy_pct=directional_accuracy,
        persistence_mae_usd_per_kg=persistence_mae,
        improvement_vs_persistence_pct=improvement,
    )


def train_one_horizon(
    csv_path: str,
    output_dir: Path,
    metal: str,
    horizon_hours: int,
    cfg: TrainingConfig,
) -> HorizonMetrics:
    frame, cols = prepare_dataset(csv_path, horizon_hours)
    if len(frame) < cfg.min_rows:
        raise ValueError(
            f"Not enough hourly rows for {metal} {horizon_hours}h: "
            f"{len(frame):,} < {cfg.min_rows:,}"
        )

    metrics = evaluate_walk_forward(frame, cols, horizon_hours, cfg)

    # Train once on every feature-complete historical row after honest OOS evaluation.
    final_model = _build_model(cfg)
    final_model.fit(frame[cols], frame[f"target_{horizon_hours}h"])

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 2,
        "metal": metal,
        "horizon_hours": horizon_hours,
        "feature_columns": cols,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(frame),
        "training_start_utc": frame.index[0].isoformat(),
        "training_end_utc": frame.index[-1].isoformat(),
        "training_config": asdict(cfg),
        "metrics": asdict(metrics),
        "model": final_model,
    }
    joblib.dump(artifact, output_dir / f"{metal}_{horizon_hours}h.joblib")
    return metrics


def train_all_horizons(
    csv_path: str,
    output_dir: Path,
    metal: str,
    cfg: TrainingConfig | None = None,
) -> list[HorizonMetrics]:
    cfg = cfg or TrainingConfig()
    metrics: list[HorizonMetrics] = []

    for horizon in HORIZONS_HOURS:
        print(f"[TRAIN] {metal} horizon={horizon}h")
        metrics.append(train_one_horizon(csv_path, output_dir, metal, horizon, cfg))

    report = {
        "metal": metal,
        "source_csv": csv_path,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_method": "expanding walk-forward TimeSeriesSplit with horizon gap",
        "metrics": [asdict(m) for m in metrics],
    }
    (output_dir / f"{metal}_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return metrics


def predict_latest(csv_path: str, artifact_path: Path) -> dict[str, float | str | int]:
    artifact = joblib.load(artifact_path)
    m1 = load_m1_csv(csv_path)
    hourly = resample_to_hourly(m1)
    frame = build_features(hourly)

    cols: list[str] = artifact["feature_columns"]
    ready = frame.dropna(subset=cols)
    if ready.empty:
        raise ValueError("No feature-complete row is available for prediction")

    row = ready.iloc[-1]
    predicted_log_return = float(artifact["model"].predict(ready.iloc[[-1]][cols])[0])
    current_close = float(row["close"])
    predicted_close = current_close * math.exp(predicted_log_return)

    return {
        "metal": str(artifact["metal"]),
        "timestamp_utc": ready.index[-1].isoformat(),
        "horizon_hours": int(artifact["horizon_hours"]),
        "current_usd_per_kg": current_close,
        "predicted_usd_per_kg": predicted_close,
        "predicted_change_pct": (predicted_close / current_close - 1.0) * 100.0,
        "validation_mae_usd_per_kg": float(artifact["metrics"]["mae_usd_per_kg"]),
        "validation_directional_accuracy_pct": float(
            artifact["metrics"]["directional_accuracy_pct"]
        ),
        "validation_improvement_vs_persistence_pct": float(
            artifact["metrics"]["improvement_vs_persistence_pct"]
        ),
    }
