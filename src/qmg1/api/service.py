from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from qmg1.config import HORIZONS_HOURS
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.predictor import ForecastPredictor

from .schemas import PredictionRequest


TARGET_PATTERNS = {
    "gold": "XAUUSD_M1_USD_PER_KG_*.csv",
    "silver": "XAGUSD_M1_USD_PER_KG_*.csv",
    "palladium": "XPDCMDUSD_M1_USD_PER_KG_*.csv",
    "platinum": "XPTCMDUSD_M1_USD_PER_KG_*.csv",
}

EXOGENOUS_PATTERNS = {
    "gold": "XAUUSD_H1_USD_PER_KG_*.csv",
    "udx": "UDXUSD_H1_NATIVE_*.csv",
    "spx": "SPXUSD_H1_NATIVE_*.csv",
    "wti": "WTIUSD_H1_NATIVE_*.csv",
}


class PredictionUnavailableError(RuntimeError):
    """Raised when persisted artifacts or serving datasets are unavailable."""


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path
    models_dir: Path
    target_data_dir: Path
    hourly_context_dir: Path

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        default_root = Path(__file__).resolve().parents[3]
        project_root = Path(os.getenv("QMG1_PROJECT_ROOT", str(default_root))).resolve()
        return cls(
            project_root=project_root,
            models_dir=Path(
                os.getenv("QMG1_MODELS_DIR", str(project_root / "models"))
            ).resolve(),
            target_data_dir=Path(
                os.getenv(
                    "QMG1_DATA_DIR",
                    str(project_root / "metals_m1_usd_per_kg" / "final"),
                )
            ).resolve(),
            hourly_context_dir=Path(
                os.getenv(
                    "QMG1_HOURLY_DIR",
                    str(project_root / "training_data" / "hourly"),
                )
            ).resolve(),
        )


class ServingDataLocator:
    """Resolve serving files without exposing arbitrary filesystem paths to callers."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings

    @staticmethod
    def _newest(directory: Path, pattern: str) -> Path | None:
        if not directory.is_dir():
            return None
        matches = sorted(
            directory.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return matches[0] if matches else None

    def target_csv(self, metal: str) -> Path | None:
        pattern = TARGET_PATTERNS.get(metal)
        if pattern is None:
            return None
        return self._newest(self.settings.target_data_dir, pattern)

    def exogenous_csvs(self) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for name, pattern in EXOGENOUS_PATTERNS.items():
            path = self._newest(self.settings.hourly_context_dir, pattern)
            if path is not None:
                resolved[name] = str(path)
        return resolved

    def has_target_data(self) -> bool:
        return any(self.target_csv(metal) is not None for metal in TARGET_PATTERNS)

    def has_hourly_context(self) -> bool:
        return bool(self.exogenous_csvs())


class ForecastApiService:
    """Application service for health/status and persisted-model inference."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.locator = ServingDataLocator(settings)
        self.repository = ModelArtifactRepository(settings.models_dir)
        self.predictor = ForecastPredictor(self.repository)

    def health(self) -> dict[str, object]:
        models_available = (
            self.settings.models_dir.is_dir()
            and any(self.settings.models_dir.glob("**/*.joblib"))
        )
        return {
            "status": "ok",
            "service": "QMG1",
            "architecture": "train-once-persist-load-predict",
            "models_available": models_available,
            "target_data_available": self.locator.has_target_data(),
            "hourly_context_available": self.locator.has_hourly_context(),
        }

    def predict(self, request: PredictionRequest) -> dict[str, float | str | int]:
        if request.horizon_hours not in HORIZONS_HOURS:
            allowed = ", ".join(str(value) for value in HORIZONS_HOURS)
            raise PredictionUnavailableError(
                f"Unsupported horizon {request.horizon_hours}. Allowed hours: {allowed}."
            )

        target_csv = self.locator.target_csv(request.metal)
        if target_csv is None:
            raise PredictionUnavailableError(
                f"Serving data for {request.metal} is not available. "
                "Mount or restore the persisted market dataset before requesting inference."
            )

        try:
            return self.predictor.predict_latest(
                csv_path=str(target_csv),
                metal=request.metal,
                horizon_hours=request.horizon_hours,
                exogenous_csv_paths=self.locator.exogenous_csvs() or None,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise PredictionUnavailableError(str(exc)) from exc
