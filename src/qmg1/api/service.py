from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from qmg1.config import HORIZONS_HOURS
from qmg1.ml.artifacts import ModelArtifactRepository
from qmg1.ml.predictor import ForecastPredictor
from qmg1.ml.serving_policy import PersistedArtifactServingPolicy
from qmg1.serving.live_price import (
    BullionVaultLivePriceProvider,
    DukascopyLivePriceProvider,
    LivePriceProvider,
    LivePriceUnavailableError,
    ResilientLivePriceProvider,
)

from .schemas import PredictionRequest


LOGGER = logging.getLogger("qmg1.api.service")


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
    """Raised when persisted artifacts or serving market data are unavailable."""


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path
    models_dir: Path
    target_data_dir: Path
    hourly_context_dir: Path
    live_cache_dir: Path | None = None
    api_key: str | None = None
    predict_requests_per_minute: int = 30
    live_cache_ttl_seconds: float = 60.0
    live_stale_ttl_seconds: float = 300.0
    required_metals: tuple[str, ...] = ("silver",)
    required_horizons: tuple[int, ...] = HORIZONS_HOURS
    production_mode: bool = False

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        default_root = Path(__file__).resolve().parents[3]
        project_root = Path(os.getenv("QMG1_PROJECT_ROOT", str(default_root))).resolve()
        required_metals = tuple(
            value.strip()
            for value in os.getenv("QMG1_REQUIRED_METALS", "silver").split(",")
            if value.strip()
        )
        required_horizons = tuple(
            int(value.strip())
            for value in os.getenv(
                "QMG1_REQUIRED_HORIZONS", ",".join(map(str, HORIZONS_HOURS))
            ).split(",")
            if value.strip()
        )
        return cls(
            project_root=project_root,
            models_dir=Path(
                os.getenv(
                    "QMG1_MODELS_DIR",
                    str(project_root / "serving_artifacts" / "models"),
                )
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
            live_cache_dir=Path(
                os.getenv("QMG1_LIVE_CACHE_DIR", "/tmp/qmg1-live")
            ).resolve(),
            api_key=os.getenv("QMG1_API_KEY") or None,
            predict_requests_per_minute=int(
                os.getenv("QMG1_PREDICT_REQUESTS_PER_MINUTE", "30")
            ),
            live_cache_ttl_seconds=float(
                os.getenv("QMG1_LIVE_CACHE_TTL_SECONDS", "60")
            ),
            live_stale_ttl_seconds=float(
                os.getenv("QMG1_LIVE_STALE_TTL_SECONDS", "86400")
            ),
            required_metals=required_metals,
            required_horizons=required_horizons,
            production_mode=os.getenv("QMG1_ENVIRONMENT", "production").lower()
            == "production",
        )


class ServingDataLocator:
    """Resolve persisted feature datasets needed by feature-dependent estimators."""

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
    """Application service for health/status and persisted trained-artifact inference."""

    def __init__(
        self,
        settings: RuntimeSettings,
        live_price_provider: LivePriceProvider | None = None,
    ) -> None:
        self.settings = settings
        self.locator = ServingDataLocator(settings)
        self.repository = ModelArtifactRepository(settings.models_dir)
        self.predictor = ForecastPredictor(self.repository)
        cache_root = settings.live_cache_dir or Path("/tmp/qmg1-live")
        self.live_price_provider = live_price_provider or ResilientLivePriceProvider(
            providers=[
                BullionVaultLivePriceProvider(),
                DukascopyLivePriceProvider(
                    cache_root / "dukascopy",
                    cache_ttl_seconds=settings.live_cache_ttl_seconds,
                    stale_ttl_seconds=settings.live_stale_ttl_seconds,
                ),
            ],
            cache_root=cache_root / "quotes",
            cache_ttl_seconds=settings.live_cache_ttl_seconds,
            stale_ttl_seconds=settings.live_stale_ttl_seconds,
        )

    def available_models(self) -> dict[str, list[int]]:
        """Expose only complete, checksum-valid artifacts produced by prior training."""

        return self.repository.available_trained()

    def readiness(self) -> tuple[bool, list[str]]:
        available = self.available_models()
        missing: list[str] = []
        for metal in self.settings.required_metals:
            horizons = set(available.get(metal, []))
            for horizon in self.settings.required_horizons:
                if horizon not in horizons:
                    missing.append(f"{metal}:{horizon}h")
        reasons = [f"missing_model:{item}" for item in missing]
        if self.settings.production_mode and not self.settings.api_key:
            reasons.append("missing_configuration:QMG1_API_KEY")
        ready = not reasons
        if not ready:
            LOGGER.warning(
                "readiness_not_ready missing=%d production_mode=%s api_key_set=%s",
                len(missing),
                self.settings.production_mode,
                bool(self.settings.api_key),
            )
        return ready, reasons

    def health(self) -> dict[str, object]:
        ready, readiness_reasons = self.readiness()
        available_models = self.available_models()
        return {
            "status": "ok",
            "service": "QMG1",
            "architecture": "train-once-persist-load-predict",
            "models_available": bool(available_models),
            "target_data_available": self.locator.has_target_data(),
            "hourly_context_available": self.locator.has_hourly_context(),
            "live_market_data_enabled": self.live_price_provider.configured,
            "ready": ready,
            "readiness_reasons": readiness_reasons,
            "available_models": available_models,
        }

    @staticmethod
    def _validate_horizon(request: PredictionRequest) -> None:
        if request.horizon_hours not in HORIZONS_HOURS:
            allowed = ", ".join(str(value) for value in HORIZONS_HOURS)
            raise PredictionUnavailableError(
                f"Unsupported horizon {request.horizon_hours}. Allowed hours: {allowed}."
            )

    def _predict_with_live_quote(
        self,
        request: PredictionRequest,
        artifact: dict[str, object],
    ) -> dict[str, float | str | int]:
        try:
            quote = self.live_price_provider.latest_quote(request.metal)
            result = self.predictor.predict_live_from_artifact(
                artifact=artifact,
                timestamp_utc=quote.timestamp_utc,
                close_usd_per_kg=quote.close_usd_per_kg,
            )
            result["market_data_source"] = quote.source
            LOGGER.info(
                "predict_live_model_completed metal=%s horizon_hours=%s "
                "strategy=%s quote_source=%s current_usd_per_kg=%.4f "
                "predicted_usd_per_kg=%.4f",
                request.metal,
                request.horizon_hours,
                result["serving_strategy"],
                quote.source,
                result["current_usd_per_kg"],
                result["predicted_usd_per_kg"],
            )
            return result
        except (
            LivePriceUnavailableError,
            FileNotFoundError,
            ValueError,
            OSError,
        ) as exc:
            LOGGER.warning(
                "predict_live_model_failed metal=%s horizon_hours=%s exc_type=%s",
                request.metal,
                request.horizon_hours,
                type(exc).__name__,
            )
            raise PredictionUnavailableError(str(exc)) from exc

    def _predict_with_artifact(
        self,
        request: PredictionRequest,
        artifact: dict[str, object],
    ) -> dict[str, float | str | int]:
        try:
            decision = PersistedArtifactServingPolicy.resolve(artifact)
        except ValueError as exc:
            raise PredictionUnavailableError(str(exc)) from exc

        LOGGER.info(
            "predict_model_path metal=%s horizon_hours=%s serving_strategy=%s "
            "governance_strategy=%s",
            request.metal,
            request.horizon_hours,
            decision.serving_strategy,
            decision.governance_strategy,
        )

        if not decision.feature_data_required:
            return self._predict_with_live_quote(request, artifact)

        target_csv = self.locator.target_csv(request.metal)
        if target_csv is None:
            LOGGER.error(
                "predict_missing_target_csv metal=%s models_dir=%s strategy=%s",
                request.metal,
                self.settings.models_dir,
                decision.serving_strategy,
            )
            raise PredictionUnavailableError(
                f"The persisted {request.metal} {decision.serving_strategy} model "
                "requires serving feature data, but its dataset is not mounted."
            )

        try:
            result = self.predictor.predict_latest_from_artifact(
                csv_path=str(target_csv),
                artifact=artifact,
                exogenous_csv_paths=self.locator.exogenous_csvs() or None,
            )
            LOGGER.info(
                "predict_completed metal=%s horizon_hours=%s current_usd_per_kg=%.4f "
                "predicted_usd_per_kg=%.4f",
                request.metal,
                request.horizon_hours,
                result["current_usd_per_kg"],
                result["predicted_usd_per_kg"],
            )
            return result
        except (FileNotFoundError, ValueError, OSError) as exc:
            LOGGER.warning(
                "predict_model_failed metal=%s horizon_hours=%s exc_type=%s",
                request.metal,
                request.horizon_hours,
                type(exc).__name__,
            )
            raise PredictionUnavailableError(str(exc)) from exc

    def predict(self, request: PredictionRequest) -> dict[str, float | str | int]:
        """Predict only from a complete artifact created by a previous training run."""

        LOGGER.info(
            "predict_request metal=%s horizon_hours=%s",
            request.metal,
            request.horizon_hours,
        )
        self._validate_horizon(request)
        try:
            artifact = self.repository.load_trained(
                request.metal,
                request.horizon_hours,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            LOGGER.warning(
                "predict_artifact_unavailable metal=%s horizon_hours=%s exc_type=%s",
                request.metal,
                request.horizon_hours,
                type(exc).__name__,
            )
            raise PredictionUnavailableError(
                f"Previously trained artifact for {request.metal} "
                f"{request.horizon_hours}h is unavailable: {exc}"
            ) from exc

        return self._predict_with_artifact(request, artifact)
