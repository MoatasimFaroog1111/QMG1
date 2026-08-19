from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from qmg1.config import HORIZONS_HOURS

from .schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)
from .service import ForecastApiService, PredictionUnavailableError


LOGGER = logging.getLogger("qmg1.api")


def build_router(service: ForecastApiService) -> APIRouter:
    router = APIRouter()

    @router.get("/api/meta", response_model=RootResponse)
    def metadata() -> RootResponse:
        return RootResponse(
            service="QMG1",
            docs="/docs",
            health="/health",
            predict="POST /predict",
            forecast_horizons_hours=list(HORIZONS_HOURS),
            available_models=service.available_models(),
        )

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse.model_validate(service.health())

    @router.get("/livez", response_model=HealthResponse)
    def liveness() -> HealthResponse:
        return HealthResponse.model_validate(service.health())

    @router.get(
        "/readyz",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
    )
    def readiness() -> HealthResponse:
        payload = service.health()
        if not payload["ready"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Required serving artifacts are unavailable.",
            )
        return HealthResponse.model_validate(payload)

    @router.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        },
    )
    def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            result = service.predict(request)
        except PredictionUnavailableError as exc:
            LOGGER.warning(
                "prediction_unavailable metal=%s horizon=%s",
                request.metal,
                request.horizon_hours,
                exc_info=exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction is temporarily unavailable.",
            ) from exc
        return PredictionResponse.model_validate(result)

    return router
