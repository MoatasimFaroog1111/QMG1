from __future__ import annotations

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


def build_router(service: ForecastApiService) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_model=RootResponse)
    def root() -> RootResponse:
        return RootResponse(
            service="QMG1",
            docs="/docs",
            health="/health",
            predict="POST /predict",
            forecast_horizons_hours=list(HORIZONS_HOURS),
        )

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse.model_validate(service.health())

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
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return PredictionResponse.model_validate(result)

    return router
