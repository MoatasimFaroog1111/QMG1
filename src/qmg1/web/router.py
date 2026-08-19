from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from qmg1.api.schemas import ErrorResponse, PredictionRequest, PredictionResponse
from qmg1.api.service import ForecastApiService, PredictionUnavailableError


LOGGER = logging.getLogger("qmg1.web")

WEB_ROOT = Path(__file__).resolve().parent
STATIC_DIR = WEB_ROOT / "static"
DASHBOARD_FILE = WEB_ROOT / "templates" / "dashboard.html"
DASHBOARD_PREDICT_PATH = "/web/predict"


def build_web_router(service: ForecastApiService) -> APIRouter:
    """Build the browser dashboard and its same-origin prediction endpoint."""

    router = APIRouter(include_in_schema=False)

    @router.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD_FILE, media_type="text/html")

    @router.post(
        DASHBOARD_PREDICT_PATH,
        response_model=PredictionResponse,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        },
    )
    def dashboard_predict(request: PredictionRequest) -> PredictionResponse:
        """Run inference server-side without exposing the external API key to JavaScript."""

        try:
            result = service.predict(request)
        except PredictionUnavailableError as exc:
            LOGGER.warning(
                "dashboard_prediction_unavailable metal=%s horizon=%s",
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
