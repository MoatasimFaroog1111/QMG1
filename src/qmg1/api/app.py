from __future__ import annotations

from fastapi import FastAPI

from .routes import build_router
from .service import ForecastApiService, RuntimeSettings


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    runtime_settings = settings or RuntimeSettings.from_environment()
    service = ForecastApiService(runtime_settings)

    application = FastAPI(
        title="QMG1 Forecast API",
        version="0.1.0",
        description=(
            "Persisted-model inference API for precious-metal forecasts in USD/kg. "
            "Training remains an offline workflow and is never triggered by an HTTP request."
        ),
    )
    application.include_router(build_router(service))
    return application


app = create_app()
