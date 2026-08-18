from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from qmg1.web.router import STATIC_DIR, build_web_router

from .routes import build_router
from .service import ForecastApiService, RuntimeSettings


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    runtime_settings = settings or RuntimeSettings.from_environment()
    service = ForecastApiService(runtime_settings)

    application = FastAPI(
        title="QMG1 Forecast API",
        version="0.2.0",
        description=(
            "Persisted-model inference API for precious-metal forecasts in USD/kg. "
            "Training remains an offline workflow and is never triggered by an HTTP request."
        ),
    )
    application.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")
    application.include_router(build_web_router())
    application.include_router(build_router(service))
    return application


app = create_app()
