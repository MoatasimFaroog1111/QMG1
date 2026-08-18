from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MetalName = Literal["gold", "silver", "palladium", "platinum"]


class RootResponse(BaseModel):
    service: str
    docs: str
    health: str
    predict: str
    forecast_horizons_hours: list[int]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    architecture: str
    models_available: bool
    target_data_available: bool
    hourly_context_available: bool
    live_market_data_enabled: bool


class PredictionRequest(BaseModel):
    metal: MetalName
    horizon_hours: int = Field(ge=1)


class PredictionResponse(BaseModel):
    metal: str
    timestamp_utc: str
    target_timestamp_utc: str
    horizon_hours: int
    active_strategy: str
    selected_challenger: str
    current_usd_per_kg: float
    predicted_usd_per_kg: float
    prediction_interval_80_low_usd_per_kg: float
    prediction_interval_80_high_usd_per_kg: float
    predicted_change_pct: float
    validation_mae_usd_per_kg: float
    validation_directional_accuracy_pct: float
    validation_improvement_vs_persistence_pct: float
    interval_note: str


class ErrorResponse(BaseModel):
    detail: str
