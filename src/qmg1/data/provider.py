from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

from .metals import MetalSpec


class HistoricalM1Provider(Protocol):
    """Boundary for replaceable historical M1 market-data providers."""

    @property
    def source_name(self) -> str: ...

    @property
    def provider_description(self) -> str: ...

    @property
    def timeframe(self) -> str: ...

    @property
    def price_side(self) -> str: ...

    def validate_runtime(self) -> None: ...

    def download(self, metal: MetalSpec, start: date, end: date) -> Path:
        """Return a provider-native CSV for [start, end)."""
        ...
