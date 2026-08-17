from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

from .metals import MetalSpec


class HistoricalM1Provider(Protocol):
    """Boundary for replaceable historical M1 market-data providers."""

    @property
    def source_name(self) -> str: ...

    def source_name_for(self, metal: MetalSpec) -> str: ...

    @property
    def provider_description(self) -> str: ...

    @property
    def timeframe(self) -> str: ...

    @property
    def price_side(self) -> str: ...

    def validate_runtime(self) -> None: ...

    def chunk_ranges(
        self,
        metal: MetalSpec,
        start: date,
        end: date,
    ) -> list[tuple[date, date]]:
        """Return preferred provider download ranges covering [start, end)."""
        ...

    def download(self, metal: MetalSpec, start: date, end: date) -> Path:
        """Return a provider-native CSV for [start, end)."""
        ...
