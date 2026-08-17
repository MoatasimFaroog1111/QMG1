from __future__ import annotations

from datetime import date
from pathlib import Path

from .dukascopy_direct import DirectDukascopyM1Downloader
from .histdata import HistDataM1Downloader
from .metals import MetalSpec


class HybridPreciousMetalsM1Provider:
    """Route bulk-friendly metals to HistData and the rest to Dukascopy."""

    def __init__(self, raw_root: Path) -> None:
        self.histdata = HistDataM1Downloader(raw_root=raw_root / "histdata")
        self.dukascopy = DirectDukascopyM1Downloader(raw_root=raw_root / "dukascopy")

    @property
    def source_name(self) -> str:
        return "HistData + Dukascopy"

    def _provider_for(self, metal: MetalSpec):
        if metal.key in {"gold", "silver"}:
            return self.histdata
        return self.dukascopy

    def source_name_for(self, metal: MetalSpec) -> str:
        return self._provider_for(metal).source_name_for(metal)

    @property
    def provider_description(self) -> str:
        return (
            "hybrid M1 provider: HistData bulk archives for gold/silver; "
            "direct Dukascopy BI5 for palladium/platinum"
        )

    @property
    def timeframe(self) -> str:
        return "m1"

    @property
    def price_side(self) -> str:
        return "bid"

    def validate_runtime(self) -> None:
        self.histdata.validate_runtime()
        self.dukascopy.validate_runtime()

    def download(self, metal: MetalSpec, start: date, end: date) -> Path:
        provider = self._provider_for(metal)
        print(f"[ROUTE] {metal.name:10s} -> {provider.source_name}")
        return provider.download(metal, start, end)
