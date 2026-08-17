from __future__ import annotations

from datetime import date
from pathlib import Path

from .dukascopy_direct import DirectDukascopyM1Downloader
from .histdata import HistDataM1Downloader
from .metals import MetalSpec


class HybridPreciousMetalsM1Provider:
    """Route each metal to the most efficient verified M1 source."""

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

    @staticmethod
    def _yearly_ranges(start: date, end: date) -> list[tuple[date, date]]:
        ranges: list[tuple[date, date]] = []
        cursor = start
        while cursor < end:
            stop = min(date(cursor.year + 1, 1, 1), end)
            ranges.append((cursor, stop))
            cursor = stop
        return ranges

    def chunk_ranges(
        self,
        metal: MetalSpec,
        start: date,
        end: date,
    ) -> list[tuple[date, date]]:
        # HistData archives already provide efficient annual/monthly bulk files.
        # Passing its complete requested UTC span in one call also preserves
        # bars that cross source-calendar boundaries after fixed-EST -> UTC
        # conversion. Dukascopy remains yearly to keep BI5 cache runs bounded.
        if metal.key in {"gold", "silver"}:
            return [(start, end)]
        return self._yearly_ranges(start, end)

    def download(self, metal: MetalSpec, start: date, end: date) -> Path:
        provider = self._provider_for(metal)
        print(f"[ROUTE] {metal.name:10s} -> {provider.source_name}")
        return provider.download(metal, start, end)
