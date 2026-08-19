from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import threading
import time

from qmg1.data.dukascopy_direct import DirectDukascopyConfig, DirectDukascopyM1Downloader
from qmg1.data.metals import METALS, MetalSpec
from qmg1.data.normalizer import TROY_OUNCES_PER_KG


class LivePriceUnavailableError(RuntimeError):
    """Raised when the serving market-data source cannot produce a usable quote."""


@dataclass(frozen=True)
class LiveQuote:
    metal: str
    timestamp_utc: datetime
    close_usd_per_kg: float
    source: str


class DukascopyLivePriceProvider:
    """Read the latest completed M1 close from Dukascopy for inference serving."""

    def __init__(
        self,
        cache_root: Path,
        lookback_days: int = 7,
        cache_ttl_seconds: float = 60.0,
        stale_ttl_seconds: float = 300.0,
    ) -> None:
        if lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        if cache_ttl_seconds <= 0 or stale_ttl_seconds < cache_ttl_seconds:
            raise ValueError("live-price cache TTL values are invalid")
        self.cache_root = cache_root
        self.lookback_days = lookback_days
        self.cache_ttl_seconds = cache_ttl_seconds
        self.stale_ttl_seconds = stale_ttl_seconds
        self._metals = {metal.key: metal for metal in METALS}
        self._cache: dict[str, tuple[float, LiveQuote]] = {}
        self._locks: dict[str, threading.Lock] = {
            metal.key: threading.Lock() for metal in METALS
        }

    @property
    def configured(self) -> bool:
        return True

    def _downloader(self, daily_cache_root: Path) -> DirectDukascopyM1Downloader:
        return DirectDukascopyM1Downloader(
            raw_root=daily_cache_root,
            config=DirectDukascopyConfig(
                timeout_seconds=15,
                max_attempts=3,
                base_backoff_seconds=0.5,
                request_pause_seconds=0.05,
                download_passes=1,
                pass_backoff_seconds=1.0,
                user_agent="QMG1/0.3 production-inference",
            ),
        )

    def _metal(self, metal_key: str) -> MetalSpec:
        try:
            return self._metals[metal_key]
        except KeyError as exc:
            raise LivePriceUnavailableError(f"Unsupported live metal: {metal_key}") from exc

    @staticmethod
    def _quote_from_csv(path: Path, metal: MetalSpec) -> LiveQuote:
        latest: dict[str, str] | None = None
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                latest = row

        if latest is None:
            raise LivePriceUnavailableError(f"No live rows were produced for {metal.name}")

        try:
            timestamp_ms = int(latest["timestamp"])
            close_per_ounce = Decimal(latest["close"])
            close_per_kg = close_per_ounce * TROY_OUNCES_PER_KG
        except (KeyError, ValueError, InvalidOperation) as exc:
            raise LivePriceUnavailableError(
                f"Malformed live Dukascopy quote for {metal.name}"
            ) from exc

        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return LiveQuote(
            metal=metal.key,
            timestamp_utc=timestamp,
            close_usd_per_kg=float(close_per_kg),
            source="Dukascopy bid M1 / last completed UTC sessions",
        )

    def latest_quote(self, metal_key: str) -> LiveQuote:
        metal = self._metal(metal_key)
        now = time.monotonic()
        cached = self._cache.get(metal_key)
        if cached and now - cached[0] <= self.cache_ttl_seconds:
            return cached[1]

        with self._locks[metal_key]:
            now = time.monotonic()
            cached = self._cache.get(metal_key)
            if cached and now - cached[0] <= self.cache_ttl_seconds:
                return cached[1]

            end_exclusive = datetime.now(timezone.utc).date()
            start = end_exclusive - timedelta(days=self.lookback_days)
            daily_cache_root = self.cache_root / end_exclusive.isoformat()
            downloader = self._downloader(daily_cache_root)
            try:
                path = downloader.download(metal, start, end_exclusive)
                quote = self._quote_from_csv(path, metal)
                self._cache[metal_key] = (time.monotonic(), quote)
                return quote
            except (RuntimeError, OSError, ValueError) as exc:
                cached = self._cache.get(metal_key)
                if cached and time.monotonic() - cached[0] <= self.stale_ttl_seconds:
                    return cached[1]
                raise LivePriceUnavailableError(
                    f"Live Dukascopy data is temporarily unavailable for {metal.name}"
                ) from exc
