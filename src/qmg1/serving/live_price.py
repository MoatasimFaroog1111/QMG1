from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import threading
import time
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

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


class LivePriceProvider(Protocol):
    @property
    def configured(self) -> bool: ...

    def latest_quote(self, metal_key: str) -> LiveQuote: ...


class BullionVaultLivePriceProvider:
    """Read a cached public BullionVault order-book midpoint in USD per kg."""

    _SECURITY_IDS = {
        "gold": "AUXLN",
        "silver": "AGXLN",
        "platinum": "PTXLN",
        "palladium": "PDXLN",
    }

    def __init__(self, timeout_seconds: float = 4.0) -> None:
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return True

    def latest_quote(self, metal_key: str) -> LiveQuote:
        try:
            security_id = self._SECURITY_IDS[metal_key]
        except KeyError as exc:
            raise LivePriceUnavailableError(
                f"Unsupported BullionVault metal: {metal_key}"
            ) from exc

        query = urlencode(
            {
                "securityId": security_id,
                "considerationCurrency": "USD",
                "quantity": "0.001",
                "marketWidth": "1",
            }
        )
        request = Request(
            f"https://www.bullionvault.com/view_market_xml.do?{query}",
            headers={"Accept": "application/xml", "User-Agent": "QMG1/0.3 production-inference"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                payload = response.read()
                observed_at = parsedate_to_datetime(response.headers["Date"])
            pitch = ET.fromstring(payload).find(
                f".//pitch[@securityId='{security_id}'][@considerationCurrency='USD']"
            )
            if pitch is None:
                raise ValueError("requested BullionVault market is absent")
            bid = Decimal(pitch.find("./buyPrices/price").attrib["limit"])
            ask = Decimal(pitch.find("./sellPrices/price").attrib["limit"])
            if bid <= 0 or ask <= 0 or bid > ask:
                raise ValueError("BullionVault returned an invalid bid/ask spread")
        except (
            ET.ParseError,
            KeyError,
            AttributeError,
            InvalidOperation,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise LivePriceUnavailableError(
                f"BullionVault data is temporarily unavailable for {metal_key}"
            ) from exc

        return LiveQuote(
            metal=metal_key,
            timestamp_utc=observed_at.astimezone(timezone.utc),
            close_usd_per_kg=float((bid + ask) / Decimal("2")),
            source="BullionVault public cached USD order-book midpoint",
        )


class ResilientLivePriceProvider:
    """Fail over between bounded providers and persist the last valid quote."""

    def __init__(
        self,
        providers: list[LivePriceProvider],
        cache_root: Path,
        *,
        cache_ttl_seconds: float = 60.0,
        stale_ttl_seconds: float = 86_400.0,
        circuit_breaker_seconds: float = 60.0,
    ) -> None:
        self.providers = providers
        self.cache_root = cache_root
        self.cache_ttl_seconds = cache_ttl_seconds
        self.stale_ttl_seconds = stale_ttl_seconds
        self.circuit_breaker_seconds = circuit_breaker_seconds
        self._cache: dict[str, LiveQuote] = {}
        self._open_until: dict[int, float] = {}
        self._locks: dict[str, threading.Lock] = {}

    @property
    def configured(self) -> bool:
        return bool(self.providers) and all(provider.configured for provider in self.providers)

    def _cache_path(self, metal_key: str) -> Path:
        return self.cache_root / f"{metal_key}.json"

    @staticmethod
    def _age_seconds(quote: LiveQuote) -> float:
        return max(0.0, (datetime.now(timezone.utc) - quote.timestamp_utc).total_seconds())

    def _read_persistent(self, metal_key: str) -> LiveQuote | None:
        try:
            payload = json.loads(self._cache_path(metal_key).read_text(encoding="utf-8"))
            quote = LiveQuote(
                metal=str(payload["metal"]),
                timestamp_utc=datetime.fromisoformat(str(payload["timestamp_utc"])),
                close_usd_per_kg=float(payload["close_usd_per_kg"]),
                source=f"{payload['source']} (persistent cache)",
            )
            if (
                quote.metal != metal_key
                or quote.timestamp_utc.tzinfo is None
                or quote.close_usd_per_kg <= 0
            ):
                return None
            return quote
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_persistent(self, quote: LiveQuote) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(quote.metal)
        temporary = path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(
                {
                    "metal": quote.metal,
                    "timestamp_utc": quote.timestamp_utc.isoformat(),
                    "close_usd_per_kg": quote.close_usd_per_kg,
                    "source": quote.source,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def latest_quote(self, metal_key: str) -> LiveQuote:
        lock = self._locks.setdefault(metal_key, threading.Lock())
        with lock:
            cached = self._cache.get(metal_key) or self._read_persistent(metal_key)
            if cached and self._age_seconds(cached) <= self.cache_ttl_seconds:
                self._cache[metal_key] = cached
                return cached

            failures: list[str] = []
            now = time.monotonic()
            for index, provider in enumerate(self.providers):
                if self._open_until.get(index, 0.0) > now:
                    continue
                try:
                    quote = provider.latest_quote(metal_key)
                    self._open_until.pop(index, None)
                    self._cache[metal_key] = quote
                    try:
                        self._write_persistent(quote)
                    except OSError:
                        pass
                    return quote
                except LivePriceUnavailableError as exc:
                    failures.append(str(exc))
                    self._open_until[index] = time.monotonic() + self.circuit_breaker_seconds

            if cached and self._age_seconds(cached) <= self.stale_ttl_seconds:
                self._cache[metal_key] = cached
                return cached
            detail = "; ".join(failures) or "all providers have open circuits"
            raise LivePriceUnavailableError(f"Live market data is unavailable: {detail}")


class DukascopyLivePriceProvider:
    """Read the latest completed M1 close from Dukascopy for inference serving."""

    def __init__(
        self,
        cache_root: Path,
        lookback_days: int = 4,
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
                timeout_seconds=4,
                max_attempts=1,
                base_backoff_seconds=0.5,
                request_pause_seconds=0.0,
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

            today = datetime.now(timezone.utc).date()
            daily_cache_root = self.cache_root / today.isoformat()
            downloader = self._downloader(daily_cache_root)
            last_error: Exception | None = None
            try:
                for days_ago in range(1, self.lookback_days + 1):
                    day = today - timedelta(days=days_ago)
                    try:
                        path = downloader.download(metal, day, day + timedelta(days=1))
                        quote = self._quote_from_csv(path, metal)
                        self._cache[metal_key] = (time.monotonic(), quote)
                        return quote
                    except (RuntimeError, OSError, ValueError) as exc:
                        last_error = exc
                raise LivePriceUnavailableError(
                    f"No recent Dukascopy session is available for {metal.name}"
                ) from last_error
            except (LivePriceUnavailableError, RuntimeError, OSError, ValueError) as exc:
                cached = self._cache.get(metal_key)
                if cached and time.monotonic() - cached[0] <= self.stale_ttl_seconds:
                    return cached[1]
                raise LivePriceUnavailableError(
                    f"Live Dukascopy data is temporarily unavailable for {metal.name}"
                ) from exc

