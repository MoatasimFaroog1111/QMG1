from __future__ import annotations

import csv
import lzma
import math
import struct
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .metals import MetalSpec


_M1_STRUCT = struct.Struct(">5if")
_SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class DirectDukascopyConfig:
    base_url: str = "https://datafeed.dukascopy.com/datafeed"
    timeframe: str = "m1"
    price_side: str = "bid"
    timeout_seconds: int = 60
    max_attempts: int = 6
    base_backoff_seconds: float = 2.0
    request_pause_seconds: float = 0.50
    download_passes: int = 3
    pass_backoff_seconds: float = 20.0
    user_agent: str = "QMG1/0.1 historical-market-data"


class DirectDukascopyM1Downloader:
    """Download and decode Dukascopy daily M1 BI5 files directly in Python.

    The provider owns only acquisition and BI5 decoding. Daily compressed
    payloads are cached before aggregate CSV construction, so interrupted or
    partially rate-limited historical runs resume from already fetched days.
    """

    def __init__(
        self,
        raw_root: Path,
        config: DirectDukascopyConfig | None = None,
    ) -> None:
        self.raw_root = raw_root
        self.config = config or DirectDukascopyConfig()

    @property
    def source_name(self) -> str:
        return "Dukascopy"

    def source_name_for(self, metal: MetalSpec) -> str:
        return self.source_name

    @property
    def provider_description(self) -> str:
        return "direct Dukascopy BI5 datafeed (Python stdlib)"

    @property
    def timeframe(self) -> str:
        return self.config.timeframe

    @property
    def price_side(self) -> str:
        return self.config.price_side

    def validate_runtime(self) -> None:
        if self.config.timeframe != "m1":
            raise RuntimeError("DirectDukascopyM1Downloader supports only M1")
        if self.config.price_side not in {"bid", "ask"}:
            raise RuntimeError(f"Unsupported price side: {self.config.price_side}")
        if self.config.max_attempts < 1 or self.config.download_passes < 1:
            raise RuntimeError("Retry counts must be positive")

    def destination_path(self, metal: MetalSpec, start: date, end: date) -> Path:
        return (
            self.raw_root
            / metal.key
            / f"{metal.downloader_instrument}_{start}_{end}_{self.timeframe}.csv"
        )

    def url_for_day(self, metal: MetalSpec, day: date) -> str:
        # Dukascopy's historical datafeed uses zero-based months in URLs.
        month_zero_based = day.month - 1
        filename = f"{self.price_side.upper()}_candles_min_1.bi5"
        return (
            f"{self.config.base_url}/{metal.downloader_instrument.upper()}/"
            f"{day.year:04d}/{month_zero_based:02d}/{day.day:02d}/{filename}"
        )

    def _cache_paths(self, metal: MetalSpec, day: date) -> tuple[Path, Path]:
        day_root = (
            self.raw_root
            / metal.key
            / "bi5"
            / f"{day.year:04d}"
            / f"{day.month:02d}"
        )
        payload = day_root / f"{day.day:02d}_{self.price_side}.bi5"
        empty_marker = day_root / f"{day.day:02d}_{self.price_side}.empty"
        return payload, empty_marker

    def _is_cached(self, metal: MetalSpec, day: date) -> bool:
        payload_path, empty_marker = self._cache_paths(metal, day)
        return (
            payload_path.exists() and payload_path.stat().st_size > 0
        ) or empty_marker.exists()

    @staticmethod
    def _iter_days(start: date, end: date):
        cursor = start
        while cursor < end:
            yield cursor
            cursor += timedelta(days=1)

    @staticmethod
    def _retry_after_seconds(exc: HTTPError) -> float | None:
        value = exc.headers.get("Retry-After") if exc.headers else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    def _fetch_day_payload(self, metal: MetalSpec, day: date) -> bytes | None:
        payload_path, empty_marker = self._cache_paths(metal, day)
        if payload_path.exists() and payload_path.stat().st_size > 0:
            return payload_path.read_bytes()
        if empty_marker.exists():
            return None

        payload_path.parent.mkdir(parents=True, exist_ok=True)
        url = self.url_for_day(metal, day)

        for attempt in range(1, self.config.max_attempts + 1):
            request = Request(url, headers={"User-Agent": self.config.user_agent})
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    status = getattr(response, "status", 200)
                    payload = response.read()
                if status != 200:
                    raise RuntimeError(f"Unexpected HTTP {status} for {url}")
                if not payload:
                    empty_marker.touch()
                    return None

                part = payload_path.with_suffix(payload_path.suffix + ".part")
                part.write_bytes(payload)
                part.replace(payload_path)
                return payload
            except HTTPError as exc:
                if exc.code == 404:
                    empty_marker.touch()
                    return None

                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.config.max_attempts:
                    raise RuntimeError(
                        f"HTTP {exc.code} downloading {metal.name} M1 for {day}: {url}"
                    ) from exc

                retry_after = self._retry_after_seconds(exc)
                delay = retry_after or self.config.base_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[HTTP {exc.code}] {metal.name:10s} {day} "
                    f"retry={attempt}/{self.config.max_attempts} sleep={delay:.1f}s"
                )
                time.sleep(delay)
            except (URLError, TimeoutError, OSError) as exc:
                if attempt >= self.config.max_attempts:
                    raise RuntimeError(
                        f"Network failure downloading {metal.name} M1 for {day}: {url}"
                    ) from exc
                delay = self.config.base_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[NET ] {metal.name:10s} {day} "
                    f"retry={attempt}/{self.config.max_attempts} sleep={delay:.1f}s"
                )
                time.sleep(delay)

        raise RuntimeError(f"Unable to download {metal.name} M1 for {day}")

    def _populate_cache(self, metal: MetalSpec, days: list[date]) -> None:
        pending = [day for day in days if not self._is_cached(metal, day)]
        if not pending:
            return

        for pass_number in range(1, self.config.download_passes + 1):
            failures: list[tuple[date, str]] = []
            total = len(pending)
            print(
                f"[PASS] {metal.name:10s} {pass_number}/{self.config.download_passes} "
                f"pending_days={total:,}"
            )

            for position, day in enumerate(pending, start=1):
                was_cached = self._is_cached(metal, day)
                try:
                    self._fetch_day_payload(metal, day)
                except RuntimeError as exc:
                    failures.append((day, str(exc)))

                if position % 25 == 0 or position == total:
                    print(
                        f"[FETCH] {metal.name:10s} pass={pass_number} "
                        f"progress={position:,}/{total:,} failures={len(failures):,}"
                    )
                if not was_cached and self.config.request_pause_seconds > 0:
                    time.sleep(self.config.request_pause_seconds)

            if not failures:
                return

            if pass_number >= self.config.download_passes:
                sample = "; ".join(
                    f"{day}: {message}" for day, message in failures[:5]
                )
                raise RuntimeError(
                    f"Unresolved Dukascopy days for {metal.name}: {len(failures)} "
                    f"after {self.config.download_passes} passes. Sample: {sample}"
                )

            pending = [day for day, _ in failures]
            delay = self.config.pass_backoff_seconds * pass_number
            print(
                f"[COOL] {metal.name:10s} unresolved={len(pending):,}; "
                f"sleeping {delay:.1f}s before retry pass"
            )
            time.sleep(delay)

    def decode_day(
        self,
        payload: bytes,
        day: date,
        decimal_factor: int,
    ) -> list[tuple[object, ...]]:
        """Decode one LZMA-compressed daily M1 BI5 payload."""
        if decimal_factor <= 0:
            raise ValueError("decimal_factor must be positive")
        try:
            decompressed = lzma.decompress(payload)
        except lzma.LZMAError as exc:
            raise ValueError(f"Invalid Dukascopy LZMA payload for {day}") from exc

        if len(decompressed) % _M1_STRUCT.size != 0:
            raise ValueError(
                f"Invalid Dukascopy M1 payload length for {day}: "
                f"{len(decompressed)} is not divisible by {_M1_STRUCT.size}"
            )

        day_start_ms = int(
            datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000
        )
        factor = float(decimal_factor)
        rows: list[tuple[object, ...]] = []

        for seconds, open_raw, close_raw, low_raw, high_raw, volume_raw in _M1_STRUCT.iter_unpack(
            decompressed
        ):
            if not 0 <= seconds < _SECONDS_PER_DAY:
                raise ValueError(f"Invalid second offset {seconds} in Dukascopy M1 data for {day}")

            open_price = open_raw / factor
            high_price = high_raw / factor
            low_price = low_raw / factor
            close_price = close_raw / factor
            volume_units = float(volume_raw) * 1_000_000.0

            values = (open_price, high_price, low_price, close_price, volume_units)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite Dukascopy M1 value for {day} at {seconds}s")
            if low_price > high_price or not (
                low_price <= open_price <= high_price
                and low_price <= close_price <= high_price
            ):
                raise ValueError(f"Invalid Dukascopy OHLC relationship for {day} at {seconds}s")

            rows.append(
                (
                    day_start_ms + seconds * 1000,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume_units,
                )
            )

        return rows

    def download(self, metal: MetalSpec, start: date, end: date) -> Path:
        destination = self.destination_path(metal, start, end)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 100:
            print(f"[SKIP] {metal.name:10s} {start} -> {end}")
            return destination

        days = list(self._iter_days(start, end))
        self._populate_cache(metal, days)

        part = destination.with_suffix(destination.suffix + ".part")
        rows_written = 0
        days_with_data = 0

        with part.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

            for day in days:
                payload = self._fetch_day_payload(metal, day)
                if payload is None:
                    continue

                rows = self.decode_day(payload, day, metal.dukascopy_decimal_factor)
                for row in rows:
                    writer.writerow(row)
                rows_written += len(rows)
                if rows:
                    days_with_data += 1
                    if days_with_data % 25 == 0:
                        print(
                            f"[BI5 ] {metal.name:10s} data_days={days_with_data:,} "
                            f"rows={rows_written:,} through={day}"
                        )

        if rows_written == 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"No Dukascopy M1 rows produced for {metal.name} {start} -> {end}")

        part.replace(destination)
        print(
            f"[DONE] {metal.name:10s} {start} -> {end} "
            f"rows={rows_written:,} data_days={days_with_data:,}"
        )
        return destination
