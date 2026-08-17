from __future__ import annotations

import csv
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

from .metals import MetalSpec


_HISTDATA_TZ = timezone(timedelta(hours=-5), name="EST-fixed")


@dataclass(frozen=True)
class HistDataConfig:
    base_url: str = (
        "https://www.histdata.com/download-free-forex-historical-data/"
        "?/ascii/1-minute-bar-quotes/"
    )
    download_url: str = "https://www.histdata.com/get.php"
    platform: str = "ASCII"
    timeframe: str = "M1"
    price_side: str = "bid"
    timeout_seconds: int = 120
    max_attempts: int = 5
    download_pause_seconds: float = 1.0


@dataclass(frozen=True)
class HistDataPeriod:
    pair: str
    year: int
    month: int | None = None

    @property
    def label(self) -> str:
        if self.month is None:
            return str(self.year)
        return f"{self.year}{self.month:02d}"

    def referer(self, base_url: str) -> str:
        if self.month is None:
            return f"{base_url}{self.pair.lower()}/{self.year}"
        return f"{base_url}{self.pair.lower()}/{self.year}/{self.month}"


class _TokenParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.token: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "input":
            return
        values = {str(key).lower(): value for key, value in attrs}
        if str(values.get("id", "")).lower() != "tk":
            return
        value = values.get("value")
        if value:
            self.token = str(value)


class HistDataM1Downloader:
    """Bulk yearly/monthly HistData adapter for XAUUSD and XAGUSD."""

    _SUPPORTED = {"gold": "XAUUSD", "silver": "XAGUSD"}

    def __init__(self, raw_root: Path, config: HistDataConfig | None = None) -> None:
        self.raw_root = raw_root
        self.config = config or HistDataConfig()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/151.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    @property
    def source_name(self) -> str:
        return "HistData"

    def source_name_for(self, metal: MetalSpec) -> str:
        self._pair_for(metal)
        return self.source_name

    @property
    def provider_description(self) -> str:
        return "HistData Generic ASCII M1 bulk archives"

    @property
    def timeframe(self) -> str:
        return "m1"

    @property
    def price_side(self) -> str:
        return self.config.price_side

    def validate_runtime(self) -> None:
        if self.config.platform != "ASCII" or self.config.timeframe != "M1":
            raise RuntimeError("HistDataM1Downloader requires ASCII M1 archives")

    def _pair_for(self, metal: MetalSpec) -> str:
        try:
            return self._SUPPORTED[metal.key]
        except KeyError as exc:
            raise ValueError(f"HistData provider does not support {metal.name}") from exc

    def destination_path(self, metal: MetalSpec, start: date, end: date) -> Path:
        return (
            self.raw_root
            / metal.key
            / f"histdata_{self._pair_for(metal).lower()}_{start}_{end}_m1.csv"
        )

    def _zip_path(self, period: HistDataPeriod) -> Path:
        return (
            self.raw_root
            / period.pair.lower()
            / "histdata_zip"
            / f"DAT_ASCII_{period.pair}_M1_{period.label}.zip"
        )

    def _periods(self, pair: str, start: date, end: date) -> list[HistDataPeriod]:
        current_year = datetime.now(timezone.utc).year
        periods: list[HistDataPeriod] = []

        for year in range(start.year, end.year + 1):
            year_start = max(start, date(year, 1, 1))
            year_end = min(end, date(year + 1, 1, 1))
            if year_start >= year_end:
                continue

            if year < current_year:
                periods.append(HistDataPeriod(pair=pair, year=year))
                continue

            month = year_start.month
            while date(year, month, 1) < year_end:
                periods.append(HistDataPeriod(pair=pair, year=year, month=month))
                if month == 12:
                    break
                month += 1

        return periods

    def _token(self, period: HistDataPeriod) -> str:
        referer = period.referer(self.config.base_url)
        response = self.session.get(
            referer,
            timeout=self.config.timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
        parser = _TokenParser()
        parser.feed(response.text)
        if not parser.token:
            raise RuntimeError(f"HistData token not found for {period.label}: {referer}")
        return parser.token

    @staticmethod
    def _validate_zip(path: Path) -> None:
        if not path.exists() or path.stat().st_size < 100 or not zipfile.is_zipfile(path):
            raise RuntimeError(f"Invalid HistData ZIP: {path}")

    def _download_period(self, period: HistDataPeriod) -> Path:
        final_path = self._zip_path(period)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            try:
                self._validate_zip(final_path)
                print(f"[SKIP] HistData {period.pair} {period.label}")
                return final_path
            except RuntimeError:
                final_path.unlink(missing_ok=True)

        part = final_path.with_suffix(final_path.suffix + ".part")
        referer = period.referer(self.config.base_url)

        for attempt in range(1, self.config.max_attempts + 1):
            part.unlink(missing_ok=True)
            try:
                payload = {
                    "tk": self._token(period),
                    "date": str(period.year),
                    "datemonth": period.label,
                    "platform": self.config.platform,
                    "timeframe": self.config.timeframe,
                    "fxpair": period.pair,
                }
                headers = {
                    "Referer": referer,
                    "Origin": "https://www.histdata.com",
                    "Content-Type": "application/x-www-form-urlencoded",
                }
                print(
                    f"[GET ] HistData {period.pair} {period.label} "
                    f"attempt={attempt}/{self.config.max_attempts}"
                )
                with self.session.post(
                    self.config.download_url,
                    data=payload,
                    headers=headers,
                    stream=True,
                    timeout=self.config.timeout_seconds,
                    allow_redirects=True,
                ) as response:
                    response.raise_for_status()
                    with part.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)

                self._validate_zip(part)
                part.replace(final_path)
                if self.config.download_pause_seconds > 0:
                    time.sleep(self.config.download_pause_seconds)
                return final_path
            except (requests.RequestException, RuntimeError) as exc:
                part.unlink(missing_ok=True)
                if attempt >= self.config.max_attempts:
                    raise RuntimeError(
                        f"HistData download failed for {period.pair} {period.label}: {exc}"
                    ) from exc
                delay = min(2**attempt, 30)
                print(
                    f"[WARN] HistData {period.pair} {period.label}: {exc}; "
                    f"retry in {delay}s",
                    file=sys.stderr,
                )
                time.sleep(delay)

        raise RuntimeError(f"Unable to download HistData {period.pair} {period.label}")

    @staticmethod
    def _timestamp_utc_ms(value: str) -> int:
        local = datetime.strptime(value.strip(), "%Y%m%d %H%M%S").replace(
            tzinfo=_HISTDATA_TZ
        )
        return int(local.astimezone(timezone.utc).timestamp() * 1000)

    def _write_archive_rows(
        self,
        archive_path: Path,
        writer: csv.writer,
        start_ms: int,
        end_ms: int,
        last_timestamp: int | None,
    ) -> tuple[int, int | None]:
        rows_written = 0
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = sorted(
                (
                    info
                    for info in archive.infolist()
                    if not info.is_dir() and info.filename.lower().endswith(".csv")
                ),
                key=lambda info: info.filename,
            )
            if not members:
                raise RuntimeError(f"No CSV found in HistData archive {archive_path}")

            for info in members:
                with archive.open(info, "r") as source:
                    text = (line.decode("utf-8-sig", errors="replace") for line in source)
                    reader = csv.reader(text, delimiter=";")
                    for row in reader:
                        if len(row) < 6:
                            continue
                        try:
                            timestamp = self._timestamp_utc_ms(row[0])
                            if not start_ms <= timestamp < end_ms:
                                continue
                            if last_timestamp is not None and timestamp <= last_timestamp:
                                continue

                            open_price = float(row[1])
                            high_price = float(row[2])
                            low_price = float(row[3])
                            close_price = float(row[4])
                            volume = float(row[5]) if row[5].strip() else 0.0
                            values = (
                                open_price,
                                high_price,
                                low_price,
                                close_price,
                                volume,
                            )
                            if not all(math.isfinite(value) for value in values):
                                continue
                            if min(open_price, high_price, low_price, close_price) <= 0:
                                continue
                            if volume < 0:
                                continue
                            if low_price > high_price or not (
                                low_price <= open_price <= high_price
                                and low_price <= close_price <= high_price
                            ):
                                continue
                        except (ValueError, OverflowError):
                            continue

                        writer.writerow(
                            [
                                timestamp,
                                open_price,
                                high_price,
                                low_price,
                                close_price,
                                volume,
                            ]
                        )
                        last_timestamp = timestamp
                        rows_written += 1

        return rows_written, last_timestamp

    def download(self, metal: MetalSpec, start: date, end: date) -> Path:
        self.validate_runtime()
        pair = self._pair_for(metal)
        destination = self.destination_path(metal, start, end)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 100:
            print(f"[SKIP] HistData {metal.name:10s} {start} -> {end}")
            return destination

        periods = self._periods(pair, start, end)
        archives = [self._download_period(period) for period in periods]
        start_ms = int(
            datetime.combine(start, datetime.min.time(), timezone.utc).timestamp() * 1000
        )
        end_ms = int(
            datetime.combine(end, datetime.min.time(), timezone.utc).timestamp() * 1000
        )

        part = destination.with_suffix(destination.suffix + ".part")
        rows_written = 0
        last_timestamp: int | None = None
        with part.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for archive in archives:
                count, last_timestamp = self._write_archive_rows(
                    archive,
                    writer,
                    start_ms,
                    end_ms,
                    last_timestamp,
                )
                rows_written += count

        if rows_written == 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"No HistData rows produced for {metal.name} {start} -> {end}")
        part.replace(destination)
        print(f"[DONE] HistData {metal.name:10s} rows={rows_written:,} {start} -> {end}")
        return destination
