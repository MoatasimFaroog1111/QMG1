from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from qmg1.data.histdata import HistDataM1Downloader
from qmg1.data.hourly import HOURLY_COLUMNS, yearly_chunks
from qmg1.data.metals import REQUESTED_END_EXCLUSIVE


@dataclass(frozen=True)
class MarketSeriesSpec:
    key: str
    name: str
    histdata_pair: str
    output_symbol: str
    effective_start: date
    unit: str


US_DOLLAR_INDEX = MarketSeriesSpec(
    key="udx",
    name="US Dollar Index",
    histdata_pair="UDXUSD",
    output_symbol="UDXUSD",
    effective_start=date(2009, 5, 1),
    unit="index_points",
)

MARKET_SERIES: dict[str, MarketSeriesSpec] = {
    US_DOLLAR_INDEX.key: US_DOLLAR_INDEX,
}


class HistDataMarketSeriesDownloader(HistDataM1Downloader):
    """Reuse HistData transport/parsing for non-metal native market series."""

    def destination_for_pair(
        self,
        series: MarketSeriesSpec,
        start: date,
        end: date,
    ) -> Path:
        return (
            self.raw_root
            / "market_series"
            / series.key
            / f"histdata_{series.histdata_pair.lower()}_{start}_{end}_m1.csv"
        )

    def download_series(
        self,
        series: MarketSeriesSpec,
        start: date,
        end: date,
    ) -> Path:
        self.validate_runtime()
        destination = self.destination_for_pair(series, start, end)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 100:
            print(f"[SKIP] HistData {series.name:16s} {start} -> {end}")
            return destination

        periods = self._periods(series.histdata_pair, start, end)
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
            raise RuntimeError(
                f"No HistData rows produced for {series.name} {start} -> {end}"
            )
        part.replace(destination)
        print(
            f"[DONE] HistData {series.name:16s} rows={rows_written:,} "
            f"{start} -> {end}"
        )
        return destination


class HourlyNativeSeriesBuilder:
    """Compact provider-native M1 OHLC into H1 without unit conversion."""

    @staticmethod
    def build(raw_csv: Path) -> pd.DataFrame:
        frame = pd.read_csv(raw_csv)
        required = {"timestamp", "open", "high", "low", "close"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing M1 columns in {raw_csv}: {missing}")

        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
        numeric = timestamps.dropna()
        if numeric.empty:
            raise ValueError(f"No valid timestamps in {raw_csv}")
        unit = "s" if float(numeric.median()) < 10_000_000_000 else "ms"
        frame["timestamp_utc"] = pd.to_datetime(
            timestamps,
            unit=unit,
            utc=True,
            errors="coerce",
        )

        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(
                frame["volume"], errors="coerce"
            ).fillna(0.0)
        else:
            frame["volume"] = 0.0

        frame = frame.dropna(
            subset=["timestamp_utc", "open", "high", "low", "close"]
        )
        frame = frame.set_index("timestamp_utc").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]

        hourly = pd.DataFrame(
            {
                "open": frame["open"].resample("1h").first(),
                "high": frame["high"].resample("1h").max(),
                "low": frame["low"].resample("1h").min(),
                "close": frame["close"].resample("1h").last(),
                "volume": frame["volume"].resample("1h").sum(min_count=1),
                "minute_count": frame["close"].resample("1h").count(),
            }
        )
        return hourly.dropna(subset=["open", "high", "low", "close"])


@dataclass(frozen=True)
class MarketSeriesResult:
    series: str
    source: str
    unit: str
    output_file: Path
    rows: int
    start_utc: str
    end_utc: str


class MarketSeriesHourlyPipeline:
    """Download native market series in yearly chunks and compact to H1."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw_root = root / "raw"
        self.final_root = root / "hourly"
        self.downloader = HistDataMarketSeriesDownloader(raw_root=self.raw_root)
        self.builder = HourlyNativeSeriesBuilder()

    @staticmethod
    def actual_end_exclusive() -> date:
        return min(REQUESTED_END_EXCLUSIVE, datetime.now(timezone.utc).date())

    def run(self, series_key: str, cleanup_raw: bool = False) -> MarketSeriesResult:
        try:
            series = MARKET_SERIES[series_key]
        except KeyError as exc:
            raise ValueError(f"Unsupported market series: {series_key}") from exc

        end_exclusive = self.actual_end_exclusive()
        if end_exclusive <= series.effective_start:
            raise RuntimeError(f"No completed data range for {series.name}")

        chunks: list[pd.DataFrame] = []
        for start, stop in yearly_chunks(series.effective_start, end_exclusive):
            raw = self.downloader.download_series(series, start, stop)
            print(f"[H1  ] {series.name:16s} {start} -> {stop}")
            hourly = self.builder.build(raw)
            if not hourly.empty:
                chunks.append(hourly)
            if cleanup_raw:
                raw.unlink(missing_ok=True)

        if not chunks:
            raise RuntimeError(f"No hourly data produced for {series.name}")

        merged = pd.concat(chunks).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.dropna(subset=HOURLY_COLUMNS[:4])

        self.final_root.mkdir(parents=True, exist_ok=True)
        end_inclusive = end_exclusive - timedelta(days=1)
        output = self.final_root / (
            f"{series.output_symbol}_H1_NATIVE_"
            f"{series.effective_start.isoformat()}_to_{end_inclusive.isoformat()}.csv"
        )
        merged.to_csv(output, index_label="timestamp_utc")

        return MarketSeriesResult(
            series=series.key,
            source="HistData",
            unit=series.unit,
            output_file=output,
            rows=len(merged),
            start_utc=merged.index[0].isoformat(),
            end_utc=merged.index[-1].isoformat(),
        )
