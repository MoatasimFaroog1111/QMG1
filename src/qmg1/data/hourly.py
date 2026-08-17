from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from qmg1.data.hybrid import HybridPreciousMetalsM1Provider
from qmg1.data.metals import METALS, REQUESTED_END_EXCLUSIVE, MetalSpec
from qmg1.data.normalizer import TROY_OUNCES_PER_KG


HOURLY_COLUMNS = ["open", "high", "low", "close", "volume", "minute_count"]


def yearly_chunks(start: date, end_exclusive: date):
    cursor = start
    while cursor < end_exclusive:
        next_year = date(cursor.year + 1, 1, 1)
        stop = min(next_year, end_exclusive)
        yield cursor, stop
        cursor = stop


def metal_by_key(key: str) -> MetalSpec:
    for metal in METALS:
        if metal.key == key:
            return metal
    raise ValueError(f"Unsupported metal: {key}")


class HourlyUsdPerKgBuilder:
    """Convert one provider-native M1 chunk to compact H1 USD/kg bars."""

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

        factor = float(TROY_OUNCES_PER_KG)
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor

        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        else:
            frame["volume"] = 0.0

        frame = frame.dropna(subset=["timestamp_utc", "open", "high", "low", "close"])
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
class HourlyTrainingDataResult:
    metal: str
    source: str
    output_file: Path
    rows: int
    start_utc: str
    end_utc: str


class HourlyTrainingDataPipeline:
    """Download yearly M1 chunks, compact immediately to H1, then merge."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw_root = root / "raw"
        self.final_root = root / "hourly"
        self.provider = HybridPreciousMetalsM1Provider(raw_root=self.raw_root)
        self.builder = HourlyUsdPerKgBuilder()

    @staticmethod
    def actual_end_exclusive() -> date:
        return min(REQUESTED_END_EXCLUSIVE, datetime.now(timezone.utc).date())

    def run(self, metal_key: str, cleanup_raw: bool = False) -> HourlyTrainingDataResult:
        metal = metal_by_key(metal_key)
        self.provider.validate_runtime()
        end_exclusive = self.actual_end_exclusive()
        if end_exclusive <= metal.effective_start:
            raise RuntimeError(f"No completed data range for {metal.name}")

        chunks: list[pd.DataFrame] = []
        for start, stop in yearly_chunks(metal.effective_start, end_exclusive):
            raw = self.provider.download(metal, start, stop)
            print(f"[H1  ] {metal.name:10s} {start} -> {stop}")
            hourly = self.builder.build(raw)
            if not hourly.empty:
                chunks.append(hourly)
            if cleanup_raw:
                raw.unlink(missing_ok=True)

        if not chunks:
            raise RuntimeError(f"No hourly data produced for {metal.name}")

        merged = pd.concat(chunks).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.dropna(subset=["open", "high", "low", "close"])

        self.final_root.mkdir(parents=True, exist_ok=True)
        end_inclusive = end_exclusive - timedelta(days=1)
        output = self.final_root / (
            f"{metal.output_symbol}_H1_USD_PER_KG_"
            f"{metal.effective_start.isoformat()}_to_{end_inclusive.isoformat()}.csv"
        )
        merged.to_csv(output, index_label="timestamp_utc")

        return HourlyTrainingDataResult(
            metal=metal.key,
            source=self.provider.source_name_for(metal),
            output_file=output,
            rows=len(merged),
            start_utc=merged.index[0].isoformat(),
            end_utc=merged.index[-1].isoformat(),
        )
