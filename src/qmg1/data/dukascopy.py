from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .metals import MetalSpec


@dataclass(frozen=True)
class DukascopyConfig:
    package_version: str = "1.50.0"
    timeframe: str = "m1"
    price_type: str = "bid"
    batch_size: int = 2
    batch_pause_ms: int = 3000
    retry_count: int = 8
    retry_pause_ms: int = 5000
    process_attempts: int = 3
    process_backoff_seconds: int = 15


class DukascopyDownloader:
    """Single-responsibility adapter around the dukascopy-node CLI."""

    def __init__(
        self,
        raw_root: Path,
        incoming_root: Path,
        config: DukascopyConfig | None = None,
    ) -> None:
        self.raw_root = raw_root
        self.incoming_root = incoming_root
        self.config = config or DukascopyConfig()

    def validate_runtime(self) -> None:
        for binary in ("node", "npx"):
            if shutil.which(binary) is None:
                raise RuntimeError(f"{binary} is required")

        version = subprocess.run(
            ["node", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lstrip("v")
        if int(version.split(".", 1)[0]) < 18:
            raise RuntimeError(f"Node.js 18+ is required; found {version}")

    def destination_path(self, metal: MetalSpec, start: date, end: date) -> Path:
        return (
            self.raw_root
            / metal.key
            / f"{metal.downloader_instrument}_{start}_{end}_{self.config.timeframe}.csv"
        )

    def _command(
        self,
        metal: MetalSpec,
        start: date,
        end: date,
        incoming: Path,
        file_stem: str,
    ) -> list[str]:
        return [
            "npx",
            "--yes",
            f"dukascopy-node@{self.config.package_version}",
            "-i",
            metal.downloader_instrument,
            "-from",
            start.isoformat(),
            "-to",
            end.isoformat(),
            "-t",
            self.config.timeframe,
            "-p",
            self.config.price_type,
            "-v",
            "-vu",
            "units",
            "-f",
            "csv",
            "-dir",
            str(incoming.resolve()),
            "-fn",
            file_stem,
            "--batch-size",
            str(self.config.batch_size),
            "--batch-pause",
            str(self.config.batch_pause_ms),
            "-r",
            str(self.config.retry_count),
            "-rp",
            str(self.config.retry_pause_ms),
            "-s",
        ]

    def _run_with_backoff(self, command: list[str], metal: MetalSpec) -> None:
        for attempt in range(1, self.config.process_attempts + 1):
            try:
                subprocess.run(command, check=True)
                return
            except subprocess.CalledProcessError as exc:
                if attempt >= self.config.process_attempts:
                    raise RuntimeError(
                        f"Dukascopy download failed after {attempt} attempts for {metal.name}"
                    ) from exc

                delay = self.config.process_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[RETRY] {metal.name:10s} process attempt "
                    f"{attempt}/{self.config.process_attempts} failed; sleeping {delay}s"
                )
                time.sleep(delay)

    def download(self, metal: MetalSpec, start: date, end: date) -> Path:
        destination = self.destination_path(metal, start, end)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and destination.stat().st_size > 100:
            print(f"[SKIP] {metal.name:10s} {start} -> {end}")
            return destination

        incoming = self.incoming_root / metal.key / f"{start}_{end}"
        shutil.rmtree(incoming, ignore_errors=True)
        incoming.mkdir(parents=True, exist_ok=True)

        file_stem = f"{metal.downloader_instrument}_{start}_{end}_{self.config.timeframe}"
        command = self._command(metal, start, end, incoming, file_stem)

        print(
            f"[GET ] {metal.name:10s} {start} -> {end} "
            f"batch={self.config.batch_size} pause={self.config.batch_pause_ms}ms"
        )
        self._run_with_backoff(command, metal)

        expected = incoming / f"{file_stem}.csv"
        candidates = [expected] if expected.exists() else list(incoming.rglob("*.csv"))
        candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
        if not candidates:
            raise RuntimeError(f"No CSV produced for {metal.name} {start} -> {end}")

        source = max(candidates, key=lambda p: p.stat().st_size)
        shutil.move(str(source), destination)
        shutil.rmtree(incoming, ignore_errors=True)
        return destination
