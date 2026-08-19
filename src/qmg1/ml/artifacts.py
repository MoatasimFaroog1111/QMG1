from __future__ import annotations

import json
import hashlib
import hmac
import os
from pathlib import Path
from typing import Any

import joblib


class ModelArtifactRepository:
    """Persistence boundary for trained artifacts and compact serving champions."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, metal: str, horizon_hours: int) -> Path:
        return self.root / metal / f"{metal}_{horizon_hours}h.joblib"

    def manifest_path_for(self, metal: str) -> Path:
        return self.root / metal / "champions.json"

    def save_manifest(self, metal: str, payload: dict[str, Any]) -> Path:
        path = self.manifest_path_for(metal)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        self._write_checksum(path)
        return path

    def save(self, metal: str, horizon_hours: int, artifact: dict[str, Any]) -> Path:
        path = self.path_for(metal, horizon_hours)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(artifact, temporary)
        os.replace(temporary, path)
        self._write_checksum(path)
        return path

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_checksum(self, path: Path) -> None:
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        temporary = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
        temporary.write_text(f"{self._digest(path)}  {path.name}\n", encoding="utf-8")
        os.replace(temporary, checksum_path)

    def _verify_checksum(self, path: Path) -> None:
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if not checksum_path.exists():
            raise ValueError(f"Missing artifact checksum: {checksum_path}")
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = self._digest(path)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"Artifact checksum mismatch: {path}")

    def has_any(self) -> bool:
        if not self.root.is_dir():
            return False
        return any(self.root.glob("**/*.joblib")) or any(
            self.root.glob("**/champions.json")
        )

    def _load_serving_manifest(self, metal: str, horizon_hours: int) -> dict[str, Any]:
        path = self.manifest_path_for(metal)
        if not path.exists():
            raise FileNotFoundError(
                f"Persisted model not found: {self.path_for(metal, horizon_hours)}"
            )

        self._verify_checksum(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        champions = payload.get("champions")
        if not isinstance(champions, dict):
            raise ValueError(f"Invalid serving champion manifest: {path}")

        artifact = champions.get(str(horizon_hours))
        if not isinstance(artifact, dict):
            raise FileNotFoundError(
                f"Persisted champion not found for {metal} {horizon_hours}h: {path}"
            )
        return artifact

    def load(self, metal: str, horizon_hours: int) -> dict[str, Any]:
        path = self.path_for(metal, horizon_hours)
        if path.exists():
            self._verify_checksum(path)
            artifact = joblib.load(path)
            if not isinstance(artifact, dict):
                raise ValueError(f"Invalid model artifact: {path}")
            return artifact
        return self._load_serving_manifest(metal, horizon_hours)

    def available(self) -> dict[str, list[int]]:
        available: dict[str, list[int]] = {}
        if not self.root.is_dir():
            return available
        for metal_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            horizons: set[int] = set()
            manifest = metal_dir / "champions.json"
            if manifest.exists():
                try:
                    self._verify_checksum(manifest)
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    horizons.update(int(value) for value in payload.get("champions", {}))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            for path in metal_dir.glob(f"{metal_dir.name}_*h.joblib"):
                try:
                    self._verify_checksum(path)
                    horizons.add(int(path.stem.rsplit("_", 1)[1][:-1]))
                except (OSError, ValueError):
                    continue
            if horizons:
                available[metal_dir.name] = sorted(horizons)
        return available
