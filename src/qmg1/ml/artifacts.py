from __future__ import annotations

import json
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

    def save(self, metal: str, horizon_hours: int, artifact: dict[str, Any]) -> Path:
        path = self.path_for(metal, horizon_hours)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        return path

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
            artifact = joblib.load(path)
            if not isinstance(artifact, dict):
                raise ValueError(f"Invalid model artifact: {path}")
            return artifact
        return self._load_serving_manifest(metal, horizon_hours)
