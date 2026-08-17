from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class ModelArtifactRepository:
    """Persistence boundary for trained forecasting artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, metal: str, horizon_hours: int) -> Path:
        return self.root / metal / f"{metal}_{horizon_hours}h.joblib"

    def save(self, metal: str, horizon_hours: int, artifact: dict[str, Any]) -> Path:
        path = self.path_for(metal, horizon_hours)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        return path

    def load(self, metal: str, horizon_hours: int) -> dict[str, Any]:
        path = self.path_for(metal, horizon_hours)
        if not path.exists():
            raise FileNotFoundError(f"Persisted model not found: {path}")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            raise ValueError(f"Invalid model artifact: {path}")
        return artifact
