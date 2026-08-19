from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import joblib


class ModelArtifactRepository:
    """Persistence boundary for trained artifacts and compact serving champions."""

    TRAINED_BUNDLE_GLOB = "trained_models_*.zip"
    ENCODED_BUNDLE_FIRST_PART_GLOB = "trained_models_*.zip.b64.part000"

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, metal: str, horizon_hours: int) -> Path:
        return self.root / metal / f"{metal}_{horizon_hours}h.joblib"

    def manifest_path_for(self, metal: str) -> Path:
        return self.root / metal / "champions.json"

    def bundle_paths_for(self, metal: str) -> list[Path]:
        directory = self.root / metal
        if not directory.is_dir():
            return []
        return sorted(
            directory.glob(self.TRAINED_BUNDLE_GLOB),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )

    def encoded_bundle_first_parts_for(self, metal: str) -> list[Path]:
        directory = self.root / metal
        if not directory.is_dir():
            return []
        return sorted(
            directory.glob(self.ENCODED_BUNDLE_FIRST_PART_GLOB),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )

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

    @staticmethod
    def _expected_checksum(checksum_path: Path) -> str:
        if not checksum_path.exists():
            raise ValueError(f"Missing artifact checksum: {checksum_path}")
        fields = checksum_path.read_text(encoding="utf-8").split()
        if not fields:
            raise ValueError(f"Invalid artifact checksum: {checksum_path}")
        return fields[0]

    def _verify_checksum(self, path: Path) -> None:
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        expected = self._expected_checksum(checksum_path)
        actual = self._digest(path)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"Artifact checksum mismatch: {path}")

    @classmethod
    def _verify_bytes_checksum(
        cls,
        data: bytes,
        checksum_path: Path,
        source_name: str,
    ) -> None:
        expected = cls._expected_checksum(checksum_path)
        actual = hashlib.sha256(data).hexdigest()
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"Artifact checksum mismatch: {source_name}")

    @staticmethod
    def _validate_artifact(artifact: object, source: object) -> dict[str, Any]:
        if not isinstance(artifact, dict):
            raise ValueError(f"Invalid model artifact: {source}")
        return artifact

    def _load_joblib_path(self, path: Path) -> dict[str, Any]:
        self._verify_checksum(path)
        return self._validate_artifact(joblib.load(path), path)

    @staticmethod
    def _archive_horizons(metal: str, archive: zipfile.ZipFile) -> set[int]:
        prefix = f"{metal}_"
        suffix = "h.joblib"
        horizons: set[int] = set()
        for member in archive.namelist():
            name = Path(member).name
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            encoded = name[len(prefix) : -len(suffix)]
            try:
                horizons.add(int(encoded))
            except ValueError:
                continue
        return horizons

    def _bundle_horizons(self, metal: str, path: Path) -> set[int]:
        self._verify_checksum(path)
        with zipfile.ZipFile(path) as archive:
            return self._archive_horizons(metal, archive)

    def _read_encoded_bundle(self, first_part: Path) -> tuple[str, bytes]:
        marker = ".b64.part"
        if marker not in first_part.name:
            raise ValueError(f"Invalid encoded bundle part name: {first_part}")
        bundle_name = first_part.name.rsplit(marker, 1)[0]
        parts = sorted(first_part.parent.glob(f"{bundle_name}.b64.part*"))
        if not parts:
            raise ValueError(f"Encoded bundle has no parts: {first_part}")

        expected_names = [
            f"{bundle_name}.b64.part{index:03d}" for index in range(len(parts))
        ]
        actual_names = [part.name for part in parts]
        if actual_names != expected_names:
            raise ValueError(
                f"Encoded bundle parts are incomplete or out of sequence: {bundle_name}"
            )

        encoded = "".join(
            part.read_text(encoding="ascii").strip() for part in parts
        )
        try:
            bundle_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 serving bundle: {bundle_name}") from exc

        checksum_path = first_part.parent / f"{bundle_name}.sha256"
        self._verify_bytes_checksum(
            bundle_bytes,
            checksum_path,
            f"{first_part.parent}/{bundle_name}",
        )
        return bundle_name, bundle_bytes

    def _encoded_bundle_horizons(self, metal: str, first_part: Path) -> set[int]:
        _, bundle_bytes = self._read_encoded_bundle(first_part)
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            return self._archive_horizons(metal, archive)

    @staticmethod
    def _load_member_from_archive(
        archive: zipfile.ZipFile,
        member: str,
        source: object,
    ) -> dict[str, Any] | None:
        names = {Path(name).name: name for name in archive.namelist()}
        archive_member = names.get(member)
        if archive_member is None:
            return None
        artifact = joblib.load(io.BytesIO(archive.read(archive_member)))
        return ModelArtifactRepository._validate_artifact(
            artifact,
            f"{source}!/{archive_member}",
        )

    def _load_from_bundle(self, metal: str, horizon_hours: int) -> dict[str, Any]:
        member = f"{metal}_{horizon_hours}h.joblib"
        raw_bundles = self.bundle_paths_for(metal)
        encoded_bundles = self.encoded_bundle_first_parts_for(metal)

        for bundle in raw_bundles:
            self._verify_checksum(bundle)
            with zipfile.ZipFile(bundle) as archive:
                artifact = self._load_member_from_archive(archive, member, bundle)
                if artifact is not None:
                    return artifact

        for first_part in encoded_bundles:
            bundle_name, bundle_bytes = self._read_encoded_bundle(first_part)
            with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
                artifact = self._load_member_from_archive(
                    archive,
                    member,
                    f"{first_part.parent}/{bundle_name}",
                )
                if artifact is not None:
                    return artifact

        if not raw_bundles and not encoded_bundles:
            raise FileNotFoundError(
                f"Persisted trained model not found: {self.path_for(metal, horizon_hours)}"
            )
        raise FileNotFoundError(
            f"Persisted trained model not found for {metal} {horizon_hours}h in serving bundles"
        )

    def load_trained(self, metal: str, horizon_hours: int) -> dict[str, Any]:
        """Load a complete previously trained artifact; never fall back to metadata-only serving."""

        path = self.path_for(metal, horizon_hours)
        if path.exists():
            return self._load_joblib_path(path)
        return self._load_from_bundle(metal, horizon_hours)

    def has_any(self) -> bool:
        if not self.root.is_dir():
            return False
        return (
            any(self.root.glob("**/*.joblib"))
            or any(self.root.glob(f"**/{self.TRAINED_BUNDLE_GLOB}"))
            or any(self.root.glob(f"**/{self.ENCODED_BUNDLE_FIRST_PART_GLOB}"))
            or any(self.root.glob("**/champions.json"))
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
        try:
            return self.load_trained(metal, horizon_hours)
        except FileNotFoundError:
            return self._load_serving_manifest(metal, horizon_hours)

    def available_trained(self) -> dict[str, list[int]]:
        """Return only checksum-valid complete trained artifacts available for inference."""

        available: dict[str, list[int]] = {}
        if not self.root.is_dir():
            return available

        for metal_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            horizons: set[int] = set()
            for path in metal_dir.glob(f"{metal_dir.name}_*h.joblib"):
                try:
                    self._verify_checksum(path)
                    horizons.add(int(path.stem.rsplit("_", 1)[1][:-1]))
                except (OSError, ValueError):
                    continue

            for bundle in self.bundle_paths_for(metal_dir.name):
                try:
                    horizons.update(self._bundle_horizons(metal_dir.name, bundle))
                except (OSError, ValueError, zipfile.BadZipFile):
                    continue

            for first_part in self.encoded_bundle_first_parts_for(metal_dir.name):
                try:
                    horizons.update(
                        self._encoded_bundle_horizons(metal_dir.name, first_part)
                    )
                except (
                    OSError,
                    ValueError,
                    binascii.Error,
                    zipfile.BadZipFile,
                ):
                    continue

            if horizons:
                available[metal_dir.name] = sorted(horizons)
        return available

    def available(self) -> dict[str, list[int]]:
        available = {
            metal: list(horizons)
            for metal, horizons in self.available_trained().items()
        }
        if not self.root.is_dir():
            return available

        for metal_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            horizons = set(available.get(metal_dir.name, []))
            manifest = metal_dir / "champions.json"
            if manifest.exists():
                try:
                    self._verify_checksum(manifest)
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    horizons.update(int(value) for value in payload.get("champions", {}))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            if horizons:
                available[metal_dir.name] = sorted(horizons)
        return available
