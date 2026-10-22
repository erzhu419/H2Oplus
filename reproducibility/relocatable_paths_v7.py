"""Explicit path remapping for hash-locked v7 evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactPathRemap:
    """Map paths below one absolute workspace root into another root."""

    source_root: Path
    target_root: Path

    @classmethod
    def build(
        cls, source_root: Path | None, target_root: Path | None
    ) -> "ArtifactPathRemap | None":
        if source_root is None and target_root is None:
            return None
        if source_root is None or target_root is None:
            raise ValueError("path remapping requires both source and target roots")
        source = source_root.expanduser()
        target = target_root.expanduser()
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError("path remapping roots must be absolute")
        source = source.resolve()
        target = target.resolve(strict=True)
        if source == target:
            raise ValueError("path remapping roots must differ")
        return cls(source_root=source, target_root=target)

    def resolve(self, raw_path: Any) -> Path:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            raise ValueError(f"evidence path is not absolute: {raw_path!r}")
        normalized = path.resolve()
        try:
            relative = normalized.relative_to(self.source_root)
        except ValueError:
            return normalized
        return (self.target_root / relative).resolve()


def resolve_evidence_path(
    raw_path: Any, remap: ArtifactPathRemap | None
) -> Path:
    path = Path(str(raw_path)).expanduser()
    if remap is not None:
        return remap.resolve(path)
    if not path.is_absolute():
        raise ValueError(f"evidence path is not absolute: {raw_path!r}")
    return path.resolve()
