"""Artifact bundle definition used for model packaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactBundle:
    name: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        name: str,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactBundle":
        root = Path(path)
        files = [p for p in root.rglob("*") if p.is_file()]
        return cls(
            name=name,
            version=version,
            metadata=metadata or {},
            files=files,
        )
