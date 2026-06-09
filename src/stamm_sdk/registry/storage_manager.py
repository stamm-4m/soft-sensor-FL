"""Simple local storage utilities for model artifacts and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StorageManager:
    def __init__(self, root_dir: str | Path = ".stamm") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, relative_path: str, payload: dict[str, Any]) -> Path:
        target = self.root_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def load_json(self, relative_path: str) -> dict[str, Any]:
        target = self.root_dir / relative_path
        return json.loads(target.read_text(encoding="utf-8"))
