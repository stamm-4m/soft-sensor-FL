"""Simple experiment logger for local and optional remote tracking."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stamm_sdk.core.session import Session


class ExperimentLogger:
    def __init__(
        self,
        session: Session | None = None,
        log_path: str | Path = ".stamm/experiments.jsonl",
    ) -> None:
        self.session = session
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def push_remote(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.session:
            raise RuntimeError("ExperimentLogger.session is not configured")
        return self.session.request(
            "POST",
            "/api/v1/experiments/",
            json=payload,
        )
