"""Stub implementation for a federated learning client node."""

from __future__ import annotations

from typing import Any


class FederatedClient:
    def fit(self, model_state: dict[str, Any], data: Any) -> dict[str, Any]:
        return {
            "status": "stub",
            "message": "Implement local FL client training",
            "model_state": model_state,
            "data_type": type(data).__name__,
        }

    def evaluate(
        self,
        model_state: dict[str, Any],
        data: Any,
    ) -> dict[str, Any]:
        return {
            "status": "stub",
            "message": "Implement local FL client evaluation",
            "model_state": model_state,
            "data_type": type(data).__name__,
        }
