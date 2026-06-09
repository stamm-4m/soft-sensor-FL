"""Stub implementation for federated aggregation logic."""

from __future__ import annotations

from typing import Any


class FederatedAggregator:
    def aggregate(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": "stub",
            "message": (
                "Implement server-side aggregation "
                "(for example FedAvg)"
            ),
            "num_updates": len(updates),
        }
