"""Typed schemas for federated learning metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FederationParticipant(BaseModel):
    node_id: str
    role: str = "client"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FederationSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    federation_id: str
    strategy: str = "fedavg"
    round_number: int = 0
    participants: list[FederationParticipant] = Field(default_factory=list)
