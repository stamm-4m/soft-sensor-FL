"""Typed schemas for model operations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    version: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionRequest(BaseModel):
    req: dict[str, Any]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
