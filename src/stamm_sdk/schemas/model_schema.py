"""Typed schemas for model operations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeatureSpec(BaseModel):
    """One input feature a model expects, as declared in soft_sensors.inputs.

    range_min/range_max come from the matching sensors/actuators catalog
    row (by name) when available — used by data/synthetic.py and
    validation/data_quality.py, not by the registry's own schema."""

    name: str
    type: str = "float"            # "float" | "int" | "category"
    lag: int = 0
    required: bool = True
    range_min: float | None = None
    range_max: float | None = None


class ModelSchema(BaseModel):
    """Normalized shape of a soft_sensors row, for FL + training use.

    Built from GET /{project_id}/metadata/{model_id} (or the soft_sensors
    CRUD row) — model_registry.inputs may be a rich
    {"features": [...]} dict or a plain list of names; get_schemas()
    normalizes either shape into this."""

    model_config = ConfigDict(extra="allow")

    model_id: str
    slug: str | None = None
    algorithm: str | None = None   # "neural_network" | "random_forest" | "linear" | ...
    framework: str | None = None   # "pytorch" | "sklearn" -- derived from algorithm, see load_model()
    features: list[FeatureSpec] = Field(default_factory=list)
    output_name: str | None = None


class PredictionRequest(BaseModel):
    req: dict[str, Any]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
