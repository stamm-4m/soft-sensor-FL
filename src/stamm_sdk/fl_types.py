"""Shared data contracts that flow through the FL pipeline.

Pulled out of federated_client.py / federated_aggregator.py into their own
module to avoid a circular import: validation/weight_validation.py and
validation/tree_validation.py both need to type-hint LocalUpdate and
AggregationResult, while federated_aggregator.py's aggregate() calls INTO
the validation module -- so these shapes can't live inside the classes
that call validation, or the classes validation would need to call back
into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from stamm_sdk.schemas.model_schema import ModelSchema


@dataclass(slots=True)
class LocalUpdate:
    participant_id: str
    round_number: int
    dataset_size: int
    kind: Literal["gradient", "trees"]
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AggregationResult:
    kind: Literal["gradient", "trees"]
    global_payload: Any
    cosine_similarities: dict[str, float] | None = None
    accepted_participants: list[str] = field(default_factory=list)
    rejected_participants: dict[str, str] = field(default_factory=dict)
    performance_vs_previous: float | None = None
    direction_ok: bool = True


@dataclass(slots=True)
class LoadedModel:
    model_id: str
    project_id: str
    schema: ModelSchema
    framework: str                        # "pytorch" | "sklearn"
    native_model: Any                     # torch.nn.Module instance, or sklearn estimator
    initial_payload: dict[str, Any]       # flattened weights/coefs at load time -- extract_update()'s delta baseline
