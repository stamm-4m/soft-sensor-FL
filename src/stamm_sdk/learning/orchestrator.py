"""Top-level entry point tying the FL pipeline together for one round.

Gate 1 (raw data validation) and the actual local training
(FederatedClient.train() / extract_update()) happen BEFORE this --
run_federation_round() is the coordinator-side step: it only ever sees
already-trained LocalUpdates, never raw client data, and is responsible
for Gate 2 + aggregation + persisting the round's result back to
model-registry.
"""

from __future__ import annotations

import pandas as pd

from stamm_sdk.learning.federated_aggregator import FederatedAggregator
from stamm_sdk.fl_types import AggregationResult, LocalUpdate
from stamm_sdk.registry.api_client import APIClient


def run_federation_round(api: APIClient, federation_id: str, round_number: int,
                          updates: list[LocalUpdate], history: list[AggregationResult],
                          reference_data: pd.DataFrame | None = None,
                          output_name: str | None = None,
                          global_model_id: str | None = None) -> AggregationResult:
    """Run Gate 2 + aggregation over this round's updates and persist a
    federation_rounds + model_contributions record of the outcome.

    Does NOT create a new soft_sensors row for the resulting global
    model -- registering the aggregated model (new version vs. in-place
    update) is a registry-side decision left to the caller. Pass the
    resulting model's id as global_model_id once it's registered, so the
    persisted round/contribution rows point at it; omit it to persist
    the round "unpublished" (e.g. for inspecting a trial round before
    committing to it).
    """
    result = FederatedAggregator().aggregate(
        updates, history, reference_data=reference_data, output_name=output_name,
    )

    accepted_size = sum(
        u.dataset_size for u in updates if u.participant_id in result.accepted_participants
    ) or 1
    contributions = {
        pid: {
            "weight": next(u.dataset_size for u in updates if u.participant_id == pid) / accepted_size,
            "metrics": {},
        }
        for pid in result.accepted_participants
    }

    api.submit_federation_round(
        federation_id=federation_id, round_number=round_number,
        global_metric_value=result.performance_vs_previous,
        contributions=contributions, global_model_id=global_model_id,
    )
    return result
