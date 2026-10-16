"""Aggregate one round's validated LocalUpdates into a new global model.

Dispatches on LocalUpdate.kind, since the aggregation math (and the Gate
2 validator used before accepting each update) is genuinely different
for the two model families -- see validation/weight_validation.py vs.
validation/tree_validation.py for why.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from stamm_sdk.fl_types import AggregationResult, LocalUpdate
from stamm_sdk.validation.tree_validation import validate_tree_update
from stamm_sdk.validation.weight_validation import validate_gradient_update


class FederatedAggregator:
    def aggregate(self, updates: list[LocalUpdate], history: list[AggregationResult],
                   reference_data: pd.DataFrame | None = None,
                   output_name: str | None = None) -> AggregationResult:
        if not updates:
            raise ValueError("aggregate() needs at least one LocalUpdate")

        kind = updates[0].kind
        if any(u.kind != kind for u in updates):
            raise ValueError("all updates in one round must be the same kind (gradient vs. trees)")

        if kind == "gradient":
            return self._aggregate_gradients(updates, history)
        return self._aggregate_trees(updates, history, reference_data, output_name)

    # ---- gradient path (neural nets / linear models) -------------------

    def _aggregate_gradients(self, updates: list[LocalUpdate],
                              history: list[AggregationResult]) -> AggregationResult:
        accepted: list[LocalUpdate] = []
        rejected: dict[str, str] = {}
        cosine_similarities: dict[str, float] = {}

        for u in updates:
            report = validate_gradient_update(u, updates, history)
            if report.passed:
                accepted.append(u)
            else:
                rejected[u.participant_id] = "; ".join(report.reasons)
            if report.cosine_to_peers:
                cosine_similarities[u.participant_id] = float(np.mean(list(report.cosine_to_peers.values())))

        if not accepted:
            raise RuntimeError("no updates in this round passed Gate 2 -- aggregation refused")

        total_size = sum(u.dataset_size for u in accepted) or 1
        global_payload = {
            k: sum(np.asarray(u.payload[k], dtype=float) * (u.dataset_size / total_size) for u in accepted)
            for k in accepted[0].payload
        }

        direction_ok = True
        if history and history[-1].global_payload:
            prev_vec = np.concatenate([np.asarray(v, dtype=float).ravel() for v in history[-1].global_payload.values()])
            new_vec = np.concatenate([np.asarray(v, dtype=float).ravel() for v in global_payload.values()])
            if prev_vec.shape == new_vec.shape:
                denom = (np.linalg.norm(prev_vec) * np.linalg.norm(new_vec)) or 1e-9
                direction_ok = bool((np.dot(prev_vec, new_vec) / denom) >= 0)

        return AggregationResult(
            kind="gradient", global_payload=global_payload,
            cosine_similarities=cosine_similarities,
            accepted_participants=[u.participant_id for u in accepted],
            rejected_participants=rejected, performance_vs_previous=None,
            direction_ok=direction_ok,
        )

    # ---- tree path (random forest / other sklearn ensembles) ----------

    def _aggregate_trees(self, updates: list[LocalUpdate], history: list[AggregationResult],
                          reference_data: pd.DataFrame | None, output_name: str | None) -> AggregationResult:
        if reference_data is None or not output_name:
            raise ValueError("aggregating tree updates needs reference_data + output_name for Gate 2 scoring")

        accepted: list[LocalUpdate] = []
        rejected: dict[str, str] = {}
        for u in updates:
            report = validate_tree_update(u, updates, reference_data, output_name)
            if report.passed:
                accepted.append(u)
            else:
                rejected[u.participant_id] = "; ".join(report.reasons)

        if not accepted:
            raise RuntimeError("no updates in this round passed Gate 2 -- aggregation refused")

        # Tree ensemble aggregation: each participant contributes a share
        # of their own fitted trees, proportional to their dataset_size,
        # into one bigger merged forest.
        total_size = sum(u.dataset_size for u in accepted) or 1
        merged_trees: list[Any] = []
        for u in accepted:
            estimators = list(u.payload.estimators_)
            share = max(1, round(len(estimators) * (u.dataset_size / total_size)))
            merged_trees.extend(estimators[:share])

        global_model = copy.deepcopy(accepted[0].payload)
        global_model.estimators_ = merged_trees
        global_model.n_estimators = len(merged_trees)

        # Same feature-selection fix as validation/tree_validation.py's
        # _score_trees -- reference_data can carry extra bookkeeping
        # columns (is_synthetic, etc.) that were never part of training.
        feature_names = getattr(global_model, "feature_names_in_", None)
        X = reference_data[list(feature_names)] if feature_names is not None \
            else reference_data.drop(columns=[output_name])
        y = reference_data[output_name]
        new_score = float(global_model.score(X, y))

        direction_ok = True
        if history:
            prev_score = history[-1].performance_vs_previous
            if prev_score is not None:
                direction_ok = new_score >= prev_score

        return AggregationResult(
            kind="trees", global_payload=global_model, cosine_similarities=None,
            accepted_participants=[u.participant_id for u in accepted],
            rejected_participants=rejected, performance_vs_previous=new_score,
            direction_ok=direction_ok,
        )
