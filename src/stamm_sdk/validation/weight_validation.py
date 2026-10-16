"""Gate 2, parametric path -- validate one client's gradient/weight update
before it's allowed anywhere near aggregation.

Two checks, both grounded in published FL defenses:

1. Norm bound: a DYNAMIC threshold (median +/- k*MAD across this round's
   updates), not a fixed absolute number -- a fixed threshold assumes a
   stable global scale, which doesn't hold across non-IID clients.
2. Cosine similarity to peers (FoolsGold-style): an attacker can rescale
   their update's magnitude freely without changing what it does, but
   not its direction without weakening their own attack -- so direction
   is much harder to fake than magnitude. Updates whose direction is
   suspiciously close to another participant's (collusion) OR far from
   the round's consensus direction both get flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stamm_sdk.fl_types import AggregationResult, LocalUpdate

NORM_MAD_MULTIPLIER = 3.0           # k in median +/- k*MAD
PEER_SIMILARITY_SUSPICION = 0.99    # near-identical direction -> likely collusion/sybils
CONSENSUS_SIMILARITY_FLOOR = 0.0    # below this vs. the round's mean direction -> flagged
DRIFT_VS_HISTORY_FLOOR = -0.5       # below this vs. previous global model -> flagged


@dataclass(slots=True)
class WeightValidationReport:
    passed: bool
    norm: float
    cosine_to_peers: dict[str, float] = field(default_factory=dict)
    drift_vs_history: float | None = None
    reasons: list[str] = field(default_factory=list)


def _flatten(payload: dict) -> np.ndarray:
    """payload is a dict of named arrays (a flattened state_dict, or a
    {"coef_": ..., "intercept_": ...}-style dict for linear models) --
    concatenate into one vector for norm/cosine math."""
    parts = [np.asarray(v, dtype=float).ravel() for v in payload.values()]
    return np.concatenate(parts) if parts else np.array([])


def compute_cosine_similarity(a: LocalUpdate, b: LocalUpdate) -> float:
    va, vb = _flatten(a.payload), _flatten(b.payload)
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def compute_norm_bound(round_updates: list[LocalUpdate]) -> tuple[float, float]:
    """(median, MAD) of ||update|| across this round's clients -- the
    actual bound a single update gets checked against is
    median +/- NORM_MAD_MULTIPLIER*MAD, computed by the caller."""
    norms = np.array([np.linalg.norm(_flatten(u.payload)) for u in round_updates])
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median))) or 1e-6
    return median, mad


def validate_gradient_update(update: LocalUpdate, round_updates: list[LocalUpdate],
                              history: list[AggregationResult]) -> WeightValidationReport:
    reasons: list[str] = []
    norm = float(np.linalg.norm(_flatten(update.payload)))

    median, mad = compute_norm_bound(round_updates)
    lower, upper = median - NORM_MAD_MULTIPLIER * mad, median + NORM_MAD_MULTIPLIER * mad
    if not (lower <= norm <= upper):
        reasons.append(f"norm {norm:.4g} outside bound [{lower:.4g}, {upper:.4g}]")

    peers = [u for u in round_updates if u.participant_id != update.participant_id]
    cosine_to_peers = {p.participant_id: compute_cosine_similarity(update, p) for p in peers}
    for pid, sim in cosine_to_peers.items():
        if sim >= PEER_SIMILARITY_SUSPICION:
            reasons.append(f"near-identical direction to participant {pid!r} (cos={sim:.4f}) -- possible collusion")

    if peers:
        all_in_round = peers + [update]
        mean_payload = {
            k: np.mean([np.asarray(u.payload.get(k, 0.0), dtype=float) for u in all_in_round], axis=0)
            for k in update.payload
        }
        consensus = LocalUpdate(participant_id="_consensus", round_number=update.round_number,
                                 dataset_size=0, kind=update.kind, payload=mean_payload)
        consensus_sim = compute_cosine_similarity(update, consensus)
        if consensus_sim < CONSENSUS_SIMILARITY_FLOOR:
            reasons.append(f"direction diverges from round consensus (cos={consensus_sim:.4f})")

    drift_vs_history: float | None = None
    if history and history[-1].global_payload:
        last_update = LocalUpdate(participant_id="_previous_global", round_number=update.round_number - 1,
                                   dataset_size=0, kind=update.kind, payload=history[-1].global_payload)
        drift_vs_history = compute_cosine_similarity(update, last_update)
        if drift_vs_history < DRIFT_VS_HISTORY_FLOOR:
            reasons.append(f"direction diverges sharply from previous global model (cos={drift_vs_history:.4f})")

    return WeightValidationReport(
        passed=not reasons, norm=norm, cosine_to_peers=cosine_to_peers,
        drift_vs_history=drift_vs_history, reasons=reasons,
    )
