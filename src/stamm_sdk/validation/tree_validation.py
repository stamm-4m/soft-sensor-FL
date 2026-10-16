"""Gate 2, non-parametric path -- validate one client's tree-ensemble
contribution before it's allowed into the merged global forest.

There's no gradient/direction for a Random Forest, so the equivalent
defense is performance-based: score the participant's trees against a
shared reference dataset and flag contributions whose accuracy is an
outlier relative to this round's other participants -- either
suspiciously bad (noisy/wrong data) or suspiciously perfect (a classic
tell for overfitting to a small poisoned sample).

Needs a labeled reference_data (features + true output column) -- see
APIClient.get_reference_dataset()'s documented gap; this gate simply
can't run without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stamm_sdk.fl_types import LocalUpdate

ACCURACY_Z_SCORE_LIMIT = 2.0       # |z| above this vs. the round's peers -> outlier
NEAR_PERFECT_SUSPICION = 0.999     # accuracy this close to 1.0 is itself a red flag


@dataclass(slots=True)
class TreeValidationReport:
    passed: bool
    accuracy: float
    accuracy_vs_round: float | None = None
    reasons: list[str] = field(default_factory=list)


def _score_trees(trees, reference_data: pd.DataFrame, output_name: str) -> float:
    """Score the submitted ensemble against reference_data, via whatever
    .score() the sklearn-compatible estimator already implements
    (accuracy for classifiers, R^2 for regressors).

    Selects columns by trees.feature_names_in_ (sklearn records this on
    any estimator fit with a DataFrame) rather than "everything except
    output_name" -- reference_data routinely carries extra bookkeeping
    columns (is_synthetic, timestamps, ids, ...) that were never part of
    training and would otherwise make sklearn reject the predict() call
    with "feature names should match those passed during fit"."""
    feature_names = getattr(trees, "feature_names_in_", None)
    X = reference_data[list(feature_names)] if feature_names is not None \
        else reference_data.drop(columns=[output_name])
    y = reference_data[output_name]
    return float(trees.score(X, y))


def validate_tree_update(update: LocalUpdate, round_updates: list[LocalUpdate],
                          reference_data: pd.DataFrame, output_name: str) -> TreeValidationReport:
    reasons: list[str] = []
    accuracy = _score_trees(update.payload, reference_data, output_name)

    if accuracy >= NEAR_PERFECT_SUSPICION:
        reasons.append(f"accuracy {accuracy:.4f} suspiciously close to perfect -- possible overfit/poisoning")

    peer_scores = [
        _score_trees(u.payload, reference_data, output_name)
        for u in round_updates if u.participant_id != update.participant_id
    ]
    accuracy_vs_round: float | None = None
    if len(peer_scores) >= 2:
        mean, std = float(np.mean(peer_scores)), float(np.std(peer_scores)) or 1e-6
        accuracy_vs_round = (accuracy - mean) / std
        if abs(accuracy_vs_round) > ACCURACY_Z_SCORE_LIMIT:
            reasons.append(f"accuracy {accuracy:.4f} is a round outlier (z={accuracy_vs_round:.2f} vs peers)")

    return TreeValidationReport(
        passed=not reasons, accuracy=accuracy,
        accuracy_vs_round=accuracy_vs_round, reasons=reasons,
    )
