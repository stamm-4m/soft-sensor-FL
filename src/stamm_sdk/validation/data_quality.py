"""Gate 1 -- validate a client's raw data before any training happens.

Two independent checks, both framework-agnostic (they run on plain
feature values, long before anything becomes a PyTorch/sklearn model):

1. Physical range check: any value outside a sensor/actuator's own
   real-world bounds is corrupt or fabricated, full stop.
2. PSI (Population Stability Index) against a reference distribution,
   when one is available: flags a client whose data has drifted too far
   from what the model was expecting, even if every individual value is
   individually "in range".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stamm_sdk.schemas.model_schema import ModelSchema

# Conventional PSI cutoff for "significant distribution shift" (credit-risk
# literature: <0.1 no change, 0.1-0.25 moderate, >0.25 significant).
PSI_DRIFT_THRESHOLD = 0.25


@dataclass(slots=True)
class DataQualityReport:
    passed: bool
    out_of_range: dict[str, int] = field(default_factory=dict)
    psi_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def compute_psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index between two numeric samples.

    Buckets are quantile edges of `reference` (so each reference bucket
    holds ~1/buckets of the reference sample by construction); `current`
    is binned into those same edges. A small epsilon replaces any 0%
    bucket so the log term never blows up.
    """
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return 0.0

    edges = np.unique(ref.quantile(np.linspace(0, 1, buckets + 1)).to_numpy())
    if len(edges) < 2:
        return 0.0  # reference has no spread -- nothing meaningful to bucket
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def validate_raw_data(data: pd.DataFrame, schema: ModelSchema,
                       reference: pd.DataFrame | None = None) -> DataQualityReport:
    """Run both checks over every feature in `schema` that's present in
    `data`. A feature missing from `data` entirely is NOT this gate's
    concern -- that's data/mapping.py's validate_mapped_data(); this gate
    assumes mapping already passed.

    PSI only runs for numeric ("float"/"int") features, and only when
    `reference` has that column -- categorical drift needs a different
    test (e.g. chi-square) and isn't implemented yet.
    """
    out_of_range: dict[str, int] = {}
    psi_scores: dict[str, float] = {}
    reasons: list[str] = []

    for feat in schema.features:
        if feat.name not in data.columns:
            continue
        column = data[feat.name]

        if feat.range_min is not None and feat.range_max is not None:
            mask = (column < feat.range_min) | (column > feat.range_max)
            count = int(mask.sum())
            if count:
                out_of_range[feat.name] = count
                reasons.append(
                    f"{feat.name}: {count} row(s) outside physical range "
                    f"[{feat.range_min}, {feat.range_max}]"
                )

        if reference is not None and feat.name in reference.columns \
                and feat.type in ("float", "int"):
            psi = compute_psi(reference[feat.name], column)
            psi_scores[feat.name] = psi
            if psi > PSI_DRIFT_THRESHOLD:
                reasons.append(
                    f"{feat.name}: PSI={psi:.3f} exceeds drift threshold {PSI_DRIFT_THRESHOLD}"
                )

    passed = not out_of_range and all(v <= PSI_DRIFT_THRESHOLD for v in psi_scores.values())
    return DataQualityReport(
        passed=passed, out_of_range=out_of_range, psi_scores=psi_scores, reasons=reasons,
    )
