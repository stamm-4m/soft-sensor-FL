"""Fill out an undersized training set with synthetic rows, so a user with
too little real data can still accumulate enough to train.

v1 scope (per design decision): bounds-only, uniform-random within each
feature's [range_min, range_max] -- no attempt yet at matching the real
data's distribution shape (that's a reasonable v2, not this one)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stamm_sdk.schemas.model_schema import ModelSchema


def _feature_bounds(feature_name: str, schema: ModelSchema,
                     existing: pd.DataFrame) -> tuple[float, float]:
    """range_min/range_max from the model schema (sourced from the
    sensors/actuators catalog via get_schemas()) if known; otherwise
    fall back to the existing real data's own min/max. Raises if neither
    is available -- generating synthetic values with literally no bound
    to anchor them to would be guessing, not synthesizing."""
    spec = next((f for f in schema.features if f.name == feature_name), None)
    if spec and spec.range_min is not None and spec.range_max is not None:
        return float(spec.range_min), float(spec.range_max)

    if feature_name in existing.columns and existing[feature_name].notna().any():
        col = existing[feature_name].dropna()
        return float(col.min()), float(col.max())

    raise ValueError(
        f"no range available for feature '{feature_name}' -- not in the "
        f"sensors/actuators catalog and no existing data to infer bounds from"
    )


def generate_synthetic_rows(schema: ModelSchema, existing: pd.DataFrame,
                             target_n: int, *, seed: int | None = None) -> pd.DataFrame:
    """Generate max(0, target_n - len(existing)) new rows, one column per
    required feature in schema, each uniform-random within that
    feature's bounds (see _feature_bounds).

    Returns only the NEW synthetic rows (not concatenated with
    ``existing``) -- the caller decides whether/how to combine them, and
    every returned row carries ``is_synthetic=True`` so synthetic rows
    stay traceable downstream (e.g. Gate 1 can choose to exclude them
    from a PSI comparison against real-data drift).
    """
    n_missing = max(0, target_n - len(existing))
    if n_missing == 0:
        return existing.iloc[0:0].assign(is_synthetic=pd.Series(dtype=bool))

    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    for feat in schema.features:
        if not feat.required:
            continue
        lo, hi = _feature_bounds(feat.name, schema, existing)
        columns[feat.name] = rng.uniform(lo, hi, size=n_missing)

    synthetic = pd.DataFrame(columns)
    synthetic["is_synthetic"] = True
    return synthetic
