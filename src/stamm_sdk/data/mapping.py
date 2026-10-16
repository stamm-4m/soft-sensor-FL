"""Map a user's raw column names onto a model's canonical feature names,
and validate the result against that model's schema before anything gets
anywhere near training."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stamm_sdk.schemas.model_schema import ModelSchema


@dataclass(slots=True)
class ValidationReport:
    passed: bool
    missing_features: list[str] = field(default_factory=list)
    type_mismatches: list[str] = field(default_factory=list)


def map_input_columns(data: pd.DataFrame, schema: ModelSchema,
                       column_map: dict[str, str]) -> pd.DataFrame:
    """Rename data's columns from the user's own names to the model's
    canonical feature names, using an explicit {user_column: feature_name}
    mapping -- never guesses/fuzzy-matches, so a wrong mapping fails loudly
    in validate_mapped_data() rather than silently training on the wrong
    column.

    Columns not mentioned in column_map are left untouched (and will show
    up as missing in validate_mapped_data() if the model needs them).
    """
    return data.rename(columns=column_map)


def validate_mapped_data(data: pd.DataFrame, schema: ModelSchema) -> ValidationReport:
    """Check that every required feature in schema is present in data
    (after map_input_columns) with a compatible dtype.

    Type compatibility is intentionally loose (int is accepted where
    float is expected, etc.) -- this is a presence+shape check, not a
    strict type system; exact value-level sanity (ranges, drift) is
    Gate 1's job in validation/data_quality.py, not this function's.
    """
    missing: list[str] = []
    type_mismatches: list[str] = []

    for feat in schema.features:
        if not feat.required:
            continue
        if feat.name not in data.columns:
            missing.append(feat.name)
            continue

        column_dtype = data[feat.name].dtype
        if feat.type == "float" and not pd.api.types.is_numeric_dtype(column_dtype):
            type_mismatches.append(f"{feat.name}: expected float-like, got {column_dtype}")
        elif feat.type == "int" and not pd.api.types.is_integer_dtype(column_dtype) \
                and not pd.api.types.is_numeric_dtype(column_dtype):
            type_mismatches.append(f"{feat.name}: expected int-like, got {column_dtype}")

    return ValidationReport(
        passed=not missing and not type_mismatches,
        missing_features=missing,
        type_mismatches=type_mismatches,
    )
