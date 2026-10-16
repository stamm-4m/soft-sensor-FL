"""Wrapper around ML registry endpoints."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from stamm_sdk.core.session import Session
from stamm_sdk.schemas.model_schema import FeatureSpec, ModelSchema

logger = logging.getLogger(__name__)

def _detect_framework(model_description: dict[str, Any]) -> str:
    """"pytorch" | "keras" | "r" | "sklearn", from model_description's own
    packages/language list -- NOT from algorithm name.

    algorithm alone is not enough: confirmed live that algorithm="rnn"
    (0009_python_penicillin_LSTM) is actually a Keras/TensorFlow model
    (packages: keras.layers.LSTM, tensorflow), not PyTorch -- guessing
    "rnn"/"lstm"/"neural_network" -> pytorch would have load_model() try
    torch.load() on a .keras file and fail confusingly. This looks at
    what's actually declared instead.
    """
    languages = {str(l.get("name", "")).lower() for l in (model_description.get("language") or [])}
    if "r" in languages:
        return "r"
    package_blob = " ".join(
        f"{p.get('package', '')} {' '.join(p.get('classes', []) or [p.get('class', '')])}"
        for p in (model_description.get("packages") or [])
    ).lower()
    if "tensorflow" in package_blob or "keras" in package_blob:
        return "keras"
    if "torch" in package_blob:
        return "pytorch"
    return "sklearn"


class APIClient:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_projects(self) -> dict[str, Any]:
        return self.session.request("GET", "/list_projects/")

    def get_project_info(self, project_id: str) -> dict[str, Any]:
        return self.session.request("GET", f"/{project_id}/project_info/")

    def get_db_config(self, project_id: str) -> dict[str, Any]:
        return self.session.request("GET", f"/{project_id}/db_config/")

    def get_references(self, project_id: str) -> dict[str, Any]:
        return self.session.request("GET", f"/{project_id}/references/")

    def get_variables(self, project_id: str) -> dict[str, Any]:
        return self.session.request("GET", f"/{project_id}/variables/")

    def list_models(self, project_id: str) -> dict[str, Any]:
        # FIX: was "/{project_id}/list_models/", which doesn't exist on the
        # registry -- the real route is list_soft_sensors/.
        return self.session.request("GET", f"/{project_id}/list_soft_sensors/")

    def get_model_metadata(self, project_id: str, model_id: str) -> dict[str, Any]:
        return self.session.request("GET", f"/{project_id}/metadata/{model_id}")

    def list_models_full(self, project_id: str) -> dict[str, Any]:
        # FIX: was "/{project_id}/models_full/", which doesn't exist -- the
        # real route is soft_sensor_full/ (singular "sensor").
        return self.session.request("GET", f"/{project_id}/soft_sensor_full/")

    def update_model(self, project_id: str, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.session.request("PUT", f"/{project_id}/update/{model_id}", json=payload)

    def predict(self, project_id: str, model_id: str, features: dict[str, Any]) -> dict[str, Any]:
        # FIX: the registry's /predict endpoint reads req.input_data, not
        # req itself -- was sending {"req": features}, which looks
        # plausible but silently never carries any feature values.
        return self.session.request(
            "POST",
            f"/{project_id}/predict/{model_id}",
            json={"req": {"input_data": features}},
        )

    # ---- New: FL / training support -----------------------------------

    def list_available_models(self) -> list[dict[str, Any]]:
        """Models the logged-in user can see, across every project they
        belong to. /list_projects/ is already lab-filtered server-side
        (see model_registry's ml_registry_router), so this just fans out
        list_models() over that set -- no separate permission check needed
        here."""
        projects = self.list_projects()
        project_rows = projects if isinstance(projects, list) else projects.get("projects", [])

        out: list[dict[str, Any]] = []
        for p in project_rows:
            # /list_projects/ returns the YAML-style code under "project_ID"
            # (capital ID, confirmed against the live registry) -- "project_id"/
            # "id" are kept as fallbacks in case a differently-cased project
            # row ever shows up, but project_ID is the real key.
            project_id = p.get("project_ID") or p.get("project_id") or p.get("id")
            if not project_id:
                continue
            # A project can exist in the DB (so it shows up here) while the
            # YAML-backed in-memory registry hasn't loaded it yet (confirmed
            # live: a freshly-added project 404s with "Project 'X' not
            # loaded" until someone POSTs /{project_id}/reload/ or the API
            # restarts) -- skip that one project instead of failing this
            # whole call over it.
            try:
                models = self.list_models(project_id)
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "list_available_models: skipping project %r (%s)",
                    project_id, exc.response.text,
                )
                continue
            model_rows = models if isinstance(models, list) else models.get("models", [])
            for m in model_rows:
                # /list_soft_sensors/ wraps everything useful under
                # "metadata" and keys the slug as "model_ID" (confirmed
                # against the live registry) -- normalize to a flat,
                # predictable shape instead of leaking that internal
                # structure to callers.
                meta = m.get("metadata") or {}
                out.append({
                    "project_id": project_id,
                    "slug": m.get("model_ID") or meta.get("ID"),
                    "name": m.get("model_name") or meta.get("name"),
                    "algorithm": meta.get("algorithm"),
                    "status": meta.get("status"),
                })
        return out

    def _sensor_actuator_ranges(self) -> dict[str, tuple[float | None, float | None]]:
        """variable -> (range_min, range_max), merged across sensors and
        actuators. Hits the generic CRUD endpoints directly (no
        project_id prefix -- these are global catalogs, not per-project).
        Used by get_schemas() only as a FALLBACK, for a feature whose
        metadata.inputs.features[].expected_range is missing -- the
        model's own metadata is the primary, more specific source."""
        ranges: dict[str, tuple[float | None, float | None]] = {}
        for table in ("sensors", "actuators"):
            rows = self.session.request("GET", f"/api/v1/{table}/", params={"limit": 1000})
            row_list = rows if isinstance(rows, list) else rows.get("items", [])
            for row in row_list:
                variable = row.get("variable")
                if variable:
                    ranges[variable] = (row.get("range_min"), row.get("range_max"))
        return ranges

    def get_schemas(self, project_id: str, model_id: str) -> ModelSchema:
        """Fetch + normalize one model's feature schema from
        GET /{project_id}/metadata/{model_id}.

        Confirmed real shape (this is the YAML-backed ml_registry
        metadata, NOT the flat soft_sensors DB row -- those are two
        different representations in this system):

            {"model_identification": {"ID": ..., "algorithm": ..., "UUID": ...},
             "inputs": {"features": [{"name", "lag", "expected_range": {"min","max"}, ...}, ...]},
             "outputs": {"variables": [...], "information": [{"name": ...}, ...]}}

        Every listed feature is treated as required -- this metadata
        shape has no per-feature "optional" flag. range_min/range_max
        come from the feature's own expected_range when present; the
        sensors/actuators catalog (_sensor_actuator_ranges) is only a
        fallback for the rare feature missing that.
        """
        row = self.get_model_metadata(project_id, model_id)
        identification = row.get("model_identification") or {}
        algorithm = identification.get("algorithm")
        framework = _detect_framework(row.get("model_description") or {})

        raw_features = (row.get("inputs") or {}).get("features") or []
        ranges_fallback: dict[str, tuple[float | None, float | None]] | None = None

        features: list[FeatureSpec] = []
        for feat in raw_features:
            name = feat.get("name")
            if not name:
                continue
            expected_range = feat.get("expected_range") or {}
            range_min, range_max = expected_range.get("min"), expected_range.get("max")
            if range_min is None or range_max is None:
                if ranges_fallback is None:
                    ranges_fallback = self._sensor_actuator_ranges()
                range_min, range_max = ranges_fallback.get(name, (range_min, range_max))
            features.append(FeatureSpec(
                name=name,
                type="float",
                lag=int(feat.get("lag") or 0),
                required=True,
                range_min=range_min,
                range_max=range_max,
            ))

        outputs = row.get("outputs") or {}
        output_info = outputs.get("information") or []
        output_name = (
            output_info[0].get("name") if output_info
            else (outputs.get("variables") or [None])[0]
        )

        return ModelSchema(
            model_id=str(identification.get("UUID") or identification.get("uuid") or model_id),
            slug=identification.get("ID") or model_id,
            algorithm=algorithm,
            framework=framework,
            features=features,
            output_name=output_name,
        )

    def get_model_artifact(self, project_id: str, model_id: str) -> bytes:
        """Raw bytes of the model's trained artifact file (.pkl/.pt/...).
        404s (as plain bytes of the error body) if artifact_path is null
        on that model row -- see load_model(), which is what actually
        checks for that case before calling this."""
        return self.session.request_raw("GET", f"/{project_id}/model_artifact/{model_id}")

    def get_reference_dataset(self, project_id: str, model_id: str, run_id: str | None = None
                               ) -> list[dict[str, Any]] | None:
        """Best-effort reference data for Gate 1 (PSI) / tree-validation
        comparisons.

        There's no dedicated "reference dataset" endpoint on the registry
        today -- this is a known gap. For now, if the caller has a known
        good run_id to compare against (e.g. a prior accepted round's
        run), this pulls its sensor_readings via the real, existing
        run_timeseries_router endpoint. Returns None (not an empty list)
        when no run_id is given, so callers can tell "no reference
        available" apart from "reference was empty".
        """
        if not run_id:
            return None
        rows = self.session.request("GET", f"/api/v1/runs/{run_id}/sensor_readings")
        return rows if isinstance(rows, list) else rows.get("items", [])

    def submit_federation_round(self, federation_id: str, round_number: int,
                                 global_metric_value: float | None,
                                 contributions: dict[str, dict[str, Any]],
                                 global_model_id: str | None = None) -> dict[str, Any]:
        """Persist one aggregation round: a federation_rounds row plus one
        model_contributions row per accepted participant.

        contributions: {contributor_model_id: {"weight": float, "metrics": {...}}}
        Both tables already exist in model_registry (see Federation /
        ModelContribution models) -- this writes through the generic CRUD
        endpoints, same pattern as everything else in this client.
        """
        round_row = self.session.request(
            "POST",
            "/api/v1/federation_rounds/",
            json={
                "federation_id": federation_id,
                "round_number": round_number,
                "global_metric_value": global_metric_value,
                "participants_expected": len(contributions),
                "participants_received": len(contributions),
                "contributions": {cid: c.get("weight") for cid, c in contributions.items()},
                "global_model_id": global_model_id,
            },
        )
        for contributor_model_id, c in contributions.items():
            if not global_model_id:
                continue
            self.session.request(
                "POST",
                "/api/v1/model_contributions/",
                json={
                    "aggregated_model_id": global_model_id,
                    "contributor_model_id": contributor_model_id,
                    "federation_id": federation_id,
                    "round_number": round_number,
                    "contribution_weight": c.get("weight", 1.0),
                    "contribution_metrics": c.get("metrics", {}),
                },
            )
        return round_row
