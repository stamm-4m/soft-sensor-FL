"""Wrapper around ML registry endpoints."""

from __future__ import annotations

from typing import Any

from stamm_sdk.core.session import Session


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
        return self.session.request("GET", f"/{project_id}/list_models/")

    def get_model_metadata(self, project_id: str, model_id: str) -> dict[str, Any]:
        return self.session.request("GET", f"/{project_id}/metadata/{model_id}")

    def list_models_full(self, project_id: str) -> dict[str, Any]:
        return self.session.request("GET", f"/{project_id}/models_full/")

    def update_model(self, project_id: str, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.session.request("PUT", f"/{project_id}/update/{model_id}", json=payload)

    def predict(self, project_id: str, model_id: str, features: dict[str, Any]) -> dict[str, Any]:
        return self.session.request(
            "POST",
            f"/{project_id}/predict/{model_id}",
            json={"req": features},
        )
