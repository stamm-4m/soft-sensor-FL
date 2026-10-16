"""HTTP session with auth helpers for the Stamm API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config


@dataclass(slots=True)
class Tokens:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class Session:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self.tokens: Tokens | None = None

    def close(self) -> None:
        self._client.close()

    def _auth_headers(self) -> dict[str, str]:
        if not self.tokens:
            return {}
        return {"Authorization": f"Bearer {self.tokens.access_token}"}

    def register(
        self,
        email: str,
        password: str,
        full_name: str,
    ) -> dict[str, Any]:
        payload = {
            "email": email,
            "password": password,
            "full_name": full_name,
        }
        response = self._client.post("/auth/register", json=payload)
        response.raise_for_status()
        return response.json()

    def login(self, email: str, password: str) -> dict[str, Any]:
        payload = {"email": email, "password": password}
        response = self._client.post("/auth/login-json", json=payload)
        response.raise_for_status()
        data = response.json()
        self.tokens = Tokens(**data)
        return data

    def refresh(self) -> dict[str, Any]:
        if not self.tokens:
            raise RuntimeError("Not authenticated. Call login first.")

        # FIX: the registry expects refresh_token in the JSON body, not as
        # a query param -- the query-param form 422s ("Field required" on
        # body), confirmed against the live registry.
        response = self._client.post(
            "/auth/refresh",
            json={"refresh_token": self.tokens.refresh_token},
        )
        response.raise_for_status()
        data = response.json()

        access_token = data.get("access_token") or self.tokens.access_token
        refresh_token = data.get("refresh_token") or self.tokens.refresh_token
        token_type = data.get("token_type") or self.tokens.token_type
        self.tokens = Tokens(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
        )
        return data

    def me(self) -> dict[str, Any]:
        response = self._client.get("/auth/me", headers=self._auth_headers())
        response.raise_for_status()
        return response.json()

    def logout(self) -> dict[str, Any]:
        if not self.tokens:
            return {"detail": "No active session"}

        # FIX: same query-param-vs-body issue as refresh() above.
        response = self._client.post(
            "/auth/logout",
            json={"refresh_token": self.tokens.refresh_token},
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        data = response.json()
        self.tokens = None
        return data

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(
            method,
            path,
            params=params,
            json=json,
            data=data,
            headers=self._auth_headers(),
        )
        response.raise_for_status()

        if response.content:
            return response.json()
        return {}

    def request_raw(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        """Like request(), but returns the raw response body instead of
        decoding it as JSON -- for binary downloads (model artifacts)."""
        response = self._client.request(
            method,
            path,
            params=params,
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        return response.content
