"""Configuration management for the SDK."""

from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_BASE_URL = "http://188.245.241.118:8080"
DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True, slots=True)
class Config:
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(
        cls,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> "Config":
        resolved_base_url = (
            base_url
            or os.getenv("STAMM_API_BASE_URL")
            or DEFAULT_BASE_URL
        )
        resolved_timeout = timeout
        if resolved_timeout is None:
            env_timeout = os.getenv("STAMM_API_TIMEOUT")
            resolved_timeout = (
                float(env_timeout) if env_timeout else DEFAULT_TIMEOUT
            )

        return cls(
            base_url=resolved_base_url.rstrip("/"),
            timeout=float(resolved_timeout),
        )
