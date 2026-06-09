"""High-level SDK client used by downstream applications."""

from __future__ import annotations

from .core.config import Config
from .core.session import Session
from .registry.api_client import APIClient


class Client:
    """Main entry point for interacting with the Stamm ML registry API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.config = Config.from_env(base_url=base_url, timeout=timeout)
        self.session = Session(config=self.config)
        self.ml = APIClient(session=self.session)

    def register(self, email: str, password: str, full_name: str) -> dict:
        return self.session.register(
            email=email,
            password=password,
            full_name=full_name,
        )

    def login(self, email: str, password: str) -> dict:
        return self.session.login(email=email, password=password)

    def logout(self) -> dict:
        return self.session.logout()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
