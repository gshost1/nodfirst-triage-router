"""Routing configuration read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

DEFAULT_ROUTER_MODE = "mock"
DEFAULT_DO_ROUTER_MODEL = "router:nodfirst"


class RouterMode(str, Enum):
    MOCK = "mock"
    CLIENT = "client"
    DO = "do"


class RouterConfigError(ValueError):
    """Raised when routing environment configuration is invalid."""


@dataclass(frozen=True)
class RouterConfig:
    """Active routing mode plus the configured tier and router model IDs.

    Tier model IDs come from the environment (populated from the verified
    DigitalOcean model list); they are never hardcoded runtime choices. The
    DO router name defaults to this project's `nodfirst` router and may be
    overridden with `DO_ROUTER_MODEL`.
    """

    mode: RouterMode
    small_model: str = ""
    mid_model: str = ""
    large_model: str = ""
    do_router_model: str = DEFAULT_DO_ROUTER_MODEL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RouterConfig:
        """Build the configuration from an environment mapping.

        Application code loads the local `.env` (via the LLM client bootstrap)
        before reading the process environment; tests pass an explicit mapping.
        """
        source = os.environ if env is None else env
        mode_raw = source.get("ROUTER_MODE", DEFAULT_ROUTER_MODE).strip().lower()
        try:
            mode = RouterMode(mode_raw or DEFAULT_ROUTER_MODE)
        except ValueError as exc:
            valid = ", ".join(m.value for m in RouterMode)
            raise RouterConfigError(
                f"ROUTER_MODE must be one of {valid}; got {mode_raw!r}."
            ) from exc

        config = cls(
            mode=mode,
            small_model=source.get("MODEL_SMALL", "").strip(),
            mid_model=source.get("MODEL_MID", "").strip(),
            large_model=source.get("MODEL_LARGE", "").strip(),
            do_router_model=(
                source.get("DO_ROUTER_MODEL", "").strip() or DEFAULT_DO_ROUTER_MODEL
            ),
        )
        config._require_tiers()
        return config

    def _require_tiers(self) -> None:
        """Client and DO modes need every tier ID; mock mode runs without them."""
        if self.mode is RouterMode.MOCK:
            return
        missing = [
            name
            for name, value in (
                ("MODEL_SMALL", self.small_model),
                ("MODEL_MID", self.mid_model),
                ("MODEL_LARGE", self.large_model),
            )
            if not value
        ]
        if missing:
            raise RouterConfigError(
                f"ROUTER_MODE={self.mode.value} requires the "
                f"{', '.join(missing)} environment variable(s) to be configured."
            )
