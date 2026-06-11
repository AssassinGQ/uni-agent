"""Mooncake Prometheus metrics backend config."""

from __future__ import annotations

from dataclasses import dataclass

from uni_agent.llm_router.config import MetricsBackendConfig


@dataclass
class MooncakePrometheusConfig(MetricsBackendConfig):
    """Config for Mooncake Prometheus metrics collector."""

    endpoints: dict[str, str] | None = None