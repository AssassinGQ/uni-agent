"""vLLM Prometheus metrics backend config."""

from __future__ import annotations

from dataclasses import dataclass

from uni_agent.llm_router.config import MetricsBackendConfig


@dataclass
class VllmPrometheusConfig(MetricsBackendConfig):
    """Config for vLLM Prometheus metrics collector."""

    prometheus_endpoints: dict[str, str] | None = None