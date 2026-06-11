"""vLLM ZMQ metrics backend config."""

from __future__ import annotations

from dataclasses import dataclass

from uni_agent.llm_router.config import MetricsBackendConfig


@dataclass
class VllmZmqConfig(MetricsBackendConfig):
    """Config for vLLM ZMQ KV-cache event subscriber."""

    zmq_endpoints: dict[str, str] | None = None