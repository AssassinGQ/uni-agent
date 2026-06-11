"""KVCache-Aware strategy config."""

from __future__ import annotations

from dataclasses import dataclass, field

from uni_agent.llm_router.config import ConfigError, StrategyConfig


@dataclass
class KVCAwareStrategyConfig(StrategyConfig):
    """Config for KVCache-Aware routing strategy.

    S = α × S_cache + (1-α) × S_load
    """

    alpha: float = 0.7
    load_threshold: float = 80.0
    layer_weights: dict[str, float] = field(default_factory=lambda: {"cpu": 1.0, "ssd": 0.25})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.load_threshold <= 0:
            raise ConfigError(f"load_threshold must be > 0, got {self.load_threshold}")
        valid_keys = {"cpu", "ssd"}
        if not set(self.layer_weights.keys()) == valid_keys:
            raise ConfigError(
                f"layer_weights keys must be {valid_keys} only, got {set(self.layer_weights.keys())}"
            )