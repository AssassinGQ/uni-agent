"""Public config dataclasses for KVCAware LLM Router.

This file contains ONLY base classes and common config — no import of
strategy-specific or backend-specific configs. Polymorphic sub-classes
live in their own submodules and are resolved at runtime via _target_ FQN.

Base classes:
  ConfigError, StrategyConfig, MetricsBackendConfig

Common configs:
  MetricsConfig, CacheStoreConfig, KVCAwareConfig

Sub-classes (resolved by Hydra _target_ at runtime):
  KVCAwareStrategyConfig  → uni_agent.llm_router.strategies.kvc_aware
  VllmPrometheusConfig    → uni_agent.llm_router.metrics.vllm_prometheus
  VllmZmqConfig           → uni_agent.llm_router.metrics.vllm_zmq
  MooncakePrometheusConfig → uni_agent.llm_router.metrics.mooncake_prometheus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf


class ConfigError(ValueError):
    """Raised when config validation fails."""


# ============================================================
# Strategy base class
# ============================================================

@dataclass
class StrategyConfig:
    """Base config for routing strategies.

    All strategy configs inherit this and must provide `weight`.
    """

    weight: float

    def __post_init__(self) -> None:
        if not (0 < self.weight <= 1):
            raise ConfigError(f"weight must be in (0, 1], got {self.weight}")


# ============================================================
# Metrics backend base class
# ============================================================

@dataclass
class MetricsBackendConfig:
    """Empty base config for metrics backends. Provides type identity."""


# ============================================================
# MetricsConfig (common cross-backend config)
# ============================================================

@dataclass
class MetricsConfig:
    """Config for metrics collection, including cross-backend common settings."""

    retry_interval: float = 5.0
    max_retries: int = 3
    timeout: float = 10.0
    degrade_policy: str = "lower_priority"
    backends: list[MetricsBackendConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.retry_interval <= 0:
            raise ConfigError(f"retry_interval must be > 0, got {self.retry_interval}")
        if self.max_retries < 0:
            raise ConfigError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.timeout <= 0:
            raise ConfigError(f"timeout must be > 0, got {self.timeout}")
        valid_policies = {"lower_priority", "exclude"}
        if self.degrade_policy not in valid_policies:
            raise ConfigError(
                f"degrade_policy must be one of {valid_policies}, got '{self.degrade_policy}'"
            )


# ============================================================
# CacheStore config
# ============================================================

_VALID_KV_CACHE_STORE_TYPES = {"list", "radix_tree"}


@dataclass
class CacheStoreConfig:
    """Config for CacheStore — passive storage layer for decoded metrics data."""

    kv_cache_store_type: str = "list"
    ttl: float = 30.0

    def __post_init__(self) -> None:
        if self.kv_cache_store_type not in _VALID_KV_CACHE_STORE_TYPES:
            raise ConfigError(
                f"kv_cache_store_type must be one of {_VALID_KV_CACHE_STORE_TYPES}, "
                f"got '{self.kv_cache_store_type}'"
            )
        if self.ttl <= 0:
            raise ConfigError(f"ttl must be > 0, got {self.ttl}")


# ============================================================
# Top-level KVCAwareConfig
# ============================================================

_DEFAULT_METRICS = MetricsConfig()
_DEFAULT_CACHE_STORE = CacheStoreConfig()


@dataclass
class KVCAwareConfig:
    """Top-level config for KVCAwareBalancer, parsed from VeRL's router_kwargs."""

    strategies: list[StrategyConfig]  # required, no default
    metrics: MetricsConfig = field(default_factory=lambda: _DEFAULT_METRICS)
    cache_store: CacheStoreConfig = field(default_factory=lambda: _DEFAULT_CACHE_STORE)

    @classmethod
    def from_router_kwargs(cls, kwargs: DictConfig | dict) -> KVCAwareConfig:
        """Two-step parsing of VeRL-transmitted router_kwargs.

        Step 1: omega_conf_to_dataclass for auto-recursive dataclass fields
                (metrics, cache_store)
        Step 2: manual traversal of strategies/backends lists, hydra.instantiate
                each _target_ entry
        """
        if not isinstance(kwargs, (DictConfig, dict)):
            raise ConfigError(f"kwargs must be DictConfig or dict, got {type(kwargs)}")

        kwargs = OmegaConf.create(kwargs)

        # Extract polymorphic lists before merge to avoid ReadonlyConfigError
        strategies_raw = _extract_list(kwargs, "strategies")
        backends_raw = _extract_nested_list(kwargs, "metrics", "backends")

        # Step 1: merge only dataclass-typed fields (metrics, cache_store)
        defaults = OmegaConf.create({
            "metrics": OmegaConf.structured(MetricsConfig),
            "cache_store": OmegaConf.structured(CacheStoreConfig),
        })
        kwargs_for_merge = OmegaConf.create(kwargs)
        if "strategies" in kwargs_for_merge:
            OmegaConf.set_struct(kwargs_for_merge, False)
            kwargs_for_merge.pop("strategies", None)
            OmegaConf.set_struct(kwargs_for_merge, True)
        if "metrics" in kwargs_for_merge and isinstance(kwargs_for_merge.metrics, DictConfig) and "backends" in kwargs_for_merge.metrics:
            OmegaConf.set_struct(kwargs_for_merge.metrics, False)
            kwargs_for_merge.metrics.pop("backends", None)
            OmegaConf.set_struct(kwargs_for_merge.metrics, True)

        # Validate that metrics and cache_store are dict-like (not primitive types)
        # None is allowed — it means "use default"
        if "metrics" in kwargs_for_merge and kwargs_for_merge.metrics is not None and not isinstance(kwargs_for_merge.metrics, (dict, DictConfig)):
            raise ConfigError(
                f"metrics must be a dict, got {type(kwargs_for_merge.metrics).__name__}"
            )
        if "cache_store" in kwargs_for_merge and kwargs_for_merge.cache_store is not None and not isinstance(kwargs_for_merge.cache_store, (dict, DictConfig)):
            raise ConfigError(
                f"cache_store must be a dict, got {type(kwargs_for_merge.cache_store).__name__}"
            )

        merged = OmegaConf.merge(defaults, kwargs_for_merge)
        config_obj = OmegaConf.to_object(merged)

        # Step 2: manual traversal for polymorphic lists
        if strategies_raw is None:
            raise ConfigError("strategies is required — must be explicitly configured")
        strategies = _parse_polymorphic_list(
            strategies_raw, StrategyConfig, "strategies"
        )

        metrics_backends = []
        if backends_raw is not None:
            metrics_backends = _parse_polymorphic_list(
                backends_raw, MetricsBackendConfig, "backends"
            )

        # Extract parsed dataclass fields from the merged dict result
        if isinstance(config_obj, dict):
            metrics_cfg = config_obj.get("metrics") or MetricsConfig()
            cache_store_cfg = config_obj.get("cache_store") or CacheStoreConfig()
        else:
            metrics_cfg = config_obj.metrics if config_obj.metrics is not None else MetricsConfig()
            cache_store_cfg = config_obj.cache_store if config_obj.cache_store is not None else CacheStoreConfig()

        metrics_cfg.backends = metrics_backends

        # Validate and construct
        result = cls(
            strategies=strategies,
            metrics=metrics_cfg,
            cache_store=cache_store_cfg,
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Validate the full config. Raises ConfigError with all violations."""
        errors: list[str] = []

        if not self.strategies:
            errors.append("strategies must be non-empty")
        elif not isinstance(self.strategies, list):
            errors.append("strategies must be a list")
        else:
            total_weight = sum(s.weight for s in self.strategies)
            if not (0.9 <= total_weight <= 1.1):
                errors.append(
                    f"sum of strategy weights must be ~1.0, got {total_weight}"
                )

        if errors:
            raise ConfigError("; ".join(errors))


def _parse_polymorphic_list(
    items: list[Any],
    base_class: type,
    list_name: str,
) -> list[Any]:
    """Parse a polymorphic list where each item has `_target_` for hydra.instantiate.

    Validates that each instantiated item is a subclass of `base_class`.
    """
    result: list[Any] = []
    if not items:
        return result

    for i, item in enumerate(items):
        if not isinstance(item, (dict, DictConfig)):
            raise ConfigError(f"{list_name}[{i}] must be a dict, got {type(item)}")

        item_conf = OmegaConf.create(item) if isinstance(item, dict) else item

        if "_target_" not in item_conf:
            raise ConfigError(
                f"{list_name}[{i}] must have '_target_' key, got keys: {list(item_conf.keys())}"
            )

        try:
            parsed = instantiate(item_conf)
        except (InstantiationException, ImportError, AttributeError) as e:
            raise ConfigError(
                f"{list_name}[{i}] failed to instantiate _target_ '{item_conf._target_}': {e}"
            ) from e

        if not isinstance(parsed, base_class):
            raise ConfigError(
                f"{list_name}[{i}] _target_ must inherit {base_class.__name__}, "
                f"got {type(parsed).__name__}"
            )

        result.append(parsed)

    return result


def _extract_list(cfg: DictConfig, key: str) -> list[Any] | None:
    """Extract a list from DictConfig, returning None if key is absent."""
    if key not in cfg:
        return None
    val = cfg[key]
    if val is None:
        return None
    if isinstance(val, (list, ListConfig)):
        return list(val)
    return val


def _extract_nested_list(cfg: DictConfig, parent: str, child: str) -> list[Any] | None:
    """Extract a nested list from cfg.parent.child, returning None if absent."""
    if parent not in cfg:
        return None
    sub = cfg[parent]
    if not isinstance(sub, DictConfig):
        return None
    return _extract_list(sub, child)