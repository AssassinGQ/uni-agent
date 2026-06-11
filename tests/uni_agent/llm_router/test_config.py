"""Tests for llm_router config dataclasses and parsing.

Per §4 of detailed_config_module.md, each module has 5 test categories:
① Input/output normal cases
② Input/output abnormal cases
③ Hydra parsing normal cases
④ Hydra parsing abnormal cases
⑤ Other cases
"""

from __future__ import annotations

import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf


# -- Public API via package __init__ --
from uni_agent.llm_router import (
    CacheStoreConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    MetricsBackendConfig,
    MetricsConfig,
    MooncakePrometheusConfig,
    StrategyConfig,
    VllmPrometheusConfig,
    VllmZmqConfig,
)


# ============================================================
# 4.1 StrategyConfig / KVCAwareStrategyConfig
# ============================================================

# -- ① Input/output normal cases --

class TestStrategyNormalInput:
    """S01-S11: 正常输入输出用例"""

    def test_s01_weight_1_0(self):
        """S01: weight = 1.0"""
        cfg = KVCAwareStrategyConfig(weight=1.0)
        assert cfg.weight == 1.0

    def test_s02_weight_0_7(self):
        """S02: weight = 0.7"""
        cfg = KVCAwareStrategyConfig(weight=0.7)
        assert cfg.weight == 0.7

    def test_s03_alpha_0_7(self):
        """S03: alpha = 0.7"""
        cfg = KVCAwareStrategyConfig(weight=1.0, alpha=0.7)
        assert cfg.alpha == 0.7

    def test_s04_alpha_0_pure_load(self):
        """S04: alpha = 0.0 → cache 权重为 0，纯 load 打分"""
        cfg = KVCAwareStrategyConfig(weight=1.0, alpha=0.0)
        assert cfg.alpha == 0.0

    def test_s05_alpha_1_pure_cache(self):
        """S05: alpha = 1.0 → load 权重为 0，纯 cache 打分"""
        cfg = KVCAwareStrategyConfig(weight=1.0, alpha=1.0)
        assert cfg.alpha == 1.0

    def test_s06_alpha_default_0_7(self):
        """S06: alpha 缺失 → 默认值 0.7"""
        cfg = KVCAwareStrategyConfig(weight=1.0)
        assert cfg.alpha == 0.7

    def test_s07_load_threshold_80(self):
        """S07: load_threshold = 80"""
        cfg = KVCAwareStrategyConfig(weight=1.0, load_threshold=80)
        assert cfg.load_threshold == 80

    def test_s08_load_threshold_default_80(self):
        """S08: load_threshold 缺失 → 默认值 80"""
        cfg = KVCAwareStrategyConfig(weight=1.0)
        assert cfg.load_threshold == 80

    def test_s09_layer_weights_dict(self):
        """S09: layer_weights = {cpu: 1.0, ssd: 0.25}"""
        cfg = KVCAwareStrategyConfig(weight=1.0, layer_weights={"cpu": 1.0, "ssd": 0.25})
        assert cfg.layer_weights == {"cpu": 1.0, "ssd": 0.25}

    def test_s10_layer_weights_default(self):
        """S10: layer_weights 缺失 → 默认值"""
        cfg = KVCAwareStrategyConfig(weight=1.0)
        assert cfg.layer_weights == {"cpu": 1.0, "ssd": 0.25}

    def test_s11_multi_strategy_weights_sum_to_1(self):
        """S11: 多策略 Σ weight ≈ 1.0"""
        s1 = KVCAwareStrategyConfig(weight=0.7)
        s2 = KVCAwareStrategyConfig(weight=0.3)
        assert s1.weight + s2.weight == pytest.approx(1.0)


# -- ② Input/output abnormal cases --

class TestStrategyAbnormalInput:
    """S12-S24: 异常输入输出用例"""

    def test_s12_weight_zero(self):
        """S12: weight = 0 → ConfigError"""
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareStrategyConfig(weight=0.0)

    def test_s13_weight_above_1(self):
        """S13: weight > 1 → ConfigError"""
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareStrategyConfig(weight=1.5)

    def test_s14_weight_negative(self):
        """S14: weight < 0 → ConfigError"""
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareStrategyConfig(weight=-1.0)

    def test_s17_load_threshold_zero(self):
        """S17: load_threshold = 0 → ConfigError"""
        with pytest.raises(ConfigError, match="load_threshold"):
            KVCAwareStrategyConfig(weight=1.0, load_threshold=0)

    def test_s18_load_threshold_negative(self):
        """S18: load_threshold < 0 → ConfigError"""
        with pytest.raises(ConfigError, match="load_threshold"):
            KVCAwareStrategyConfig(weight=1.0, load_threshold=-1)

    def test_s19_layer_weights_non_cpu_ssd_key(self):
        """S19: layer_weights 含非 cpu/ssd 键 → ConfigError"""
        with pytest.raises(ConfigError, match="layer_weights"):
            KVCAwareStrategyConfig(weight=1.0, layer_weights={"cpu": 1.0, "disk": 0.5})

    def test_s20_layer_weights_missing_key(self):
        """S20: layer_weights 缺少 cpu 或 ssd → ConfigError"""
        with pytest.raises(ConfigError, match="layer_weights"):
            KVCAwareStrategyConfig(weight=1.0, layer_weights={"cpu": 1.0})

    def test_s21_multi_strategy_weights_not_sum_to_1(self):
        """S21: 多策略 Σ weight ≠ 1.0 → ConfigError (at validate level)"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 0.4},
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 0.4},
            ],
        })
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_s22_strategies_empty_list(self):
        """S22: strategies 为空列表 → ConfigError"""
        kwargs = OmegaConf.create({"strategies": []})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_s23_strategies_not_list(self):
        """S23: strategies 不是 list → ConfigError"""
        kwargs = OmegaConf.create({"strategies": "kvc_aware"})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_s24_strategy_item_not_dict(self):
        """S24: strategy item 不是 dict → ConfigError"""
        kwargs = OmegaConf.create({"strategies": ["kvc_aware"]})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_router_kwargs(kwargs)


# ============================================================
# 4.2 MetricsConfig / MetricsBackendConfig
# ============================================================

# -- ① Input/output normal cases --

class TestMetricsNormalInput:
    """M01-M16: 正常输入输出用例"""

    def test_m01_retry_interval_5(self):
        """M01: retry_interval = 5.0"""
        cfg = MetricsConfig(retry_interval=5.0)
        assert cfg.retry_interval == 5.0

    def test_m02_retry_interval_default(self):
        """M02: retry_interval 缺失 → 默认值 5.0"""
        cfg = MetricsConfig()
        assert cfg.retry_interval == 5.0

    def test_m03_max_retries_3(self):
        """M03: max_retries = 3"""
        cfg = MetricsConfig(max_retries=3)
        assert cfg.max_retries == 3

    def test_m04_max_retries_0(self):
        """M04: max_retries = 0 → 正常解析（不重试）"""
        cfg = MetricsConfig(max_retries=0)
        assert cfg.max_retries == 0

    def test_m05_max_retries_default(self):
        """M05: max_retries 缺失 → 默认值 3"""
        cfg = MetricsConfig()
        assert cfg.max_retries == 3

    def test_m06_timeout_10(self):
        """M06: timeout = 10.0"""
        cfg = MetricsConfig(timeout=10.0)
        assert cfg.timeout == 10.0

    def test_m07_timeout_default(self):
        """M07: timeout 缺失 → 默认值 10.0"""
        cfg = MetricsConfig()
        assert cfg.timeout == 10.0

    def test_m08_degrade_policy_lower_priority(self):
        """M08: degrade_policy = lower_priority"""
        cfg = MetricsConfig(degrade_policy="lower_priority")
        assert cfg.degrade_policy == "lower_priority"

    def test_m09_degrade_policy_exclude(self):
        """M09: degrade_policy = exclude"""
        cfg = MetricsConfig(degrade_policy="exclude")
        assert cfg.degrade_policy == "exclude"

    def test_m10_degrade_policy_default(self):
        """M10: degrade_policy 缺失 → 默认值 lower_priority"""
        cfg = MetricsConfig()
        assert cfg.degrade_policy == "lower_priority"

    def test_m11_vllm_prometheus_endpoints_dict(self):
        """M11: prometheus_endpoints = dict"""
        cfg = VllmPrometheusConfig(prometheus_endpoints={"r0": "http://host/metrics"})
        assert cfg.prometheus_endpoints == {"r0": "http://host/metrics"}

    def test_m12_vllm_prometheus_endpoints_none(self):
        """M12: prometheus_endpoints = None → 自动推断"""
        cfg = VllmPrometheusConfig(prometheus_endpoints=None)
        assert cfg.prometheus_endpoints is None

    def test_m13_vllm_zmq_endpoints_dict(self):
        """M13: zmq_endpoints = dict"""
        cfg = VllmZmqConfig(zmq_endpoints={"r0": "tcp://host:5556"})
        assert cfg.zmq_endpoints == {"r0": "tcp://host:5556"}

    def test_m14_vllm_zmq_endpoints_none(self):
        """M14: zmq_endpoints = None"""
        cfg = VllmZmqConfig(zmq_endpoints=None)
        assert cfg.zmq_endpoints is None

    def test_m15_mooncake_endpoints_dict(self):
        """M15: endpoints (Mooncake) = dict"""
        cfg = MooncakePrometheusConfig(endpoints={"mooncake": "http://host/metrics"})
        assert cfg.endpoints == {"mooncake": "http://host/metrics"}

    def test_m16_mooncake_endpoints_none(self):
        """M16: endpoints (Mooncake) = None"""
        cfg = MooncakePrometheusConfig(endpoints=None)
        assert cfg.endpoints is None


# -- ② Input/output abnormal cases --

class TestMetricsAbnormalInput:
    """M17-M23: 异常输入输出用例"""

    def test_m17_retry_interval_zero(self):
        """M17: retry_interval = 0 → ConfigError"""
        with pytest.raises(ConfigError, match="retry_interval"):
            MetricsConfig(retry_interval=0)

    def test_m18_retry_interval_negative(self):
        """M18: retry_interval < 0 → ConfigError"""
        with pytest.raises(ConfigError, match="retry_interval"):
            MetricsConfig(retry_interval=-1)

    def test_m19_max_retries_negative(self):
        """M19: max_retries < 0 → ConfigError"""
        with pytest.raises(ConfigError, match="max_retries"):
            MetricsConfig(max_retries=-1)

    def test_m20_timeout_zero(self):
        """M20: timeout = 0 → ConfigError"""
        with pytest.raises(ConfigError, match="timeout"):
            MetricsConfig(timeout=0)

    def test_m21_timeout_negative(self):
        """M21: timeout < 0 → ConfigError"""
        with pytest.raises(ConfigError, match="timeout"):
            MetricsConfig(timeout=-1)

    def test_m22_degrade_policy_invalid(self):
        """M22: degrade_policy = random → ConfigError"""
        with pytest.raises(ConfigError, match="degrade_policy"):
            MetricsConfig(degrade_policy="random")

    def test_m23_metrics_not_dict(self):
        """M23: metrics 不是 dict → ConfigError"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "metrics": "vllm",
        })
        with pytest.raises(ConfigError, match="metrics"):
            KVCAwareConfig.from_router_kwargs(kwargs)


# ============================================================
# 4.3 CacheStoreConfig
# ============================================================

# -- ① Input/output normal cases --

class TestCacheStoreNormalInput:
    """C01-C05: 正常输入输出用例"""

    def test_c01_kv_cache_store_type_list(self):
        """C01: kv_cache_store_type = list"""
        cfg = CacheStoreConfig(kv_cache_store_type="list")
        assert cfg.kv_cache_store_type == "list"

    def test_c02_kv_cache_store_type_radix_tree(self):
        """C02: kv_cache_store_type = radix_tree"""
        cfg = CacheStoreConfig(kv_cache_store_type="radix_tree")
        assert cfg.kv_cache_store_type == "radix_tree"

    def test_c03_kv_cache_store_type_default(self):
        """C03: kv_cache_store_type 缺失 → 默认值 list"""
        cfg = CacheStoreConfig()
        assert cfg.kv_cache_store_type == "list"

    def test_c04_ttl_30(self):
        """C04: ttl = 30.0"""
        cfg = CacheStoreConfig(ttl=30.0)
        assert cfg.ttl == 30.0

    def test_c05_ttl_default(self):
        """C05: ttl 缺失 → 默认值 30.0"""
        cfg = CacheStoreConfig()
        assert cfg.ttl == 30.0


# -- ② Input/output abnormal cases --

class TestCacheStoreAbnormalInput:
    """C06-C09: 异常输入输出用例"""

    def test_c06_kv_cache_store_type_unknown(self):
        """C06: kv_cache_store_type = unknown → ConfigError"""
        with pytest.raises(ConfigError, match="kv_cache_store_type"):
            CacheStoreConfig(kv_cache_store_type="unknown")

    def test_c07_ttl_zero(self):
        """C07: ttl = 0 → ConfigError"""
        with pytest.raises(ConfigError, match="ttl"):
            CacheStoreConfig(ttl=0)

    def test_c08_ttl_negative(self):
        """C08: ttl < 0 → ConfigError"""
        with pytest.raises(ConfigError, match="ttl"):
            CacheStoreConfig(ttl=-1)

    def test_c09_cache_store_not_dict(self):
        """C09: cache_store 不是 dict → ConfigError"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "cache_store": "list",
        })
        with pytest.raises(ConfigError, match="cache_store"):
            KVCAwareConfig.from_router_kwargs(kwargs)


# ============================================================
# 4.4 KVCAwareConfig top-level + Hydra parsing
# ============================================================

# -- ③ Hydra parsing normal cases --

class TestHydraParsingNormal:
    """S25-S27, M24-M25, K03-K06: Hydra 解析正常用例"""

    def test_s25_strategy_instantiate(self):
        """S25: strategy 带 _target_ instantiate 为 KVCAwareStrategyConfig"""
        entry = OmegaConf.create({
            "_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig",
            "weight": 1.0,
            "alpha": 0.7,
        })
        from hydra.utils import instantiate
        result = instantiate(entry)
        assert isinstance(result, KVCAwareStrategyConfig)

    def test_s26_strategy_inherit_base(self):
        """S26: instantiate 后可通过基类访问 weight"""
        entry = OmegaConf.create({
            "_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig",
            "weight": 1.0,
            "alpha": 0.7,
        })
        from hydra.utils import instantiate
        result = instantiate(entry)
        assert isinstance(result, StrategyConfig)
        assert result.weight == 1.0

    def test_s27_strategy_defaults_filled(self):
        """S27: instantiate 默认值填充"""
        entry = OmegaConf.create({
            "_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig",
            "weight": 1.0,
        })
        from hydra.utils import instantiate
        result = instantiate(entry)
        assert result.alpha == 0.7
        assert result.load_threshold == 80
        assert result.layer_weights == {"cpu": 1.0, "ssd": 0.25}

    def test_m24_backend_instantiate(self):
        """M24: backend 带 _target_ instantiate 为 VllmPrometheusConfig"""
        entry = OmegaConf.create({
            "_target_": "uni_agent.llm_router.metrics.vllm_prometheus.VllmPrometheusConfig",
            "prometheus_endpoints": None,
        })
        from hydra.utils import instantiate
        result = instantiate(entry)
        assert isinstance(result, VllmPrometheusConfig)

    def test_m25_backend_inherit_base(self):
        """M25: backend instantiate 后 isinstance MetricsBackendConfig"""
        entry = OmegaConf.create({
            "_target_": "uni_agent.llm_router.metrics.vllm_prometheus.VllmPrometheusConfig",
            "prometheus_endpoints": None,
        })
        from hydra.utils import instantiate
        result = instantiate(entry)
        assert isinstance(result, MetricsBackendConfig)

    def test_k03_omega_conf_to_dataclass_top_level(self):
        """K03: 顶层 omega_conf_to_dataclass → KVCAwareConfig"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0, "alpha": 0.7},
            ],
            "metrics": {
                "retry_interval": 5, "max_retries": 3, "timeout": 10,
                "degrade_policy": "lower_priority",
                "backends": [
                    {"_target_": "uni_agent.llm_router.metrics.vllm_prometheus.VllmPrometheusConfig", "prometheus_endpoints": None},
                ],
            },
            "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result, KVCAwareConfig)

    def test_k04_metrics_auto_recursive_parse(self):
        """K04: metrics 字段自动递归解析为 MetricsConfig"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "metrics": {"retry_interval": 5, "max_retries": 3},
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result.metrics, MetricsConfig)
        assert result.metrics.retry_interval == 5.0
        assert result.metrics.max_retries == 3

    def test_k05_cache_store_auto_recursive_parse(self):
        """K05: cache_store 字段自动递归解析为 CacheStoreConfig"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "cache_store": {"kv_cache_store_type": "radix_tree", "ttl": 60},
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result.cache_store, CacheStoreConfig)
        assert result.cache_store.kv_cache_store_type == "radix_tree"
        assert result.cache_store.ttl == 60.0

    def test_k06_omegaconf_dictconfig_input(self):
        """K06: OmegaConf DictConfig 输入 → instantiate 正常"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
        })
        # OmegaConf.create already produces DictConfig
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)


# -- ④ Hydra parsing abnormal cases --

class TestHydraParsingAbnormal:
    """S28-S32, M26-M28, K07-K10: Hydra 解析异常用例"""

    def test_s28_target_module_not_exist(self):
        """S28: _target_ FQN 模块不存在 → InstantiationException"""
        entry = OmegaConf.create({"_target_": "nonexistent.Module.Class"})
        from hydra.utils import instantiate
        with pytest.raises((InstantiationException, ImportError, ConfigError)):
            instantiate(entry)

    def test_s29_target_class_not_exist(self):
        """S29: _target_ FQN 类不存在 → InstantiationException"""
        entry = OmegaConf.create({
            "_target_": "uni_agent.llm_router.config.NonExistClass",
        })
        from hydra.utils import instantiate
        with pytest.raises((InstantiationException, AttributeError, ConfigError)):
            instantiate(entry)

    def test_s30_target_missing(self):
        """S30: _target_ 缺失 → instantiate 报错"""
        entry = OmegaConf.create({"weight": 1.0})
        # hydra.instantiate without _target_ returns the dict itself, but
        # from_router_kwargs should detect missing _target_ and raise
        kwargs = OmegaConf.create({"strategies": [{"weight": 1.0}]})
        with pytest.raises(ConfigError, match="_target_"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_s31_target_not_strategy_subclass(self):
        """S31: _target_ 指向非 StrategyConfig 子类 → ConfigError"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.config.CacheStoreConfig", "weight": 1.0},
            ],
        })
        # Either fails to instantiate (wrong kwargs) or fails isinstance check
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_m26_backend_target_module_not_exist(self):
        """M26: backends _target_ 模块不存在 → InstantiationException"""
        entry = OmegaConf.create({"_target_": "nonexistent.Module.Class"})
        from hydra.utils import instantiate
        with pytest.raises((InstantiationException, ImportError, ConfigError)):
            instantiate(entry)

    def test_m27_backend_target_not_backend_subclass(self):
        """M27: backends _target_ 指向非 MetricsBackendConfig 子类 → ConfigError"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "metrics": {
                "backends": [
                    {"_target_": "uni_agent.llm_router.config.CacheStoreConfig"},
                ],
            },
        })
        with pytest.raises(ConfigError, match="MetricsBackendConfig"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_m28_backend_target_missing(self):
        """M28: backends _target_ 缺失 → ConfigError"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "metrics": {
                "backends": [{"prometheus_endpoints": None}],
            },
        })
        with pytest.raises(ConfigError, match="_target_"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_k09_strategies_remain_dict_after_top_parse(self):
        """K09: 顶层解析后 strategies[0] 仍为 dict（含 _target_），需手动遍历 instantiate"""
        # After omega_conf_to_dataclass, strategies list items remain as DictConfig
        # from_router_kwargs must manually traverse and instantiate each _target_ item
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0, "alpha": 0.7},
            ],
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)

    def test_k10_backends_remain_dict_after_top_parse(self):
        """K10: 顶层解析后 backends[0] 仍为 dict（含 _target_），需手动遍历 instantiate"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "metrics": {
                "backends": [
                    {"_target_": "uni_agent.llm_router.metrics.vllm_prometheus.VllmPrometheusConfig", "prometheus_endpoints": None},
                ],
            },
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result.metrics.backends[0], VllmPrometheusConfig)


# -- ⑤ Other cases --

class TestStrategyOther:
    """S33-S34: 其他用例"""

    def test_s33_empty_kwargs_no_strategy(self):
        """S33: 空 kwargs → ConfigError: strategies is required"""
        kwargs = OmegaConf.create({})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_router_kwargs(kwargs)

    def test_s34_strategies_null_error(self):
        """S34: strategies 为 null → ConfigError: strategies is required"""
        kwargs = OmegaConf.create({"strategies": None})
        with pytest.raises(ConfigError, match="strategies"):
            KVCAwareConfig.from_router_kwargs(kwargs)


class TestMetricsOther:
    """M29-M30: 其他用例"""

    def test_m29_metrics_defaults_with_strategy(self):
        """M29: strategies + 空 metrics → metrics 默认值"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert result.metrics.retry_interval == 5.0
        assert result.metrics.max_retries == 3
        assert result.metrics.timeout == 10.0
        assert result.metrics.degrade_policy == "lower_priority"

    def test_m30_metrics_null_default(self):
        """M30: metrics 为 null → 默认值"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "metrics": None,
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result.metrics, MetricsConfig)


class TestCacheStoreOther:
    """C10-C11: 其他用例"""

    def test_c10_cache_store_defaults_with_strategy(self):
        """C10: strategies + 空 cache_store → cache_store 默认值"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert result.cache_store.kv_cache_store_type == "list"
        assert result.cache_store.ttl == 30.0

    def test_c11_cache_store_null_default(self):
        """C11: cache_store 为 null → 默认值"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
            "cache_store": None,
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result.cache_store, CacheStoreConfig)


class TestKVCAwareTopLevel:
    """K01-K02, K11-K12: KVCAwareConfig 顶层用例"""

    def test_k01_full_kwargs_normal_parse(self):
        """K01: 完整 kwargs → 正常解析"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0, "alpha": 0.7},
            ],
            "metrics": {
                "retry_interval": 5, "max_retries": 3, "timeout": 10,
                "degrade_policy": "lower_priority",
                "backends": [
                    {"_target_": "uni_agent.llm_router.metrics.vllm_prometheus.VllmPrometheusConfig", "prometheus_endpoints": None},
                ],
            },
            "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert isinstance(result, KVCAwareConfig)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert isinstance(result.metrics, MetricsConfig)
        assert isinstance(result.metrics.backends[0], VllmPrometheusConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)

    def test_k02_multi_error_aggregation(self):
        """K02: 多错误聚合"""
        kwargs = OmegaConf.create({
            "strategies": [],
            "metrics": {"max_retries": -1},
            "cache_store": {"ttl": 0},
        })
        with pytest.raises(ConfigError) as exc_info:
            KVCAwareConfig.from_router_kwargs(kwargs)
        # Error message should contain multiple field names
        error_msg = str(exc_info.value)
        assert "strategies" in error_msg or "max_retries" in error_msg or "ttl" in error_msg

    def test_k11_minimal_kwargs_all_defaults(self):
        """K11: 仅 strategies → metrics/cache_store 默认值"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 1.0},
            ],
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        assert len(result.strategies) == 1
        assert isinstance(result.metrics, MetricsConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)

    def test_k12_manual_instantiate_results(self):
        """K12: 手动遍历 instantiate 后 strategies/backends 变为 dataclass"""
        kwargs = OmegaConf.create({
            "strategies": [
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 0.7},
                {"_target_": "uni_agent.llm_router.strategies.kvc_aware.KVCAwareStrategyConfig", "weight": 0.3},
            ],
            "metrics": {
                "backends": [
                    {"_target_": "uni_agent.llm_router.metrics.vllm_prometheus.VllmPrometheusConfig"},
                    {"_target_": "uni_agent.llm_router.metrics.vllm_zmq.VllmZmqConfig"},
                ],
            },
        })
        result = KVCAwareConfig.from_router_kwargs(kwargs)
        # All strategies are KVCAwareStrategyConfig instances
        for s in result.strategies:
            assert isinstance(s, StrategyConfig)
        # All backends are MetricsBackendConfig instances
        for b in result.metrics.backends:
            assert isinstance(b, MetricsBackendConfig)


# ============================================================
# YAML config file loading
# ============================================================

class TestYamlConfigLoading:
    """Loading default/example YAML file and parsing to KVCAwareConfig."""

    def test_load_default_yaml(self):
        """从默认 YAML 文件加载并解析为 KVCAwareConfig"""
        from pathlib import Path

        yaml_path = Path(__file__).parent.parent.parent.parent / "uni_agent" / "llm_router" / "configs" / "default.yaml"
        assert yaml_path.exists(), f"default.yaml not found at {yaml_path}"

        kwargs = OmegaConf.load(yaml_path)
        # The YAML wraps config under `router.router_kwargs`
        router_kwargs = kwargs.router.router_kwargs
        result = KVCAwareConfig.from_router_kwargs(router_kwargs)

        assert isinstance(result, KVCAwareConfig)
        assert len(result.strategies) == 1
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert result.strategies[0].alpha == 0.7
        assert result.strategies[0].load_threshold == 80
        assert isinstance(result.metrics, MetricsConfig)
        assert result.metrics.degrade_policy == "lower_priority"
        assert len(result.metrics.backends) == 1
        assert isinstance(result.metrics.backends[0], VllmPrometheusConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)
        assert result.cache_store.kv_cache_store_type == "list"
        assert result.cache_store.ttl == 30.0