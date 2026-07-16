#!/usr/bin/env python3
"""Dry-run §3.2 startup prints for GLM-5.2-like PD (no NPU / weights needed).

Mimics Prefill (P) and Decode (D) connector registration + asymmetric config,
capturing the [SFA-PD-LEARN] prints added for learning.

Usage:
  python scripts/sfa_pd_learn_startup_trace.py
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Stub heavy deps (Windows / no vLLM / no torch environments)
# ---------------------------------------------------------------------------


def _ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as package-like when needed
    sys.modules[name] = mod
    return mod


def _load_module(mod_name: str, file_path: Path):
    """Load a module file without executing package __init__.py side effects."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs() -> None:
    # Pre-register package shells so `import vllm_ascend.xxx` does not run
    # vllm_ascend/__init__.py (which pulls real vllm).
    for name in [
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.kv_transfer",
        "vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload",
        "vllm_ascend.distributed.kv_transfer.kv_pool",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store",
        "vllm_ascend.distributed.kv_transfer.kv_p2p",
    ]:
        _ensure_module(name)

    # vllm core stubs
    vllm = _ensure_module("vllm")
    vllm.envs = SimpleNamespace()
    vllm_logger = _ensure_module("vllm.logger")
    logger = MagicMock()
    logger.info_once = MagicMock()
    logger.warning_once = MagicMock()
    logger.warning = MagicMock()
    logger.info = MagicMock()
    logger.exception = MagicMock()
    vllm_logger.logger = logger
    vllm.logger = vllm_logger

    math_utils = _ensure_module("vllm.utils.math_utils")
    math_utils.cdiv = lambda a, b: (a + b - 1) // b
    math_utils.round_down = lambda a, b: a - (a % b)
    utils = _ensure_module("vllm.utils")
    utils.math_utils = math_utils

    factory_mod = _ensure_module("vllm.distributed.kv_transfer.kv_connector.factory")

    class KVConnectorFactory:
        _registry: dict = {}

        @classmethod
        def register_connector(cls, name, module_path, class_name):
            cls._registry[name] = (module_path, class_name)

    factory_mod.KVConnectorFactory = KVConnectorFactory

    base_mod = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorRole:
        SCHEDULER = "SCHEDULER"
        WORKER = "WORKER"

    class KVConnectorBase_V1:
        def __init__(self, vllm_config=None, role=None, kv_cache_config=None):
            self.vllm_config = vllm_config
            self.role = role
            self.kv_cache_config = kv_cache_config

    class SupportsHMA:
        pass

    def supports_hma(_c):
        return True

    base_mod.KVConnectorRole = KVConnectorRole
    base_mod.KVConnectorBase_V1 = KVConnectorBase_V1
    base_mod.KVConnectorMetadata = object
    base_mod.SupportsHMA = SupportsHMA
    base_mod.supports_hma = supports_hma

    multi_mod = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector")

    class AscendStoreConnector:
        """Stand-in for AscendStoreConnector (no NPU/memcache in dry-run)."""

        pass

    class MultiConnector(KVConnectorBase_V1, SupportsHMA):
        def __init__(self, vllm_config, role, kv_cache_config):
            super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
            connector_mod = sys.modules["vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.connector"]
            SFAPDCpuOffloadConnector = connector_mod.SFAPDCpuOffloadConnector

            self._connectors = []
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
            for child in extra.get("connectors", []):
                name = child.get("kv_connector")
                child_ktc = SimpleNamespace(
                    kv_connector=name,
                    kv_role=vllm_config.kv_transfer_config.kv_role,
                    is_kv_producer=vllm_config.kv_transfer_config.is_kv_producer,
                    is_kv_consumer=vllm_config.kv_transfer_config.is_kv_consumer,
                    engine_id=vllm_config.kv_transfer_config.engine_id,
                    kv_connector_extra_config=child.get("kv_connector_extra_config") or {},
                )
                child_cfg = SimpleNamespace(
                    kv_transfer_config=child_ktc,
                    additional_config=vllm_config.additional_config,
                    scheduler_config=vllm_config.scheduler_config,
                    parallel_config=vllm_config.parallel_config,
                    model_config=vllm_config.model_config,
                )
                if name == "SFAPDCpuOffloadConnector":
                    self._connectors.append(SFAPDCpuOffloadConnector(child_cfg, role, kv_cache_config))
                elif name == "AscendStoreConnector":
                    self._connectors.append(AscendStoreConnector())
                else:
                    self._connectors.append(SimpleNamespace())

    multi_mod.MultiConnector = MultiConnector

    for name in [
        "vllm.config",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.kv_cache_manager",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.output",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.request",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm.forward_context",
        "vllm.utils.network_utils",
    ]:
        _ensure_module(name)

    sys.modules["vllm.config"].VllmConfig = type("VllmConfig", (), {})
    sys.modules["vllm.v1.kv_cache_interface"].KVCacheConfig = type("KVCacheConfig", (), {})
    sys.modules["vllm.v1.core.kv_cache_manager"].KVCacheBlocks = type("KVCacheBlocks", (), {})
    sys.modules["vllm.v1.core.sched.output"].SchedulerOutput = type("SchedulerOutput", (), {})
    sys.modules["vllm.utils.network_utils"].get_ip = lambda: "127.0.0.1"
    sys.modules["vllm.distributed"].get_tensor_model_parallel_rank = lambda: 0

    torch = _ensure_module("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.npu = MagicMock()
    _ensure_module("httpx")

    utils_mod = _ensure_module("vllm_ascend.utils")
    utils_mod.enable_sp = lambda *a, **k: False
    utils_mod.clear_enable_sp = lambda: None

    envs_mod = _ensure_module("vllm_ascend.envs")
    envs_mod.VLLM_ASCEND_BALANCE_SCHEDULING = False
    envs_mod.VLLM_ASCEND_ENABLE_FLASHCOMM1 = False
    envs_mod.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE = False
    envs_mod.VLLM_ASCEND_ENABLE_FUSED_MC2 = False
    envs_mod.VLLM_ASCEND_ENABLE_MLAPO = False
    envs_mod.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE = 0
    envs_mod.MSMONITOR_USE_DAEMON = False
    envs_mod.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK = False
    envs_mod.VLLM_ASCEND_KV_TRANSFER_BACKEND = "memfabric"

    # MooncakeLayerwiseConnector symbols imported by ascend_multi_connector
    mooncake_lw = _ensure_module(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector"
    )
    mooncake_lw.MooncakeLayerwiseConnector = type("MooncakeLayerwiseConnector", (), {})
    mooncake_lw.MooncakeLayerwiseConnectorScheduler = type(
        "MooncakeLayerwiseConnectorScheduler", (), {}
    )


_install_stubs()

# Load real instrumented modules by file path (skip package __init__ side effects).
_ascend_config = _load_module(
    "vllm_ascend.ascend_config",
    REPO_ROOT / "vllm_ascend" / "ascend_config.py",
)
_layerwise_config = _load_module(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config",
    REPO_ROOT
    / "vllm_ascend"
    / "distributed"
    / "kv_transfer"
    / "kv_pool"
    / "ascend_store"
    / "layerwise_config.py",
)
# Stub scheduler/worker modules before loading connector (avoids torch/memfabric).
_sched_stub = _ensure_module("vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler")
_sched_stub.SFAPDCpuOffloadScheduler = MagicMock
_sched_stub.SFAPDProducerScheduler = MagicMock
_worker_stub = _ensure_module("vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker")
_worker_stub.SFAPDCpuOffloadConsumerWorker = MagicMock
_worker_stub.SFAPDCpuOffloadProducerWorker = MagicMock

_connector = _load_module(
    "vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.connector",
    REPO_ROOT / "vllm_ascend" / "distributed" / "kv_transfer" / "sfa_pd_cpu_offload" / "connector.py",
)
_register = _load_module(
    "vllm_ascend.distributed.kv_transfer.__init__",
    REPO_ROOT / "vllm_ascend" / "distributed" / "kv_transfer" / "__init__.py",
)
_multi = _load_module(
    "vllm_ascend.distributed.kv_transfer.ascend_multi_connector",
    REPO_ROOT / "vllm_ascend" / "distributed" / "kv_transfer" / "ascend_multi_connector.py",
)


def _patch_ascend_config_light_init() -> None:
    """Keep §3.2 config prints without pulling the full AscendConfig graph."""

    RealLRU = _ascend_config.LRUResidentCacheConfig

    def _light_init(self, vllm_config):
        self.vllm_config = vllm_config
        # Markers required by _is_ascend_config_initialized / get_ascend_config.
        self.ascend_compilation_config = object()
        self.eplb_config = object()
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
        self.use_offload = bool(additional_config.get("use_offload", False))
        print(
            f"[SFA-PD-LEARN][①配置] AscendConfig.use_offload={self.use_offload} "
            f"（来自 additional_config；P=false / D=true 为 PD 硬约束）"
        )
        self.lru_resident_cache_config = RealLRU(additional_config.get("lru_resident_cache_config", {}))

    _ascend_config.AscendConfig.__init__ = _light_init


_patch_ascend_config_light_init()

# GLM-5.2 dense transformer depth used for mate-map demo (shared buffers M=2).
# Real GLM-5.2 has more layers; M=2 reuse math is the same shape.
GLM52_NUM_LAYERS = 6
LAYERWISE_NUM_SHARED_BUFFERS = 2


def _make_kv_cache_config(num_layers: int = GLM52_NUM_LAYERS):
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=layer_names)])


def _make_vllm_config(*, kv_role: str, use_offload: bool, engine_id: str, with_ascend_store: bool):
    """Build a GLM-5.2 PD-like VllmConfig stand-in (design doc §2.6.7)."""
    connectors = [
        {
            "kv_connector": "SFAPDCpuOffloadConnector",
            "kv_connector_extra_config": {
                "use_layerwise": True,
                "transfer_backend": "memfabric",
            },
        }
    ]
    if with_ascend_store:
        connectors.append(
            {
                "kv_connector": "AscendStoreConnector",
                "kv_connector_extra_config": {
                    "backend": "memcache",
                    "use_layerwise": True,
                    "layerwise_num_shared_buffers": LAYERWISE_NUM_SHARED_BUFFERS,
                },
            }
        )

    kv_transfer_config = SimpleNamespace(
        kv_connector="MultiConnector",
        kv_role=kv_role,
        is_kv_producer=(kv_role == "kv_producer"),
        is_kv_consumer=(kv_role == "kv_consumer"),
        engine_id=engine_id,
        kv_connector_extra_config={"connectors": connectors},
        get_from_extra_config=lambda key, default=None: (connectors and {}) or default,
    )

    additional_config = {
        "use_offload": use_offload,
        "enable_sparse_c8": False,
        "lru_resident_cache_config": {
            "enabled": use_offload,  # D enables resident LRU
            "buffer_size": 2048,
            "topk": 2048,
        },
        "refresh": True,  # force AscendConfig rebuild per node
    }

    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        additional_config=additional_config,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=True,
            max_num_batched_tokens=4096,
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            tensor_parallel_size=16 if kv_role == "kv_producer" else 4,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            is_deepseek_mla=True,
            get_total_num_kv_heads=lambda: 1,
        ),
    )


def _trace_node(node_label: str, *, kv_role: str, use_offload: bool, with_ascend_store: bool) -> str:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    clear_ascend_config = _ascend_config.clear_ascend_config
    register_connector = _register.register_connector
    AscendMultiConnector = _multi.AscendMultiConnector
    connector_mod = _connector

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 72)
        print(f" GLM-5.2 PD dry-run · {node_label}")
        print(f" kv_role={kv_role}  use_offload={use_offload}  Multi+AscendStore={with_ascend_store}")
        print("=" * 72)

        # --- ①注册 ---
        register_connector()

        # --- ①配置 + ①Multi + ①SFAPD ---
        clear_ascend_config()
        vllm_config = _make_vllm_config(
            kv_role=kv_role,
            use_offload=use_offload,
            engine_id="glm52-pd-0",
            with_ascend_store=with_ascend_store,
        )
        kv_cache_config = _make_kv_cache_config()

        # Patch scheduler/worker constructors so connector __init__ does not
        # pull memfabric / mooncake / NPU.
        with (
            patch.object(connector_mod, "SFAPDProducerScheduler", MagicMock(return_value=MagicMock())),
            patch.object(connector_mod, "SFAPDCpuOffloadScheduler", MagicMock(return_value=MagicMock())),
            patch.object(connector_mod, "SFAPDCpuOffloadProducerWorker", MagicMock(return_value=MagicMock())),
            patch.object(connector_mod, "SFAPDCpuOffloadConsumerWorker", MagicMock(return_value=MagicMock())),
        ):
            print("\n--- construct AscendMultiConnector(SCHEDULER) ---")
            AscendMultiConnector(vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config)

            print("\n--- construct AscendMultiConnector(WORKER) ---")
            AscendMultiConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)

        print("\n[done] §3.2 Connector 注册与 PD 不对称配置 路径已走完")
        print()

    return buf.getvalue()


def main() -> int:
    out_dir = REPO_ROOT / "scripts" / "learn_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # P: producer, use_offload=false, Multi(SFAPD + AscendStore GVA)
    p_log = _trace_node(
        "P 节点 (Prefill / kv_producer)",
        kv_role="kv_producer",
        use_offload=False,
        with_ascend_store=True,
    )
    # D: consumer, use_offload=true, Multi(SFAPD); AscendStore 不作为 D 层复用路径
    d_log = _trace_node(
        "D 节点 (Decode / kv_consumer)",
        kv_role="kv_consumer",
        use_offload=True,
        with_ascend_store=False,
    )

    p_path = out_dir / "glm52_pd_P_node_startup_learn.log"
    d_path = out_dir / "glm52_pd_D_node_startup_learn.log"
    p_path.write_text(p_log, encoding="utf-8")
    d_path.write_text(d_log, encoding="utf-8")

    print(p_log)
    print(d_log)
    print(f"[saved] P log -> {p_path}")
    print(f"[saved] D log -> {d_path}")
    return 0


if __name__ == "__main__":
    # Avoid AscendConfig picking up stale singleton across P/D in one process.
    os.environ.setdefault("VLLM_ASCEND_LEARN_TRACE", "1")
    raise SystemExit(main())
