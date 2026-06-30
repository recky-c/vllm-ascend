from collections.abc import Iterator

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionBackend  # type: ignore
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandler
from vllm_ascend.kv_debug import kv_debug_log


class NPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, kv_cache_config)

        num_cpu_blocks = self.extra_config.get("num_cpu_blocks")
        if not num_cpu_blocks:
            raise Exception("num_cpu_blocks must be specified in kv_connector_extra_config")
        self.num_cpu_blocks: int = num_cpu_blocks
        kv_debug_log(
            logger,
            "NPUOffloadingSpec.__init__: num_cpu_blocks=%s "
            "block_size_factor=%s extra_config=%s",
            self.num_cpu_blocks,
            self.block_size_factor,
            self.extra_config,
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handler: OffloadingHandler | None = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = kv_events_config is not None and kv_events_config.enable_kv_cache_events
            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor
            kv_debug_log(
                logger,
                "NPUOffloadingSpec.get_manager: gpu_block_size=%s "
                "offloaded_block_size=%s num_cpu_blocks=%s enable_events=%s",
                gpu_block_size,
                offloaded_block_size,
                self.num_cpu_blocks,
                enable_events,
            )
            self._manager = CPUOffloadingManager(
                block_size=offloaded_block_size,
                num_blocks=self.num_cpu_blocks,
                enable_events=enable_events,
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            kv_debug_log(
                logger,
                "NPUOffloadingSpec.get_handlers: gpu_block_size=%s "
                "cpu_block_size=%s num_cpu_blocks=%s kv_cache_layers=%s",
                gpu_block_size,
                gpu_block_size * self.block_size_factor,
                self.num_cpu_blocks,
                list(kv_caches.keys()),
            )
            self._handler = CpuNpuOffloadingHandler(
                attn_backends=attn_backends,
                gpu_block_size=gpu_block_size,
                cpu_block_size=gpu_block_size * self.block_size_factor,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
