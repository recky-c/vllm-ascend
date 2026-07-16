# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""PD-disaggregated SFA KV-transfer connector.

On the Decode node (``kv_consumer``), remote Prefill exposes its KV and Decode
pulls the bulk MLA KV into a CPU pinned offload pool; the indexer KV lands in
HBM. The D-side load path (LRU-resident H2D) is reused from
:class:`SFAKVOffloadWorker`.
"""

from typing import TYPE_CHECKING, Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler import (
    SFAPDCpuOffloadScheduler,
    SFAPDProducerScheduler,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker import (
    SFAPDCpuOffloadConsumerWorker,
    SFAPDCpuOffloadProducerWorker,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.request import Request

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


class SFAPDCpuOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """One connector class branching on ``role`` and ``kv_role``.

    * SCHEDULER + producer : P-side metadata for memfabric pull notifications.
    * SCHEDULER + consumer : D-side CPU-block allocation / advertisement tracking.
    * WORKER + producer    : P-side layer-wise READ_READY notifications.
    * WORKER + consumer    : D-side : composes ``SFAKVOffloadWorker`` (LRU load +
      CPU pool) + memfabric pull read + indexer/main split registration.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_consumer = vllm_config.kv_transfer_config.is_kv_consumer
        # SFA path is layer-wise on both sides.
        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", True)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        # Layer-reuse mate map. For a layer that time-multiplexes a shared HBM
        # slot, the "mate" is the slot's previous occupant whose KV D must finish
        # reading before this layer may overwrite the slot. ``prefetch_layer_map``
        # maps each reusing layer -> its mate; empty when layer reuse is disabled
        # (so the gate below becomes a no-op, matching the no-reuse behavior).
        self._reuse_mate_map: dict[int, int | None] = {}

        # Guard the asymmetric use_offload assumption (the launch scripts must
        # set it via --additional-config). Fail fast at startup rather than
        # producing confusing mid-run failures.
        #   P (producer)  : use_offload=false — producer exposes standard
        #                    paged KV, not the offload 5-tuple.
        #   D (consumer)  : use_offload=true  — drives the SFA offload code path
        #                    (5-tuple kv_cache, num_offloaded_blocks, LRU load).
        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        # AscendConfig may not be initialized yet at connector construction
        # time; init_ascend_config is idempotent (no-op if already done).
        init_ascend_config(vllm_config)
        ascend_use_offload = get_ascend_config().use_offload
        node = "P(producer)" if self.is_producer else "D(consumer)"
        print(
            f"[SFA-PD-LEARN][①SFAPD] SFAPDCpuOffloadConnector.__init__ "
            f"node={node} role={role} kv_role={self.kv_role} "
            f"use_offload={ascend_use_offload} use_layerwise={self.use_layerwise} "
            f"engine_id={self.engine_id}"
        )
        if self.is_producer:
            assert not ascend_use_offload, (
                "SFAPDCpuOffloadConnector producer (P) must run with "
                "use_offload=false (set --additional-config "
                "'{\"use_offload\": false}')."
            )
            print(
                "[SFA-PD-LEARN][①SFAPD] PD不对称断言通过: "
                "P 必须 use_offload=false（标准 paged HBM，非 5-tuple）"
            )
        if self.is_consumer:
            assert ascend_use_offload, (
                "SFAPDCpuOffloadConnector consumer (D) must run with "
                "use_offload=true (set --additional-config "
                "'{\"use_offload\": true, ...}')."
            )
            print(
                "[SFA-PD-LEARN][①SFAPD] PD不对称断言通过: "
                "D 必须 use_offload=true（5/6-tuple + CPU池 + LRU）"
            )

        if role == KVConnectorRole.SCHEDULER:
            # Producer scheduler prepares P-side metadata; consumer scheduler
            # allocates and advertises D-side CPU blocks.
            if self.is_producer:
                self.connector_scheduler = SFAPDProducerScheduler(vllm_config, kv_cache_config, str(self.engine_id))
                quad = "SCHEDULER×producer → SFAPDProducerScheduler"
            else:
                self.connector_scheduler = SFAPDCpuOffloadScheduler(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                )
                quad = "SCHEDULER×consumer → SFAPDCpuOffloadScheduler"
            self.connector_worker = None
        else:
            self.connector_scheduler = None
            if self.is_producer:
                self.connector_worker = SFAPDCpuOffloadProducerWorker(vllm_config, kv_cache_config, str(self.engine_id))
                quad = "WORKER×producer → SFAPDCpuOffloadProducerWorker"
            else:
                self.connector_worker = SFAPDCpuOffloadConsumerWorker(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                )
                quad = "WORKER×consumer → SFAPDCpuOffloadConsumerWorker"
        print(f"[SFA-PD-LEARN][①SFAPD] 四象限分支: {quad}")

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        print(
            f"[SFA-PD-LEARN][⑨释放] connector.request_finished: "
            f"req={request.request_id} → scheduler 释放 CPU tracker "
            f"（正常结束或 preempt）"
        )
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def set_gva_layerwise_reuse_plan(self, reuse_mate_map: dict[int, int | None]) -> None:
        """Install the AscendStore GVA reuse gates on the producer connector."""

        if self.is_producer:
            self._reuse_mate_map = reuse_mate_map
            print(
                f"[SFA-PD-LEARN][①SFAPD] set_gva_layerwise_reuse_plan(P): "
                f"mate_map={dict(reuse_mate_map)} "
                f"（复用层进场前须等 mate 的 READ_DONE）"
            )
        else:
            print(
                "[SFA-PD-LEARN][①SFAPD] set_gva_layerwise_reuse_plan(D): "
                "忽略（mate 门控只挂在 P producer）"
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        if self.is_consumer:
            return self.connector_worker.get_finished(finished_req_ids)
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Per-layer gate called before each layer's attention computation.

        D-side: no-op (SFA loads inside ``prepare_lru_resident_and_load``).
        P-side: **buffer-reuse gate** — before this layer overwrites a shared HBM
        slot, ensure D has finished reading the slot's *previous occupant* (the
        reuse mate) by waiting on the mate's send-done event. Waiting on this
        layer's OWN event would not protect the slot (that event tracks a
        different layer), so it must be the mate. Layers that do not reuse a
        slot (independent layers / first occupant) have no mate and skip.

        The per-layer send-done events are cleared when a READ_READY_BATCH is
        sent and set again by the pipelined MembPull send thread when READ_DONE
        arrives. Events are initially set, so the first occupant does not block.
        """
        if not self.is_producer:
            return
        match = _LAYER_IDX_RE.search(layer_name)
        if match is None:
            return
        layer_idx = int(match.group(1))
        mate = self._reuse_mate_map.get(layer_idx)
        if mate is None:
            print(
                f"[SFA-PD-LEARN][③SFAPD门控] wait_for_layer_load: "
                f"layer={layer_idx}({layer_name}) mate=None → skip "
                f"（独立层/槽内首占，无需等 READ_DONE）"
            )
            return  # independent / first occupant of its slot: nothing to gate.
        print(
            f"[SFA-PD-LEARN][③SFAPD门控] wait_for_layer_load: "
            f"layer={layer_idx}({layer_name}) → wait mate={mate} READ_DONE "
            f"（复用槽：D 必须先读走旧层再允许本层覆盖）"
        )
        self.wait_for_layer_send(mate)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        # SFA attention calls this every forward, including profiling / graph
        # capture where no per-step connector metadata is bound. Nothing to save
        # then; skip rather than trip _get_connector_metadata's assert.
        if not self.has_connector_metadata():
            print(
                f"[SFA-PD-LEARN][③SFAPD-READY] save_kv_layer: "
                f"layer={layer_name} skip（无 connector metadata，profiling/capture）"
            )
            return
        print(
            f"[SFA-PD-LEARN][③SFAPD-READY] save_kv_layer: "
            f"layer={layer_name} → ProducerWorker "
            f"（记 NPU event + 入队 SendTask → READ_READY）"
        )
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._get_connector_metadata())

    def wait_for_save(self):
        # P side has no worker wait_for_save hook (completion is tracked via
        # READ_DONE/layer_send_done_events). D side composes SFAKVOffloadWorker,
        # so keep its normal HBM->CPU save synchronization semantics.
        if self.is_consumer and self.connector_worker is not None:
            self.connector_worker.wait_for_save()

    # ------------------------------------------------------------------
    # SFA duck-typed hooks (attention/utils.py) — D side only
    # ------------------------------------------------------------------
    def set_req_ids(self, req_ids: list):
        if self.connector_worker is not None:
            self.connector_worker.set_req_ids(req_ids)

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        topk_indices: torch.Tensor,
        current_slots: torch.Tensor,
        req_ids: torch.Tensor,
        token_to_req: torch.Tensor | None = None,
        capturing: bool = False,
    ) -> bool:
        assert self.connector_worker is not None
        n = getattr(self, "_learn_8_lru_n", 0)
        if n < 3:
            self._learn_8_lru_n = n + 1
            print(
                f"[SFA-PD-LEARN][⑧转交] connector.prepare_lru_resident_and_load: "
                f"layer={layer_name} tokens={num_tokens} reqs={num_reqs} "
                f"→ worker （#{n + 1}/3）"
            )
        return self.connector_worker.prepare_lru_resident_and_load(
            layer_name,
            num_tokens,
            num_reqs,
            topk_indices,
            current_slots,
            req_ids,
            token_to_req,
            capturing,
        )

    # Phase 3: real per-req CPU-block count for the solution-1 threshold.
    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_num_cpu_blocks(req_ids)

    # P-side buffer-reuse gate: block until D has read a layer's source KV buffer,
    # so the buffer may be reused by a later layer.
    def wait_for_layer_send(self, layer_idx: int) -> None:
        worker = self.connector_worker
        if worker is None or not hasattr(worker, "wait_for_layer_send"):
            return
        print(
            f"[SFA-PD-LEARN][③SFAPD门控] wait_for_layer_send: "
            f"delegate → ProducerWorker.wait_for_layer_send(mate={layer_idx})"
        )
        worker.wait_for_layer_send(layer_idx)
