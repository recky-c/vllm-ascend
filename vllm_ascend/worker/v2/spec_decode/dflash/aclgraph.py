from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (  # type: ignore[import-not-found]
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import FIAParamProvider
from vllm_ascend.compilation.acl_graph import (
    set_draft_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.utils import use_updatable_graph
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes, model_capture_wrapper
from vllm_ascend.worker.v2.utils import communicator_switch


class DFlashAclGraphManager(DFlashCudaGraphManager):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        speculator: Any = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
        )

        # It is set by AscendDFlashSpeculator.init_cudagraph_manager after creation,
        # because upstream's init_cudagraph_manager creates the manager without it.
        self.speculator = speculator
        # The attention backend keys its per-size graph params by the actual
        # captured token counts (rounded up to decode_query_len when using
        # speculative decoding), so derive them from the capture descriptors
        # instead of the raw config sizes.
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        # DFlash's parallel drafting forward has its own dedicated draft graph
        # path, independent of Eagle's prefill/decode split, so it always uses
        # the default draft params bucket (is_draft_model_prefill stays False in
        # both capture and replay to keep them consistent).
        if super().needs_capture():
            set_draft_graph_params(self.capture_sizes)

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool] = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture ACL graphs for DFlash."""
        with communicator_switch(), model_capture_wrapper(self.speculator, False):
            super().capture(
                forward_fn,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                causal,
                progress_bar_desc,
            )

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        if (
            getattr(self.speculator, "_deferred_metadata_enabled", False)
            and getattr(self.speculator, "_deferred_draft_attn_metadata", None) is not None
        ):
            return self._replay_with_deferred_metadata(desc)
        attn_backend = list(self.speculator.attn_backends.values())[0]
        draft_attn_metadatas = self.speculator.build_draft_attn_metadatas(
            desc.num_reqs,
            self.speculator.input_batch.seq_lens_cpu_upper_bound,
        )
        if use_updatable_graph(attn_backend):
            if getattr(self.speculator, "_exact_draft_metadata_active", False):
                self._prepare_neutral_tasks(desc)
            return self._updatable_graph_replay(desc, draft_attn_metadatas)
        else:
            # This will be removed once the refactoring is fully complete.
            return self._graph_replay(desc, attn_backend, num_tokens, draft_attn_metadatas)

    def _prepare_neutral_tasks(self, desc) -> None:
        if not getattr(self.speculator, "_deferred_metadata_enabled", False):
            return
        graph = self.graphs[desc]
        if not isinstance(graph, UpdatableGraph) or not graph.tasks:
            return
        if getattr(graph, "_dflash_neutral_tasks", None) is not None:
            return
        tasks = []
        width = self.speculator.num_query_per_req
        for task in graph.tasks:
            kwargs = task.kwargs
            provider = task.provider
            if (
                type(provider) is not FIAParamProvider
                or not provider.is_draft_model
                or provider.is_non_pa
                or task.operation != torch_npu.npu_fused_infer_attention_score.out
                or kwargs.get("input_layout") != "TND"
                or not isinstance(kwargs.get("block_table"), torch.Tensor)
                or not kwargs.get("block_size")
                or kwargs["block_size"] < width
            ):
                return
            neutral_kv = {}
            for name in ("key", "value"):
                caches = kwargs.get(name)
                if isinstance(caches, torch.Tensor):
                    cache = caches
                elif isinstance(caches, (tuple, list)) and len(caches) == 1:
                    cache = caches[0]
                else:
                    return
                neutral = self._make_neutral_cache(cache)
                if neutral is None:
                    return
                neutral_kv[name] = neutral if isinstance(caches, torch.Tensor) else [neutral]
            tasks.append(
                task.bind(
                    {
                        **neutral_kv,
                        "block_table": torch.zeros_like(kwargs["block_table"]),
                        "actual_seq_lengths": [(row + 1) * width for row in range(desc.num_reqs)],
                        "actual_seq_lengths_kv": [width] * desc.num_reqs,
                    }
                )
            )
        # Keep graph-owned independent cache/table tensors alive before replay.
        # Unsupported task kinds keep the ordinary metadata-first order.
        graph._dflash_neutral_tasks = tuple(tasks)

    @staticmethod
    def _make_neutral_cache(cache: Any) -> torch.Tensor | None:
        """Own one zero page without changing a live hybrid cache's strides."""
        if (
            not isinstance(cache, torch.Tensor)
            or cache.ndim not in (3, 4)
            or cache.shape[0] < 1
            or cache.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or torch_npu.get_npu_format(cache) != 2  # ND; packed/NZ layouts keep the ordinary order.
        ):
            return None
        payload = 1
        for size, stride in zip(reversed(cache.shape[1:]), reversed(cache.stride()[1:])):
            if size < 1 or stride != payload:
                return None
            payload *= size
        page_stride = cache.stride(0)
        if page_stride < payload:
            return None
        # Combined K/V and padded hybrid pages have a larger outer stride.
        # Reserve that entire page in separate storage, including its gap;
        # block zero is the only page referenced by the neutral block table.
        backing = torch.zeros(page_stride, dtype=cache.dtype, device=cache.device)
        return backing.as_strided((1, *cache.shape[1:]), cache.stride(), storage_offset=0)

    def _replay_with_deferred_metadata(self, desc):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        neutral_tasks = graph._dflash_neutral_tasks
        # The input dependency must precede replay. Waiting on the replay
        # itself would cycle at its first external FIA update event.
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        try:
            with torch.npu.stream(self.update_stream):
                metadata = self.speculator.build_draft_attn_metadatas(
                    desc.num_reqs,
                    self.speculator.input_batch.seq_lens_cpu_upper_bound,
                )
                resolved = graph.resolve_tasks(ContextSource(metadata[0]))
                graph.update(self.update_stream, resolved)
        except Exception as error:
            # Release every waiting task with legal independent zero KV. Never
            # let old lengths read newly rebound physical tables on failure.
            try:
                graph.update(self.update_stream, neutral_tasks)
                torch.npu.current_stream().synchronize()
            except Exception as cleanup_error:
                error.add_note(f"DFlash graph cleanup also failed: {cleanup_error!r}")
            self.speculator._deferred_draft_attn_metadata = None
            self.speculator._draft_attn_metadata_for_graph = None
            self.speculator._draft_metadata_template = None
            raise
        return ret

    def _graph_replay(self, desc, attn_backend, num_tokens, draft_attn_metadatas):
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)

        with set_forward_context(
            self.speculator.model_state.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,  # Full graph model don't need batch_descriptor
            slot_mapping=None,
        ):
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = False
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return ret

    def _updatable_graph_replay(self, desc, draft_attn_metadatas):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        resolved_tasks = graph.resolve_tasks(ContextSource(draft_attn_metadatas[0]))
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        graph.update(self.update_stream, resolved_tasks)
        return ret
