# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.pcp_manager import RankSegment
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import (
    AscendInputBatch,
    AscendInputBuffers,
)
from vllm_ascend.worker.v2.pcp.layout import count_decode_prefix_tokens


class HybridLinearBatchPartitioner:
    """Build the contiguous causal main view for Hybrid PCP."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        req_states: RequestState,
        max_num_reqs: int,
        max_num_tokens: int,
    ) -> None:
        self._vllm_config = vllm_config
        self._pcp_world_size = pcp_world_size
        self._pcp_rank = pcp_rank
        self._device = device
        self._req_states = req_states
        self._input_buffers = AscendInputBuffers(
            max_num_reqs,
            max_num_tokens,
            device,
        )

    def partition(
        self,
        global_batch: AscendInputBatch,
        lengths_by_rank: np.ndarray,
    ) -> tuple[AscendInputBatch, tuple[np.ndarray, ...]]:
        input_buffers = self._input_buffers
        num_reqs = global_batch.num_reqs
        is_prefilling = global_batch.is_prefilling_np
        num_scheduled_tokens = global_batch.num_scheduled_tokens
        decode_tokens = count_decode_prefix_tokens(
            num_scheduled_tokens,
            is_prefilling,
        )
        per_rank_tokens = lengths_by_rank.sum(axis=0)
        linear_num_tokens_padded = int(per_rank_tokens.max())
        if linear_num_tokens_padded > input_buffers.max_num_tokens:
            raise RuntimeError(
                "Hybrid PCP linear token count exceeds the MRV2 input "
                f"buffer: {linear_num_tokens_padded} > "
                f"{input_buffers.max_num_tokens}."
            )

        local_counts = lengths_by_rank[:, self._pcp_rank]
        rank_starts = lengths_by_rank[:, : self._pcp_rank].sum(axis=1)
        local_start_pos = global_batch.num_computed_tokens_np + np.where(
            is_prefilling,
            rank_starts,
            0,
        )

        prefill_padding = linear_num_tokens_padded - decode_tokens
        linear_prefill_by_rank: list[np.ndarray] = []
        gather_indices_by_rank: list[np.ndarray] = []
        for rank in range(self._pcp_world_size):
            prefill_map = np.full(
                prefill_padding,
                -1,
                dtype=np.int64,
            )
            gather_indices: list[np.ndarray] = []
            prefill_offset = 0
            for req_idx, count in enumerate(lengths_by_rank[:, rank]):
                count = int(count)
                if count == 0:
                    continue
                global_start = int(global_batch.query_start_loc_np[req_idx])
                if is_prefilling[req_idx]:
                    global_start += int(lengths_by_rank[req_idx, :rank].sum())
                    indices = np.arange(
                        global_start,
                        global_start + count,
                        dtype=np.int64,
                    )
                    prefill_map[prefill_offset : prefill_offset + count] = indices - decode_tokens
                    prefill_offset += count
                else:
                    indices = np.arange(
                        global_start,
                        global_start + count,
                        dtype=np.int64,
                    )
                gather_indices.append(indices)
            gather_indices_by_rank.append(
                np.concatenate(gather_indices) if gather_indices else np.empty(0, dtype=np.int64)
            )
            linear_prefill_by_rank.append(prefill_map)

        local_gather_indices = gather_indices_by_rank[self._pcp_rank]
        padded_gather_indices = np.zeros(
            linear_num_tokens_padded,
            dtype=np.int64,
        )
        padded_gather_indices[: local_gather_indices.shape[0]] = local_gather_indices
        gather_idx = async_copy_to_gpu(
            padded_gather_indices,
            device=self._device,
        )
        torch.index_select(
            global_batch.input_ids,
            0,
            gather_idx,
            out=input_buffers.input_ids[:linear_num_tokens_padded],
        )

        query_start_loc_np = np.empty(
            input_buffers.max_num_reqs + 1,
            dtype=np.int32,
        )
        query_start_loc_np[0] = 0
        np.cumsum(
            local_counts,
            out=query_start_loc_np[1 : num_reqs + 1],
        )
        query_start_loc_np[num_reqs + 1 :] = local_counts.sum()
        async_copy_to_gpu(
            query_start_loc_np,
            out=input_buffers.query_start_loc,
        )
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        prepare_pos_seq_lens(
            global_batch.idx_mapping,
            query_start_loc,
            async_copy_to_gpu(local_start_pos, device=self._device),
            input_buffers.positions,
            input_buffers.seq_lens[:num_reqs],
        )
        is_padding = input_buffers.is_padding[:linear_num_tokens_padded]
        local_num_tokens = int(local_counts.sum())
        is_padding[:local_num_tokens].fill_(False)
        is_padding[local_num_tokens:].fill_(True)
        if linear_num_tokens_padded > local_num_tokens:
            input_buffers.input_ids[local_num_tokens:linear_num_tokens_padded].zero_()
            input_buffers.positions[local_num_tokens:linear_num_tokens_padded].zero_()

        # Sampling uses the restored global batch. Keep a valid local value for
        # model code that inspects logits_indices before restoration.
        logits_indices = combine_sampled_and_draft_tokens(
            input_buffers.input_ids,
            global_batch.idx_mapping,
            self._req_states.last_sampled_tokens,
            query_start_loc,
            input_buffers.seq_lens[:num_reqs],
            self._req_states.prefill_len.gpu,
            self._req_states.draft_tokens,
            global_batch.cu_num_logits,
            global_batch.cu_num_logits_np[-1],
            1,
        )

        local_prefill_len = global_batch.prefill_len_np
        local_computed_prefill = np.minimum(
            local_start_pos,
            local_prefill_len,
        )
        local_is_prefilling = local_computed_prefill < local_prefill_len
        seq_lens_np = local_start_pos + local_counts
        input_buffers.seq_lens_np[:num_reqs] = seq_lens_np
        input_buffers.seq_lens_np[num_reqs:] = 0
        attn_state = build_attn_state(
            self._vllm_config,
            seq_lens_np,
            num_reqs,
            local_counts,
            local_counts,
        )

        local_batch = replace(
            global_batch,
            num_reqs_after_padding=num_reqs,
            num_scheduled_tokens=local_counts,
            num_tokens=local_num_tokens,
            num_tokens_after_padding=linear_num_tokens_padded,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np[: num_reqs + 1],
            seq_lens=input_buffers.seq_lens[:num_reqs],
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=local_start_pos,
            num_computed_prefill_tokens_np=local_computed_prefill,
            is_prefilling_np=local_is_prefilling,
            input_ids=input_buffers.input_ids[:linear_num_tokens_padded],
            positions=input_buffers.positions[:linear_num_tokens_padded],
            is_padding=is_padding,
            logits_indices=logits_indices,
            prompt_lens=None,
            seq_lens_np=input_buffers.seq_lens_np[:num_reqs],
            attn_state=attn_state,
        )
        return local_batch, tuple(linear_prefill_by_rank)


def build_fa_prefill_maps(
    *,
    global_batch: AscendInputBatch,
    decode_tokens: int,
    fa_tokens_padded: int,
    pcp_world_size: int,
    get_rank_segments: Callable[
        [
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        list[RankSegment],
    ],
) -> tuple[np.ndarray, ...]:
    prefill_padding = fa_tokens_padded - decode_tokens
    maps: list[np.ndarray] = []
    for rank in range(pcp_world_size):
        segments = get_rank_segments(
            rank,
            global_batch.num_scheduled_tokens,
            global_batch.num_computed_tokens_np,
            global_batch.is_prefilling_np,
            global_batch.query_start_loc_np,
        )
        mapping = np.full(prefill_padding, -1, dtype=np.int64)
        for segment in segments:
            if not global_batch.is_prefilling_np[segment.global_batch_req_idx]:
                continue
            local_start = segment.rank_local_batch_slice.start - decode_tokens
            local_stop = local_start + segment.num_tokens
            mapping[local_start:local_stop] = np.arange(
                segment.global_batch_slice.start - decode_tokens,
                segment.global_batch_slice.stop - decode_tokens,
                dtype=np.int64,
            )
        maps.append(mapping)
    return tuple(maps)
