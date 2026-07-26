# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.utils import enable_sp, enable_sp_by_pass
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


def validate_hybrid_batch(input_batch: AscendInputBatch) -> None:
    continued_prefill = input_batch.is_prefilling_np & (input_batch.num_computed_tokens_np > 0)
    if np.any(continued_prefill):
        request_indices = np.flatnonzero(continued_prefill).tolist()
        raise NotImplementedError(
            "MRV2 Hybrid PCP first stage does not support continued "
            "prefill or prefix-cache hits; affected batch rows: "
            f"{request_indices}."
        )
    empty_prefill = input_batch.is_prefilling_np & (input_batch.num_scheduled_tokens <= 0)
    if np.any(empty_prefill):
        request_indices = np.flatnonzero(empty_prefill).tolist()
        raise RuntimeError(
            "Hybrid PCP requires every scheduled prefill request to own at "
            f"least one token; affected batch rows: {request_indices}."
        )


def validate_hybrid_pcp_config(
    vllm_config: VllmConfig,
    supports_mm_inputs: bool,
) -> None:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config

    unsupported: list[str] = []
    if parallel_config.pipeline_parallel_size > 1:
        unsupported.append("pipeline parallelism")
    if parallel_config.decode_context_parallel_size > 1:
        unsupported.append("decode context parallelism")
    if parallel_config.enable_expert_parallel:
        unsupported.append("expert parallelism")
    if getattr(parallel_config, "data_parallel_size", 1) > 1:
        unsupported.append("data parallelism")
    if enable_sp(vllm_config) or enable_sp_by_pass():
        unsupported.append("FlashComm1 / sequence parallelism")
    if model_config.is_encoder_decoder:
        unsupported.append("encoder-decoder models")
    if supports_mm_inputs:
        unsupported.append("multimodal inputs")
    if vllm_config.lora_config is not None:
        unsupported.append("LoRA")
    if vllm_config.speculative_config is not None:
        unsupported.append("speculative decoding")
    if scheduler_config.enable_chunked_prefill:
        unsupported.append("chunked prefill")
    if cache_config.enable_prefix_caching:
        unsupported.append("prefix caching")
    if cache_config.mamba_cache_mode != "none":
        unsupported.append(f"mamba_cache_mode={cache_config.mamba_cache_mode}")
    if model_config.quantization is not None:
        unsupported.append(f"quantization={model_config.quantization}")
    if model_config.get_sliding_window() is not None:
        unsupported.append("sliding-window attention")
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is not None and getattr(kv_transfer_config, "kv_connector", None) is not None:
        unsupported.append("KV transfer / disaggregated serving")
    if vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
        unsupported.append("full CUDA/ACL graphs")
    if model_config.dtype not in (torch.float16, torch.bfloat16):
        unsupported.append(f"dtype={model_config.dtype}")

    if unsupported:
        raise NotImplementedError(
            "MRV2 Hybrid PCP first-stage capability does not support: " + ", ".join(unsupported) + "."
        )

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
    ):
        raise NotImplementedError("MRV2 Hybrid PCP supports eager and PIECEWISE graph modes only.")
