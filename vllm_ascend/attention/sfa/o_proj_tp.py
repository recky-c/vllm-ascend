"""DSA-CP mixed-mode o_proj TP gather helpers.

In SFA DSA-CP mixed execution, the same model instance can run both
decode-only and prefill/mixed batches:
- Decode-only batches all-to-all the SFA output in the TP group, then
  run the original TP-sharded o_proj.
- Prefill/mixed batches produce SFA output that is not directly
  compatible with TP-sharded o_proj, so each rank all-gathers the TP
  o_proj shards and input-sharded quant params before running o_proj.

The original TP parameter storage remains the persistent source of
truth. The o_proj_tp_* tensors alias that storage, while the
o_proj_full_* tensors are temporary gather destinations reused across
forwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention.sfa.constants import O_PROJ_ACLNN_INPUT_PARAMS

if TYPE_CHECKING:
    from vllm_ascend.attention.sfa.impl import AscendSFAImpl


class OProjTPGather:
    """Owns the shared full-weight gather pool; mutates host impl attributes."""

    # Shared across layers with the same gather layout to save memory.
    pools: dict[tuple[str, int | None, torch.dtype, int, tuple[int, ...]], torch.Tensor] = {}

    @classmethod
    def init_params(cls, host: AscendSFAImpl) -> None:
        sample = host.o_proj.weight
        host.o_proj_full_weight_gather_dim = 1 if cls._is_o_proj_unquantized(host) else 0
        if host.o_proj_full_weight_gather_dim == 0:
            full_shape = (sample.shape[0] * host.tp_size, sample.shape[1])
            gather_shape = full_shape
        else:
            full_shape = (sample.shape[0], sample.shape[1] * host.tp_size)
            gather_shape = (sample.shape[1] * host.tp_size, sample.shape[0])
        # Main and MTP layers can use different quantized o_proj weight layouts,
        # so key the shared full-gather pool by gather dimension, dtype, and shape.
        pool_key = (
            sample.device.type,
            sample.device.index,
            sample.dtype,
            host.o_proj_full_weight_gather_dim,
            full_shape,
        )
        if pool_key not in cls.pools:
            cls.pools[pool_key] = torch.empty(gather_shape, dtype=sample.dtype, device=sample.device)
        host.o_proj_full_gather_pool = cls.pools[pool_key]
        if host.o_proj_full_weight_gather_dim == 0:
            host.o_proj_full_pool = host.o_proj_full_gather_pool
        else:
            host.o_proj_full_pool = host.o_proj_full_gather_pool.transpose(0, 1)

        # TP tensors alias the original parameter storage. The TP shard remains
        # the single source of truth; full-weight tensors below are temporary
        # gather destinations only.
        host.o_proj_tp_weight = host.o_proj.weight.detach()
        if host.o_proj_full_weight_gather_dim == 0:
            host.o_proj_tp_weight_gather_input = host.o_proj_tp_weight
        else:
            # Communication scratch only: all_gather_into_tensor concatenates on
            # dim0, while unquantized row-parallel o_proj is sharded on dim1.
            host.o_proj_tp_weight_gather_input = host.o_proj_tp_weight.transpose(0, 1).contiguous()
        host.o_proj_tp_aclnn_input_params = {}
        host.o_proj_full_aclnn_input_params = {}
        for param_name in O_PROJ_ACLNN_INPUT_PARAMS:
            param = getattr(host.o_proj, param_name, None)
            if param is None:
                continue
            host.o_proj_tp_aclnn_input_params[param_name] = param.detach()
            host.o_proj_full_aclnn_input_params[param_name] = param.repeat(host.tp_size)

        host.o_proj_tp_input_sharded_quant_params = {}
        host.o_proj_full_input_sharded_quant_params = {}
        for param_name, param in cls.iter_input_sharded_quant_params(host):
            host.o_proj_tp_input_sharded_quant_params[param_name] = param.detach()
            host.o_proj_full_input_sharded_quant_params[param_name] = torch.empty(
                (param.shape[0] * host.tp_size, *param.shape[1:]), dtype=param.dtype, device=param.device
            )

    @staticmethod
    def iter_input_sharded_quant_params(host: AscendSFAImpl):
        if not isinstance(host.o_proj, nn.Module):
            return
        for param_name, param in host.o_proj.named_parameters(recurse=False):
            if param_name == "weight" or param_name in O_PROJ_ACLNN_INPUT_PARAMS:
                continue
            if getattr(param, "input_dim", None) == 1:
                yield param_name, param

    @staticmethod
    def switch_params(host: AscendSFAImpl, params: dict[str, torch.Tensor]) -> None:
        for param_name, param in params.items():
            getattr(host.o_proj, param_name).set_(param)

    @staticmethod
    def get_linear_method(host: AscendSFAImpl):
        quant_method = host.o_proj.quant_method
        return getattr(quant_method, "quant_method", quant_method)

    @classmethod
    def _is_o_proj_unquantized(cls, host: AscendSFAImpl) -> bool:
        return isinstance(cls.get_linear_method(host), UnquantizedLinearMethod)

    @classmethod
    def apply_full_weight(cls, host: AscendSFAImpl, attn_output: torch.Tensor) -> torch.Tensor:
        return cls.get_linear_method(host).apply(host.o_proj, attn_output)

    @classmethod
    def handle_weight_switch_and_forward(
        cls,
        host: AscendSFAImpl,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_full_handle: torch.distributed.Work | None,
        o_proj_full_param_handles: list[torch.distributed.Work | None] | None,
        should_shard_weight: bool,
    ) -> tuple[torch.Tensor, bool]:
        """Switch o_proj between TP-mode and Full-mode, then run forward."""
        if should_shard_weight:
            if o_proj_full_handle is not None:
                o_proj_full_handle.wait()
            for handle in o_proj_full_param_handles or []:
                if handle is not None:
                    handle.wait()

            # Temporarily switch o_proj to the gathered full-weight view for
            # prefill/mixed DSA-CP, whose attention output is not TP-sharded.
            host.o_proj.weight.set_(host.o_proj_full_pool)
            cls.switch_params(host, host.o_proj_full_aclnn_input_params)
            cls.switch_params(host, host.o_proj_full_input_sharded_quant_params)
            output[...] = cls.apply_full_weight(host, attn_output)
            # Restore TP aliases so later decode batches keep using TP storage.
            host.o_proj.weight.set_(host.o_proj_tp_weight)
            cls.switch_params(host, host.o_proj_tp_aclnn_input_params)
            cls.switch_params(host, host.o_proj_tp_input_sharded_quant_params)

            return output, False

        # For decode scenario: all-to-all o_proj input activations.
        # Reshape: [batch * seq, tp_size, head_dim] -> [tp_size, batch * seq, head_dim]
        send = (
            attn_output.view(-1, host.tp_size, host.num_heads * host.v_head_dim)
            .permute(1, 0, 2)
            .reshape(-1, host.num_heads * host.v_head_dim)
        )

        attn_output = torch.empty_like(send)
        torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)

        return attn_output, True
