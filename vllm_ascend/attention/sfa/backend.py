import vllm.envs as envs_vllm
from vllm.v1.attention.backend import AttentionBackend  # type: ignore

from vllm_ascend.attention.sfa.constants import SFA_KERNEL_BLOCK_SIZE
from vllm_ascend.attention.utils import enable_cp
from vllm_ascend.utils import enable_sfa_dcp_replicated_indexer


class AscendSFABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        if enable_sfa_dcp_replicated_indexer():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder

            return AscendSFADCPMetadataBuilder
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPMetadataBuilder

            return AscendSFACPMetadataBuilder
        from vllm_ascend.attention.sfa.builder import AscendSFAMetadataBuilder

        return AscendSFAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls():
        if enable_sfa_dcp_replicated_indexer():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl

            return AscendSFADCPImpl
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPImpl

            return AscendSFACPImpl
        from vllm_ascend.attention.sfa.impl import AscendSFAImpl

        return AscendSFAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [SFA_KERNEL_BLOCK_SIZE]
