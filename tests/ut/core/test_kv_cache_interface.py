import torch

from vllm_ascend.core.kv_cache_interface import (
    OFFLOAD_C8_INDEXER_BLOCK_MULTIPLIER,
    OFFLOAD_C8_KV_PAD_DIM,
    AscendMLAAttentionSpec,
    OffloadMLAAttentionSpec,
    offload_c8_indexer_page_size_bytes,
    offload_c8_main_page_size_bytes,
)
from vllm_ascend.utils import calc_split_factor

# DSV3.2 / GLM5.2 A3 dims
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
INDEXER_PAD_DIM = INDEX_HEAD_DIM * QK_ROPE_HEAD_DIM // KV_LORA_RANK  # 16
MLA_BLOCK_SIZE = 128
MAIN_BYTES_PER_TOKEN = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2  # bf16 main MLA: 1152


def test_offload_c8_main_page_size_bytes():
    page = offload_c8_main_page_size_bytes(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        torch.bfloat16,
    )
    # spec_0: 128 * (512+64+8) * 2
    assert page == MLA_BLOCK_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM + OFFLOAD_C8_KV_PAD_DIM) * 2


def test_offload_c8_indexer_page_size_bytes():
    page = offload_c8_indexer_page_size_bytes(
        MLA_BLOCK_SIZE,
        INDEX_HEAD_DIM,
        INDEXER_PAD_DIM,
        torch.float16,
    )
    # spec_1: 128 * (128 int8 + 16 pad + 2 fp16)
    assert page == MLA_BLOCK_SIZE * (INDEX_HEAD_DIM + INDEXER_PAD_DIM + 2)


def test_offload_c8_page_ratio_is_eight_to_one():
    main_page = offload_c8_main_page_size_bytes(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        torch.bfloat16,
    )
    indexer_page = offload_c8_indexer_page_size_bytes(
        MLA_BLOCK_SIZE,
        INDEX_HEAD_DIM,
        INDEXER_PAD_DIM,
        torch.float16,
    )
    assert main_page == indexer_page * OFFLOAD_C8_INDEXER_BLOCK_MULTIPLIER


def test_offload_mla_spec_kv_pad_dim_page_size():
    spec = OffloadMLAAttentionSpec(
        block_size=MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        kv_pad_dim=OFFLOAD_C8_KV_PAD_DIM,
    )
    assert spec.page_size_bytes == offload_c8_main_page_size_bytes(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        torch.bfloat16,
    )


def test_ascend_mla_offload_unified_pool_c8_page_size():
    spec = AscendMLAAttentionSpec(
        block_size=MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=INDEX_HEAD_DIM + INDEXER_PAD_DIM,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        sparse_head_dim=(KV_LORA_RANK, QK_ROPE_HEAD_DIM, INDEX_HEAD_DIM),
        scale_dim=INDEXER_PAD_DIM,
        cache_sparse_c8=True,
        offload_unified_pool_c8=True,
    )
    assert spec.page_size_bytes == offload_c8_indexer_page_size_bytes(
        MLA_BLOCK_SIZE,
        INDEX_HEAD_DIM,
        INDEXER_PAD_DIM,
        torch.float16,
    )


def test_offload_c8_indexer_block_multiplier_matches_token_ratio():
    # 8 main blocks (128 tokens each) == 1 indexer block (1024 tokens)
    assert (
        MLA_BLOCK_SIZE * OFFLOAD_C8_INDEXER_BLOCK_MULTIPLIER
        == MLA_BLOCK_SIZE * KV_LORA_RANK // INDEX_HEAD_DIM * 2
    )


def test_offload_c8_pool_three_way_split_matches_main_page():
    """C8 offload allocates raw_k/raw_v/raw_scale from one pool page."""
    k_dim, v_dim = KV_LORA_RANK, QK_ROPE_HEAD_DIM
    factors = calc_split_factor([k_dim, v_dim, OFFLOAD_C8_KV_PAD_DIM])
    pool_bytes = offload_c8_main_page_size_bytes(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        torch.bfloat16,
    )
    k_bytes = int(pool_bytes // factors[0])
    v_bytes = int(pool_bytes // factors[1])
    scale_bytes = int(pool_bytes // factors[2])
    assert k_bytes + v_bytes + scale_bytes == pool_bytes
    assert k_bytes == MLA_BLOCK_SIZE * KV_LORA_RANK * 2
    assert v_bytes == MLA_BLOCK_SIZE * QK_ROPE_HEAD_DIM * 2
    assert scale_bytes == MLA_BLOCK_SIZE * OFFLOAD_C8_KV_PAD_DIM * 2
