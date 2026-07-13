import torch

from vllm_ascend.core.kv_cache_interface import (
    compute_offload_sparse_c8_layout,
    make_offload_indexer_mla_spec,
    make_offload_main_mla_spec,
    offload_c8_indexer_page_size_bytes,
    offload_c8_main_page_size_bytes,
    offload_indexer_kernel_block_size,
)

# DSV3.2 / GLM5.2 A3 dims
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
MLA_BLOCK_SIZE = 128
INDEXER_BLOCK_SIZE = MLA_BLOCK_SIZE * 4
DSV32_C8_LAYOUT = compute_offload_sparse_c8_layout(
    KV_LORA_RANK,
    QK_ROPE_HEAD_DIM,
    INDEX_HEAD_DIM,
    torch.bfloat16,
    scale_dtype=torch.float16,
)


def test_offload_c8_main_page_size_bytes():
    page = offload_c8_main_page_size_bytes(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        torch.bfloat16,
        c8_layout=DSV32_C8_LAYOUT,
    )
    # spec_0: 128 * packed CKV 656 bytes/token.
    assert page == MLA_BLOCK_SIZE * 656


def test_offload_c8_indexer_page_size_bytes():
    page = offload_c8_indexer_page_size_bytes(
        INDEXER_BLOCK_SIZE,
        INDEX_HEAD_DIM,
        DSV32_C8_LAYOUT.indexer_pad_dim,
        torch.float16,
    )
    # spec_1 accounting: 512 * (128 + 34 + 2) bytes.
    assert page == INDEXER_BLOCK_SIZE * 164


def test_offload_c8_page_ratio_is_four_to_one():
    main_page = offload_c8_main_page_size_bytes(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        torch.bfloat16,
        c8_layout=DSV32_C8_LAYOUT,
    )
    indexer_page = offload_c8_indexer_page_size_bytes(
        INDEXER_BLOCK_SIZE,
        INDEX_HEAD_DIM,
        DSV32_C8_LAYOUT.indexer_pad_dim,
        torch.float16,
    )
    assert main_page == indexer_page


def test_offload_mla_spec_c8_page_bytes_per_token():
    spec = make_offload_main_mla_spec(
        block_size=MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        index_head_dim=INDEX_HEAD_DIM,
        c8_unified_pool=True,
        scale_dtype=torch.float16,
    )
    assert spec.page_size_bytes == offload_c8_main_page_size_bytes(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        torch.bfloat16,
        c8_layout=DSV32_C8_LAYOUT,
    )


def test_ascend_mla_offload_c8_page_bytes_per_token():
    spec = make_offload_indexer_mla_spec(
        block_size=INDEXER_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=INDEX_HEAD_DIM + DSV32_C8_LAYOUT.indexer_pad_dim,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        index_head_dim=INDEX_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        sparse_head_dim=(KV_LORA_RANK, QK_ROPE_HEAD_DIM, INDEX_HEAD_DIM),
        c8_unified_pool=True,
        scale_dtype=torch.float16,
    )
    assert spec.page_size_bytes == offload_c8_indexer_page_size_bytes(
        INDEXER_BLOCK_SIZE,
        INDEX_HEAD_DIM,
        DSV32_C8_LAYOUT.indexer_pad_dim,
        torch.float16,
    )


def test_offload_c8_indexer_block_multiplier_matches_token_ratio():
    # 4 main blocks (128 tokens each) == 1 indexer block (512 tokens)
    assert offload_indexer_kernel_block_size(
        MLA_BLOCK_SIZE,
        KV_LORA_RANK,
        INDEX_HEAD_DIM,
        c8_layout=DSV32_C8_LAYOUT,
    ) == MLA_BLOCK_SIZE * DSV32_C8_LAYOUT.page_block_multiplier


def test_compute_offload_sparse_c8_layout_dsv32_values():
    assert DSV32_C8_LAYOUT.packed_head_dim == 656
    assert DSV32_C8_LAYOUT.main_bytes_per_token == 656
    assert DSV32_C8_LAYOUT.indexer_bytes_per_token == 164
    assert DSV32_C8_LAYOUT.indexer_pad_dim == 34
    assert DSV32_C8_LAYOUT.page_block_multiplier == 4
