"""Named constants shared across the SFA attention backend modules."""

# token count limits within bmm_transpose operator
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024

O_PROJ_ACLNN_INPUT_PARAMS = (
    "aclnn_input_scale",
    "aclnn_input_scale_reciprocal",
    "aclnn_input_offset",
)

# KV cache block size supported by the SFA kernels (see
# AscendSFABackend.get_supported_kernel_block_sizes).
SFA_KERNEL_BLOCK_SIZE = 128

# npu_fused_infer_attention_score with TND layout supports at most this many
# tokens per decode request (1 + num_speculative_tokens).
TND_LAYOUT_MAX_DECODE_THRESHOLD = 16

# (rows, cols) tiling block used by transdata() when packing MLAPO weights
# into the fractal NZ friendly layout.
TRANSDATA_BLOCK_SIZE = (16, 32)

# Hadamard rotation size applied to the C8 lightning-indexer q/k activations
# before int8 quantization. Must match the indexer head_dim (128).
HADAMARD_DIM = 128

# npu_dynamic_block_quant expects an integer enum for the destination dtype
# when quantizing to int8; ``1`` is the operator's int8 sentinel.
NPU_QUANT_DST_TYPE_INT8 = 1

# Default rmsnorm epsilon used by custom_kv_rmsnorm_rope when the caller does
# not provide the layer's variance_epsilon.
DEFAULT_KV_RMSNORM_EPSILON = 1e-5
