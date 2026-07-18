"""Named constants shared across the SFA attention backend modules."""

# token count limits within bmm_transpose operator
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024

O_PROJ_ACLNN_INPUT_PARAMS = (
    "aclnn_input_scale",
    "aclnn_input_scale_reciprocal",
    "aclnn_input_offset",
)
