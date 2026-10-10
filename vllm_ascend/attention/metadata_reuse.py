# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def tensor_view_key(tensor: torch.Tensor | None) -> tuple | None:
    """Identify a read-only input view within a single metadata build.

    The caller must discard its cache after each build: a stable address does
    not imply stable contents across decoding steps. Include shape and strides
    so different views of the same allocation cannot accidentally share data.
    """
    if tensor is None:
        return None
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device)
