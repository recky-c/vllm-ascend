# SPDX-License-Identifier: Apache-2.0
"""Experimental A3 P2P allocation pool. Allocate before binding KV views.

The tensor constructors below are torch_npu private APIs, verified only with
torch_npu 2.10.0.post4. An explicit pool owns allocations; tensor wrappers do
not free them. The caller must drain every user before closing the pool.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.kvpp_memory import DEFAULT_IPC_POOL_BUDGET_BYTES

P2P_ALIGNMENT = 2 * 1024 * 1024
ACL_MEM_MALLOC_HUGE_ONLY_P2P = 4
IPC_ENABLE_PEER_ACCESS = 1
EXPORT_KEY_BYTES = 65


class IpcRuntime:
    def __init__(self) -> None:
        self.library = ctypes.CDLL("libascendcl.so")
        signatures = {
            "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
            "aclrtFree": [ctypes.c_void_p],
            "aclrtDeviceGetBareTgid": [ctypes.POINTER(ctypes.c_int32)],
            "aclrtIpcMemGetExportKey": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
                                      ctypes.c_size_t, ctypes.c_uint64],
            "aclrtIpcMemSetImportPid": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t],
            "aclrtIpcMemImportByKey": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_uint64],
            "aclrtIpcMemClose": [ctypes.c_char_p],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int

    def call(self, name: str, *arguments: Any) -> None:
        result = getattr(self.library, name)(*arguments)
        if result != 0:
            raise RuntimeError(f"{name} failed: {result}")

    def pid(self) -> int:
        result = ctypes.c_int32()
        self.call("aclrtDeviceGetBareTgid", ctypes.byref(result))
        return result.value


@dataclass
class IpcAllocation:
    runtime: IpcRuntime
    identifier: int
    epoch: int
    pointer: int
    length: int
    tensor: Any
    export_key: bytes | None = None

    def export(self, reader_pids: list[int]) -> dict[str, Any]:
        if self.export_key is not None:
            raise RuntimeError("Allocation is already exported")
        key = ctypes.create_string_buffer(EXPORT_KEY_BYTES)
        self.runtime.call("aclrtIpcMemGetExportKey", self.pointer, self.length, key, len(key), 0)
        self.export_key = key.value
        pids = (ctypes.c_int32 * len(reader_pids))(*reader_pids)
        self.runtime.call("aclrtIpcMemSetImportPid", self.export_key, pids, len(pids))
        return {"allocation": self.identifier, "epoch": self.epoch,
                "length": self.length, "key": self.export_key.decode()}


class IpcAllocationPool:
    def __init__(self, device: torch.device, epoch: int = 1,
                 budget_bytes: int = DEFAULT_IPC_POOL_BUDGET_BYTES) -> None:
        self.device = device
        self.epoch = epoch
        self.budget_bytes = budget_bytes
        self.runtime = IpcRuntime()
        self.allocations: list[IpcAllocation] = []
        self.closed = False
        self.active_transports = 0

    def allocate(self, num_bytes: int) -> torch.Tensor:
        if self.closed or num_bytes <= 0:
            raise ValueError("Invalid allocation or closed pool")
        length = (num_bytes + P2P_ALIGNMENT - 1) // P2P_ALIGNMENT * P2P_ALIGNMENT
        retained_bytes = sum(allocation.length for allocation in self.allocations)
        if retained_bytes + length > self.budget_bytes:
            raise MemoryError(f"IPC pool budget exceeded before aclrtMalloc: requested={length}, "
                              f"retained={retained_bytes}, budget={self.budget_bytes}")
        pointer = ctypes.c_void_p()
        self.runtime.call("aclrtMalloc", ctypes.byref(pointer), length, ACL_MEM_MALLOC_HUGE_ONLY_P2P)
        try:
            storage = torch_npu._C._construct_storage_from_data_pointer(pointer.value, self.device, length)
            metadata = {"nbytes": length, "storage_offset": 0, "npu_format": 2,
                        "size": torch.Size([length]), "stride": [1], "dtype": torch.int8,
                        "data_ptr": pointer.value, "device": self.device, "layout": torch.strided,
                        "memory_format": torch.contiguous_format, "requires_grad": False}
            tensor = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
        except Exception:
            self.runtime.call("aclrtFree", pointer)
            raise
        self.allocations.append(IpcAllocation(self.runtime, len(self.allocations), self.epoch,
                                              pointer.value, length, tensor))
        return tensor[:num_bytes].zero_()

    def resolve(self, tensor: torch.Tensor, extent: int) -> tuple[IpcAllocation, int]:
        pointer = tensor.data_ptr()
        for allocation in self.allocations:
            offset = pointer - allocation.pointer
            if 0 <= offset < allocation.length and 0 <= extent <= allocation.length - offset:
                return allocation, offset
        raise RuntimeError("KV tensor is not backed by this IPC allocation pool; refusing a staging fallback")

    def close_exports(self) -> None:
        for allocation in self.allocations:
            if allocation.export_key is not None:
                self.runtime.call("aclrtIpcMemClose", allocation.export_key)
                allocation.export_key = None

    def close(self) -> None:
        if self.closed:
            return
        if self.active_transports:
            raise RuntimeError("Drain and close every IPC transport before freeing its allocation pool")
        if any(item.export_key is not None for item in self.allocations):
            raise RuntimeError("Close remote importers and export keys before freeing KV")
        # External attention/connector users must already have been drained.
        torch.npu.synchronize()
        for allocation in self.allocations:
            allocation.tensor = None
            self.runtime.call("aclrtFree", allocation.pointer)
        self.allocations.clear()
        self.closed = True


class IpcMapping:
    def __init__(self, runtime: IpcRuntime, metadata: dict[str, Any]) -> None:
        self.runtime = runtime
        self.metadata = metadata
        self.key = metadata["key"].encode()
        pointer = ctypes.c_void_p()
        runtime.call("aclrtIpcMemImportByKey", ctypes.byref(pointer), self.key, IPC_ENABLE_PEER_ACCESS)
        self.pointer = pointer.value
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.runtime.call("aclrtIpcMemClose", self.key)
            self.closed = True
