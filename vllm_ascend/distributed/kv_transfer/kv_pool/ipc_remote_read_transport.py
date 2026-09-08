# SPDX-License-Identifier: Apache-2.0
"""Experimental eager SFA reads from owner memory, with explicit read leases."""
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.distributed.kv_transfer.kv_pool.ipc_pull_transport import IpcPullKVPPTransport


class IpcRemoteReadKVPPTransport(IpcPullKVPPTransport):
    uses_remote_read = True

    def initialize_transport(self, caches: dict[str, Any], bundles: tuple, max_active_pages: int) -> None:
        super().initialize_transport(caches, bundles, max_active_pages)
        self.remote_views = {}
        self.read_epoch = 0
        self.active_read = None
        self.remote_tensor_reads = 0
        for bundle in bundles:
            layer = bundle[0]
            owner = self.owners[layer]
            views = {}
            for name in bundle:
                templates = (caches[name],) if isinstance(caches[name], torch.Tensor) else tuple(caches[name])
                for template, source in zip(templates, self.peer_metadata[owner]['regions'][name], strict=True):
                    rows = template.shape[0] // self.num_pages
                    if (template[:rows].numel() * template.element_size() != source['length']
                            or template.stride(0) * template.element_size() * rows != source['stride']):
                        raise ValueError('Remote and local cache geometry mismatch')
                    if owner == self.rank:
                        view = template
                    else:
                        mapping = self.mappings[owner, source['allocation']]
                        pointer = mapping.pointer + source['offset']
                        extent = (self.num_pages - 1) * source['stride'] + source['length']
                        if source['offset'] + extent > mapping.metadata['length']:
                            raise ValueError('Remote tensor exceeds imported allocation')
                        storage = torch_npu._C._construct_storage_from_data_pointer(pointer, self.pool.device, extent)
                        metadata = {'nbytes': extent, 'storage_offset': 0, 'npu_format': 2,
                                    'size': template.size(), 'stride': list(template.stride()),
                                    'dtype': template.dtype, 'data_ptr': pointer, 'device': self.pool.device,
                                    'layout': torch.strided, 'memory_format': torch.contiguous_format,
                                    'requires_grad': False}
                        view = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
                        if view.data_ptr() != pointer:
                            raise RuntimeError('Remote tensor unexpectedly materialized locally')
                    views[template.data_ptr()] = view
            self.remote_views[layer] = views
        # This baseline has no background prefetch collectives. Data remains
        # resident at the owner; no MTE copy or HCCL payload transfer is issued.
        self._gather('remote-views-ready')

    def prefetch(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError('Remote-read transport must not use copy-prefetch scheduling')

    def remote_read_cache(self, layer: str, caches: tuple) -> tuple:
        if self.active_read is not None:
            raise RuntimeError('Previous remote reader has not released its lease')
        # Current chunk writes stay in local caches. Publish after owner writes
        # complete, including DSA-CP all-gather and indexer/scale cache stores.
        torch.npu.current_stream().synchronize()
        self.read_epoch += 1
        generation = (self.read_epoch, layer)
        if any(value != generation for value in self._gather(generation)):
            raise RuntimeError('Remote-read generation mismatch')
        result = tuple(self.remote_views[layer][tensor.data_ptr()] for tensor in caches)
        self.active_read = generation
        if self.rank != self.owners[layer]:
            self.remote_tensor_reads += len(result)
        return result

    def finish_remote_reads(self) -> None:
        if self.active_read is None:
            return
        # No owner may overwrite a tail page until all remote kernels finish.
        torch.npu.current_stream().synchronize()
        generation = self.active_read
        if any(value != generation for value in self._gather(generation)):
            raise RuntimeError('Remote-read completion mismatch')
        self.active_read = None

    def close(self) -> None:
        if self.closed:
            return
        self.finish_remote_reads()
        print(f'KVPP_REMOTE_READ rank={self.rank} layers={self.read_epoch} '
              f'remote_tensor_reads={self.remote_tensor_reads} payload_copy_calls=0', flush=True)
        self.remote_views.clear()
        super().close()
