# SPDX-License-Identifier: Apache-2.0
"""Correctness-only worker extension; never load this during TTFT runs."""

import functools
import json
import os
import threading
import time
from pathlib import Path

from vllm_ascend.core.kv_cache_placement import build_kvpp_layer_layout
from vllm_ascend.worker.worker import NPUWorker


class BroadcastAuditWorker(NPUWorker):
    def init_device(self):
        super().init_device()
        self._audit_lock = threading.Lock()
        self._audit_forward = -1
        root = Path("/home/recky/a3-kv-transfer-20260907/phase3/results/audit")
        root.mkdir(parents=True, exist_ok=True)
        self._audit_path = root / f"{time.time_ns()}-{os.getpid()}-rank{self.rank}.jsonl"
        print(f"KVPP audit: {self._audit_path}", flush=True)
        self._audit("worker", use_v2_model_runner=self.use_v2_model_runner)
        if not self.use_v2_model_runner:
            runner = self.model_runner
            original = runner._prepare_inputs

            @functools.wraps(original)
            def prepare_inputs(*args, **kwargs):
                result = original(*args, **kwargs)
                batch = runner.input_batch
                self._audit(
                    "inputs",
                    req_ids=list(batch.req_ids),
                    computed=batch.num_computed_tokens_cpu[: batch.num_reqs].tolist(),
                )
                return result

            runner._prepare_inputs = prepare_inputs

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if self.use_v2_model_runner:
            state = self.model_runner.model_state
            original = state.prepare_attn

            @functools.wraps(original)
            def prepare_attn(input_batch, *args, **kwargs):
                self._audit(
                    "inputs",
                    computed=input_batch.num_computed_tokens_np[: input_batch.num_reqs].tolist(),
                    dummy=state.kvpp_is_dummy_run,
                )
                return original(input_batch, *args, **kwargs)

            state.prepare_attn = prepare_attn
        return result

    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        plan = self._kvpp_cache_allocation_plan
        if plan is not None:
            for name, bundle in plan.layer_bundles.items():
                layout, size = build_kvpp_layer_layout(bundle, plan.tensor_specs, kv_cache_config.num_blocks)
                component_bytes = sum(length for parts in layout.values() for _, length in parts)
                self._audit(
                    "layout",
                    layer=name,
                    owner=plan.layer_owner_ranks.get(name),
                    num_blocks=kv_cache_config.num_blocks,
                    layer_span_bytes=size,
                    component_bytes=component_bytes,
                    padding_bytes=size - component_bytes,
                    components={
                        cache: [
                            dict(dtype=str(spec.dtype), offset=offset, bytes=length)
                            for spec, (offset, length) in zip(plan.tensor_specs[cache], parts)
                        ]
                        for cache, parts in layout.items()
                    },
                )
        runtime = self.model_runner.kvpp
        if not self.use_v2_model_runner:
            runtime = runtime._kvpp_runtime
        original = runtime.prepare_forward

        @functools.wraps(original)
        def prepare_forward(has_history):
            self._audit_forward += 1
            self._audit("prepare", has_history=has_history)
            return original(has_history)

        runtime.prepare_forward = prepare_forward
        if runtime.scheduler is not None:
            transport = runtime.scheduler.transport
            original_prefetch = transport.prefetch

            @functools.wraps(original_prefetch)
            def prefetch(layer_name, *args, **kwargs):
                buffers = transport._layer_buffers[layer_name]
                self._audit(
                    "broadcast",
                    layer=layer_name,
                    calls=len(buffers),
                    payload_bytes=sum(buffer.numel() * buffer.element_size() for buffer in buffers),
                )
                return original_prefetch(layer_name, *args, **kwargs)

            transport.prefetch = prefetch

    def _audit(self, event, **values):
        with self._audit_lock, self._audit_path.open("a") as output:
            forward = self._audit_forward + int(event == "inputs")
            output.write(json.dumps(dict(event=event, rank=self.rank, forward=forward, **values)) + "\n")
