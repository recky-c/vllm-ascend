# A3 KVPP implementation archive — 2026-09-08

This branch preserves the implementations and recorded hardware results, not a production release. The three archive commits retain the tested milestones independently; the latest tree includes all four selectable transport variants.

| Variant | Additional configuration | Implementation |
|---|---|---|
| MTE Pull | `enable_kvpp=true`, `kvpp_transport=ipc_pull` | [ipc_pull_transport.py](../../vllm_ascend/distributed/kv_transfer/kv_pool/ipc_pull_transport.py) |
| Packed active-page Broadcast | `enable_kvpp=true`, `kvpp_transport=ipc_broadcast`, `kvpp_broadcast_full_pages=false` | [ipc_broadcast_transport.py](../../vllm_ascend/distributed/kv_transfer/kv_pool/ipc_broadcast_transport.py) |
| Per-tensor full-page Broadcast | `enable_kvpp=true`, `kvpp_transport=ipc_broadcast`, `kvpp_broadcast_full_pages=true` | [ipc_fullpage_broadcast_transport.py](../../vllm_ascend/distributed/kv_transfer/kv_pool/ipc_fullpage_broadcast_transport.py) |
| Direct remote KV read | `enable_kvpp=true`, `kvpp_transport=ipc_pull`, `kvpp_remote_read=true` | [ipc_remote_read_transport.py](../../vllm_ascend/distributed/kv_transfer/kv_pool/ipc_remote_read_transport.py) |

Unset experimental booleans default to false. Presets also supply the existing IPC kernel library path and tested environment. Direct remote read selects a different class despite reusing the `ipc_pull` configuration family; it does not execute the Pull payload-copy path.

**Per-layer contiguous storage with exactly one Broadcast per layer is planned and is not implemented here.** The tested full-page variant performs one Broadcast per cache tensor, potentially 1–3 calls per layer.

## Results and validation

- [All schemes and configuration caveats](ALL_SCHEMES_TTFT.md)
- [Off / Pull / packed Broadcast](RESULTS_ONLINE_64K128K.md)
- [Full-page versus active-page Broadcast](RESULTS_FULLPAGES_64K128K.md)
- [Direct remote read versus Broadcast](RESULTS_REMOTE_READ_64K128K.md)
- [Named benchmark presets](presets/)

Latest measured direct-read TTFT: 59.246 / 129.453 seconds at 64K / 128K. Its paired Broadcast reference: 8.602 / 18.402 seconds. Full-page TTFT from the preceding comparison: 8.545 / 18.358 seconds. The reports retain individual samples, earlier results and limitations; small cross-run differences do not establish an advantage.

Tests used GLM-5.2-w4a8c8 on A3, 8 physical cards / 16 chips, TP16, DP1, EP, DSA-CP, asynchronous scheduling and chunk32K. The deprecated FlashComm flag was set but the deployed SP path was not active at DP1. Requests generated one token; matching first tokens are not a full accuracy evaluation.

Full-page validation: 43 unit tests and 10800 NPU data checks in 2/16-rank lifecycle tests. Direct-read validation: 38 unit tests and 3552 NPU data checks. These are previously completed hardware tests, not reruns performed while archiving.

## Source and dependencies

- [MTE copy source and CMake entry](../../tools/kvpp_ipc_copy/)
- [IPC allocation, mapping and lifecycle](../../vllm_ascend/distributed/kv_transfer/kv_pool/)
- [Scheduler integration](../../vllm_ascend/worker/v2/kvpp.py)
- [Hardware lifecycle tests](../../tests/e2e/kvpp/)

The snapshots were based on commit `1c3733d577f276fd5e4b7f0c3ff61cb60c56b309`. `base-source-sha256.json`, `full-source-sha256.json` and `direct-source-sha256.json` identify the respective local tested snapshots with CRLF normalized to LF. Source semantics, including experimental code and remaining cleanup work, are preserved.

The C++ copy library requires CANN / AscendC and explicit CANN path and SOC version. Tests used CANN 9.1.0, Ascend910_9362 and torch_npu 2.10.0.post4. The measured `libkv_copy.so` SHA256 was `c82e5f1831a42db7456eb2914218569a891b5fce1b0ad04b9dc783b8274523c7`.

The full model runs reused the existing vLLM runtime and matching custom OPP installation named in the presets. They did not rebuild every native operator from this tree. Model weights, binaries, credentials and machine-session state are not included. The full-page and remote-read prototypes still inherit initialization dependencies on the existing IPC/copy infrastructure.
