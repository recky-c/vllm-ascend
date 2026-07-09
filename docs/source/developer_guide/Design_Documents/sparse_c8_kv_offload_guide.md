# Sparse C8 / Lightning Indexer 与 KV Offload 设计开发指南

本文说明在 vLLM Ascend 上如何通过 **KV Offload + LIC8** 支撑 GLM5.2 等 DSA 模型长序列 decode。

- 分支：**`feat/lic8-kv-offload`**
- 基线分支：[`feat/sfa-offload-layerwise-reuse`](https://github.com/ader47/vllm-ascend/commits/feat/sfa-offload-layerwise-reuse/)

---
## 1. 架构原则

长序列 decode 的显存瓶颈在 **main MLA KV**（`kv_lora` + `k_rope`），不在 indexer。方案把两类数据拆开处理：

| 数据 | 存放 | 作用 |
|------|------|------|
| Indexer `k_li`（LIC8 时 + `k_li_scale`） | **HBM 全量常驻** | Lightning Indexer 粗筛 TopK，充当「目录」 |
| Main MLA KV | **前缀 CPU pool，尾部 HBM** | 真正参与 Sparse FA 的 K/V |
| `topk_buffer_k/v` | **HBM 工作区** | TopK miss 时 CPU→HBM 的 staging，不是历史 KV |

三条硬约束（LIC8 适配不得破坏）：

1. Indexer 不进 CPU pool、不走 D2H、不进 LRU resident。
2. 只有 main MLA 参与 offload / H2D。
3. 从 CPU 拉回哪些 main KV，由 indexer TopK + `num_offloaded_blocks` 决定。

LIC8 的意义：indexer 压到 8bit+scale 后，全量历史 index 仍能放进 HBM，main MLA 才有空间 offload 到 CPU。

---

## 2. 系统架构

### 2.1 逻辑分层

```text
┌─────────────────────────────────────────────────────────────┐
│  Scheduler                                                   │
│  OffloadMLAAttentionManager / CPUBlockManager               │
│  block 分配、null-pad、cpu_block_table                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ SchedulerOutput + connector_meta
┌──────────────────────────▼──────────────────────────────────┐
│  ModelRunner (model_runner_v1.py)                            │
│  kv_cache 五元组创建、kernel_block_sizes、offload metadata    │
│  num_offloaded_blocks / indexer_block_table / slot_mapping    │
└──────────────────────────┬──────────────────────────────────┘
                           │ kv_cache tuple per layer
┌──────────────────────────▼──────────────────────────────────┐
│  Attention (sfa_v1.py)                                       │
│  scatter → Lightning Indexer → topk 切分 → Sparse FA           │
│  offload 时调用 prepare_lru_resident_and_load                │
└──────────────────────────┬──────────────────────────────────┘
                           │ save / load 请求
┌──────────────────────────▼──────────────────────────────────┐
│  KV Transfer                                                 │
│  单机: SFAKVOffloadWorker (CPU pool, layerwise save)         │
│  PD:   SFAPDCpuOffloadConsumerWorker + MembPullReadThread    │
└─────────────────────────────────────────────────────────────┘
```

各层通过 **`kv_cache` 元组下标** 和 **attention metadata**（`block_table`、`indexer_block_table`、`slot_mapping`、`indexer_slot_mapping`、`num_offloaded_blocks`）耦合。LIC8 适配的本质是扩展元组契约，并保持 metadata 语义不变。

### 2.2 内存拓扑（单机 / PD D 侧 decode）

```text
                    ┌────────────────── HBM ──────────────────┐
                    │  indexer key [2]、LIC8 scale [5]  全量历史   │
                    │  main MLA 尾部 [0]/[1]  未 offload 块     │
                    │  resident [3]/[4]  LRU staging（C8 不下标漂移）│
                    └───────────────────────────────────────────┘
                                      ▲ H2D miss only
                                      │
                    ┌────────────────── CPU DRAM ─────────────┐
                    │  k_caches_cpu / v_caches_cpu              │
                    │  main MLA 前缀（按 block paged）           │
                    └───────────────────────────────────────────┘
```

PD D 侧 main MLA 的 **vLLM HBM pool 不按全前缀 sizing**：前缀在 CPU pool，HBM 仅保留每请求尾部块（`OffloadMLAAttentionSpec.max_memory_usage_bytes`，`kv_consumer` 时 O(1) 估算，见 `kv_cache_interface.py` ~260）。

### 2.3 单机 offload 组件关系

```text
SFAKVOffloadConnector
├── Scheduler: SFAKVOffloadlScheduler
│     └── CPUBlockManager (cpu_block_num；当前 scheduler×4 vs worker×1，见 §4.5)
└── Worker: SFAKVOffloadWorker
      ├── register_kv_caches: 从五元组取 [0]/[1] → CPU pool 注册
      │                       [3]/[4] → topk_buffers（不进 CPU pool）
      ├── prepare_lru_resident_and_load + cpu_sparse_attn.cpp
      └── KVCacheStoreLayerSendingThread: 按层 main MLA D2H (save_kv_layer)
```

门禁：`use_sparse ∧ use_offload`（`model_runner_v1.py` ~358、~4467）；worker 仅支持 **layerwise**（`sfa_kv_offload_worker.py` ~327）。

### 2.4 PD 分离架构

```text
  Prefill (P)                          Decode (D)
  kv_producer                          kv_consumer
  use_offload=false                    use_offload=true
  标准 paged KV (mooncake layerwise)    五元组 + SFAKVOffloadWorker

  P 每层 forward 完成
       │
       ▼ READ_READY(layer, p_block_ids)
  D MembPullReadThread._do_read
       ├─ indexer: P HBM [2] ──pull──► D HBM kv_cache[2]
       └─ main:    P [0]/[1] ──pull──► D CPU pool (+ partial 尾块留 D HBM)
       ▼ READ_DONE
  P layer_transfer_finished_events.set()   ← gate P 侧 layerwise buffer 复用

  D decode 与单机相同: Indexer → TopK → LRU → Sparse FA
```

P 不发五元组；D 在 `SFAPDCpuOffloadConsumerWorker` 内组合 `SFAKVOffloadWorker` + MembPull。`OffloadMLAAttentionManager` 对 remote-prefill 请求在 block_table 前 **null-pad** CPU 前缀，与 `num_offloaded_blocks` 配合 mask（`single_type_kv_cache_manager.py`）。

PD 并行约束：`DCP × PCP == 1`（`sfa_pd_cpu_offload/scheduler.py` ~103），否则 null-pad 宽度与 attention mask 阈值不一致。

启动：`run_sfa_pd_{prefill,decode,proxy}.sh`；顺序 D → P → proxy。

### 2.5 双 KV Cache Group 与 indexer 三层语义

vLLM 侧分两个 group，与五元组 `[2]` 的物理存储不是一一对应：

| 层级 | 作用 | 代码 |
|------|------|------|
| **Group 0** | indexer 预算与 slot 分组；`indexer.k_cache` synthetic 层 | `_get_kv_cache_spec` ~5092；allocate 时 skip indexer 层 |
| **kernel_block_sizes** | non-C8：`[[512],[128]]`；C8：`[[1024],[128]]` | `model_runner_v1.py` |
| **五/六元组 [2]/[5]** | scatter / Lightning Indexer 读写的 indexer 视图 | reshape `_reshape_kv_cache_tensors` |

Group1 为 MLA 层：`OffloadMLAAttentionSpec` + `OffloadMLAAttentionManager`。

### 2.6 层间接口：`kv_cache` 元组契约

**现网（非 C8）**：`use_sparse ∧ use_offload` 时每层五元组。

```text
[0]=kv_lora/k_nope   HBM 尾部
[1]=k_rope
[2]=dsa_k_indexer    HBM 全量（non-C8：raw_k bf16 view；C8：池内 int8 视图，§6.2）
[3]=topk_buffer_k    LRU
[4]=topk_buffer_v
[5]=k_li_scale       仅 C8；池内 raw_scale fp16 视图（§6.2）
```

创建：`model_runner_v1.py` ~4467–4532。注册：`sfa_kv_offload_worker.py` ~309（`len==5`，仅 `[0]/[1]` 进 offload 路径）。

**LIC8 目标：六元组**——`[0]–[4]` 与现网五元组**完全一致**，仅在队尾追加 `[5]=k_li_scale`（§6.7）。resident 保持在 `[3]/[4]`，无需因开 C8 改 LRU 下标。

逻辑槽位命名：

```text
main_k[0]  main_v[1]  indexer_k[2]  resident_k[3]  resident_v[4]  indexer_s[5]  ← 仅 C8
```

元组长度含义见 §3.3。`device_op.py`、PD MembPull、`sfa_v1.py` scatter 共用上述下标（§6.1）。

### 2.7 LIC8 接入后的架构变化（目标态）

采用 **队尾追加 scale**，`[0]–[4]` 与现网五元组对齐，降低适配面：

```text
现网五元组 [0..4]              LIC8 六元组
[0][1] main HBM 尾      ──►    [0][1]  不变
[2] indexer (bf16 view / C8 int8 池内视图)  ──►    [2] indexer_k
[3][4] LRU              ──►    [3][4]  不变
（无）                  ──►    [5] indexer_s（池内 scale 视图）
CPU pool / save         ──►    仍只 [0]/[1]
MembPull indexer        ──►    [2] key leg + [5] scale leg
```

相对非 offload A3 四元组（scale 在 `[3]`），offload 六元组把 scale **挪到 `[5]`**，以便 resident 继续占用 `[3]/[4]`。

---

## 3. 基础概念

### 3.1 MLA / Sparse / SFA

- **MLA**：存 `kv_lora` + `k_rope`，非完整多头 K/V。
- **Sparse / DSA**：每 token 只 attend TopK（常见 2048）历史。
- **SFA**：Ascend 实现路径（`sfa_v1.py`）。

### 3.2 LIC8（非 offload 基线布局）

```text
# A3 非 C8          [0]=kv_lora  [1]=k_rope  [2]=k_li bf16
# A3 C8             [0..1] + [2]=k_li int8  [3]=k_li_scale fp16
# A5 C8             [0]=merged CKV fp8  [1]=k_li fp8  [2]=scale fp32
```

| 平台 | 算子 | 当前 len |
|------|------|----------|
| A3 | `npu_lightning_indexer_quant` | 4 |
| A5 | `npu_quant_lightning_indexer` | 3 |

量化前 Hadamard（`q_hadamard` / `k_hadamard`）。A5 quant 路径当前用 `attn_metadata.block_table` 而非 `indexer_block_table`（`device_op.py` ~1470），offload 适配须改。

### 3.3 元组长度 3 / 4 / 5 / 6

| 场景 | 元组 | 相对 offload 五元组 |
|------|------|---------------------|
| 非 C8、无 offload | 3 | — |
| C8、无 offload（A3） | 4 | scale 在 `[3]`（非 offload 布局） |
| 无 C8、offload（现网） | 5 | `[0]main [1]main [2]idx [3]res_k [4]res_v` |
| C8 + offload（目标） | 6 | **`[0]–[4]` 同上，仅追加 `[5]=indexer_s`** |

### 3.4 Scatter 与 LRU

- **Scatter**：main 用 `slot_mapping`，indexer 用 `indexer_slot_mapping`（offload 双 group 不同 block 粒度）。
- **LRU**：`lru_resident_cache_config.buffer_size` 容量；miss 地址由 `cpu_sparse_attn.cpp` 计算，`prepare_lru_resident_and_load` 执行 H2D。

---

## 4. 实现要点

### 4.1 五元组与 `[2]` view

```text
dsa_block_size = mla_block_size * kv_lora_rank // index_head_dim
```

`[2]` 与 `[0]` 共享 `raw_k_tensor`（~4498）。DSV3.2 典型 `dsa_block_size=512` ═ `kernel_block_sizes[0]`。

CPU pool（独立分配）：`k_caches_cpu`/`v_caches_cpu`，`[cpu_block_num, block_size, 1, 512|64]` bf16。

### 4.2 P / D 配置（架构相关项）

| 节点 | `kv_role` | `use_offload` | `kv_cache` 布局 |
|------|-----------|---------------|-----------------|
| P | `kv_producer` | **false** | 标准 paged（可含非 offload LIC8 四/三元组） |
| D | `kv_consumer` | **true** | 五元组（目标六元组） |

D 另需 `lru_resident_cache_config`（`buffer_size ≥ topk`，`topk` 等于模型 sparse topk）。

### 4.3 Decode 逐步数据流

```text
scatter([2] indexer, [0]/[1] main 尾部)
  → Lightning Indexer([2]) → topk_indices
  → num_offloaded_blocks 切 npu_mask / cpu_mask
  → prepare_lru_resident_and_load → [3]/[4] resident
  → npu_sparse_flash_attention(NPU路径 + resident路径) → LSE merge
  → save_kv_layer: 满 block 的 [0]/[1] → CPU pool
```

### 4.4 PD MembPull 落点（`_do_read`）

- Main full blocks：P `[0]/[1]` → D `k_caches_cpu`/`v_caches_cpu`（按 `offload_id` 索引 pool，见 `959368d`）。
- Main partial：P `[0]/[1]` → D HBM 尾块。
- Indexer：P `p_base_addrs[2]` → D `_indexer_tensors`（五元组 `[2]`）。LIC8 前**尚无 scale leg**；该缺口已由 §6.6 / 提交 `a07bcd96` 实现（P `p_base_addrs[3]` → D `[5]`）。

同步：P `layer_transfer_finished_events`（MembPull 完成后）；D `layer_save_finished_events`（D2H save）。

### 4.5 实现缺口与互斥（LIC8 前）

| 项 | 说明 | 锚点 | 状态 |
|----|------|------|------|
| LIC8 + offload | 五元组 `len==5` vs C8 assert A3 `len∈{3,4}`、A5 `len==3` | `sfa_v1.py` ~1805；`device_op.py` | 已修复（§6.4，分支 `assert len in (5,6)`） |
| scale scatter | offload 下误用 `slot_mapping` | `sfa_v1.py` ~1821 | 已修复（§6.3，改用 `indexer_slot_mapping`） |
| cpu_block 倍数 | worker `×1`，scheduler `×4` | worker ~334；scheduler ~80 | 未处理（非 LIC8 范围，另行跟踪） |
| MTP decode | 需 `token_to_req` | `sfa_v1.py` ~1908 | 未处理（目标场景未启用 MTP 时无碍） |

### 4.6 模块索引

| 模块 | 路径 |
|------|------|
| Offload worker / connector | `sfa_kv_offload/{sfa_kv_offload_worker,sfa_kv_offload_connector}.py` |
| PD | `sfa_pd_cpu_offload/{connector,worker,scheduler}.py` |
| Layerwise | `kv_p2p/mooncake_layerwise_connector.py` |
| Attention | `attention/sfa_v1.py`、`attention/utils.py` |
| Runner | `worker/model_runner_v1.py` |
| 规格 | `core/kv_cache_interface.py`、`core/single_type_kv_cache_manager.py` |
| 算子 | `device/device_op.py`、`sfa_kv_offload/cpu_sparse_attn.cpp` |

---

## 5. 代码阅读路径

### 5.1 布局与规格

- `model_runner_v1.py` ~358–362：`use_sparse`
- `model_runner_v1.py` ~4467–4532：五元组
- `model_runner_v1.py` ~5072–5099：双 group spec
- `model_runner_v1.py` ~811–837：`num_offloaded_blocks`、`cpu_blocks_map`
- `kv_cache_interface.py`：`OffloadMLAAttentionSpec.max_memory_usage_bytes`
- `single_type_kv_cache_manager.py`：null-pad
- `sfa_kv_offload_worker.py` ~301–351：`register_kv_caches`

### 5.2 前向

- `sfa_v1.py`：`use_offload`、`indexer_slot_mapping`、`_get_topk_buffer`、scatter、`indexer_select_post_process`
- `attention/utils.py`：`maybe_prepare_lru_resident_and_load*`

### 5.3 Offload 与 LRU

- `sfa_kv_offload_worker.py`：`prepare_lru_resident_and_load`、`save_kv_layer`
- `sfa_kv_offload/kv_transfer.py`：`KVCacheStoreLayerSendingThread`
- `cpu_sparse_attn.cpp`

### 5.4 PD

- `sfa_pd_cpu_offload/connector.py`、`worker.py`（`MembPullReadThread`、`_do_read`）
- `mooncake_layerwise_connector.py`：`layer_transfer_finished_events`

### 5.5 非 offload LIC8 对照

- `model_runner_v1.py`：`use_sparse and not use_offload` 的 C8 allocate/reshape
- `device_op.py`：A3/A5 quant Lightning Indexer

---

## 6. LIC8 适配 Offload（已实现）

**现状**：`enable_sparse_c8 ∧ use_offload` 可跑通；六元组 + 统一 paged pool（C8 8:1 unify）。

### 6.1 逻辑槽位

```text
# 非 C8 offload（五元组）
main_k[0]  main_v[1]  indexer_k[2]  resident_k[3]  resident_v[4]

# C8 + offload（六元组）
main_k[0]  main_v[1]  indexer_k[2]  resident_k[3]  resident_v[4]  indexer_s[5]
```

`register_kv_caches` / `_get_topk_buffer` / MembPull 对 `[0]–[4]` 下标不变；C8 追加 `[5]`。

涉及文件：`model_runner_v1.py`、`kv_cache_interface.py`、`sfa_v1.py`、`sfa_kv_offload_worker.py`、`device_op.py`、`sfa_pd_cpu_offload/worker.py`。

### 6.2 统一 paged pool 布局（C8 核心）

**原则**：main 与 indexer **共用 vLLM paged pool**；每个物理 page **要么写 main，要么写 index**，不同时写。不再 pool 外单独分配 `dsa_k` / `scale`。

#### spec_0（main group，`make_offload_main_mla_spec` / `OffloadMLAAttentionSpec`）

C8 时通过 ``page_bytes_per_token`` 记账（与 spec_1 同一套
``block_size * num_kv_heads * page_bytes_per_token`` 公式）。``kv_pad_dim``
与 ``page_block_multiplier`` 由 ``compute_offload_sparse_c8_layout`` 从模型维度
推导（``page_block_multiplier = 2 * kv_lora_rank // index_head_dim``，
``kv_pad_dim`` 补齐 spec_0 与 spec_1 的 8:1 page 比例），不再硬编码常量。

```text
block_size = 128
# per token: (k 512 + v 64 + pad 8) × 2 bytes (bf16)
# pad = (page_block_multiplier * spec_1_bytes - (k+v)*2) / 2
page_bytes = 128 × (512 + 64 + 8) × 2 = 149504
```

池内切分（每层 raw 三元组）：

```text
raw_k     → k_cache [num_blocks, 128, 512] bf16
raw_v     → v_cache [num_blocks, 128, 64]  bf16
raw_scale → pad 区  [num_blocks, 128, 8]  bf16（main 不用；index 作 scale）
```

#### spec_1（indexer group，`make_offload_indexer_mla_spec` / `AscendMLAAttentionSpec`）

C8 时同样设置 ``page_bytes_per_token``（spec_1 byte-mix 布局）。

```text
# per token: (idx 128 + pad 16 + scale 2) × 1 byte (int8 byte-mix accounting)
# scale 物理上是 1 个 fp16（2 bytes），在 spec_1 公式里按 2 bytes 计入
page_bytes = 128 × (128 + 16 + 2) = 18688
main_page : indexer_page = 8 : 1   # unify_kv_cache_spec_page_size
unify 后 indexer block_size = 1024  # kernel_block_sizes[0]
```

C8 reshape 视图（index 专用 page）：

```text
raw_k     → dsa_k [num_blocks, 1024, 128] int8   # quant scatter 写入
raw_scale → scale [num_blocks, 1024, 1]   fp16
```

non-C8 offload 仍用 `raw_k` bf16 view：`dsa_k [num_blocks, 512, 128]`，`kernel_block_sizes=[[512],[128]]`。

#### 分配路径（`_allocate_kv_cache_tensors`）

- non-C8：`(raw_k, raw_v)` 二路切分。
- C8：`(raw_k, raw_v, raw_scale)` 三路切分，
  ``calc_split_factor([kv_lora_rank, qk_rope_head_dim, kv_pad_dim])``（``kv_pad_dim``
  由 ``compute_offload_sparse_c8_layout`` 推导），**无** `_allocate_sparse_c8_indexer_tensors`。

### 6.3 scatter（sfa_v1）

- `k_li` → `[2]`，`k_li_scale` → `[5]`；均用 `indexer_slot_mapping`。
- `slot_mapping` 仅用于 main `[0]/[1]`。

### 6.4 Lightning Indexer 读（device_op）

- A3 非 offload：`[2]` key、`[3]` scale，`len==4`。
- A3 offload：`[2]` key、`[5]` scale，`len==6`。
- A5 offload：六元组映射；quant 路径用 `indexer_block_table`。

### 6.5 CPU pool / LRU 隔离

- CPU pool 仍只注册 `[0]/[1]`；`topk_buffers` 仍取 `[3]/[4]`。
- `[2]`、`[5]` 不进 CPU pool 或 resident。

### 6.6 PD Pull scale

- D 侧 indexer：`[2]` key + `[5]` scale；P 侧 `base_addrs[3]` → D `[5]`（非 offload A3 四元组 scale 在 `[3]`）。

### 6.7 六元组逻辑布局

```text
[0]=kv_lora/k_nope     # 可卸
[1]=k_rope/k_pe
[2]=k_li (int8)        # 池内 raw_k 视图，常驻 HBM
[3]=topk_buffer_k      # LRU
[4]=topk_buffer_v
[5]=k_li_scale         # 池内 raw_scale 视图，常驻 HBM
```

### 6.8 启动日志与验证

`GPU KV cache size` 来自 vLLM `get_kv_cache_capacity`：

```text
GPU KV cache size ≈ max_concurrency × max_model_len
max_concurrency   = num_blocks / num_block_per_request
```

**观测要点**（相同 `gpu_memory_utilization` / `max_model_len`，仅切换 `enable_sparse_c8`）：

| 项 | 说明 |
|----|------|
| 启动 | 无 `unify_kv_cache_spec_page_size` / page 整除报错 |
| `GPU KV cache size` | C8 相对 non-C8 **略增**（~10% 量级）属预期：indexer 每请求占 block 从 `ceil(M/512)` 降到 `ceil(M/1024)` |
| 不应期待 2× | main `ceil(M/128)` 仍主导；unify 后 `num_blocks` 仍按大 page（149504B）规划 |
| 真实收益 | 无 pool 外 dsa 分配、物理 indexer 为 int8+scale、长上下文更稳 |

**实测样例**（`max_model_len=10000`，offload + GLM5.2 类配置）：

| C8 | GPU KV cache size | Maximum concurrency @ 10k |
|----|-------------------|---------------------------|
| 关 | 114,848 tokens | 11.48× |
| 开 | 125,842 tokens | 12.58× |

验证 grep：

```bash
grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency|num_blocks:|SFA KV offload" serve.log
```

单元测试：

```bash
pytest -sv tests/ut/core/test_kv_cache_interface.py
pytest -sv tests/ut/distributed/kv_transfer/sfa_kv_offload/
```
---

## 7. 速查

| 项 | 说明 |
|----|------|
| 门禁 | `use_sparse ∧ use_offload` |
| 五元组 | `[0]main [1]main [2]idx [3]res_k [4]res_v` |
| 六元组 | 五元组 + **`[5]idx_s`**（§6.7） |
| C8 pool | spec_0/spec_1 **8:1 unify**；池内 `raw_k/raw_v/raw_scale`（§6.2） |
| kernel_block_sizes | non-C8 `[[512],[128]]`；C8 `[[1024],[128]]` |
| Indexer | key=`[2]`；LIC8 scale=**`[5]`**（非 `[3]`） |
| LRU | 恒为 `[3]/[4]`，开 C8 不变 |
| PD | MembPull `[2]`+`[5]`；P `layer_transfer_finished` |

---

## 8. 方案摘要

> **`feat/lic8-kv-offload`**：在 SFA offload 五元组上扩展 LIC8 六元组（队尾 `[5]=scale`）；C8 下 indexer key/scale 走 **统一 paged pool**（spec_0/spec_1，8:1 unify），废除 pool 外 dsa 分配；PD MembPull 拉 `[2]`+`[5]`。启动验证见 §6.8。