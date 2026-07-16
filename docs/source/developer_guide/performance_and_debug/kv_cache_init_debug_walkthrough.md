# KV Cache 初始化调试指南

本文档说明 vLLM Ascend 启动时 KV cache 的初始化流程，按 **函数模块** 组织，每个模块内分别对照两份真实 `[KV_DEBUG]` 日志：

| 模型 | 日志文件 | 架构特点 |
|------|----------|----------|
| Qwen3-8B-W8A8 | `kv_debug_20260710_144119.log` | 标准 GQA `Attention`，36 层 spec 相同 |
| GLM-5.2-w8a8 | `kv_debug_20260710_170613.log` | `MLAAttention` + Sparse DSA，78 层，两种 page_size |

## 如何启用调试日志

在 `debug/kv-cache-memory-inspect` 分支中，`[KV_DEBUG]` 分两阶段：

### 初始化阶段（默认开启）

| 文件 | 函数 |
|------|------|
| `vllm_ascend/worker/model_runner_v1.py` | `get_kv_cache_spec()`、`initialize_kv_cache()`、`initialize_kv_cache_tensors()` |
| `vllm_ascend/worker/worker.py` | `determine_available_memory()` |

### 使用阶段（本 debug 分支默认开启）

无需再设 `VLLM_ASCEND_KV_USAGE_DEBUG`。usage 日志与 init 日志一样走 `vllm.logger`，启动后即可在 serve tee 日志里看到。

启动时应先出现 patch 加载标记：

```text
[KV_DEBUG]: usage patch applied (KVCacheManager.allocate_slots/free)
```

| 日志关键字 | 插入点 | 含义 |
|------------|--------|------|
| `usage.allocate` | `KVCacheManager.allocate_slots`（platform patch） | Scheduler 为请求从 block pool 拿新 block |
| `usage.free` | `KVCacheManager.free` | 请求结束，block 归还 pool |
| `usage.slot_mapping` | `NPUModelRunner._build_attention_metadata` | 本 step 的 `block_table` / `slot_mapping` |
| `usage.kv_write` | Dense `reshape_and_cache` / SFA `exec_kv` / MLA preprocess | 按 `slot` 把本 step 的 K/V 写入 cache tensor（看 `backend=`） |
| `usage.kv_read` | Dense paged/FIA / SFA sparse flash / MLA decode | 按 `block_table` 从 cache 里找历史 K/V（看 `backend=`） |

节流：前 8 个 forward step 全打；之后每 32 step 打一次；`kv_write`/`kv_read` 各最多 24 条，避免热路径刷屏。

过滤日志：

```bash
grep '\[KV_DEBUG\]' /path/to/kv_debug_*.log
grep 'usage\.' /path/to/kv_debug_*.log
```

> **说明**：环境变量 `VLLM_KV_DEBUG=1` 控制的是 upstream vLLM 的 `kv_debug_log`，与本文 ascend `[KV_DEBUG]` 打印是两套机制。

## 总览：初始化调用链

Engine 在 `EngineCore._initialize_kv_caches()`（`vllm/v1/engine/core.py`）中完成 KV cache 初始化：

```text
① get_kv_cache_spec()          → 每层 KV 格式（KVCacheSpec）
② determine_available_memory()  → 可用 KV 显存预算（bytes）
③ get_kv_cache_configs()        → 计算 num_blocks，生成 KVCacheConfig（Engine 侧）
④ initialize_kv_cache()         → 在 NPU 上分配并 reshape tensor
```

数据流：

```text
KVCacheSpec（每层格式）
    block_size, page_size_bytes, num_kv_heads, head_size, dtype ...
         ↓
available_memory（能给 KV 用的字节数）
         ↓
KVCacheConfig（分配计划）
    num_blocks, kv_cache_groups[], kv_cache_tensors[]
         ↓
NPU tensor（每层实际布局）
```

### 两模型关键数字一览

| 字段 | Qwen3-8B | GLM-5.2 |
|------|----------|---------|
| 并行 | 单卡 | TP16 |
| `gpu_memory_utilization` | 0.8 | 0.9 |
| 模型权重 | 10.52 GiB | 47.39 GiB |
| `use_sparse` | False | **True** |
| 层数（KV spec） | 36 | **78** |
| Spec 类型 | `FullAttentionSpec` | `AscendMLAAttentionSpec` |
| 单层 `page_size_bytes` | 524,288 B（固定） | 180,224 B 或 147,456 B |
| `available_kv` | 38.26 GiB | **6.82 GiB** |
| `num_blocks` | 2176 | **599** |
| 总 token 容量 | 278,528 | **76,672** |
| 8192 长度最大并发 | 34.00x | **9.36x** |
| `attn_groups` / `kv_cache_groups` | 1 | 1 |
| `use_hybrid_blocks` | False | False |

---

## 模块一：`get_kv_cache_spec()`

**调用链：**

```text
EngineCore._initialize_kv_caches()
  → model_executor.get_kv_cache_specs()   # collective_rpc
  → Worker.get_kv_cache_spec()
  → NPUModelRunner.get_kv_cache_spec()
```

**源码：** `vllm_ascend/worker/model_runner_v1.py` → `get_kv_cache_spec()`

**职责：** 扫描模型中所有 attention 相关模块，为需要 KV cache 的层生成 `KVCacheSpec` 字典。此阶段只描述 **格式**，不分配内存，也不确定 `num_blocks`。

### 1.1 入口 flags

| 字段 | 含义 |
|------|------|
| `use_compress` | config 含 `compress_ratios` 时为 True（DeepSeek V4 等） |
| `use_sparse` | config 含 `index_topk` 时为 True（DeepSeek V3.2 / GLM5 等 DSA） |
| `block_size` | 来自 `--block-size`，每个 KV block 存储的 **token 数** |

#### Qwen3-8B

```text
[KV_DEBUG]: start assembly flags use_compress=False use_sparse=False block_size=128
```

Dense 标准 attention，不走 MLA / Sparse 分支。

#### GLM-5.2

```text
[KV_DEBUG]: start assembly flags use_compress=False use_sparse=True block_size=128
```

`use_sparse=True` 触发 `MLAAttention.sparse` 分支，生成 `AscendMLAAttentionSpec`。

### 1.2 `attn_layers` 扫描

**源码：** `get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)`

| 字段 | 含义 |
|------|------|
| `attn_layers` | static forward context 中的 attention 模块表 |
| key | 层名，如 `model.layers.0.self_attn.attn` |
| value（类型） | 决定 spec 生成分支：`Attention`、`MLAAttention` 等 |

#### Qwen3-8B

```text
[KV_DEBUG]: attn_layers count=36 modules={'model.layers.0.self_attn.attn': 'Attention', ...}
```

36 层全部为 `Attention`，与模型层数一一对应。

#### GLM-5.2

```text
[KV_DEBUG]: attn_layers count=99 modules={
  'model.layers.0.self_attn.indexer.k_cache': 'DeepseekV32IndexerCache',
  'model.layers.0.self_attn.attn': 'MLAAttention',
  ...
}
```

| 现象 | 说明 |
|------|------|
| count=99 > 层数 78 | 部分层额外注册了 `DeepseekV32IndexerCache` |
| `DeepseekV32IndexerCache` | 扫描到但 **不单独分配 KV**；Ascend 用 `IndexerWrapper`，indexer key 合并进 MLA 的 tuple cache |
| 最终 spec 层数 | `final kv_cache_spec layers=78`（仅 `MLAAttention` 层） |

### 1.3 每层 spec 生成

根据模块类型走不同分支（`model_runner_v1.py`）：

| 模块类型 | 分支 | 典型模型 |
|----------|------|----------|
| `Attention` | `branch=Attention` | Qwen3、Llama |
| `MLAAttention` + `use_sparse` | `branch=MLAAttention.sparse` | GLM5、DeepSeek V3.2 |
| `MLAAttention` | `branch=MLAAttention` | DeepSeek V3（无 sparse） |

#### Qwen3-8B — `branch=Attention`

每层相同：

```text
[KV_DEBUG]: layer=model.layers.0.self_attn.attn branch=Attention spec={
  'type': 'FullAttentionSpec',
  'block_size': 128,
  'storage_block_size': 128,
  'page_size_bytes': 524288,
  'num_kv_heads': 8,
  'head_size': 128,
  'dtype': 'torch.bfloat16'
}
```

**`page_size_bytes` 计算**（`AttentionSpec.real_page_size_bytes`）：

```text
= 2 × block_size × num_kv_heads × head_size × dtype_size
= 2 × 128 × 8 × 128 × 2(bf16) = 524,288 字节
```

#### GLM-5.2 — `branch=MLAAttention.sparse`

两种 spec，取决于该层是否有独立 indexer（`has_indexer`）：

**有 indexer 的层**（layer 0,1,2, 6,10,14,…,74，共 21 层）：

```text
spec={
  'type': 'AscendMLAAttentionSpec',
  'page_size_bytes': 180224,
  'num_kv_heads': 1,
  'head_size': 704,    # 512(kv_lora) + 64(k_rope) + 128(index_head)
  'dtype': 'torch.bfloat16'
}
```

**无 indexer 的层**（复用邻近层 top-k，共 57 层）：

```text
spec={
  'type': 'AscendMLAAttentionSpec',
  'page_size_bytes': 147456,
  'num_kv_heads': 1,
  'head_size': 576,    # 512(kv_lora) + 64(k_rope)
  'dtype': 'torch.bfloat16'
}
```

**`page_size_bytes` 计算**（MLA，非传统 K+V）：

```text
= block_size × num_kv_heads × head_size × dtype_size
= 128 × 1 × head_size × 2(bf16)

有 indexer: 128 × 704 × 2 = 180,224
无 indexer: 128 × 576 × 2 = 147,456
```

### 1.4 最终汇总

```text
[KV_DEBUG]: final kv_cache_spec layers=N summary={...}
```

| | Qwen3-8B | GLM-5.2 |
|--|----------|---------|
| `layers` | 36 | 78 |
| 返回给 Engine | `dict[..., FullAttentionSpec]` | `dict[..., AscendMLAAttentionSpec]` |

---

## 模块二：`determine_available_memory()`

**调用链：**

```text
EngineCore._initialize_kv_caches()
  → model_executor.determine_available_memory()
  → Worker.determine_available_memory()
```

**源码：** `vllm_ascend/worker/worker.py` → `determine_available_memory()`

**职责：** 通过 dummy forward（`profile_run()`）测量激活峰值，计算 **可用于 KV cache 的字节预算**，返回给 Engine 用于 `num_blocks` 计算。

两模型走同一套逻辑，仅 `utilization` / 权重 / 激活峰值等数值不同。下面以 **Qwen3-8B** 串讲全过程；GLM-5.2 日志形态相同。

### 2.1 Profiling 起点

| 字段 | 含义 |
|------|------|
| `init_total` | NPU 总显存 |
| `init_free` | 启动时空闲显存 |
| `gpu_memory_utilization` | `--gpu-memory-utilization` |
| `requested` | 显存上限 = `total × utilization` |
| `model_weights` | 已加载权重占用 |
| `kv_cache_memory_bytes` | `--kv-cache-memory`；`None` 表示自动 profiling |

```text
[KV_DEBUG] determine_available_memory start:
  init_free=60.89 GiB init_total=61.27 GiB requested=49.02 GiB
  gpu_memory_utilization=0.8000 model_weights=10.52 GiB kv_cache_memory_bytes=None
```

### 2.2 Profiling 分解

| 字段 | 含义 |
|------|------|
| `torch_peak_increase` | dummy forward 激活显存峰值增量 |
| `non_torch_increase` | 非 PyTorch 分配（CANN/驱动等） |
| `weights_memory` | 模型权重 |
| `non_kv_cache_memory` | 不可用于 KV = 上三项之和 |
| `available_kv` | KV 预算 = `requested - non_kv` |

**核心公式：**

```text
non_kv_cache_memory = weights_memory + torch_peak_increase + non_torch_increase
available_kv        = requested_memory - non_kv_cache_memory
```

```text
[KV_DEBUG] determine_available_memory profile breakdown:
  torch_peak_increase=0.22 GiB non_torch_increase=0.02 GiB weights_memory=10.52 GiB
  non_kv_cache_memory=10.76 GiB
  formula=requested(49.02)-non_kv(10.76) => available_kv=38.26 GiB
```

### 2.3 返回值

```text
[KV_DEBUG] determine_available_memory return=41076813824 bytes (38.26 GiB)
```

返回值传入 Engine 的 `get_kv_cache_configs()`。

### 2.4 两模型数值对照

流程相同；差异只在配置与 profiling 结果：

| | Qwen3-8B | GLM-5.2 |
|--|----------|---------|
| `gpu_memory_utilization` | 0.8 | 0.9 |
| `model_weights` | 10.52 GiB | 47.39 GiB |
| `available_kv` | 38.26 GiB | 6.82 GiB |

权重越大，`non_kv` 越高，留给 KV 的预算越小（GLM-5.2 最明显）。

---

## 模块三：`get_kv_cache_configs()`（Engine 侧）

**源码：** `vllm/v1/core/kv_cache_utils.py`

| 函数 | 作用 |
|------|------|
| `get_kv_cache_configs()` | 合并各 worker 的 spec，再为每个 worker 生成 `KVCacheConfig` |
| `get_kv_cache_groups()` | **分组**：把层划成 `kv_cache_groups` |
| `get_kv_cache_config_from_groups()` | 按 group 算 `num_blocks`、规划 `kv_cache_tensors` |

**职责：** 将各层 `KVCacheSpec` 分成若干 `kv_cache_groups`，再根据 `available_memory` 计算 `num_blocks`，生成 `KVCacheConfig`。

> 此阶段日志 **不带** `[KV_DEBUG]` 前缀，由 upstream Engine 打印。

### 3.1 分组逻辑（`get_kv_cache_groups`）

核心问题：哪些层可以共用同一套 block table / 同一类分配策略？

Engine 按下面顺序尝试（命中即返回）：

```text
get_kv_cache_groups(kv_cache_spec)
  │
  ├─ ① is_kv_cache_spec_uniform？
  │     所有层 Spec 可 merge 成同一种
  │     → 1 个 group，包含全部层
  │     例：Qwen3-8B（36 × 相同 FullAttentionSpec）
  │
  ├─ ② UniformTypeKVCacheSpecs.from_specs？
  │     层类型相同（都需要相同数量的 token slot），
  │     但单层 page_size 可以不同
  │     → 1 个 group，spec 为 UniformTypeKVCacheSpecs
  │       group.page_size_bytes = Σ 各层 page_size_bytes
  │     例：GLM-5.2（78 × AscendMLAAttentionSpec，两种 page）
  │
  └─ ③ 其他路径（DeepSeek V4 / Hybrid Mamba 等）
        本文不展开；基础 Dense / MLA Sparse 走 ① 或 ② 即可
```

> 本文两个样例都是 **1 个 `kv_cache_group`**。路径 ① / ② 的差别在于：Qwen3 各层 page 相同；GLM5.2 各层 page 不同，但类型统一，用 `UniformTypeKVCacheSpecs` 把各层 page **相加** 得到 group 级 page。

### 3.2 从 group 到内存规划（`get_kv_cache_config_from_groups`）

本文样例均为单 group。`UniformTypeKVCacheSpecs` 下：

```text
num_blocks = available_memory // group.page_size_bytes
每层一个物理 tensor，size = 该层 page_size_bytes × num_blocks
```

这里的 **tensor** 指计划分配的一块连续 KV 显存（`KVCacheTensor`：含 `size_bytes` 与 `shared_by`），真正 `torch.zeros` 在模块四执行。

#### Qwen3-8B — 路径 ①

```text
group.page_size_bytes = 36 × 524,288 = 18,874,368 字节/block（≈ 18.0 MiB）
num_blocks = available // 18,874,368 = 2176
每层 tensor size = 524,288 × 2176（共 36 个 tensor）
```

#### GLM-5.2 — 路径 ②

```text
group.page_size_bytes = 21 × 180,224 + 57 × 147,456
                      = 12,189,696 字节/block（≈ 11.6 MiB）
num_blocks = available // 12,189,696 = 599
每层 tensor size = 该层 page × 599（共 78 个 tensor，大小因层而异）
```

### 3.3 `num_blocks` 对照

| | Qwen3-8B | GLM-5.2 |
|--|----------|---------|
| 路径 | ① uniform spec | ② uniform type |
| 计算 | avail // (36×524288) | avail // Σpage |
| 结果 | **2176** | **599** |

### 3.4 总容量与并发

```python
size_bytes[layer_i] = per_layer_page_size_bytes × num_blocks
```

#### Qwen3-8B

```text
单层 size_bytes = 524,288 × 2176 = 1,140,850,688 B ≈ 1.06 GiB
36 层合计       ≈ 38.26 GiB（= available_kv）

总 token 容量   = 2176 × 128 = 278,528
```

```text
GPU KV cache size: 278,528 tokens
Maximum concurrency for 8,192 tokens per request: 34.00x
```

#### GLM-5.2

```text
有 indexer 层 = 180,224 × 599 ≈ 103.0 MiB/层
无 indexer 层 = 147,456 × 599 ≈ 84.3 MiB/层
78 层合计     ≈ 7.30 GiB（≈ available_kv）

总 token 容量   = 599 × 128 = 76,672
```

```text
GPU KV cache size: 76,672 tokens
Maximum concurrency for 8,192 tokens per request: 9.36x
```

并发估算：

```text
单请求 block 数 = ceil(max_model_len / block_size) = ceil(8192/128) = 64
最大并发 ≈ num_blocks / 64

Qwen3:  2176 / 64 = 34x
GLM5.2: 599 / 64  ≈ 9.36x
```

---

## 模块四：`initialize_kv_cache()`

**调用链：**

```text
Worker.initialize_from_config(kv_cache_config)
  → NPUModelRunner.initialize_kv_cache()              # model_runner_v1.py
       ├─ may_add_encoder_only_layers_to_kv_cache_config()
       ├─ maybe_add_kv_sharing_layers_to_kv_cache_groups()
       ├─ initialize_attn_backend()
       ├─ may_reinitialize_input_batch()
       └─ initialize_kv_cache_tensors()
            ├─ _allocate_kv_cache_tensors()            # torch.zeros raw buffer
            └─ _reshape_kv_cache_tensors()            # view 成推理布局
```

**职责：** 收到 Engine 下发的 `KVCacheConfig`，在 NPU 上分配内存、reshape、绑定到各 attention 层。

### 4.1 输入 config（函数入口）

打印：`[KV_DEBUG]: initialize_kv_cache input config=...`（`model_runner_v1.py:4030`）

| 顶层字段 | 含义 |
|----------|------|
| `num_blocks` | 全局 KV block 池大小 |
| `kv_cache_groups[]` | 分组信息：层名列表 + 合并后的 spec |
| `kv_cache_tensors[]` | 每层要申请的 raw buffer：`tensor_id`、`size_bytes`、`shared_by` |

#### Qwen3-8B

```text
num_blocks: 2176
kv_cache_groups[0].layer_names: 36 层 model.layers.0~35.self_attn.attn
kv_cache_tensors[0].size_bytes: 1140850688  (= 524288 × 2176)
```

#### GLM-5.2

```text
num_blocks: 599
kv_cache_groups[0].layer_names: 78 层 model.layers.0~77.self_attn.attn
kv_cache_tensors[i].size_bytes: 因层而异（180224×599 或 147456×599）
```

### 4.2 encoder / kv-sharing 调整

打印：`[KV_DEBUG]: after encoder/sharing adjustments config=...`（`model_runner_v1.py:4041`）

两模型在本例中 **config 均不变**（无 encoder-only 补层、无 kv-sharing）。

### 4.3 attention backend 初始化

打印（`model_runner_v1.py:4049`、`4059`）：

```text
[KV_DEBUG]: attn_groups=1 use_hybrid_blocks=False need_accepted_tokens pending
[KV_DEBUG]: need_accepted_tokens=False
```

| 字段 | 含义 | Qwen3-8B / GLM5.2 |
|------|------|-------------------|
| `attn_groups` | attention backend 分组数 | 均为 **1** |
| `use_hybrid_blocks` | 多组不同 page 格式（如 Mamba+Attention） | 均为 **False** |
| `need_accepted_tokens` | 是否含 `MambaSpec`、需额外字段 | 均为 **False** |

> GLM5.2 虽有两种 `page_size_bytes`，但都属于 `AscendMLAAttentionSpec`、同一 backend，故仍是 `attn_groups=1`。

### 4.4 `initialize_kv_cache_tensors()` — raw 分配

打印：`[KV_DEBUG]: raw tensors allocated layers=N total_bytes=...`（`model_runner_v1.py:4142`）

此时内存为 **一维 int8 原始 buffer**，尚未 reshape。

#### Qwen3-8B

```text
layers=36  total_bytes=41070624768（≈ 38.26 GiB）

model.layers.0.self_attn.attn: tuple of 2 parts
  K: shape=[570425344]  dtype=int8  bytes=570425344
  V: shape=[570425344]  dtype=int8  bytes=570425344
```

验证：`570425344 × 2 × 36 = 41,070,624,768 ≈ available_kv` ✓

#### GLM-5.2

```text
layers=78  total_bytes=7301627904（≈ 6.80 GiB）

有 indexer 层（如 layer 0）: tuple of 3 parts
  part0: bytes=78512128
  part1: bytes=9814016
  part2: bytes=19628032

无 indexer 层（如 layer 3）: tuple of 2 parts
  part0: bytes=78512128
  part1: bytes=9814016
```

验证：`7301627904 / 599 ≈ 12,189,696 = group.page_size_bytes` ✓

### 4.5 `initialize_kv_cache_tensors()` — reshape

打印：`[KV_DEBUG]: reshaped kv caches=...`（`model_runner_v1.py:4177`）

一维 int8 buffer 被 view 成 attention kernel 需要的多维 tensor。**bytes 不变，仅重新解释内存布局。**

#### Qwen3-8B — 标准 K/V 四维

```text
model.layers.0.self_attn.attn: tuple of 2 parts
  K: shape=[2176, 128, 8, 128]  dtype=bfloat16  bytes=570425344
  V: shape=[2176, 128, 8, 128]  dtype=bfloat16  bytes=570425344
```

```text
[2176,  128,  8,   128]
  │      │    │     └── head_size
  │      │    └──────── num_kv_heads（GQA）
  │      └───────────── block_size（每 block 128 token）
  └──────────────────── num_blocks
```

#### GLM-5.2 — MLA 三元组（非 K/V）

**有 indexer 的层**（layer 0）：

```text
tuple of 3 parts:
  [599, 128, 1, 512]  bf16  → kv_lora
  [599, 128, 1, 64]   bf16  → k_rope
  [599, 128, 1, 128]  bf16  → indexer key
```

**无 indexer 的层**（layer 3）：

```text
tuple of 2 parts:
  [599, 128, 1, 512]  bf16  → kv_lora
  [599, 128, 1, 64]   bf16  → k_rope
```

```text
[599,  128,  1,  D]
  │     │    │   └── latent 维度（512 / 64 / 128）
  │     │    └────── num_kv_heads=1（MLA 压缩为 1）
  │     └─────────── block_size
  └───────────────── num_blocks
```

### 4.6 完成

打印：`[KV_DEBUG]: initialize_kv_cache done allocated=... shared_layers={}`（`model_runner_v1.py:4087`）

| | Qwen3-8B | GLM-5.2 |
|--|----------|---------|
| `allocated` | 36 层 K+V tuple | 78 层 tuple（2 或 3 parts） |
| `shared_layers` | `{}` | `{}` |

此后 KV cache 初始化完成，Engine 进入 warmup / serve。

### 4.7 阶段时间线（两模型通用）

| 顺序 | 打印关键词 | 代码位置 | 内存状态 |
|------|-----------|----------|----------|
| ① | `input config` | `initialize_kv_cache` 入口 | 仅 config，未分配 |
| ② | `after encoder/sharing` | 补 group 之后 | 仍未分配 |
| ③ | `attn_groups=...` | `initialize_attn_backend` 之后 | 仍未分配 |
| ④ | `need_accepted_tokens=...` | backend 初始化完毕 | 仍未分配 |
| ⑤ | `initialize_kv_cache_tensors start` | 进入分配子函数 | 即将 malloc |
| ⑥ | `raw tensors allocated` | `_allocate_kv_cache_tensors` 之后 | **已 malloc**，int8 一维 |
| ⑦ | `reshaped kv caches` | `_reshape_kv_cache_tensors` 之后 | **已 reshape**，bf16 多维 |
| ⑧ | `initialize_kv_cache done` | 全流程结束 | 可推理 |

---

## 模型类型速查（扩展阅读）

按 `[KV_DEBUG]` 日志特征区分常见架构：

| 类型 | `use_sparse` | 模块类型 | Spec | `attn_groups` | 代表模型 |
|------|-------------|----------|------|---------------|----------|
| Dense GQA | False | `Attention` | `FullAttentionSpec` | 1 | Qwen3-8B |
| MLA + Sparse | True | `MLAAttention` | `AscendMLAAttentionSpec` | 1 | GLM5.2、DeepSeek V3.2 |
| MLA（无 sparse） | False | `MLAAttention` | `MLAAttentionSpec` | 1 | DeepSeek V3 |
| Compressed | `use_compress=True` | 多种 | 压缩 spec | 视模型 | DeepSeek V4 |

---

## 字段关系总图

```text
┌─ 配置层 ─────────────────────────────────────────────┐
│ block_size=128          每 block 128 tokens          │
│ gpu_memory_utilization  显存上限 → requested         │
│ max_model_len           单请求最长 token 数           │
└────────────────────────────────────────────────────┘
         ↓
┌─ Spec 层（get_kv_cache_spec）────────────────────────┐
│ Qwen3:  page_size=524288, 8 heads, head=128         │
│ GLM5:   page_size=180224|147456, 1 head, MLA latent │
└────────────────────────────────────────────────────┘
         ↓
┌─ 预算层（determine_available_memory）───────────────┐
│ Qwen3:  available_kv=38.26 GiB, weights=10.52 GiB   │
│ GLM5:   available_kv=6.82 GiB,  weights=47.39 GiB   │
└────────────────────────────────────────────────────┘
         ↓
┌─ Config 层（get_kv_cache_configs）───────────────────┐
│ Qwen3:  num_blocks=2176, tokens=278528, groups=1    │
│ GLM5:   num_blocks=599,  tokens=76672,  groups=1    │
└────────────────────────────────────────────────────┘
         ↓
┌─ Tensor 层（initialize_kv_cache_tensors）────────────┐
│ Qwen3:  [num_blocks,128,8,128] ×2 (K+V)             │
│ GLM5:   tuple [num_blocks,128,1,512|64|128]         │
└────────────────────────────────────────────────────┘
```

---

## 使用阶段：初始化之后内存怎么被用

`initialize_kv_cache` 只做一件事：在 NPU 上准备好一块 **按 block 编号的大数组**。
真正「谁占用哪个 block、本 step 写到哪个 slot、decode 去哪读历史」发生在推理循环里。

```text
Scheduler / KVCacheManager          ModelRunner                 Attention
─────────────────────────          ───────────                 ─────────
allocate_slots → block ids    →    block_table + slot_mapping → kv_write (reshape_and_cache)
                                   (slot=block*bs+offset)       kv_read  (paged/FIA via block_table)
free → 归还 block pool
```

### 1. Scheduler 怎么 allocate / free

- `allocate_slots(request, num_new_tokens, ...)`：从 `block_pool` 取空闲 block，挂到该请求的 block table。
- `free(request)`：请求结束时把其 block 还回 pool，`free_blocks` 增加。
- 日志示例：

```text
[KV_DEBUG]: usage.allocate req_id=... ok=True num_new_tokens=128 ... free_blocks 2176->2175 new_blocks=[[42]]
[KV_DEBUG]: usage.free req_id=... free_blocks 2000->2001
```

注意：这里分配的是 **逻辑 block id**，不是重新 `malloc` NPU 显存。物理 tensor 在 init 时已经固定大小。

### 2. slot_mapping 怎么传

每个 forward step，`NPUModelRunner` 根据 Scheduler 下发的 block table 计算：

```text
slot = block_id * block_size + offset_in_block
```

- `block_table[req][i]`：该请求第 i 个物理 block id
- `slot_mapping[token]`：该 token 的 K/V 应写入 / 对应的全局 slot

日志：

```text
[KV_DEBUG]: usage.slot_mapping step=1 gid=0 ... formula='slot=block_id*block_size+offset'
  block_table_preview=[[42, 43]] slot_mapping_preview=[5376, 5377, ...]
```

### 3. Attention 怎么写 / 读

| 阶段 | Dense / Qwen3-8B | SFA / GLM-5.2 | MLA（无 sparse） |
|------|------------------|--------------|-------------------|
| 写 | `reshape_and_cache` → `backend=Dense` | `npu_kv_rmsnorm_rope_cache` / MLAPO → `backend=SFA` | preprocess rope_cache → `backend=MLA` |
| 读 (decode) | paged / FIA + `block_tables` | sparse flash attn + `block_table`（+ indexer topk） | FIA v2 + `decode.block_table` |

日志里用 `backend=` 区分路径：

```text
[KV_DEBUG]: usage.kv_write backend=Dense ...   # Qwen3-8B
[KV_DEBUG]: usage.kv_read  backend=Dense ...
[KV_DEBUG]: usage.kv_write backend=SFA ...     # GLM-5.2
[KV_DEBUG]: usage.kv_read  backend=SFA ...
```


Decode **不靠 token 下标连续寻址**，而是：

1. 用 `block_table` 找到该序列占用过的 block id 列表；
2. 在已分配的 `key_cache` / `value_cache`（或 MLA latent cache）里按 block 取历史；
3. 本 step 新 token 先经 `slot_mapping` 写入，再参与 attention。

### 4. 对照日志读一条请求生命周期

```text
usage.allocate     → 拿到 block 42
usage.slot_mapping → slot 含 42*128+offset
usage.kv_write     → reshape_and_cache 写到这些 slot
usage.kv_read      → decode 用 block_table=[[42,...]] 读回
usage.free         → block 42 回到 pool
```

---

## 源码索引

| 主题 | 路径 |
|------|------|
| Engine 初始化入口 | `vllm/v1/engine/core.py` → `_initialize_kv_caches()` |
| CLI `block_size` | `vllm/config/cache.py` → `CacheConfig.block_size` |
| Spec 定义与 `page_size_bytes` | `vllm/v1/kv_cache_interface.py` |
| Ascend MLA Sparse spec | `vllm_ascend/core/kv_cache_interface.py` → `AscendMLAAttentionSpec` |
| Attention 生成 spec | `vllm/model_executor/layers/attention/attention.py` → `get_kv_cache_spec()` |
| 计算 `num_blocks` | `vllm/v1/core/kv_cache_utils.py` → `get_kv_cache_config_from_groups()` |
| Ascend spec 组装 | `vllm_ascend/worker/model_runner_v1.py` → `get_kv_cache_spec()` |
| 内存 profiling | `vllm_ascend/worker/worker.py` → `determine_available_memory()` |
| NPU 分配与 reshape | `vllm_ascend/worker/model_runner_v1.py` → `initialize_kv_cache()` |
| Scheduler allocate/free | `vllm/v1/core/kv_cache_manager.py`（Ascend patch: `patch_kv_cache_manager_debug.py`） |
| slot_mapping 计算 | `vllm/v1/worker/block_table.py` |
| Dense 写/读（Qwen3） | `vllm_ascend/attention/attention_v1.py`（`backend=Dense`） |
| SFA 写/读（GLM5.2） | `vllm_ascend/attention/sfa_v1.py`（`backend=SFA`） |
| MLA 写/读（无 sparse） | `vllm_ascend/attention/mla_v1.py`（`backend=MLA`） |
| 使用阶段日志 helper | `vllm_ascend/kv_usage_debug.py` |

---

## 相关文档

- [KV Cache Pool Guide](../Design_Documents/KV_Cache_Pool_Guide.md)
- [Service Profiling Guide](service_profiling_guide.md)
