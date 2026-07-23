# Hybrid 模型学习笔记（vllm-ascend / pcp-integration）

> 仓库：`d:\code\vllm-ascend`，分支：`feat/model-runner-v2`（基于 `li1how/pcp-integration`）  
> 目标：一步步搞清 hybrid（FA + GDN/Mamba）与普通模型的差异，并为后续 MRv2 开发做铺垫。

---

## 学习路线


| 章节  | 主题                                     | 状态   |
| --- | -------------------------------------- | ---- |
| 1   | 启动服务：hybrid 有哪些特殊点                     | ✅    |
| 2   | 请求处理：按阶段拆开看差异                          | ✅ 2.1–2.3 + sample/accepted 日志 |
| 3   | PCP hybrid 特殊路径                        | 待写   |
| 4   | 与 MRv2 的差异 / 缺口                        | 待写   |


---

## 前置概念（30 秒）

**Hybrid** 在这里主要指：同一套 decoder 里交错着

- `full_attention` → 普通 FA，cache 是 K/V pages  
- `linear_attention` → GDN（Gated Delta Net），cache 是 **conv_state + ssm_state**（用 `MambaSpec` 描述）

典型模型：`qwen3_next` / `qwen3_5` / `qwen3_5_moe`。

MoE 只替换 FFN，和「是不是 hybrid attention」正交；hybrid 讨论的是 **attention + cache 布局**。

---



# 第 1 章：启动服务时，hybrid 有哪些特殊点

把启动想成一条时间线。下面每一步都标出：**普通模型怎么做** vs **hybrid 多了什么**。

```
CLI / API
  → 解析 VllmConfig
  → try_verify_and_update_config          ★ hybrid 配置对齐
  → Worker.init_device → ModelRunner
  → load_model（建层：FA / GDN / MLP|MoE）
  → determine_available_memory（profile）
  → Engine 根据各层 get_kv_cache_spec 分组  ★ 多 KV group
  → Worker.initialize_from_config
      → initialize_kv_cache               ★ 双 backend + 共享 page 布局
  → 服务就绪，开始收请求
```

---



## Step 0：怎么判断「这是 hybrid」

入口属性：`model_config.is_hybrid`（`vllm/config/model.py`）。

逻辑大意：

1. 模型 registry 标记 `_model_info.is_hybrid`
2. 若 HF config 有 `layer_types`，且不全是 `"attention"`，才算真正 hybrid
  （避免 granite 这类「标了 hybrid 但其实全是 attention」的假阳性）

**对照**：Dense / 普通 MoE → `is_hybrid=False`；Qwen3.5（含 GDN）→ `True`。

---



## Step 1：配置阶段 —— page / block 对齐（最关键）

调用链：

```
VllmConfig.try_verify_and_update_config
  → if model_config.is_hybrid:
        HybridAttentionMambaModelConfig.verify_and_update_config(self)
```

Ascend 把这个方法 **patch** 成了：

`vllm_ascend/patch/platform/patch_mamba_config.py`

### Hybrid 特殊点 A：不能随便用默认 block_size=128

平台通用逻辑（`vllm_ascend/utils.py`）里：若 `is_hybrid`，**直接 return**，不做「prefix cache / chunked prefill 时强制 block_size=128」。

原因：hybrid 的 `block_size` 要由 **SSM page 与 attn token page 对齐公式**决定，不能被通用默认值覆盖。

### Hybrid 特殊点 B：自动抬 FA 的 `block_size`

`patch_mamba_config.verify_and_update_config` 会：

1. 从模型类取 mamba state shape/dtype → 算出 `ssm_block_page_size`、`conv_block_page_size`
2. 算出单个 token 的 attn K page 大小
3. 求一个 `attn_block_size`（常为 128 的倍数），使得：
  `attn_single_token_k_page_size * attn_block_size == ssm_block_page_size`
4. 若用户没设或设得太小 → **覆盖** `cache_config.block_size = attn_block_size`

日志里常能看到类似：

> Setting attention block size to N tokens to ensure that attention page size is >= mamba page size.



### Hybrid 特殊点 C：`mamba_page_size_padded`（就是 padding）

对齐后通常：

```
mamba_page_size_padded = attn_page_size + conv_block_page_size
```

这样 Ascend 上才能把 FA 的 K/V 和 Mamba 的 conv/ssm **塞进同一套物理 page / 共享 raw buffer**（后面 initialize_kv_cache 用）。

布局直觉（启动后 reshape 时也会再写一遍）：

```
tensor1: [(kv_padding), conv           , ...]
tensor2: [k           , ssm            , ...]
tensor3: [v           , (mamba_padding), ...]
```



### Hybrid 特殊点 D：`mamba_cache_mode` / `mamba_block_size`


| 场景                                         | 行为                                                               |
| ------------------------------------------ | ---------------------------------------------------------------- |
| 使用 AscendStore 等 kv store + hybrid manager | 强制 `mamba_cache_mode="align"`                                    |
| prefix caching + `align`                   | `mamba_block_size = block_size`（和 FA 对齐，方便按块管理）                  |
| 否则                                         | `mamba_block_size = max_model_len`（一个 request 常只需很少 state block） |


`align` 的含义（简化）：Mamba state 按与 attention 对齐的 block 语义管理，便于 prefix cache / PD 传输；不是「多算一层」。

### Hybrid 特殊点 E：平台能力与限制

- `NPUPlatform.support_hybrid_kv_cache() → True`（允许 hybrid KV manager）
- hybrid + `kv_load_failure_policy=recompute`：**断言不支持**

---



## Step 2：Worker / ModelRunner 创建

`worker.py` → `init_device()`：

- 默认：`model_runner_v1.NPUModelRunner`（**完整 hybrid 路径**）
- `VLLM_USE_V2_MODEL_RUNNER`：`v2.NPUModelRunner`（开发中；hybrid ACLGraph / Ascend PCP 等仍有缺口）

**学习建议**：先跟 v1 启动路径；MRv2 对照放在第 4 章。

启动后 hybrid 相关标志会在 runner 上陆续出现：

- `use_hybrid_blocks`：`len(attn_groups) > 1`
- `need_accepted_tokens`：存在 `MambaSpec`（spec decode 要用）
- `hybrid_with_attn_and_mamba`：同一 raw tensor 被 FA+Mamba 层共享
- `_has_gdn`：模型里有 GDN（影响 metadata 的 query_start_loc 等）

---



## Step 3：load_model —— 层结构差异

每个 decoder layer 大致：

```
input_layernorm
  → if linear_attention:  linear_attn (GDN)     # MambaSpec
    elif full_attention:  self_attn (FA)        # AttentionSpec
  → post_attention_layernorm
  → mlp  (Dense MLP 或 SparseMoE)               # 与 hybrid 正交
```

Ascend 对 Qwen3.5 的 patch：`vllm_ascend/patch/worker/patch_qwen3_5.py`。

**启动时 hybrid 特殊点**：

- 注册 **两套** attention backend（FA + GDN）
- 每层通过 `get_kv_cache_spec()` 上报自己的 cache 规格：
  - FA → `FullAttentionSpec` / `AttentionSpec`
  - GDN → `MambaSpec(shapes=(conv_shape, ssm_shape), ...)`

MoE 层（若有）只影响 FFN 权重与 EP 通信，**不增加 KV group**。

---



## Step 4：内存 profile → 算能开多少 KV block

与普通模型类似：`determine_available_memory` → `profile_run`。

Hybrid 间接影响：

- `block_size` 往往更大（为对齐 SSM）→ **同样字节预算下，block 数更少**，前缀命中粒度也可能更粗
- 每页还要覆盖 mamba padding → 有效利用率低于「纯 FA 理论值」

---



## Step 5：Engine 侧 —— 多 KV cache group（调度器视角）

各层 `get_kv_cache_spec` 汇总后，`get_kv_cache_groups`（`vllm/v1/core/kv_cache_utils.py`）：


| 模型                   | group 数（典型）                  |
| -------------------- | ---------------------------- |
| Dense / 普通 MoE       | 1（全是 AttentionSpec）          |
| Hybrid Qwen3.x       | ≥2：一组 FA + 一组 MambaSpec      |
| DeepSeek-V4 等多 ratio | 也可能多组（另一类「hybrid KV」，不是 GDN） |


Ascend 还会 patch：

- `AscendHybridKVCacheCoordinator`：跨 FA+Mamba 的 prefix hit / PD 部分命中
- Mamba manager：prefix + PCP/DCP 下的 mamba block 行为

**调度器特殊点**：同一 request 要在 **多个 group** 上同时持有 block；FA 按 token 增长，Mamba 按 state block（常很少）增长。

若 `disable_hybrid_kv_cache_manager=True`，会走 unify 路径，把异构 spec 硬拧成可单组管理的形式（功能受限，hybrid 一般不要关）。

---



## Step 6：`initialize_kv_cache` —— Runner 真正建 cache

入口：`Worker.initialize_from_config` → `NPUModelRunner.initialize_kv_cache`。

### 6.1 初始化双 backend

```python
self.initialize_attn_backend(kv_cache_config)
self.use_hybrid_blocks = len(self.attn_groups) > 1
self.need_accepted_tokens = any(MambaSpec in groups)
```



### 6.2 按 group 准备 BlockTable / kernel_block_sizes

- FA group：`kernel_block_sizes` 取 backend 支持的 kernel size（常 128）；物理 block 可能更大 → **virtual block splitting**（`use_hybrid_blocks` 的另一层含义）
- Mamba group：`kernel_block_sizes=[0]` → **禁用 slot mapping 计算**



### 6.3 分配 / reshape 共享 raw tensor

若 `hybrid_with_attn_and_mamba`：

- 一块（或按 shared_by 共享的）`int8` raw buffer
- 再 view 成 K、V、conv、ssm（含 padding）

这是 Ascend hybrid 启动期最「硬件味道」的一步：为了连续内存与 page 对齐。

### 6.4 Spec decode / Eagle3 额外动作

若 eagle3 且 `num_speculative_tokens > 1` 且需要 zeroing：可能 `_init_kv_zero_meta()`，避免 mamba block 重用时脏状态导致 NaN。

---



## Step 7：启动完成时，你应具备的心智模型

服务起来后，hybrid 相对普通模型，启动阶段已经固定了这些事实：

```
┌─────────────────────────────────────────────────────┐
│  Scheduler / KV Manager                             │
│   group0: AttentionSpec  (FA layers)                │
│   group1: MambaSpec      (GDN layers)               │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  ModelRunner                                        │
│   attn_groups[0] → FA backend + BlockTable(slots)   │
│   attn_groups[1] → GDN backend + BlockTable(state)  │
│   共享 page 对齐的 raw KV tensors                     │
└─────────────────────────────────────────────────────┘
```

**一句话**：启动时 hybrid 的特殊性，几乎全是在回答 ——  
「两套完全不同的 cache，如何在同一套 block 池、同一块 NPU 内存里对齐、分组、可调度。」

---



## 对照表：启动期 hybrid vs 普通模型


| 环节                 | 普通 Dense/MoE | Hybrid (FA+GDN)                                                    |
| ------------------ | ------------ | ------------------------------------------------------------------ |
| `is_hybrid`        | False        | True                                                               |
| block_size         | 常默认 128      | 由 SSM/attn 对齐公式决定，可能 >128                                          |
| mamba_* 配置         | 基本不用         | `mamba_page_size_padded` / `mamba_block_size` / `mamba_cache_mode` |
| KV groups          | 1            | ≥2                                                                 |
| Attention backends | 1            | FA + GDN                                                           |
| slot mapping       | 全层共用一份       | 只对 FA group 算；Mamba 不算                                             |
| KV 内存布局            | 标准 K/V       | K/V 与 conv/ssm 共享对齐 page（含 padding）                                |
| 不支持项               | —            | hybrid + kv load `recompute`                                       |


---



## 本地跟读时如何抓启动日志

代码里已加学习用日志，统一前缀：`[HYBRID-STARTUP]`。

启动后过滤：

```bash
# 全量 hybrid 启动日志
grep '\[HYBRID-STARTUP\]' your_server.log

# 只看配置对齐 / 内存 / 分组
grep '\[HYBRID-STARTUP\]\[config\]' your_server.log
grep '\[HYBRID-STARTUP\]\[memory\]' your_server.log
grep '\[HYBRID-STARTUP\]\[groups\]' your_server.log
grep '\[HYBRID-STARTUP\]\[runner\]' your_server.log
```

标签含义：

| 标签 | 阶段 | 主要文件 |
|------|------|----------|
| `[config]` | page/block 对齐、跳过默认 block_size | `patch_mamba_config.py`, `utils.py` |
| `[memory]` | profile 可用 KV 内存、共享 raw tensor | `worker.py`, `model_runner_v1.py` |
| `[groups]` | KV group / tensor 拆分 | `worker.py` → `initialize_from_config` |
| `[runner]` | attn backend、InputBatch、cache shape 样例 | `model_runner_v1.py` |

建议阅读顺序：`config` → `memory`(profile) → `groups` → `runner`。

---

## 建议你本地跟读的文件（按启动顺序）

1. `vllm/config/model.py` — `is_hybrid`
2. `vllm/config/vllm.py` — `try_verify_and_update_config` 里 hybrid 分支
3. `vllm_ascend/patch/platform/patch_mamba_config.py` — **整文件精读**
4. `vllm_ascend/utils.py` — `is_hybrid` 时跳过默认 block_size
5. `vllm_ascend/worker/worker.py` — `init_device` / `initialize_from_config`
6. `vllm_ascend/patch/worker/patch_qwen3_5.py` — 层分叉
7. `vllm_ascend/worker/model_runner_v1.py` — `initialize_kv_cache`、`may_reinitialize_input_batch`、hybrid reshape
8. `vllm_ascend/worker/block_table.py` — MultiGroup + `is_mamba_group` skip slot mapping

---



---

# 第 2 章：用户输入之后 —— 处理阶段怎么拆

服务起来之后，一条用户请求大致走：

```
① API / tokenize / Engine.add_request     （进 waiting 队列）
② Scheduler.schedule()
     · get_computed_blocks  （prefix cache 命中）
     · allocate_slots       （给本步新 token 分 block）   ★ 第一个 hybrid 差异点
③ Worker.execute_model
     · _update_states       （把多组 block_id 填进 MultiGroupBlockTable）
     · _prepare_inputs      （positions / slot_mapping；Mamba 跳过）
     · preprocess_mamba?    （仅 mamba_cache_mode=align）
     · _build_attention_metadata  （FA + GDN 两份）
     · model.forward        （层内 FA vs GDN）
     · sample
④ 返回 token → 下一轮 schedule
```

下面只先钉死 **第一个差异点**；后面阶段按你的节奏继续拆。

---

## 2.1 Scheduler 里 hybrid 到底差在哪？（不只是 slot）

### 一句话结论

**不是只有 slot。**  
`allocate_slots` / 多 group `block_ids` 是最大、最显眼的差异，但 `schedule()` 里还有：

1. **Chunk 切分边界**（`mamba_cache_mode=="align"` → `_mamba_block_aligned_split`）
2. **Prefix hit 协商**（多 group 取共同命中；Ascend PD 还有 per-group 特判）
3. **Mamba 的 cache/free 语义**（稀疏 state、null block、同 step 不能复用刚 cache 的块）
4. **输出形状**：`block_ids: tuple[list[int], ...]` 多维；`num_common_prefix_blocks` 按 group（Mamba 常为 0）

Token 预算、优先级、max_running、encoder 调度等 **控制流本身不变**。

---

### Scheduler 初始化时就记下的 hybrid 开关

`scheduler.py` 构造时：

```python
self.has_mamba_layers = kv_cache_config.has_mamba_layers
self.need_mamba_block_aligned_split = (
    self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
)
```

- 有 GDN/Mamba → `has_mamba_layers=True`
- 再叠加 `align`（Ascend 在 kv-store / 某些配置下会强制）→ 打开 **chunk 对齐裁剪**

KV 侧挂的是 `HybridKVCacheCoordinator` / Ascend 的 `AscendHybridKVCacheCoordinator`，下面每个 group 一个 manager（FA vs `MambaManager`），**共享同一个 BlockPool**。

---

### `schedule()` 逐步：哪里 same / 哪里 different

把一步 schedule 拆开（RUNNING 续跑 + WAITING 准入）：

| 步骤 | Hybrid？ | 说明 |
|------|----------|------|
| 选 request、算 `num_new_tokens`（含 spec） | **same** | 和是否 hybrid 无关 |
| Encoder 输入调度 | **same** | 无 mamba 分支 |
| **`_mamba_block_aligned_split`** | **diff（align）** | 把本步 chunk 裁到可 cache 的 block/hash/shared-prefix 边界；裁成 0 → 本请求本步 skip |
| **`allocate_slots` + preempt** | **diff（核心）** | 见下一节；preempt **策略**（FCFS/PRIORITY）same，**释放语义** Mamba 不同 |
| WAITING：max_running / LoRA | **same** | |
| **`get_computed_blocks` / prefix hit** | **diff** | 多 group 协商命中长度；可设 `shared_prefix_boundary` |
| PD connector matched tokens | **API same；输入可能不同** | Ascend hybrid+mamba：local hit 常取 FA 侧，避免被 Mamba 压成 0 |
| Watermark / full_isl 准入门闩 | **规则 same；计数更严** | 需求 = Σ 各 group 需要的 blocks |
| 组装 `SchedulerOutput` | **形状不同** | 多 group `block_ids` |

**第一个分叉**仍是：走进 schedule 后第一次碰到 mamba/hybrid 钩子（常见是 align 的 chunk split，或直接 `get_computed_blocks`/`allocate_slots`）。相对「用户请求刚进来」，**调度分块**仍是整条链路的第一个实质差异点。

---

### 差异 1：Slot / block 分配（最大一块）

普通模型：1 个 group → `block_ids` 基本是「一组 list」。  
Hybrid：一次 `allocate_slots` 对 **每个 group** 调各自 manager，结果是：

```text
KVCacheBlocks.blocks = (
  [FA_block_0, FA_block_1, ...],   # group 0 AttentionSpec
  [Mamba_state_or_null, ...],      # group 1 MambaSpec
)
# NewRequestData.block_ids / CachedRequestData.new_block_ids 同形：
# tuple[list[int], ...]
```

含义：

- FA：随序列长度涨，和你熟悉的 slot/page 一样  
- Mamba：往往只要 **很少 state block**（`align` 时 running 一步常≈1 个新 state + null padding）；**不是**按 token 填满整条序列的 K/V  

两边共用一个 BlockPool → 算「够不够」时要把两组需求 **加总**，所以同样 prompt，hybrid 更容易因总 free 不足被拒（策略没变，**有效更严**）。

关掉 prefix cache：**仍然要给两组分块**。Prefix 只是额外放大 hit 协商的差异。

> 注意术语：Scheduler 管的是 **block ids**；Worker 里的 **slot_mapping** 是下一步（FA 把 token→slot）。Scheduler 并不算 slot_mapping。口语里说「分 slot」多半指分 block。

---

### 差异 2：Chunk 边界（只有 align 时）

`need_mamba_block_aligned_split` 为真时，RUNNING / WAITING 在 `allocate_slots` **之前**可能调用 `_mamba_block_aligned_split`：

- 目的：本步结束位置必须落在 Mamba 可注册/可对齐的边界（block / hash / shared_prefix）
- 后果：本步 `num_new_tokens` 被裁短；极端裁到 0 → 本请求本轮不进 batch  

这是 **调度决策差异**，不是「多分几个 block」这么简单——它直接改变「这一步算多少 token」。

---

### 差异 3：Prefix hit 协商

- 单 group：一条 hit 长度即可  
- Hybrid：`find_longest_cache_hit` 要让各 attention 类型的命中 **收敛到可共用的长度**；可能带 `shared_prefix_boundary`（后续 chunk split 还要用）  
- Ascend PD：`find_longest_cache_hit_per_group` **跳过 Mamba 参与压 min-hit**，local computed 常用 FA hit，避免 D 侧无 mamba APC 时把 FA 命中打成 0  

---

### 差异 4：Free / cache 语义（藏在 allocate 底下）

由 `MambaManager` 承担，scheduler 通过 `allocate_slots` / `new_step_starts` / free 触发：

- 不需要像 FA 那样长期保留整段历史 K/V；常只保留「当前有效 state」  
- `align`：null block 占位、稀疏表；刚 cache 的 state **同一步不能被别的请求命中复用**（`new_step_starts` 清标记）  
- Preempt/free 时 Mamba 的「跳过多少 token / 清哪些 state」和 FA 不同  

---

### 差异 5：输出里还能看到什么

| 字段 | Hybrid 表现 |
|------|-------------|
| `NewRequestData.block_ids` | `tuple` 长度 ≥2 |
| `CachedRequestData.new_block_ids` | 每请求一项也是多 group tuple |
| `num_common_prefix_blocks` | `list[int]`，**按 group**；Mamba 组 cascade 常为 **0** |

Worker 收到后才能填 `MultiGroupBlockTable`；没有这层多 group 输出，后面的双 metadata 无从谈起。

---

### Scheduler 里 **不变** 的（避免过度解读）

- `max_num_scheduled_tokens` / token_budget 会计方式  
- FCFS vs PRIORITY、选谁 preempt  
- `max_num_running_reqs`、LoRA 限额  
- Encoder / EC 调度逻辑  
- Spec decode 在 schedule 层的「要不要带 draft」控制流（Mamba 对 lookahead **占块**另说，但调度开关同一套）  
- Async / PP / pause 等状态机  

---

### 对照表（回答「只有 slot 吗」）

| 类别 | 只有 slot？ | 实际 |
|------|-------------|------|
| 多 group 分 block | 是「slot/block」类 | **最大差异** |
| align 裁 chunk | **否** | 改变本步 token 数 |
| prefix hit 协商 | **否** | 改变 computed tokens / 边界 |
| Mamba free/cache | **否** | 改变块生命周期 |
| 输出多维 `block_ids` | 是 block 的形状 | Worker 契约 |
| 优先级 / budget | — | **不变** |

---

### 建议跟读顺序

1. `scheduler.py`：`has_mamba_layers` / `_mamba_block_aligned_split` / `allocate_slots` 调用点  
2. `kv_cache_manager.py`：`allocate_slots`、`get_computed_blocks`  
3. `patch_kv_cache_coordinator.py`：Ascend Hybrid + PD per-group  
4. `single_type_kv_cache_manager.py`：`MambaManager`（align / null / free）  
5. `sched/output.py`：`block_ids: tuple[list[int], ...]`

### 本地抓 Scheduler 学习日志

已在 `vllm/v1/core/sched/scheduler.py`（以及 Ascend PD prefix 路径）加学习日志。

| 前缀 | 含义 |
|------|------|
| `[SCHED][begin]` / `[SCHED][end]` | 本步 schedule 做了什么（队列、预算、结果摘要） |
| `[HYBRID-SCHED][align]` | mamba align 裁 chunk（hybrid 差异） |
| `[HYBRID-SCHED][prefix]` | prefix hit 协商 / PD per-group（hybrid 差异） |
| `[HYBRID-SCHED][alloc]` | 某请求分到的 **多 group** blocks（hybrid 差异） |
| `[HYBRID-SCHED][end]` | 本步结束 + new_req 的 `block_ids` 各组长度 |

开关 `VLLM_HYBRID_SCHED_DEBUG`：

```bash
# hybrid 模型：默认就会打（unset 时自动开）
# 强制打开（含非 hybrid，看通用 [SCHED]）
export VLLM_HYBRID_SCHED_DEBUG=1

# 强制关闭
export VLLM_HYBRID_SCHED_DEBUG=0

grep -E '\[SCHED\]|\[HYBRID-SCHED\]' your_server.log
```

建议读序：`[begin]` →（若有）`[align]` / `[prefix]` → `[alloc]` → `[end]`。  
对比 hybrid：看 `[alloc]` 里 `g0:n=...; g1:n=...` 是否两组都有；看 `[end]` 的 `group_lens`。

注意：若开了 `VLLM_ASCEND_BALANCE_SCHEDULING=1`，会走拷贝版 `schedule()`，begin/alloc/end 可能看不到；学习时先关掉。`[align]` 仍可能出现（基类方法）。

### 自测

1. `mamba_cache_mode!="align"` 时，scheduler 还会走 `_mamba_block_aligned_split` 吗？还会多 group 分块吗？  
2. `num_common_prefix_blocks` 对 Mamba group 为什么经常是 0？  
3. 一次 `allocate_slots` 失败，是 FA 不够、Mamba 不够，还是 **加总** 后 BlockPool 不够？

---

## 2.2 下一步：Worker 接到 SchedulerOutput

Scheduler 返回后，引擎把 `SchedulerOutput` 发给 Worker → `NPUModelRunner.execute_model`。  
**下一阶段拆成两小步：**

```
③a  _update_states     ← 多组 block_ids 写入 MultiGroupBlockTable
③b  _prepare_inputs    ← 算 positions；FA 算 slot_mapping，Mamba 跳过
③c  _build_attention_metadata + model.forward  ← 见 2.3
```

---

### 2.2.1 `_update_states`：把多维 block_ids 落盘

调用链：

```
execute_model(scheduler_output)
  → NPUModelRunner._update_states   # Ascend 薄封装
      → GPUModelRunner._update_states  # 真正干活（vllm）
```

对 **新请求**：`NewRequestData.block_ids` 是 `tuple[list[int], ...]`（每 group 一份）  
对 **续跑请求**：`CachedRequestData.new_block_ids[i]` 同样是按 group 的 tuple（或 None）

关键写入逻辑（上游）：

```python
# 续跑：按 group zip 追加
for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
    block_ids.extend(new_ids)

# 落到 InputBatch 的多表
self.input_batch.block_table.append_row(new_block_ids, req_index)
```

`block_table` 实际是 **`MultiGroupBlockTable`**：`block_tables[0]`=FA，`block_tables[1]`=Mamba。

| | 普通模型 | Hybrid |
|--|----------|--------|
| `block_ids` 形状 | 长度 1 的 tuple | 长度 ≥2 |
| `append_row` | 写一张表 | **按 group 写多张表** |
| Runner 侧 Ascend | 几乎只 `super()` | 差异在数据结构，不在 Ascend 特判 |

**这一步的 hybrid 差异**：不是新算法，而是 **消费 Scheduler 已经造好的多 group 形状**。若这里只接了一组，后面 metadata/forward 全错。

---

### 2.2.2 `_prepare_inputs`：第一个「算出来就不一样」的点

在 Ascend `model_runner_v1._prepare_inputs` 里：

1. `block_table.commit_block_table`（CPU→NPU）
2. 算 `positions` / `query_start_loc` / `seq_lens`（和普通模型同思路）
3. **`compute_slot_mapping`** ← hybrid 在这里分叉

```python
# MultiGroupBlockTable.compute_slot_mapping
for i, block_table in enumerate(self.block_tables):
    if block_table.is_mamba_group:
        continue   # ★ Mamba/GDN 不算 slot
    block_table.compute_slot_mapping(...)
```

含义：

- **FA group**：`slot = f(block_table, position)`，给 `reshape_and_cache` 用  
- **Mamba group**：跳过；后面靠 metadata 里的 **`state_indices` / `cache_indices`** 索引 conv/ssm  

所以：

- Scheduler 差异 ≈ **分多组 block**  
- Runner 第一个实质差异 ≈ **只给 FA 算 slot_mapping**（Mamba 不走 token→slot）

（若开 PCP，slot_mapping 会在 split 前就算；hybrid PCP 还有线性切分，那是第 3 章。）

---

### 2.2.3 和「再下一步」的边界

`_prepare_inputs` 之后（**已在 2.3 展开**）：

| 步骤 | hybrid 点 |
|------|-----------|
| `preprocess_mamba` | 仅 `mamba_cache_mode=="align"`（主干可先跳过） |
| `_build_attention_metadata` | 按 group 建 FA meta + GDN meta |
| `model.forward` | 层内 `full_attention` vs `linear_attention` |
| sample / `need_accepted_tokens` | GDN + spec 时同步 accepted（后续） |

---

### 跟读文件

1. `vllm/v1/worker/gpu_model_runner.py` → `_update_states`（`block_ids` / `append_row`）  
2. `vllm_ascend/worker/model_runner_v1.py` → `_update_states`、`_prepare_inputs`  
3. `vllm_ascend/worker/block_table.py` → `MultiGroupBlockTable.compute_slot_mapping` 的 `is_mamba_group` skip  

### 自测

1. `_update_states` 里 `zip(req_state.block_ids, new_block_ids)` 若一边长度 1 一边长度 2 会怎样？  
2. 为什么 Mamba 不算 slot_mapping，却仍要在 Scheduler 分 block？  
3. `commit_block_table` 和 `compute_slot_mapping` 谁先谁后？为什么？

### 本地抓 Worker 主干日志（update_states / prepare_inputs）

日志在 Ascend：`model_runner_v1.py` + `block_table.py`。只打主干，不掺 PCP/async 细节。

| 前缀 | 含义 |
|------|------|
| `[HYBRID-WORKER][update_states][begin]` | 收到的 new/cached `block_ids` 多 group 形状 |
| `[HYBRID-WORKER][update_states][end]` | 写入后 batch 与各表 row0 块数、`mamba=` |
| `[HYBRID-WORKER][prepare_inputs][begin]` | 进门摘要 |
| `[HYBRID-WORKER][prepare_inputs][commit]` | 多组 block_table commit |
| `[HYBRID-WORKER][prepare_inputs][positions]` | attn_state / pos 样例 |
| `[HYBRID-WORKER][prepare_inputs][input_ids]` | 本步 token id 样例 |
| `[HYBRID-WORKER][prepare_inputs][query_start_loc]` | FA 的 qsl |
| `[HYBRID-WORKER][prepare_inputs][gdn_qsl]` | GDN 专用 unpadded qsl（hybrid） |
| `[HYBRID-WORKER][prepare_inputs][seq_lens_input_ids]` | seq_lens + 拷到 NPU |
| `[HYBRID-WORKER][slot_mapping]` | 每组 COMPUTE(fa) / SKIP(mamba) |
| `[HYBRID-WORKER][prepare_inputs][slot_mapping]` | 汇总 FA 已算 / Mamba 跳过 |
| `[HYBRID-WORKER][prepare_inputs][logits_spec]` | logits_indices / spec metadata |
| `[HYBRID-WORKER][metadata][begin]` | 开始按 group 建双 metadata |
| `[HYBRID-WORKER][metadata][cm_base]` | 公共 CommonAttentionMetadata（默认 group0 FA） |
| `[HYBRID-WORKER][metadata][gdn_qsl_swap]` | GDN 组换用 unpadded qsl |
| `[HYBRID-WORKER][metadata][swap_tables]` | gid>0 换本组 block_table/slot |
| `[HYBRID-WORKER][metadata][build_extra]` | GDN+spec 传入 accepted_tokens |
| `[HYBRID-WORKER][metadata][build]` / `[build_done]` | builder.build 前后 |
| `[HYBRID-WORKER][metadata][group]` | 每组：spec、builder、meta_type、slot vs state_indices |
| `[HYBRID-WORKER][metadata][end]` | 按类型统计层数（hybrid 应 ≥2 种 meta） |

```bash
# hybrid 默认开；强制开/关：
export VLLM_HYBRID_WORKER_DEBUG=1
# export VLLM_HYBRID_WORKER_DEBUG=0

grep '\[HYBRID-WORKER\]' your_server.log
# 或一起看 schedule：
grep -E '\[HYBRID-SCHED\]|\[HYBRID-WORKER\]' your_server.log
```

建议对照：`[update_states][begin]` 的 `g0/g1` → `[end]` 的 `mamba=True/False` → `[slot_mapping]` 的 SKIP vs COMPUTE → `[metadata][group]` 的 FA slot vs GDN `state_indices` → `[forward][layer]` 的 FA/GDN 分叉。

---

## 2.3 `_build_attention_metadata` + `model.forward`

到这一步，batch 里已经有：多 group 的 block table、FA 的 `slot_mapping`、（hybrid 时）GDN 的 unpadded `query_start_loc`。  
**还差两件事**：给每层挂上正确的 attention metadata，再进 decoder 按 `layer_type` 走 FA 或 GDN。

```
_prepare_inputs 结束
  → （可选）preprocess_mamba   # 仅 mamba_cache_mode=="align"；主干可先跳过
  → _build_attention_metadata  # 按 KV group 建 FA meta + GDN meta
  → set_ascend_forward_context(attn_metadata, ...)
  → model.forward
       → DecoderLayer:
            full_attention  → self_attn → FA（读 slot_mapping / K/V）
            linear_attention → linear_attn → GDN（读 state_indices / conv+ssm）
```

---

### 2.3.1 `_build_attention_metadata`：一份 common，两套 builder

调用：`NPUModelRunner._build_attention_metadata`（`model_runner_v1.py`）。

骨架：

1. 用 **group0（通常 FA）** 的 block_table / slot_mapping 建 `CommonAttentionMetadata`（`cm_base`）
2. **对每个 `kv_cache_gid`**：
   - `cm = copy(cm_base)`
   - 若是 GDN 组：把 `query_start_loc` 换成 unpadded 的 `gdn_query_start_loc`（日志：`[gdn_qsl_swap]`）
   - 若 `gid > 0`：换成**本组**的 `block_table_tensor` / `slot_mapping`（Mamba 的 slot 往往是占位；真索引在 meta 里）
   - 对该 group 下每个 `attn_group`：`builder.build(...)` → 得到一份 metadata 对象
   - **同 group 的所有 layer_name 共享同一份 meta 对象**

| | FA group | GDN / Mamba group |
|--|----------|-------------------|
| Builder | Ascend FA builder | `GDNAttentionMetadataBuilder` |
| 关键字段 | `slot_mapping`、block table | `non_spec_state_indices_tensor`（+ spec 时 `spec_state_indices_*`） |
| 语义 | token → KV page 槽位 | request → mamba **page / state 行** |
| 日志 | `[metadata][group] ... has_slot_mapping=...` | `[metadata][group] ... non_spec_state_indices.shape=...` |

`[metadata][end]` 里 hybrid 应看到 **≥2 种** meta type（例如 FA 一种 + `GDNAttentionMetadata` 一种）。

**和普通模型的差异**：普通模型通常只有一个 group、一种 meta；hybrid 必须保证 **层名 → 正确类型的 meta**，否则 GDN 拿 FA meta（或反过来）会直接炸。

---

### 2.3.2 Forward 入口：context 里挂着整本 `attn_metadata` dict

```
execute_model
  → set_ascend_forward_context(attn_metadata=..., ...)
  → _model_forward → self.model(...)
```

层内算子通过 `get_forward_context().attn_metadata[self.prefix]` 取**本层**那一份。  
所以：metadata 建错 = 所有层一起错；某一 group 的 layer 挂错 dict key = 单层错。

主干日志：

| 前缀 | 含义 |
|------|------|
| `[HYBRID-WORKER][forward][begin]` | 进 forward 前：按 meta 类型统计层数 |
| `[HYBRID-WORKER][forward][end]` | 出 forward：`hidden_states` shape |

---

### 2.3.3 Decoder 分叉：`full_attention` / `linear_attention` / MoE

Ascend patch：`vllm_ascend/patch/worker/patch_qwen3_5.py`。

一层里顺序固定：

```
input_layernorm
  → attention 分支（二选一）
       full_attention  → self_attn (FA)
       linear_attention → linear_attn (GDN)
  → (optional attn layer_scale)
  → post_attention_layernorm
  → mlp   ← 与 attention 正交：dense MLP 或 SparseMoE
  → (optional ffn layer_scale)
  → 返回 (hidden, residual)
```

| 分支 | 输入 | 怎么处理 | 输出 |
|------|------|----------|------|
| FA | norm 后 hidden `[T,H]` | qkv→rope→**Attention backend**（`slot_mapping` 写/读 K/V）→o_proj | 同 shape hidden |
| GDN | 同上 | in_proj→**gdn_core**（conv+ssm，`state_indices`）→norm+out_proj | 同 shape hidden |
| MoE | post_attn_norm 后 hidden | gate/router→FusedMoE（+可选 shared_expert） | 同 shape hidden |
| dense MLP | 同上 | gate_up / down | 同 shape hidden |

**条数控制（防刷屏）**

| 变量 | 作用 |
|------|------|
| `VLLM_HYBRID_WORKER_DEBUG=0` | 关掉整类 worker/forward 学习日志 |
| `VLLM_HYBRID_FWD_LOG_N` | **每种 kind 每步最多打 N 条**（默认 `1`；`0`=层 I/O 全关） |

每步 `execute_model` 进 forward 前会 `reset_hybrid_fwd_log()`。  
kind 包括：`full_attention` / `linear_attention` / `fa_io` / `gdn_io` / `gdn_core`，以及 FFN 按挂靠的 attention 分开：`moe_after_full_attention` / `moe_after_linear_attention`（dense 则为 `mlp_after_*`）。  
因此默认 `N=1` 时，MoE 会各打一条「挂在首个 FA 后」和「挂在首个 GDN 后」。

| 日志 | 含义 |
|------|------|
| `[forward][io] kind=full_attention` | Decoder：FA 层 in/out |
| `[forward][io] kind=full_attention.detail` | FA 内部：q/k/v/attn_out/out |
| `[forward][io] kind=linear_attention` | Decoder：GDN 层 in/out |
| `[forward][io] kind=linear_attention.detail` | GDN 内部：mixed_qkv / core_out / out |
| `[forward][gdn_core]` | state_indices + conv/ssm cache shape |
| `[forward][io] kind=moe` | MoE in/out + experts/top_k（`after_attn_type=` 标明挂在 FA 还是 GDN 后） |
| `[forward][io] kind=mlp` | 非 MoE 的 dense MLP in/out |

```bash
# 默认每种 1 条；想多看几层：
export VLLM_HYBRID_FWD_LOG_N=2

grep '\[HYBRID-WORKER\]\[forward\]\[io\]' your_server.log
```

`gdn_core` 仍值得盯：确认用的是 `state_indices`，不是 FA 的 `slot_mapping`。

---

### 2.3.4 GDN 核心在干什么（主干直觉）

`vllm_ascend/ops/gdn.py` → `AscendGatedDeltaNetAttention._forward_core`：

1. 从 context 取 `GDNAttentionMetadata`
2. 截到 `num_actual_tokens`
3. **Causal conv1d**：读写 `kv_cache[0]`（conv state），索引来自 state / cache indices
4. **Gated delta / SSM 更新**：读写 `kv_cache[1]`（ssm state）
5. 输出再经 norm + `out_proj`

对照 FA：FA 是 `reshape_and_cache` + attention kernel；GDN 是 **状态递推**，没有 token→KV slot 那条路。

---

### 跟读文件

1. `vllm_ascend/worker/model_runner_v1.py` → `_build_attention_metadata`、`execute_model` 里 `set_ascend_forward_context` / `_model_forward`  
2. `vllm_ascend/patch/worker/patch_qwen3_5.py` → `AscendQwen3_5DecoderLayer.forward`  
3. `vllm_ascend/ops/gdn.py` → `forward` / `_forward_core`（`state_indices`、`kv_cache[0/1]`）  

### 自测

1. 同一步里 FA 层和 GDN 层拿到的 `attn_metadata[prefix]` 为什么不是同一个 Python 对象？  
2. 为什么 GDN 需要 `gdn_query_start_loc`（unpadded），而 FA 可以用 padded qsl？  
3. `state_indices` 和 `slot_mapping` 各索引的是什么？长度量纲各是什么（token vs request）？

### 本地抓 forward 主干日志

开关与 2.2 相同：`VLLM_HYBRID_WORKER_DEBUG`（unset 时 hybrid 默认开）。

```bash
grep -E '\[HYBRID-WORKER\]\[(metadata|forward)\]' your_server.log
```

建议一条请求对照顺序：

1. `[metadata][end]` → 出现 FA + GDN 两种 type  
2. `[forward][begin]` → 层数统计与上一步一致  
3. `[forward][io] full_attention` → `[io] full_attention.detail`  
4. `[forward][io] linear_attention` → `[io] linear_attention.detail` → `[gdn_core]`  
5. `[forward][io] moe` 或 `mlp`（看 `after_attn_type`）  
6. `[forward][end]` → hidden shape  
7. `[logits]` → `[sample][begin/end]` →（hybrid）`[accepted][after_update]` → 下一步 `[accepted][prepare_inputs_sync]`

运维/另一台机器交接：见同目录 `AGENT_HANDOFF.md`。

---

## 下一节预告

**第 3 章 PCP hybrid**：跨卡切分、linear 路径与 FA 路径如何对齐（特性路径，默认 decode 学习可先跳过）。

---

## 自测问题（第 1 章）

1. 为什么 Ascend 上 hybrid 不能简单用 `block_size=128`？
2. `mamba_page_size_padded` 补的是什么？不补会怎样？
3. 启动后为什么会有两个 `attn_groups`？MoE 会不会再加一个 group？
4. `kernel_block_sizes=[0]` 对 Mamba group 意味着什么？
5. MRv2 启动同一条链路时，哪一步最可能和 v1 行为不一致？（可先猜，第 4 章验证）

---
