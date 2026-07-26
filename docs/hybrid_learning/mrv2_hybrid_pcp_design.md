# Ascend MRv2 Hybrid PCP — V2 适配设计方案

> 状态：架构方向已确认，本文作为实施与评审基线；代码尚未按本文实施。  
> 仓库：`vllm-ascend`（MRv2） / 对照：上游 `vllm` MRv2 PCP、Ascend V1 hybrid PCP。  
> 核心边界：公共框架面向 **MLA/GQA + GDN**（收窄）；切分等后端无关逻辑可先按 MLA 公共契约开发；第一阶段不做 chunked prefill / spec / PD / MM / PP / LoRA / DCP / EP；只承诺 BF16/FP16、普通因果 attention、`mamba_cache_mode=none`；目标门禁可删，未支持组合由 capability fail-fast；M0～M5 以 eager 为必达，piecewise 作为紧随其后的 M6 必做目标。

---

## 0. 文档约定与最终决策摘要

1. 上游 MRv2 `PCPManager` 是宿主框架；不把 Ascend V1 runner 深耦合路径整体搬入 V2。  
2. Ascend V1 用于复用 Hybrid 线性切分、GDN 接缝和进/出 FA 桥接算法。  
3. token 切分与公共字段不区分 MLA/GQA；差异收敛在 attention backend；无后端时可先复用 MLA 公共字段开发切分/双视图。  
4. `AscendPCPManager` 统一持有 linear/FA 双视图及转换索引，并通过只读视图对外提供。  
5. 不重写上游 `execute_model()`；只增加 `prepare_inputs` / `prepare_attn` / ModelState 的窄接口；由 Ascend Runner 的 `prepare_attn()` 生成并绑定只读 `HybridPreparedStep`，同时返回上游旧签名兼容的 per-group 合并结果。  
6. 先语义移植 `Wangbei25/vllm-ascend:mrv2` Hybrid 主干并验证 `pcp=1`，再叠加 PCP（O3=A）。  
7. 图模式：M0～M5 以 eager 为必达；M6 必须补齐 piecewise，通信桥允许 graph break；full graph 不做。  
8. 不做 chunked prefill；验收由负责人真机自行验收，文档只保留实现约束。  
9. 目标门禁（MLA-only / 禁 hybrid+PCP 等）均可删除，未支持组合用 capability fail-fast。  
10. EP：**架构按后续必叠 EP 设计**，但第一阶段关闭 EP；MoE 只认 linear pad + valid-token 契约，不把 EP=False 写进 Manager。
11. Hybrid FA 参考 V1 采用 **单次 linear QKV/latent AG**：bridge 产出本卡 FA query 与已全局化 cache inputs，通过 `CacheWritePlan` 告知 backend 不再二次 AG K/V；纯 decode 旁路 bridge。
12. GQA prefill 采用 **cache-first**：先在每个 PCP rank 写入完整本地 KV cache，再由 virtual-row Q 通过 block table 读取。
13. group 分类使用 backend/builder capability strategy，不用 `model_type` 白名单，也不把 `AttentionSpec` / `MambaSpec` 的 `isinstance` 当成长期扩展协议。

历史讨论材料保留用于解释取舍；若历史文字与本节及 §5 的最终决策冲突，以最终决策为准。

---

## 1. 问题定义与第一版范围

### 1.1 我们要解决什么

在 **Model Runner V2** 上，让 **hybrid 模型**（FA + GDN/Mamba）支持 **PCP（Prefill Context Parallel）**。

> PCP 的目标是：prefill 时沿 sequence 维把计算分到多张卡上，缩短长上下文 prefill 时延。  
> （本阶段 FA KV **不分片**，见 C4/C9。）

### 1.2 第一版可叠加 / 不叠加（已确认）

| 类别 | 第一版 | 说明 |
|------|--------|------|
| **PCP + TP** | ✅ 做 | 并行主叠加 |
| **MLA + GDN** | ✅ 做 | 公共契约可先按 MLA 字段开发 |
| **GQA + GDN** | ✅ 做 | 同一双视图/bridge；后端 adapter |
| dense / MoE Hybrid | ✅ 做 | MoE 走 linear 主路径 + pad/unpad |
| **Expert Parallel** | ❌ 第一阶段不打开 | 架构和 valid-token 契约必须可叠 EP；后续工作收敛在 MoE 通信/dispatch adapter，不改双视图 |
| dynamic EPLB | ❌ 第一阶段不打开 | 延续当前 MRv2 限制；valid-mask 过滤 hook/UT 第一阶段完成 |
| eager | ✅ M0～M5 必达 | 第一条功能闭环 |
| piecewise | ✅ M6 必达 | 紧随 eager；通信桥允许 graph break；full graph 不做 |
| **chunked prefill** | ❌ 不做 | 要求 `enable_chunked_prefill=False`；配置 fail-fast |
| speculative decoding | ❌ 不做 | 不叠加 |
| PD / KV connector | ❌ 不做 | 不叠加 |
| MM / PP / LoRA | ❌ 不做 | 不叠加 |
| PCP + DCP | ❌ 不做 | 不叠加 |
| full ACLGraph hybrid | ❌ 不做 | — |
| quant / sliding window | ❌ 第一阶段不做 | capability fail-fast；后续 backend adapter 扩展 |
| Hybrid prefix cache | ❌ 第一阶段不做 | 要求 `enable_prefix_caching=False`；continued prefill/prefix hit fail-fast |

### 1.3 第一版模型范围（已确认）

公共框架面向 **KV-attention（MLA/GQA）+ GDN**，不以 `model_type` 白名单收窄：

- 实施顺序：无后端时先按 MLA 公共字段做切分/双视图 → 接 GDN → 再接 GQA adapter。  
- M0～M5 先完成 eager，M6 必须补齐 piecewise。  
- 第一阶段只承诺 BF16/FP16、普通因果 attention、`mamba_cache_mode=none`、`enable_prefix_caching=False`；quant、sliding window、Hybrid prefix cache 后置。
- 经典纯 Mamba1/2、非 GDN 线性层和其它 Hybrid 结构不作为第一版验收对象。

### 1.4 成功标准（已确认方向）

1. 语义移植 MRv2 Hybrid 主干，并在 `pcp_size=1` 下完成真实 forward。  
2. MLA+GDN、GQA+GDN 在 `pcp_size>1` 下完成 prefill/decode；M0～M5 eager 必达，M6 piecewise 必达。  
3. dense / MoE Hybrid + PCP+TP 可运行；第一阶段 EP 关闭，但 MoE valid-token 契约可直接叠加后续 EP。  
4. 不支持组合 fail-fast。  
5. **验收**：负责人后续真机自行验收；本文不绑定详细验收矩阵为交付门禁。

---

## 2. 三份基线各自是什么（材料，不决策）

便于对齐术语，本节只陈述现状，不做方案选择。

### 2.1 上游 vLLM MRv2 PCP（已与代码核对）

- 入口：`vllm/v1/worker/gpu/pcp_manager.py`
- 模型限制：当前 `validate_config` → **MLA only**；实施时直接替换为 backend/group capability 校验，不保留全局模型白名单
- 切分算法：针对 **普通 FA/MLA 的 KV attention** → **DualChunkSwap**（`num_chunks = 2 * pcp_world_size`，rank `r` 拿 chunk `r` 与 `2*pcp-1-r`）
- Decode：不切，各 rank **复制**同一段 decode tokens

#### 2.1.1 与 Ascend PCP V1 的关键差异（你的分析 → **正确**）

**差异 1：切完之后的 batch 形态 = virtual batch，不拼回「一个 req」**

上游每个真实 prefill req，在本 rank 上会生成 **最多 2 个 `RankSegment`**（head 一段 + tail 一段）。  
`partition_batch` 里：

```text
num_local_reqs = len(local_segments)   # 不是 global num_reqs
```

并且构造 local `InputBatch` 时：

- `req_ids` 可以 **重复同一个真实 req_id**（head 行、tail 行各一次）
- `num_scheduled_tokens` **按 segment**，不是按真实 req 合并
- buffer 预留：`max_num_local_reqs = 2 * max_num_reqs`（见 `PCPManager.__init__`）

含义：下游 attention **假装这是两个（或多个）独立 query row** 在算，而不是 V1 那种「仍是 1 个 req，positions 里 head|tail 拼在一段里」。

对照 Ascend V1：`update_tokens_for_pcp` 返回的仍是 **按原 req 数** 的 `pcp_tokens` + 一段 positions（head/tail 已拼进同一 query），`num_reqs` 不变。

**差异 2：社区 PCP 不切分 KV 存储**

- Scheduler 建 KV manager 时传 **`pcp_world_size=1`**（`scheduler.py`），PCP **不参与** block/KV 分片计数
- DCP 才做 sequence 维交错分片
- PCP 路径上 slot：对 global positions 算 slot，再 gather 到各 rank 布局；`gathered_kv_write_mask` 用于展开 slot 表去重（**不是**「只有 rank0 的 GPU 更新 KV」——各卡仍写本地全量副本，见 §4.8.2）

对照 Ascend V1：`slot_mapping` 用 `cp_size = pcp * dcp` 交错，**PCP 参与 FA KV 分片**。

#### 2.1.2 上游计算路径（简图）

```text
global InputBatch（真实 req）
  → DualChunk → 每 rank 若干 RankSegment
  → local InputBatch（virtual rows：1 真实 prefill ≈ 2 local reqs）
  → forward（按 virtual batch 做 attention）
  → all_gather hidden + restore_idx → 回真实 token 序
  → 用 global_batch sample
```

### 2.2 Ascend V1 hybrid PCP

- 入口：`pcp_utils.py` + `model_runner_v1._prepare_inputs` + `attention_cp`
- DualChunk（及 hybrid 时再加线性）后，**仍保持原 req 维度**，同一 req 内拼 positions
- FA KV：**PCP 参与** slot 交错分片
- hybrid 额外：线性布局给 GDN + FA enter/exit 桥

### 2.3 Ascend MRv2 现状（与 PCP/hybrid 相关）

- `worker/v2/pcp/pcp_manager.py`：薄封装社区 DualChunk，刷新 `seq_lens_np` / `attn_state`
- 当前 `pcp-integration` 仍走社区 `validate_config`（MLA-only）——按 C17 **实施时删除**，改为 capability fail-fast
- `Wangbei25/vllm-ascend:mrv2` 已提供可语义移植的 MRv2 Hybrid 基线：
  - `58d1b6502`：保留 `MambaSpec`、Hybrid cache allocate/reshape、`AscendMambaHybridModelState`、Mamba state 生命周期
  - `291658e90`：修正 MRv2 KV/state zeroing 初始化条件
- 引入策略：只语义移植 Hybrid 相关提交，不整体合并该分支中的 EPLB、scheduler、FIA 等无关改动
- 该分支 UT 主要覆盖 cache/spec/ModelState 选择，尚未覆盖 Hybrid PCP 双视图和真实模型 forward
- 实施顺序已定：先在 `pcp=1` 验证 Hybrid 主干，再叠加 PCP

---

## 2.4 已确认的上游事实（本轮）

| # | 结论 | 状态 |
|---|------|------|
| U1 | 上游切分是 DualChunk，面向普通 KV attention（当前产品面 MLA） | 已确认 |
| U2 | 切完用 **virtual batch**（segment ≈ 假 req），不拼回单 req | 已确认 |
| U3 | 社区 **不按 PCP 切 KV 存储**；Ascend V1 **会** | 已确认 |
| U4 | 目标门禁均可删（含 MLA-only / Hybrid 禁 PCP）；未支持组合改由 capability / 配置 fail-fast | 已确认 |

---

## 2.5 Hybrid 模型盘点：有哪些、`full_attention` 是什么（本轮讨论）

口径：`IsHybrid` = 同一模型里交替（或并存）**有状态的线性/SSM 层** + **标准 KV attention 层**。  
下面按「FA 层到底用什么 Attention」分组（上游 vLLM 现状）。

### A. Ascend V1 hybrid PCP 已点名的目标族（优先）

| model_type / 类 | linear / SSM 层 | **full_attention 层** | FA 底层 |
|-----------------|-----------------|------------------------|---------|
| `qwen3_next` → `Qwen3NextForCausalLM` | `QwenGatedDeltaNetAttention`（GDN） | `Qwen3NextAttention` | **标准 `Attention`（GQA MHA，非 MLA）** + RoPE；可选 `attn_output_gate` |
| `qwen3_5` → `Qwen3_5ForConditionalGeneration` | 同上 GDN（`gqa_interleaved_layout=False`） | **复用 `Qwen3NextAttention`** | **同上：标准 `Attention`** |
| `qwen3_5_moe` → `Qwen3_5MoeForConditionalGeneration` | 同上 GDN | **复用 `Qwen3NextAttention`** | **同上：标准 `Attention`**；FFN 侧为 Sparse MoE |

要点：

- 配置侧层类型字符串就是 `full_attention` / `linear_attention`（`layer_types`）。
- **不是 MLA**；上游今天只放行 MLA。本方案 **删除目标门禁**，GQA 走同一公共切分契约 + backend adapter。
- Ascend V1 开关：`pcp_use_hybrid_attn` 仅认这三个 `model_type`（V2 不以白名单收窄）。

### B. 同类「FA = 标准 Attention + GDN/线性」但未进 Ascend PCP 名单

| 模型 | linear 侧 | full_attention 侧 |
|------|-----------|-------------------|
| `OlmoHybridForCausalLM` | Olmo GDN | `OlmoHybridAttention` → **标准 `Attention`** |
| `Lfm2*` / ColBERT-Lfm2 | short conv / 类 mamba | `Lfm2Attention` → **标准 `Attention`** |

### C. FA = 标准 Attention + Mamba1/2（经典 hybrid）

| 模型 | SSM | full_attention |
|------|-----|----------------|
| `JambaForCausalLM` | Mamba1 | `Attention` |
| `GraniteMoeHybridForCausalLM` | Mamba2（layer 名也可叫 linear_attention） | `GraniteMoeHybridAttention` → `Attention` |
| `NemotronH*` | Mamba2 | `NemotronHAttention` → `Attention` |
| `FalconH1*` | Mamba2（与 attn **并行** 分支，形态略不同） | `Attention` |
| `Zamba2*` / `Plamo2*` 等 | 各自 SSM | 多为标准 `Attention` |

### D. FA = **MLA** + 线性/GDN（和上游 PCP「MLA only」同一 FA 族，但是 hybrid）

| 模型 | linear 侧 | full_attention 侧 |
|------|-----------|-------------------|
| `KimiLinearForCausalLM` | Kimi GDN | `KimiMLAAttention` → **MLA** |
| `BailingMoeV2_5ForCausalLM` | Bailing linear attn | `BailingMoeV25MLAAttention` → **MLA** |

这类 FA 与上游 DualChunk PCP **更接近**（都是 MLA KV），但 **linear 因果** 仍是 hybrid 难题；Ascend V1 hybrid PCP **没有**把它们列入名单。

### E. 最终范围结论

> **公共框架范围 = KV-attention（MLA/GQA）+ GDN。**  
> 对 PCP token 切分和双视图索引语义而言，GQA 与 MLA 一致：统一走 DualChunk/virtual-batch。二者的 bridge 操作数、projection/RoPE 边界、KV 物理布局、cache 写入、metadata 类型和计算 kernel 均允许由 backend adapter 分别实现；不能将差异缩写成“只有 KV 存储形态不同”。

实施上先使用上游 MLA 的公共字段语义和流程闭环，再接入 GQA adapter：

| 层次 | 是否区分 MLA/GQA |
|------|------------------|
| `AscendPCPManager`、linear/FA 双视图、转换索引 | 不区分 |
| GDN 接缝、MoE pad/unpad、sample restore | 不区分 |
| FA 公共输入：virtual rows、positions、query_start、seq_lens、block table、global/gathered slot（由 plan 选择） | 不区分 |
| metadata 具体 dataclass、KV tensor/cache spec、kernel | backend 内区分 |

“复用 MLA 字段”指复用公共字段契约，不要求 GQA 直接依赖 `MLACommonMetadata` 具体类型。GQA adapter 应从同一份 FA 公共视图构建自己的标准 Attention metadata。

---


## 3. 核心分叉的决策记录

本节保留拍板过程用于解释取舍，所有选项均已关闭，不再是开放问题。

### 3.1 MRv2 的「宿主框架」跟谁

| 选项 | 含义 |
|------|------|
| A | **跟上游 MRv2**：一切挂在 `PCPManager.partition_batch` / restore 上，Ascend 只扩展 |
| B | **跟 Ascend V1**：在 V2 runner 里再造一套类似 `update_tokens_for_pcp` 的深耦合路径 |
| C | **混合**：框架跟上游（global/local batch），算法语义参考 V1（线性+FA 桥） |

你的选择：`A`（已确认）——高内聚低耦合；算法语义上 hybrid 双布局仍用 V1（见 C7），但**挂载点跟上游 PCPManager**，不把 V1 深耦合路径整段搬进 V2 runner。

### 3.2 hybrid 下 prefill token 怎么切

| 选项 | 含义 |
|------|------|
| A | **只 DualChunk**（与上游 MLA 相同切法；GDN 如何接要另设计） |
| B | **只线性**（GDN 友好；FA 负载与上游 DualChunk 不同） |
| C | **V1 式双布局**：本地线性跑 GDN；FA 进场再变成 DualChunk 语义 |
| D | **其他**（例如 GDN 只在 rank0 跑全序列、FA 走 PCP） |

你的选择：`C`（已确认，见 C7）

要点（C7；细节以 C12/C29/C30 为准）：

- 平时 / GDN：线性局部 hidden，**层间不 AG 全序列**
- 进 FA（Hybrid 2B）：**单次** entry QKV/latent AG → 本卡 DualChunk Q + global cache inputs；backend **不再**二次 AG K/V
- 出 FA：AG out → scatter 回线性局部；纯 decode 旁路 bridge
- KV 存储跟上游：**不分片**；每卡写完整本地副本（`CacheWritePlan`，见 §4.5/§4.7）

### 3.3 FA 的 KV 存储：PCP 要不要分片

| 选项 | 含义 |
|------|------|
| A | **跟上游**：PCP 不分片 KV，主要切计算；DCP 才分片 |
| B | **跟 Ascend V1**：PCP 也参与 `slot_mapping` 交错分片 |
| C | **本阶段先不分片**，只切计算，存储策略后置 |

你的选择：`A`（已确认）——**本阶段不考虑 DCP**。

### 3.4 Mamba/GDN state 存储（已确认）

先把「state」和「FA 的 KV」分开：

| | FA 的 KV cache | GDN/Mamba 的 state |
|--|----------------|-------------------|
| 存什么 | 每个 token 的 K、V（随序列变长） | 每个 **请求** 一份递推状态（`conv_state` / SSM state 等），长度不随「本卡 token 数」线性涨成序列页 |
| PCP 切 token 后 | 前面已定：跟上游，**不分片、每卡全量副本** | **本方案定案：请求级 state 不分片/可复制** |

**为什么会有这个问题？**

Prefill 时序列被切到多张卡：rank0 算前半，rank1 算后半（线性布局）。  
GDN 是因果递推：rank1 要接着 rank0 末尾的状态往下算，不能各算各的从零开始。

所以至少有两层意思（常被混在一起）：

1. **算的时候（接缝）**：rank>0 要从「前面的卡」拿到初始 state（V1 里有 AG `last_width` / `initial_state_mode` 一类逻辑）——这是 **通信怎么接**，见 §4.4。  
2. **存的时候（本小节）**：这份「请求级 state」在显存里怎么放？

常见两种：

| 选项 | 含义 | 直观 |
|------|------|------|
| A | **不分片 / 按请求复制或单份持有**：每卡（或逻辑上）都能对着「这个 req」读写完整 state；不把 state 按 token 交错切到不同卡 | 和「FA KV 不分片」同一精神；实现简单 |
| B | **按 PCP 切 state**：只有部分卡存/写某段递推结果 | 省一点显存，但和接缝、调度强耦合，本阶段一般不划算 |

上游对 DCP+hybrid 的精神也是：**full-attn KV 可分片，Mamba state 复制**（`kv_cache_coordinator` 注释）；我们 PCP 且先不做 DCP，更应 state 不分片。

你的选择：`A`（已确认）——**V2 跟随 V1：state 不分片 / 可复制；接缝逻辑跟 V1（conv last_width AG + SSM AG 修正）**
#### 对照：V1 hybrid PCP 里 GDN state 实际怎么做

**存储（§3.4 的「存」）**

- GDN cache = `conv_state`（`kv_cache[0]`）+ `ssm_state`（`kv_cache[1]`），`MambaSpec`
- 按 **请求** 用 `state_indices` / `cache_indices` 索引，**不是** FA 的 token 级 `slot_mapping` 交错分片
- 各 PCP rank 本地都有这套缓冲；接缝结束后各卡写入同一份「序列末尾」state（decode 用）→ 精神上就是 **state 不分片 / 可复制**

**计算接缝（「算时怎么接」）**

| 组件 | V1 做法 | 代码 |
|------|---------|------|
| conv | 本卡 `extract_last_width` → PCP AG → rank>0 把 **上一卡 last_width** 写入本地 `conv_state` 再跑；rank>0 强制 `initial_state_mode=True`；结束后各卡写回 **最后一卡** 的 last_width | `ops/gdn.py`、`gdn_attn_builder.py` |
| SSM | 各卡先本地 `chunk_gated_delta_rule` → AG `final_state`+`h_update` → 递推修正 → rank>0 用修正 initial **重跑 h**；`final_state` 取末卡结果写回 | `ops/triton/fla/chunk.py` |

一句话：**V1 不按 PCP 切 state 存储；靠 AG + 注入/重算接缝，并让各卡落同一份末状态。** 表中的“最后一卡/末卡”是 V1 常规非空假设；V2 必须按 §4.4 扩展为“最后一个有真实 token 的 rank”。

---

## 4. 最终架构设计

本节是实施主规范。阅读顺序：

1. 分支与代码基线（§4.1）  
2. V1 对照材料（§4.2）  
3. 最终数据流（§4.3）  
4. GDN 接缝（§4.4）  
5. 双视图与公共字段 / `HybridPreparedStep`（§4.5）  
6. Runner / ModelState 窄接口与 capability（§4.6）  
7. MLA/GQA backend adapter（§4.7）  
8. MoE / EP / Decode 写盘 / 线性 L1（§4.8）  
9. 图模式与 buffer 容量（§4.9）  
10. 校验边界（§4.10）

### 4.1 分支基线

| 仓库 | 远程分支 | 本地分支 | HEAD（拉取时） |
|------|----------|----------|----------------|
| `vllm-ascend` | `li1how/pcp-integration` | `pcp-integration` | `414d80ec0` feat SFA PCP under MRV2 |
| `vllm` | `li1how/pcp-integration` | `pcp-integration` | `4ddfbd678` MRV2 virtual-batch PCP for MLA |
| Hybrid 主干来源 | `Wangbei25/vllm-ascend:mrv2` | 只作语义移植源 | `58d1b6502` + `291658e90` |

**注意**：`vllm-ascend` 的 `pcp-integration` **已删除 V1 PCP**（`pcp_utils.py` 移除；非 V2 runner 开 PCP 会直接报错）。  
下文「V1 数据流」依据删除前实现（如 `feat/hybrid-learning-debug` / commit `aa34edd1f` 一代）与设计文档，作对照材料，**不是**当前 `pcp-integration` 可运行路径。

Hybrid 主干采用选择性语义移植：

1. 移植 `58d1b6502` 的 `MambaSpec`、cache allocate/reshape、HybridModelState 和 state lifecycle。  
2. 移植 `291658e90` 的 MRv2 zeroing 条件修复。  
3. 不整体带入源分支的 EPLB、scheduler、FIA 等无关提交。  
4. 首先用真实模型验证 `pcp=1`，成功后才启用 Hybrid PCP。

### 4.2 V1 PCP 数据流（对照）

#### 总览

```text
SchedulerOutput（全局真实 req）
        │
        ▼
model_runner_v1._prepare_inputs
        │  PCPManager.init_batch_info
        │  update_tokens_for_pcp  ──► DualChunk（+ hybrid 时再写线性布局）
        │  positions / query / query_mapping（本卡局部）
        │  _get_pcp_metadata → AscendPrefillContextParallelMetadata → long_seq_metadata
        ▼
build attn metadata（挂 prefill_context_parallel_metadata）
        ▼
model.forward（各层）
  ├─ GDN（hybrid）：吃线性局部 hidden；接缝 AG last_width / SSM 修正
  └─ FA GQA/MLA：
        reshape_and_cache 内 AG KV（或 hybrid AG QKV + 桥）
        → DualChunk FA（head/tail × mask/nomask）
        → hybrid 时 AG out + exit_fa 回线性
        │  层间 hidden 保持本卡局部，不整网 AG
        ▼
forward 结束
        │  get_restore_hidden_states：AG + restore_idx → 全局原序
        ▼
sample / postprocess
```

#### 阶段拆解

| 阶段 | 谁做 | 数据形态 | 关键产物 |
|------|------|----------|----------|
| 1. 切分 | `PCPManager.update_tokens_for_pcp` | 仍按 **原 req 数**（非 virtual-batch） | DualChunk `pcp_tokens`/`positions`；hybrid 另有线性 tokens/positions |
| 2. 元数据 | `_get_pcp_metadata` 等 | 批级 `AscendPrefillContextParallelMetadata` | restore / FA 桥 idx、seqlens… |
| 3. slot | `BlockTable` + PCP 交错 | **KV 分片**（非本卡 `-1`） | 与 DualChunk positions 对齐的 slot |
| 4. 层间 | runner 不 AG hidden | 本卡局部 token 流 | — |
| 5a. FA | `attention_cp` | 临时 AG 本轮 KV（hybrid 再桥到 DualChunk Q） | 本卡 DualChunk 输出；hybrid 再散回线性 |
| 5b. GDN | `gdn.py` / `chunk.py` | 线性局部 + state 接缝 | 末状态各卡对齐 |
| 6. 收尾 | `get_restore_hidden_states` | AG 局部 hidden → 原序 | 给 sample |

#### 与上游 MRv2 / 当前 `pcp-integration` 的差异（数据流视角）

| | Ascend V1（已删） | 上游 / 当前 pcp-integration 方向 |
|--|-------------------|----------------------------------|
| 切分后 batch | 原 req 维，positions 拼 head\|tail | **virtual-batch**（segment≈假 req） |
| 宿主 | `pcp_utils.PCPManager` 深耦合 runner | `v1/worker/gpu/pcp_manager.PCPManager` |
| KV | PCP 分片 | 不分片，AG 后每卡全量副本 |
| hybrid | 线性+FA 桥在 V1 Manager | 当前分支几乎只有 MRv2 壳，hybrid 桥待做 |

---

### 4.3 最终数据流（已确认）

> 推荐：**主路径 = 线性**；**FA = 旁路 DualChunk（上游）**；桥只在进/出 FA；层间不 AG 全序列。

#### 为什么这样定

| 选项 | 问题 |
|------|------|
| 主路径 DualChunk，GDN 每次再变线性 | GDN 层更多、接缝更重，每层都桥 → 差 |
| 整网每层 AG 全序列 | 吃掉 PCP 收益 |
| **主路径线性 + FA 旁路 DualChunk（推荐）** | GDN 零布局税；FA 层少；FA 可纯跟上游 |

#### 一 step 时序

```text
global InputBatch（真实 req）
        │
        ▼
AscendPCPManager.partition_hybrid()
  ├─ 上游 DualChunk partition → 保留 fa_batch + gathered slot（FA/KV）
  ├─ 线性 partition           → 主 local_batch（GDN/MLP/默认）
  └─ 填 HybridPCPLayout
        │
        ▼
prepare_attn（按 attn group）
  ├─ FA 组：fa_batch + block table → FA metadata
  ├─ cache write：global/original slot mapping → CacheWritePlan
  ├─ GDN 组：linear batch → GDN metadata
  └─ 生成并绑定只读 HybridPreparedStep；向 execute_model 返回兼容旧签名的合并结果
        │
        ▼
forward（hidden 默认线性）
  ├─ GDN：直接算 + V1 接缝；state 不分片
  ├─ FA（薄包装）：
  │    decode: 纯 decode 直接旁路 bridge
  │    enter: 只对 prefill 做 pad → 单次 AG → hybrid_linear_ag_restore_idx → 原序
  │           → hybrid_global_to_fa_idx → 本卡 DualChunk Q
  │           → 原序 K/V(或 MLA latent KV) + global slot → PREFILL/ALREADY_GLOBAL segment
  │    FA:    不再二次 AG K/V；每卡先写完整本地 KV cache
  │           → 本卡 DualChunk/virtual-row Q 通过 block table 读 cache
  │    exit:  AG out → hybrid_fa_to_linear_idx → 线性
  └─ 层间：不 AG 全序列
        │
        ▼
sample：pure decode 直接用本地前缀；prefill 段 AG + restore
        → decode 前缀与原序 prefill 拼回 global_batch → sample
```

#### 职责

| 组件 | 职责 |
|------|------|
| `AscendPCPManager` | 两套 partition、私有 step builder、`HybridPCPLayout`、batch/index/slot buffer、global/gathered slot、sample restore；不分配 layer-specific QKV/latent workspace |
| Ascend Runner `prepare_attn()` | 完成 per-group prepare，发布只读 `HybridPreparedStep`，绑定给 ModelState，并返回旧签名兼容结果 |
| FA 薄包装 | 只做 enter/exit 和 `CacheWritePlan` 分发；不理解 GDN，不通过全局单例读取 Manager |
| GDN | 只认线性 + V1 接缝 |
| HybridModelState | 一次性消费 `HybridPreparedStep`；按 group capability 从 FA/linear 视图构造 metadata、投影 bridge/plan，并按 layer name 合并 |
| Runner | 标准 `partition → prepare_attn → forward → restore`，只增加窄接口，不重写 `execute_model()` |

你的选择：`同意推荐`（已确认，见 C12）

---

### 4.4 GDN 接缝落点（已确认 → C13）

> 前提：C12 主路径 **线性**。`pcp-integration` 上 V1 接缝代码 **仍在** `ops/gdn.py` / `chunk.py` / `gdn_attn_builder.py`（PCP `world_size>1` 分支未删）。  
> **定案：继续用 V1 接缝方案**——算法留 ops，Manager 只保线性。

#### V1 / 现状接缝在哪（已存在，建议继续留在 ops）

| 接缝 | 落点文件 | 做什么 | 假设的 token 布局 |
|------|----------|--------|-------------------|
| conv | `ops/gdn.py` | `extract_last_width` → PCP AG → rank>0 写入上一卡 `conv_state`；结束后写回**最后有效 rank** 的 last_width（V1 常写物理末卡；V2 见约束 4 / C35） | **线性局部** prefill |
| conv metadata | `ops/gdn_attn_builder.py` | rank>0 prefill 强制 `initial_state_mode=True`；零长度 rank 不得误置 | 同上 |
| SSM / chunk | `ops/triton/fla/chunk.py` | AG `final_state`+`h_update` → 递推修正 → rank>0 重跑 h；末状态取**最后有效 rank** 写回 | **线性局部** prefill |

这些 **不依赖** `HybridPCPLayout` 的 FA 桥 idx；只依赖「本卡是因果连续的线性段」+ `state_indices` / `query_start_loc`。

#### 推荐落点（与 C12 对齐）

```text
AscendPCPManager
  └─ 产出 linear local_batch + HybridPCPLayout
           │
prepare_attn（GDN 组）
  └─ 只用 linear batch 建 GDN metadata
     （gdn_attn_builder 内已有 rank>0 initial_state_mode）
           │
forward · GDN 层
  └─ 直接进现有 gdn.py / chunk.py 的 PCP 分支  ← 接缝落点（保持）
     禁止把 DualChunk/fa_batch 喂给 GDN
           │
FA 层
  └─ 只走 HybridPCPLayout 桥 + 上游 FA；不碰 GDN 接缝
```

| 该放哪 | 结论 |
|--------|------|
| 接缝算法 | **留在 ops**（复用现有 PCP 分支），不搬进 `PCPManager`、不新建第二套 |
| metadata 改写 | **留在 `gdn_attn_builder`**（`initial_state_mode`） |
| Manager 职责 | 保证 GDN 只看到 **linear** batch；**不**把接缝逻辑吞进 Manager |
| FA 桥 | 仅 `HybridPCPLayout`；与 GDN 接缝正交 |

#### 必须守住的约束

1. **GDN 的 `query_start_loc` / positions / state_indices 必须对线性布局**；若误用 DualChunk virtual-batch，接缝数学全错。  
2. Decode：各卡复制同一 decode token 时，接缝 prefill 分支不走或走 V1 已有 decode 路径（现有代码已分 prefill/decode）。  
3. 第一版不做 chunked prefill 时，SSM 接缝保持现有「整段 linear prefill」假设即可。  
4. 线性切分允许某请求在某 rank 上为零 token；零长度 rank 对该请求必须执行 **state identity/透传**，不能生成未初始化 state。最终写盘取“最后一个有真实 token 的 rank”的状态，而不是无条件取物理 `rank=-1`。  
5. `seq_len < pcp_size`、多请求中部分请求零长度、整 rank 仅 decode/零 prefill 都必须有显式 mask；conv `last_width`、SSM `final_state/h_update` 的 AG shape 固定，但无效行不得覆盖有效状态。  
6. GDN 外层 hidden 行数是 `linear_num_tokens_padded`，metadata 的有效 token 数是 `linear_num_tokens`；尾部 `linear_valid_mask=false`。conv/SSM/state write 必须显式忽略尾部，而不是把 padded row 当成某请求的 token。

#### 实现加固（不新增结构）

| 项 | 说明 |
|----|------|
| 显式 assert | **必须**：GDN metadata/forward 在 `pcp>1` 时断言 `input_layout=CONTIGUOUS_CAUSAL_STATE`、outer rows=`linear_num_tokens_padded`、actual tokens=`linear_num_tokens`；不读取 Manager，防误用 fa_batch |
| 命名/注释 | **必须**：标明「依赖 C12 线性主路径」 |
| **不**把 `last_width` AG 抽到 Manager | 会破坏「算子侧接缝」内聚，且与 FA 桥生命周期不同 |

你的选择：`同意接缝留 ops + Manager 只保线性`（已确认，见 C13）

---

### 4.5 双视图与新增字段（实施规范）

> 口径修订（你的反馈）：  
> 1. hybrid / 线性字段 **命名必须带 `hybrid_` 或 `linear_`**  
> 2. 线性 AG 需要对齐 pad；`max_num_tokens_across_pcp` **不要与上游 DualChunk 的 max 混用/重复命名**  
> 3. 进 FA：**线性 AG → 一次还原全局序 → 一次取 DualChunk**，不必 V1 那套三件套  
> 4. DualChunk pad / unpad **先复用上游 MLA PCP 已有机制**；GQA adapter 遵守同一公共契约，hybrid 不重复造  
> 5. hybrid 字段用 **独立结构体** 挂到 `AscendPCPManager`，不散落  
> 6. MLA/GQA 共享公共 FA 视图和 bridge 字段；具体 metadata dataclass 由 backend 自己构建

#### 0. 两阶段生命周期与挂载方式（1A）

`prepare_inputs()` 与 `prepare_attn()` 之间，block tables / slot mappings 尚未就绪，不能把一个未填完的对象同时宣称为“只读视图”。采用私有 builder + 最终不可变 step context：

```text
AscendPCPManager(PCPManager)
  └── _hybrid_builder: HybridStepBuilder | None     # Manager 私有、当前 step 可写
        ├── linear_batch
        ├── fa_batch
        └── bridge: HybridPCPLayout

AscendModelRunner.prepare_attn(linear_batch)
  ├── manager.prepare_hybrid_attn()
  ├── finalize → HybridPreparedStep                 # 冻结、单 step 只读
  ├── model_state.bind_hybrid_step(prepared_step)   # 显式绑定，不反向读 Runner
  └── return prepared_step.legacy_block_tables,
             prepared_step.legacy_slot_mappings     # 保持 execute_model() 旧签名

HybridPreparedStep
  ├── step_id
  ├── linear_batch
  ├── fa_batch
  ├── group_inputs: tuple[PreparedGroupInputs, ...]
  ├── bridge: HybridPCPBridgeView
  ├── forward_view: HybridPCPForwardView
  ├── legacy_block_tables
  └── legacy_slot_mappings

PreparedGroupInputs
  ├── group_id / capability
  ├── batch_view / block_table
  ├── cache_write_plan: CacheWritePlan | None
  └── legacy_slot_row

HybridPCPForwardView
  ├── linear_num_tokens
  ├── linear_num_tokens_padded
  └── linear_valid_mask
```

约束：

- `linear_batch` 与 `fa_batch` 使用独立、预分配的 buffer；构造其中一份不得覆盖另一份。  
- Manager 只持有私有 builder；`finalize()` 后立即清除 builder 并返回带单调 `step_id` 的只读 `HybridPreparedStep`。Runner/ModelState 只能消费 finalized 对象，backend 只能读由 ModelState 投影出的 metadata。  
- `PreparedGroupInputs` 按 `kv_cache_group_id` 保存该 group 的 batch view、block table、per-group `CacheWritePlan`、兼容 slot row 和 capability；不以模型类型分类。不同 FA group 可以有不同 block size/cache 格式，禁止把 plan 提升成 step 级单例。  
- `legacy_block_tables` 是 tuple，可让不同 group 持有不同请求行数；FA group 使用 virtual rows，GDN group 使用 linear rows。  
- `legacy_slot_mappings` 仍保持二维 `[num_kv_cache_groups, max_group_slot_len]`，每个 group 写入自己的有效前缀，其余填 `PAD_SLOT_ID`；这样现有 `build_slot_mappings_by_layer()` 无需修改。  
- legacy 返回值只是 `PreparedGroupInputs` 内同一底层 buffer 的兼容投影，不能独立重算；FA backend 以 per-group plan 为准，`set_forward_context(slot_mapping=...)` 中对应 layer 的 row 必须与 plan 的有效 segment 一致，尾部只能是 `PAD_SLOT_ID`。  
- `HybridPCPForwardView` 只包含 MoE/通用算子需要的 linear token count 与 valid mask，不暴露 FA bridge idx；它与 `HybridPCPBridgeView` 分离，避免 MoE 依赖 attention 布局细节。  
- `preprocess_state()` 收到合并结果后只消费 GDN/Mamba group 的 linear block table；FA group 的 slot/cache 输入从 `HybridPreparedStep` 获取。  
- `AscendInputBatch` 不增加 hybrid 索引字段。

#### 1. 新结构体 `HybridPCPLayout`（仅 hybrid 增量）

| 字段（命名规范） | 含义 | 原因 |
|------------------|------|------|
| `linear_num_tokens: int` | 本卡线性布局 **未 pad** token 数 | GDN/默认 forward 真实长度 |
| `linear_num_tokens_padded: int` | 本 step 线性 AG 对齐长度 = `max(各卡 linear_num_tokens)` | **线性要 pad**：各卡线性段长度可能不同，AG 前必须对齐；这是 **linear 专用 max**，≠ 上游 DualChunk 的 `max(per_rank DualChunk tokens)`，故 **不叫** `max_num_tokens_across_pcp`，避免与上游重复/混淆 |
| `hybrid_linear_ag_restore_idx` | `AG(线性 prefill pad 后)` → **全局/原序 prefill** | AG 结果是 `[rank0_pad \| rank1_pad \| …]`，**不是**原序；需要 **一次** index 还原。有了原序后才能取 DualChunk；sample restore 也复用该 idx |
| `hybrid_global_to_fa_idx` | 全局/原序 prefill → **本卡 FA（DualChunk）prefill token** | 进 FA 只需这一次选取（对齐你提的「有原序后保存一个 DualChunk 下标即可」）；对应 V1 `pcp_fa_query_idx`，**不再**要 `enter_fa`+`fa_padding` 两套 |
| `hybrid_fa_to_linear_idx` | `AG(FA/DualChunk prefill 输出)` → **本卡线性 prefill** | 出 FA 回 GDN；对应 V1 `pcp_exit_fa_scatter_idx` |

`linear_positions` **不进入** `HybridPCPLayout`：唯一数据源是 `linear_batch.positions`。若实现需要快捷访问，只允许保存同一底层 buffer 的只读 view，并断言 `data_ptr`/切片一致，禁止复制第二份。

混合 batch 约定 decode 在前，且 PCP 各 rank 的 decode 请求/position 完全相同。令 `D = num_decode_tokens`，则只对 `[D:linear_num_tokens]` 的 prefill 段通信，AG pad 长度为 `linear_num_tokens_padded - D`；三个 bridge idx 的通信域均只覆盖 prefill。decode 前缀直接拼回 linear/FA 结果。若 rank 间 `D`、decode request ids 或 positions 不一致，必须在 collective 前 fail-fast。

下式沿用 §4.9.1 的记号：`L=linear_num_tokens_padded`、`F=fa_batch.num_tokens_after_padding`、`T=global real tokens`。索引 shape 必须固定：

```text
hybrid_linear_ag_restore_idx.shape[0] = T - D       # 只产出 global real prefill
hybrid_global_to_fa_idx.shape[0]      = F - D       # 含 FA local padding rows
hybrid_fa_to_linear_idx.shape[0]      = L - D       # 含 linear local padding rows
```

后两个 idx 的 padding 位可填安全索引 0，但对应 `fa_batch.is_padding` / `linear_valid_mask` 必须为 false，cache/state/output 消费端不得读取其数值。pure decode 时三者都是空 view 且 bridge early-return；禁止为了避免零长 tensor 伪造一个“有效 token”。

**开关**：以当前 step 是否存在 `HybridPreparedStep` 为准，不再增加 `pcp_use_hybrid_attn` 状态位，避免两个开关漂移。

#### 2. 进 FA：单次 AG + query/cache 两种消费视图（2B）

```text
本卡 linear QKV / MLA latent tuple
  → 纯 decode：不做 bridge，直接使用本地复制 token
  → 含 prefill：
       仅 prefill 部分 pad 到 (linear_num_tokens_padded - num_decode_tokens)
       → 一次 PCP all_gather
       → hybrid_linear_ag_restore_idx → 全局/原序 tensor tuple
       ├─ query：hybrid_global_to_fa_idx → 本卡 DualChunk/virtual-row Q
       └─ cache inputs：保持全局/原序
            + global/original slot mapping
            → CacheWritePlan(prefill=ALREADY_GLOBAL)
            → 每个 PCP rank 写完整本地 KV cache
       → 本卡 Q 通过 FA block table 读取 cache 并计算
       → AG FA 输出
       → hybrid_fa_to_linear_idx → 本卡 linear，继续 GDN
```

- 「线性按顺序拼接就是原序」：对 **单卡本地** 线性段成立；**跨卡 AG 后** 是按 rank 拼接的，必须 `hybrid_linear_ag_restore_idx` 一次。  
- V1 的 `pcp_fa_padding_restore_idx`：把原序扩成 DualChunk **工作区**；V2 的 query 直接使用 `fa_batch`/`hybrid_global_to_fa_idx`，hybrid 侧不再复制该字段。  
- `CacheWritePlan` 不是新的切分索引袋；它按 decode/prefill segment 描述 cache inputs 的分布状态和与之对齐的 slot mapping，防止 MLA/GQA backend 对已全局化 KV 再做一次 AG。  
- 混合 decode+prefill batch 中，decode 保留本地复制语义，只 gather/restore prefill；输出 bridge 同理只处理需要恢复的 prefill，decode 输出直接留在 linear 前缀。

`CacheWritePlan` 最小契约：

| 字段 | 含义 |
|------|------|
| `segments` | 有序 `CacheWriteSegment`；fresh prefill、decode 或 mixed batch 使用同一协议 |
| `segment.kind` | `DECODE` 或 `PREFILL` |
| `segment.distribution` | `LOCAL_REPLICATED`、`LOCAL_FA` 或 `ALREADY_GLOBAL` |
| `segment.input_range` | 指向 bridge 返回 cache payload 中该 segment 的有效范围；禁止靠 `num_tokens` 猜 |
| `segment.slot_mapping` | 与该 segment cache inputs 同序；Hybrid prefill 使用 global/original slot，decode 使用本地 slot |
| `segment.valid_mask` | 输入范围包含 bridge/graph padding时必填并禁止写 cache；仅当 `input_range` 已是无 padding 的精确紧凑区间时可为 `None` |

Hybrid 2B 的 mixed plan 是 `DECODE/LOCAL_REPLICATED + PREFILL/ALREADY_GLOBAL`；pure decode 只有前一段且不进入 bridge。非 Hybrid 的纯 MLA PCP 可使用 `PREFILL/LOCAL_FA + gathered slot + backend AG`。三种路径由同一 per-group `CacheWritePlan` 统一表达，而不是在 backend 中读取 Manager 状态。

#### 3. 明确不建 / 复用上游（回应反馈 2、4）

| 不建的 hybrid 字段 | 原因 |
|--------------------|------|
| `max_num_tokens_across_pcp`（旧名） | 与上游 DualChunk `padded_num_tokens = max(per_rank_*)` **概念易混**；线性侧改用 `linear_num_tokens_padded` |
| `pcp_unpad_mask` / DualChunk pad 位 | 上游 DualChunk 已有 `is_padding`、`_padded_gather_idx`、local pad；**只加 hybrid 桥** |
| `pcp_enter_fa_restore_idx` + `pcp_fa_padding_restore_idx` 双套 | 合并为 `hybrid_linear_ag_restore_idx` + `hybrid_global_to_fa_idx` |
| `AscendPCPMetadata` 全套 q_head/tail、kv mask/nomask | MLA 跟上游；GQA 使用相同 virtual-batch 公共字段，不搬 V1 分片算子索引 |
| GDN 新 PCP dataclass 字段 | 接缝改 `initial_state_mode` 等现有字段即可 |

#### 4. bridge 只读视图（必须、仍用 hybrid_ 前缀）

经公共 forward context / attention common metadata 投影 **bridge 只读视图**，供 MLA/GQA adapter 进/出 FA。它应是独立小结构，不塞入 GDN metadata：

| 投影字段 | 来源 |
|----------|------|
| `hybrid_linear_ag_restore_idx` | 同上 |
| `hybrid_global_to_fa_idx` | 同上 |
| `hybrid_fa_to_linear_idx` | 同上 |
| `linear_num_tokens` / `linear_num_tokens_padded` | 同上 |

**不要**把上游 DualChunk 的 pad/slot 再抄一份进 hybrid 袋。这里的三个转换索引由 FA 外层 bridge 消费，不进入 MLA/GQA 算法内部；cache 输入分布和 slot 对齐关系放在独立 `CacheWritePlan`。

#### 5. `AscendInputBatch`

- **不加** hybrid 索引字段；布局状态只在当前 step 的 `HybridPreparedStep`。  
- 主 forward batch 由 Manager 产出（线性主路径）；FA 临时 DualChunk/virtual-batch 由上游逻辑 + bridge idx 完成。

#### 6. GQA / MLA 公共字段契约

**结论分两层：**

| 场景 | 是否够 |
|------|--------|
| 纯 MLA PCP（非 hybrid），跟上游 | **够**。不靠 `HybridPCPLayout`，吃上游 `PCPManager` 即可 |
| 纯 GQA PCP（非 hybrid） | 切分/公共视图够；仍需 GQA PCP backend adapter |
| hybrid：GDN 线性 + MLA/GQA | `HybridPCPLayout` 对公共 bridge 足够；FA 算法内部不再增加 V1 那袋切分索引 |

**MLA/GQA 共同消费的公共数据（不必再往 hybrid 袋里复制）：**

| 来源 | 字段 / 数据 | 层内用途 |
|------|-------------|---------|
| `PCPManager.prepare_attn` | global/original 与 **gathered** 两种 `slot_mappings` | Hybrid plan 的 prefill segment 使用 global/original；非 Hybrid 上游 PCP 使用 gathered |
| DualChunk `InputBatch` | `positions` / `query_start_loc` / `num_tokens` / `num_tokens_after_padding` / `is_padding` / virtual `num_reqs` | 正常 FA；local pad |
| 层 metadata（builder 填） | `num_decode_tokens` / `num_decodes` / `num_prefills` / `slot_mapping` / `seq_lens` / `block_tables` | AG 时区分 decode/prefill；注意力本身 |
| backend adapter | `pcp_world_size` / `pcp_rank`、`CacheWritePlan`、backend capability | 判断是否仍需 KV AG、cache 写入与计算 |

MLA/GQA 公共框架不读取 `q_head_idx`、`kv_*_mask/nomask` 等 V1 分片 KV 算子字段。V1 GQA 仅用于参考进/出 FA 桥接位置与数学关系；V2 backend 采用 virtual-batch + 每卡完整 KV 副本。

**hybrid 下 FA 还缺的不是「更多 FA 索引字段」，而是数据流约束：**

| # | 缺口 | 说明 |
|---|------|------|
| G1 | Manager 要同时有 **DualChunk 视图**、global slot 与 gathered slot | FA query 吃 DualChunk local batch；Hybrid 2B cache write 吃 global/original slot；非 Hybrid 上游 PCP 保留 gathered slot |
| G2 | FA 外包一层用 bridge idx | `hybrid_*` 在进/出 FA 时用；不进入 MLA/GQA 算法 metadata，可作为公共只读 bridge context |
| G3 | 主路径若是 linear，sample restore | 不能只用上游 DualChunk `_hidden_restore_idx`；pure decode 使用本地复制前缀，prefill 段 AG 后复用 `hybrid_linear_ag_restore_idx`，再按 decode-first 顺序拼回 global batch |
| G4 | Ascend GQA backend | 不搬 V1「PCP 分片 KV + mask/nomask」路径；消费 `ALREADY_GLOBAL` cache inputs，先写完整 cache，再让 virtual-row Q 读 cache |

**对 `HybridPCPLayout` 五字段的判定：**  
给「线性 ↔ FA」索引桥 **足够**；cache 输入分布不塞入 layout，而由 `CacheWritePlan` 表达；FA 层算子内部的 group 数据由 `PreparedGroupInputs` 提供。

### 4.6 Runner / ModelState 窄接口

保持上游执行骨架：

```text
prepare_inputs
→ prepare_attn
→ model_state.preprocess_state
→ model_state.prepare_attn
→ model forward
→ restore_for_sampling
```

Hybrid PCP 只在已有扩展点增加显式数据传递：

```text
prepare_inputs(global_batch)
  → AscendPCPManager 调用上游 partition 生成并保存 fa_batch
  → 使用独立 buffer 生成并保存 linear_batch
  → 生成 HybridPCPLayout
  → 对模型主路径返回 linear_batch

prepare_attn(linear_batch)
  → Ascend Runner 调用 Manager 完成私有 HybridStepBuilder
  → KV-attention group 准备 FA block tables、FA slot 与 per-group CacheWritePlan
  → GDN state group 准备 linear block tables/slot
  → 按 group capability 生成 PreparedGroupInputs
  → finalize 为只读 HybridPreparedStep
  → model_state.bind_hybrid_step(prepared_step)
  → 返回 legacy_block_tables / legacy_slot_mappings

AscendMambaHybridModelState.prepare_attn()
  → 一次性读取已绑定的 HybridPreparedStep
  → capability=DUAL_CHUNK_VIRTUAL 的 group 使用 fa inputs
  → capability=CONTIGUOUS_CAUSAL_STATE 的 group 使用 linear inputs
  → 分组构建后按 layer_name 合并 metadata dict，并把 bridge/plan 投影进相应 metadata
  → 返回带 hybrid_forward_view 属性的 HybridAttentionMetadataMap（仍是 dict 子类）
  → finally 清除 ModelState 的一次性绑定；backend forward 只读 metadata，不再读 ModelState/Manager
```

约束：

- 不复制或重写 `execute_model()`。  
- ModelState 不持有 Runner，不反向读取 Runner 私有状态。  
- backend 不通过全局单例访问 Manager。  
- `bind_hybrid_step()` 是显式、单 step、只读、一次性依赖注入；重复 bind 同一 `step_id` 或覆盖未消费的当前 step 必须报错。`ModelState.prepare_attn()` 必须在 `finally` 中 clear；`preprocess_state()` 若抛异常也必须 clear 后原样重抛。新一 step 的 `prepare_attn()` 只允许清理更旧 `step_id` 的异常残留，并记录诊断，不能静默把同 step 覆盖掉。投影到 metadata 的 tensor view 由该次 forward 持有，backend 禁止延迟读取 ModelState/Manager。这样无需给 `execute_model()` 增加清理分支。  
- `HybridAttentionMetadataMap` 保持 `dict[str, Any]` 行为，只额外挂同一份只读 `HybridPCPForwardView`。现有 `AscendPlatform.set_additional_forward_context()` 从该属性取值，并写入当前 forward context 的 `hybrid_pcp_forward_view`；若需兼容旧字段名 `max_tokens_across_pcp`，**取值必须等于 `linear_num_tokens_padded`**，禁止填 DualChunk/`fa_batch` 的 max（避免与 §4.5 命名禁令冲突）。MoE 通过现有 `_EXTRA_CTX` 代理读取。禁止用模块全局变量或 Manager 反向引用传递 valid mask。  
- KV group 分类依据 backend/builder capability strategy，不使用 `model_type` 白名单，也不以 `isinstance(AttentionSpec/MambaSpec)` 作为长期协议。cache spec 只负责物理存储信息。  
- `preprocess_state` 始终使用 linear request 行和 Mamba/GDN block table。  
- `restore_for_sampling` 覆盖上游 FA restore：pure decode 不通信；mixed/full prefill 只 AG prefill 段并复用 `hybrid_linear_ag_restore_idx`，最后把本地 decode 复制前缀与原序 prefill 拼成 global batch。禁止为 sample 再增加第四套 restore idx。

dummy/profile/capture 不能绕开该契约：覆盖已有 `prepare_dummy_attn()` 扩展点，构造同类型的 dummy `HybridPreparedStep`，所有 **cache-write segment** 的 slot 填 `PAD_SLOT_ID`、`valid_mask=false`，避免污染 cache；`HybridPCPForwardView.linear_valid_mask` 则按本次 profile/capture 要实际执行的 dummy rows 置真，graph 尾部 padding 置假，保证 MoE kernel/通信确实被 warmup。再走同一 bind/consume 流程，不得让 backend 在无 step 时猜布局。

Ascend 现有 MoE `profile_run()` 可能调用 `_dummy_run(..., skip_attn=True)`；Hybrid PCP 必须在这一现有窄 hook 中强制 `skip_attn=False`（或直接 fail-fast），使 dummy step 和 forward view 能建立，不能新增模块全局变量绕过。第一阶段 full graph 本身 fail-fast；上述约束同时覆盖 eager/piecewise 的 profile warmup，仍不需要重写 `execute_model()`。

#### 4.6.1 group capability（4A）

最小 capability 不是单一 `supports_pcp: bool`，而是 backend/builder 注册的只读策略：

```text
PCPGroupCapability
  ├── input_layout:
  │     DUAL_CHUNK_VIRTUAL | CONTIGUOUS_CAUSAL_STATE
  ├── accepted_cache_input_distributions:
  │     frozenset[LOCAL_REPLICATED | LOCAL_FA | ALREADY_GLOBAL]
  │     # GDN 等无 KV cache 的 group 为空集
  ├── cache_replication:
  │     REPLICATED_PER_PCP_RANK
  ├── supports_piecewise: bool
  └── supported_features:
        dtype / quant / sliding_window / prefix_cache / dcp / ...
```

- MLA/GQA backend 声明 `DUAL_CHUNK_VIRTUAL`；GDN builder 声明 `CONTIGUOUS_CAUSAL_STATE`。  
- `PCPGroupCapability` 定义在 worker/v2 的中立接口模块；backend/builder 初始化时注册，Runner 只缓存按 `kv_cache_group_id` 对齐的不可变描述符并传给 Manager。Manager 只能依赖上述枚举/协议，禁止 import MLA、GQA、GDN 具体实现。  
- Hybrid 2B 的 per-group plan 可把 prefill segment 收窄为 `ALREADY_GLOBAL`，但不能与 backend 静态 capability 声明冲突。  
- 同一 kv-cache group 内若有多个 attention builders/layer specs，初始化时必须逐一取 capability 并验证公共字段完全兼容；不能只看第一个 builder。若 input layout、cache distribution/replication 或 feature 支持不同，应由 KV-cache 配置拆组，否则 fail-fast。  
- 新 backend 只需注册 capability 和 adapter；Manager、Runner、GDN 接缝及 bridge 索引不增加模型特判。  
- 校验分两阶段：模型加载前做粗粒度配置 fail-fast；backend/group 初始化后逐 group 校验 capability。不得因为早期拿不到 backend 实例而退回 `model_type` 白名单。

#### 4.6.2 模块归属与依赖方向

下表约束的是职责和 import 方向；若实施时文件名因主干变化调整，边界不能变化：

| 逻辑模块（建议落点） | 持有什么 | 禁止依赖 |
|----------------------|----------|----------|
| `attention/context_parallel/hybrid_pcp/contracts.py` | backend-facing `PCPGroupCapability`、layout/distribution 枚举和 bridge protocol | Worker/Runner、模型类、具体 backend 实现 |
| `attention/context_parallel/hybrid_pcp/{bridge,gqa_adapter,mla_adapter,gdn_adapter}.py` | packed payload enter/exit；projection、cache-first、GDN state 接缝与 backend kernel 适配 | Runner/Manager 私有状态、模型白名单 |
| `worker/v2/pcp/contracts.py` | frozen step/layout/cache-plan/view 数据结构 | 模型类、具体 MLA/GQA/GDN backend、Runner 实例 |
| `worker/v2/pcp/{layout,batch,cache_plan}.py` | 纯布局索引、linear batch 变换、cache write plan | 具体 attention/MoE/GDN 实现 |
| `worker/v2/pcp/{pcp_manager,hybrid_pcp_manager,capability,validation}.py` | 标准/Hybrid Manager、组合根、预分配 buffer、global/gathered slot、restore、能力解析和 fail-fast | 具体 attention/MoE/GDN 实现 |
| `worker/v2/pcp/group_preparer.py` | 按 capability 选择 group 输入视图和 cache plan；可注入的 layout strategy registry | Runner、模型类型、attention 数学 |
| `worker/v2/model_runner.py` | `prepare_attn()` / `prepare_dummy_attn()` / profile 窄 hook；finalize、bind、legacy 投影 | attention 数学、GDN 接缝 |
| `worker/v2/model_states/mamba_hybrid.py` | 一次性 consume step；按 group 构建 metadata；`HybridAttentionMetadataMap` | Runner/Manager 反向引用 |
| 现有 `attention_v1.py` / `mla_v1.py` | PCP capability 声明和 adapter 薄分发入口 | Worker/V2 合同、Manager、布局构建算法 |
| 现有 `ops/gdn.py` / `chunk.py` / GDN builder | 对 `gdn_adapter` 的薄调用和 linear metadata | FA bridge、Manager |
| `ops/fused_moe/hybrid_pcp.py` + 现有 MoE prepare/finalize | 独立 compact/restore 生命周期；MoE 主路径只保留薄调用 | FA layout/index、Manager |
| `platform.py` additional forward context | 从 metadata map 投影只读 `HybridPCPForwardView` | Manager、step builder |

依赖只能单向流动：

```text
backend/builder capability
        ↓
Runner 的 immutable group descriptors
        ↓
PCPManager / HybridStepBuilder
        ↓ finalize
HybridPreparedStep
        ↓ one-shot consume
ModelState → per-layer metadata / forward view
        ↓
MLA/GQA adapter、GDN、MoE
```

禁止反向箭头。尤其禁止 backend 调 `get_pcp_manager()`、Manager import backend 类、MoE 读取 FA bridge idx，或为单个模型在 Runner 增加条件分支。

### 4.7 MLA/GQA backend adapter

#### 4.7.1 公共 token-layout bridge

bridge 只操作 token 维，不理解 attention 数学。backend 在 projection/RoPE 边界提供一个可 AG 的 packed QKV/latent payload，bridge 只返回 query/global-cache 两种 token view，具体 unpack 仍归 backend：

```text
enter(packed_payload, per_group_cache_write_plan)
  → pure decode: 原样返回本地 packed payload
  → prefill:
       pad → 单次 PCP all_gather → hybrid_linear_ag_restore_idx
       ├─ query_view  = global[hybrid_global_to_fa_idx]
       └─ cache_view  = global
          prefill_segment.distribution = ALREADY_GLOBAL

exit(attention_output)
  → pure decode: 原样返回
  → prefill: PCP all_gather → hybrid_fa_to_linear_idx → 本卡 linear output
```

`enter()` 返回只读 `BridgedAttentionInputs`：

| 字段 | 行数 / 顺序 | 所有者 |
|------|-------------|--------|
| `query_packed` | `F`；local replicated decode 前缀 + 本卡 FA prefill + FA padding | bridge 产 view，backend unpack |
| `cache_packed` | `T`；local replicated decode 前缀 + global/original real prefill | bridge 产 view，backend unpack |
| `cache_write_plan` | per-group segments，range 必须精确覆盖 `cache_packed` 的有效行且不重叠 | Manager 构造，backend 消费 |
| `unpack_descriptor` | packed Q/K/V 或 MLA latent 的尾维切片；不含 token-layout 索引 | backend 构造并校验 |

这里的“单次 AG”是指 **每个 FA 层、每个 prefill forward 恰好一次 entry payload collective**；exit output AG 另计，step 级长度/索引元数据通信不计为 payload AG。GQA 应优先使用 fused packed QKV，MLA 使用可一次通信的 packed latent/cache operands。position 可导出的 RoPE/cos/sin 等在 global/FA positions 上重算；不能重算且无法与 payload 同 dtype/shape 打包的 token-wise tensor，第一阶段应 capability fail-fast，禁止 backend 静默增加第二次 entry payload AG。

进/出时机参考 Ascend V1 Hybrid GQA：

- GQA：Q/K/V projection 与 RoPE 后进入 bridge；attention 输出退出 bridge 后继续 gate / `o_proj` / residual。  
- MLA：在 Ascend MLA adapter 的 latent projection 与 attention 边界进入；attention 输出退出后继续 `o_proj`。  
- 不包整个 DecoderLayer，不把 bridge 算法放入 Manager。
- RoPE cos/sin、position 派生量、valid/padding mask 等，只要与被转换 tensor 的第 0 维对齐，就必须随 packed payload 转换，或在目标布局中按 global/`fa_batch.positions` 明确重算；禁止沿用 linear 顺序的辅助 tensor 配 FA query。per-token quant scale 因 7A 暂不支持；后续启用 quant 时须通过 capability 声明 pack/recompute 方案。  
- 编译优化不得把 projection/RoPE/cache-write 跨过 bridge 边界后仍假设原 token 顺序；backend capability 必须明确其 fusion 是否兼容 Hybrid bridge。

#### 4.7.1.1 外层 linear shape 与内层 FA shape 契约

```text
model input / forward context / BatchDescriptor / layer output = linear shape
backend FA temporary query/output                         = FA shape
backend cache inputs                                      = global/original shape
```

- 上式的 linear shape 指 `linear_num_tokens_padded` 行；所有有效计算/状态更新再由 `linear_num_tokens` 与 `linear_valid_mask` 收窄。  
- `set_forward_context(num_tokens/is_padding/positions)` 保持 linear 主路径语义，不在 attention 层内动态切换全局 context。  
- 标准 Attention 外层 output 仍按 linear query 分配；adapter 在内部使用预分配 FA temporary，`exit()` 后才写入外层 output。  
- Ascend MLA 不得用 `_EXTRA_CTX.num_tokens` 推导 FA temporary 长度；FA 长度取 backend metadata，linear 输出长度取 bridge view。  
- `o_proj`、attention gate 和 residual 只消费 `exit()` 后的 linear tensor。  
- piecewise graph 的 `BatchDescriptor` 仍按 linear shape 选图；FA temporary 的 shape bucket 独立由 backend metadata/capability 管理。

#### 4.7.2 MLA adapter

- 复用上游 MRv2 MLA virtual-batch、metadata 和 DualChunk attention 数学。  
- Hybrid 2B 收到 prefill segment `ALREADY_GLOBAL` 时，直接使用 global latent KV + global slot 写完整 cache，**跳过**上游 `maybe_gather_*_cache_inputs` 的第二次 KV AG。  
- 非 Hybrid MLA PCP 仍走上游 `LOCAL_FA + gathered slot + KV AG`，两条路径在 `CacheWritePlan` 分发处汇合。  
- MLA 算法内部不新增 Hybrid 切分字段。  
- Hybrid 增量只有公共 bridge、cache-write plan 分发和 FA temporary→linear output 适配。

#### 4.7.3 GQA adapter（3A：cache-first）

GQA 使用与 MLA 相同的 FA 公共输入和 bridge，但不得直接依赖 `MLACommonMetadata` 具体类型。GQA backend 自行完成：

1. 从 `PreparedGroupInputs` 构建标准 Attention metadata。  
2. Hybrid prefill 消费 per-group plan 的 `PREFILL/ALREADY_GLOBAL` segment，不再 AG K/V；decode 消费 `DECODE/LOCAL_REPLICATED` segment。  
3. 使用 global/original slot mapping，让每个 PCP rank 先写入完整本地 KV cache 副本。  
4. virtual-row 只作用于 Q；每个 head/tail row 通过自己的 block table、seq end 和 query start 读取同一真实请求的完整 cache，保证因果上界。  
5. 第一阶段禁止退回“给每个 virtual row 复制整份 K/V”的 `PrefillNoCache` 方案；若底层 FIA 的 no-cache 接口无法表达 Q/KV 不同 row 拓扑，必须选择 cache-hit/cache-first kernel path。  
6. 调用 GQA/FIA kernel，返回本卡 FA temporary output，再走公共 `exit()` 写回 linear output。  
7. 处理 TP 下 `num_heads` / `num_kv_heads` 的本地形状。

V1 `attention_cp.py` 只用于参考 Hybrid QKV enter/exit 和因果数学；不直接搬其「PCP 分片 KV + mask/nomask + Out/LSE 合并」架构。

初期公共框架实现允许直接由 MLA 路径填充并验证 `PreparedGroupInputs`；GQA 接入时消费同一公共 step context，不新增第二套切分字段。若 GQA 接入需要大改 Manager、Runner、GDN 或 bridge 索引，说明公共契约未正确收敛，必须重新评审，而不是在 backend 外增加模型特判。

### 4.8 MoE 与 EP

第一阶段支持 dense 与 MoE Hybrid，组合为 PCP+TP，**EP 关闭**。MoE 在 **linear 主路径**上消费 token；与 FA DualChunk 正交。PCP 下各卡 `linear_num_tokens` 可能不等，通信前必须局部 pad/unpad，并显式携带 valid-token 语义：

```text
linear hidden
→ 构造 local_valid_mask = arange < linear_num_tokens
→ 参考 V1 prepare/finalize：hidden、router_logits、input_ids/scale 等
  在进入 PCP collective 前统一 pad 到 linear_num_tokens_padded
→ PCP AG 后同步 AG valid_mask
→ topk/dispatch 前 compact valid token，或使用 backend 明确支持的 invalid-token 语义
→ MoE dispatch/combine（第一阶段 PCP AG/RS + TP；后续可替换 EP A2A/MC2）
→ 立即 unpad 回 linear_num_tokens
```

要求：

- `valid_token_mask` 是 MoE adapter 的显式输入，不加入 `HybridPCPLayout`；来源是当前 forward context 的只读 `HybridPCPForwardView`，由 ModelState 经 `HybridAttentionMetadataMap` 投影，不能在 MoE 中读取 Manager/ModelState。  
- 优先在 topk/dispatch 前按 gathered valid mask compact；若 kernel 支持 invalid expert id，可令 padding 的 `topk_id=-1`、weight=0 并由 dispatcher 跳过。只有当 dispatcher 明确保证 zero-weight token 不计 capacity/EPLB 时，才允许 deterministic dummy expert + weight=0；否则该路径必须 fail-fast，不能靠数值为零假设“不会占容量”。  
- padding token 不产生有效专家贡献，不进入 EPLB 统计；EPLB 记录前必须按 gathered valid mask 过滤。  
- padding 不更新任何模型状态、不影响真实 token 输出。  
- 不能让整网 forward 永久保持 padded hidden。  
- **禁止**把「EP=False」写进 `HybridPCPLayout` / Manager 契约。
- shared expert、外置 gate、量化 scale、`input_ids` 等所有 token-wise MoE 支路必须使用同一 pad/mask/unpad adapter，不能只补 routed hidden；shared-expert 的无效行也必须在 combine 前归零。

#### 4.8.1 EP 工作量与架构冲击（评估结论）

| 层次 | 对双视图 / GDN / FA 桥 | 对 MoE 层 |
|------|------------------------|-----------|
| 打开 EP | **冲击小**（仍吃 linear 局部 token） | **冲击中～大**（进程组、dispatch 域、与 PCP/TP 组合、pad 对齐） |
| 工作量粗估 | 不改 Manager/layout 五字段 | MoE 通信路径联调为主（数人日～数周，视现有 MC2/A2A 成熟度） |

结论（与拍板一致）：

1. **第一阶段关闭 EP**，验收 dense/MoE + PCP+TP。  
2. **架构按后续必叠 EP 设计**：MoE 只认 `linear_num_tokens(_padded)` + `valid_token_mask`；禁止把 EP=False 写进 Manager/双视图契约。  
3. pad/unpad、valid mask、EPLB 过滤与 token 维契约第一阶段一次到位；后续 EP 只联调进程组与 dispatch/combine，不倒逼改 DualChunk、GDN 接缝或 `HybridPCPLayout`。

### 4.8.2 Decode 写盘：FA KV vs GDN state（V1 对照 → 规范）

两类缓存 **分开写**；GDN **不**套用 FA 的 `gathered_kv_write_mask`：

| | FA KV cache | GDN conv/SSM state |
|--|-------------|-------------------|
| 粒度 | token 级 slot | **请求级** `state_indices` |
| Prefill（V1） | 分片存储：只写本卡 slot；算前 AG 本轮 KV | 接缝后 **各卡都写** 同一份序列末状态（V1 常取物理末 rank；**V2 必须取最后有效 rank**，C35） |
| Decode（V1） | 各卡用本地分片 KV 算 → out+lse AG 合并；写 cache 仍按本卡 slot | **无 PCP AG**：各卡复制同一 decode token，对 **本地已对齐的 state** 各自 `recurrent` 更新（状态副本一致则结果一致） |
| Prefill（上游/本方案 FA） | AG 后 **每卡本地** 写全量副本 | （GDN 不走这条） |
| Decode（上游/本方案 FA） | token **各卡复制计算**；K/V **各卡写入自己的本地全量 cache**（不是只有 rank0 刷 KV） | **跟 V1**：各卡本地更新 state；不套 FA 的 write_mask |

**澄清 `gathered_kv_write_mask`（易误解）：**

- decode 在每张卡的 local batch 里都会出现一份 → 拼成「按 rank 展开的 gathered 布局」时，同一 decode **slot 会重复出现**。  
- `write_mask` 在展开布局里 **只保留一份有效 decode slot 记录**（与 rank0 那段对应），避免 slot 表重复；`pcp.py` 注释是 *Keep replicated decode writes local*。  
- **纯 decode**：走 early return，各卡用 **本地** K/V + **本地** `slot_mapping` 写自己的 cache → **rank0/rank1 都会更新本地 KV**。  
- **混合 batch**：prefill 段 AG；decode 段各卡仍用本地算出的 K/V 写入，slot 取 gathered 表里那份有效 decode slot（数值上各卡写到相同全局 position/slot，但落在各自显存副本里）。

一句话：

```text
FA KV decode：各卡都算、各卡都写【本地全量副本】；write_mask ≠「只有 rank0 更新 KV」
GDN state：跟 V1（prefill 接缝对齐；decode 各卡本地递推）
```

### 4.8.3 线性 partition 算法（方案选择）

| 选项 | 做法 | 评价 |
|------|------|------|
| **L1（= V1 hybrid 线性；推荐）** | `_get_cp_local_seq_lens(..., dcp=1, interleave=1)` → 每卡 **连续因果段**；**保持原 `num_reqs`**；decode **复制** | V1 `pcp_use_hybrid_attn` 主路径就是这个，**不是** L2 |
| L2 | 从 DualChunk `RankSegment` 反推再拼成线性 | **V1 没这么做**；易与因果段错位；不推荐 |
| L3 | 简单均分 `L/pcp` | 边界易漂 |

说明：V1 **非 hybrid** 主路径才是 DualChunk（原 req 维拼 head\|tail）；hybrid 时主路径改线性，FA 靠 enter/exit 索引桥到 DualChunk 语义。  
**定案 L1**：`fa_batch` = 上游 `partition_batch`（virtual-batch）；`linear_batch` = V1 线性（原 req 维）。

### 4.9 图模式

里程碑顺序已定（6A）：

1. M0～M5：**eager 必达**，先完成 MLA/GQA+GDN、dense/MoE、PCP+TP 的正确性闭环。  
2. M6：**piecewise 必达**，紧随 eager；不是改变公共接口的第二套实现。  
3. full graph：不支持。

| 区域 | 处理 |
|------|------|
| linear↔FA bridge / CacheWritePlan 通信 | graph break（piecewise） |
| GDN PCP 接缝通信 | graph break |
| MoE pad/unpad 与通信 | graph break |
| 计算主体 | 允许 piecewise capture |
| full graph | 不支持 |

双视图、索引和临时 buffer 预分配；图内不动态分配。`BatchDescriptor` 使用 linear token bucket；FA temporary 使用 backend 独立 bucket，二者不能共用一个 token-count 变量。

#### 4.9.1 buffer 容量与越界契约

记：

```text
P = pcp_world_size
L = linear_num_tokens_padded
F = fa_batch.num_tokens_after_padding
T = 当前 step 全局真实 token 数
D = 各 rank 相同的 decode token 数
```

保守容量下界（允许多分配，禁止少分配）：

| buffer | 所有者 | 最小 token 行容量 |
|--------|--------|-------------------|
| linear/FA InputBatch 的 token/position/index | Manager | 分别按 `L` / `F` 及 virtual-row 上限 |
| 外层 linear hidden/output | Runner/模型现有 buffer | `L` |
| entry packed payload AG workspace | 公共 bridge + backend adapter | `P * (L - D)`；若实现选择整 batch 通信则取 `P * L`，但必须过滤重复 decode |
| global/original cache inputs | backend adapter | `T`，且 `T <= P * L` |
| FA local query/output temporary | backend adapter | `F` |
| FA output AG workspace | 公共 bridge + backend adapter | `P * (F - D)`；保守预分配可取 `P * F` |
| legacy merged slot row | Manager | `max(T, P * F, L)`；无效尾部填 `PAD_SLOT_ID` |
| MoE hidden/router/mask collective workspace | MoE prepare/finalize | `P * L`（再叠 DP/EP 时由 MoE adapter 扩展） |

- 初始化时按 scheduler 最大 token/request 配置计算上界，并把 DualChunk 每请求 padding、virtual rows 上限 `2 * max_num_reqs` 纳入容量。  
- 每 step 在 collective 之前检查 `D <= min(L,F)`、`L/F/T` 与预分配上界；越界必须 fail-fast，禁止截断、复用另一视图 buffer 或临时动态扩容。  
- entry payload 是 backend 打包后的单 tensor；Q/K/V 或 latent 的尾维拆分由 backend 的只读 unpack descriptor 解释。positions/mask 等重算辅助 buffer 单独预分配，不与 payload 尾维混用。  
- 上表是跨模块的 row-capacity 契约，不代表 Manager 统一持有所有 tensor；layer/hidden/head/dtype 相关 workspace 必须由对应 backend adapter 预分配，禁止做一个由 Manager 按最大模型维度分配的“万能 buffer”。  
- eager 与 piecewise 使用同一容量公式；piecewise 只额外要求 shape bucket 稳定。

### 4.10 配置与 capability 校验

- **目标门禁均可删除**（上游 MLA-only、Ascend「禁 hybrid+PCP」等）。  
- 第一阶段支持矩阵（7A）：BF16/FP16、普通因果 MLA/GQA、`mamba_cache_mode=none`、dense/MoE、PCP+TP、eager；M6 增加 piecewise。  
- 未支持组合 **fail-fast**：`enable_chunked_prefill=True`/chunked prefill、continued prefill、`enable_prefix_caching=True`/prefix-cache hit、`mamba_cache_mode=align/all`、quant、sliding window、spec、PD/KV connector、MM、PP、LoRA、DCP、EP、dynamic EPLB、full graph。  
- 不使用 `model_type` 白名单；初始化前做配置级校验，backend/group 建立后使用 §4.6.1 capability 逐组校验。  
- 运行时若出现 `num_computed_tokens>0` 的 prefill（包括 prefix-cache hit、恢复请求或其它 continued prefill），第一阶段必须在 partition 前 fail-fast，不能把它误当成 fresh full prefill。  
- capability fail-fast 必须报告具体 layer/group、backend、feature 和当前配置，不能只报“Hybrid PCP unsupported”。  
- 分布式 fail-fast 必须 **rank-consistent 且早于 payload collective**。只依赖全局配置/调度输入的检查在 partition 前完成；依赖本地 token 数、decode ids/hash 或 buffer 容量的检查，由 Manager 每 step 一次性用固定小标量 collective 汇总状态，再让所有 rank 同步通过或同步报错，不能在每层重复。禁止某 rank 已抛异常而其它 rank 进入 QKV/MoE/GDN payload collective。  
- §5.2 的 V1 `attention_cp` 材料 **仅对照**，实施 **禁止** 搬「PCP 分片 KV + mask/nomask + Out/LSE 合并」进 V2 FA（见 §4.7）。

---

## 5. 最终决策清单

| # | 结论 | 日期 |
|---|------|------|
| C1 | 本讨论主范围：**KV-attention（GQA/MLA 的 token 切分同构）+ GDN** | 2026-07-24 |
| C2 | PCP **切分**：凡 FA 为 KV 结构 → 统一 **DualChunk**；不因 GQA/MLA 换切法 | 2026-07-24 |
| C3 | GQA vs MLA 的 **token 切分语义一致**；bridge 操作数、projection/RoPE 边界、KV/cache、metadata 和 kernel 均由 backend adapter 区分 | 2026-07-24 |
| C4 | V2 跟上游：**PCP 不切分 KV**；每卡存全量副本 | 2026-07-24 |
| C5 | 全量 KV 来源：PCP collective 后由每个 rank 写入自己的完整本地 cache；Hybrid 用 V1 式单次 QKV/latent AG，非 Hybrid MLA 保留上游 local-KV AG | 2026-07-24 |
| C6 | **层间 hidden 不聚合**；请求级 restore 只在整网 forward 后 / sample 前 | 2026-07-24 |
| C7 | Hybrid **沿用 V1 布局桥**：仅进/出 FA 时 AG + DualChunk 再切；平时（含 GDN）保持线性局部 hidden | 2026-07-24 |
| C8 | 宿主框架：**跟上游 MRv2 PCPManager**（高内聚低耦合）；不把 V1 runner 深耦合路径整段搬进 V2 | 2026-07-24 |
| C9 | FA KV：**跟上游不分片**；本阶段 **不考虑 DCP** | 2026-07-24 |
| C10 | GDN state：**跟 V1**——不分片/可复制；接缝跟 V1（conv + SSM） | 2026-07-24 |
| C11 | MLA PCP 对齐上游；GQA 复用同一 virtual-batch/完整 KV 公共契约并新增 backend adapter；GDN 对齐 V1 线性接缝 | 2026-07-25 |
| C12 | **数据流定案**：主路径线性；FA 旁路 DualChunk；Hybrid 进 FA 单次 AG 得到本地 Q + global cache inputs，backend 不二次 AG K/V；层间不 AG hidden | 2026-07-25 |
| C13 | GDN 接缝：**继续用 V1**——`gdn.py` / `chunk.py` / `gdn_attn_builder` 留 ops；Manager 只保线性主路径；不新建第二套 | 2026-07-25 |
| C14 | 第一阶段叠加：**PCP+TP**；不做 chunked/continued prefill、prefix hit、spec/PD/MM/PP/LoRA/DCP/EP/full graph | 2026-07-25 |
| C15 | 第一版结构收窄：**MLA/GQA + GDN**；dense+MoE；不以 model_type 白名单；经典纯 Mamba 等非本版验收 | 2026-07-25 |
| C16 | FA 的 DualChunk/virtual-batch 数学按上游；Hybrid cache 写入通过 `CacheWritePlan` 复用 V1 单 AG；无后端时可先按 MLA 公共字段开发切分/双视图 | 2026-07-25 |
| C17 | **目标门禁均可删**；未支持组合 capability fail-fast | 2026-07-25 |
| C18 | 验收：负责人后续 **真机自行验收**；文档不绑定详细矩阵为门禁 | 2026-07-25 |
| C19 | Hybrid 主干：语义移植 `58d1b6502`+`291658e90`；先 `pcp=1` 再 PCP（O3=A） | 2026-07-25 |
| C20 | 双视图：Manager 使用私有 `HybridStepBuilder`；Runner `prepare_attn()` finalize 只读 `HybridPreparedStep`；独立预分配 buffer | 2026-07-25 |
| C21 | Runner：不重写 `execute_model()`；`prepare_attn()` 显式 bind step，并返回旧签名兼容的 per-group 合并结果（1A） | 2026-07-25 |
| C22 | HybridModelState：按 group capability 选 FA/linear view 并合并 metadata | 2026-07-25 |
| C23 | MLA/GQA：切分/索引语义一致；bridge 操作数与 cache/metadata/kernel 差异在 backend | 2026-07-25 |
| C24 | 进/出 FA 时机参考 V1 Hybrid GQA | 2026-07-25 |
| C25 | MoE：第一阶段支持 PCP+TP、关闭 EP；pad/unpad + valid-token/EPLB 过滤一次到位，架构可后续叠 EP（5A） | 2026-07-25 |
| C26 | 图模式：M0～M5 eager 必达，M6 piecewise 必达；通信区允许 graph break；full graph 不做（6A） | 2026-07-25 |
| C27 | Decode：FA/GDN **各卡都算、各卡都写本地副本**；`write_mask` 只做 gathered slot 去重，≠ 仅 rank0 更新 KV；GDN 不套 FA mask | 2026-07-25 |
| C28 | 线性 partition：**L1** = 复用 V1 `_get_cp_local_seq_lens` 连续段 + 原 req 维；FA 仍上游 DualChunk | 2026-07-25 |
| C29 | Hybrid FA bridge：每 FA 层使用 V1 式 **单次 entry QKV/latent payload AG**；plan 的 `PREFILL/ALREADY_GLOBAL` segment 阻止 backend 二次 AG；纯 decode 旁路（2B） | 2026-07-25 |
| C30 | GQA prefill：**cache-first**；先写每卡完整 KV cache，virtual-row Q 再按 block table 读 cache（3A） | 2026-07-25 |
| C31 | 扩展协议：backend/builder 注册 `PCPGroupCapability`；不用模型白名单或长期 `isinstance` 分支（4A） | 2026-07-25 |
| C32 | 首阶段 backend 范围：BF16/FP16、普通因果 attention、`mamba_cache_mode=none`、`enable_prefix_caching=False`；quant/sliding/prefix 后置（7A） | 2026-07-25 |
| C33 | 外层模型/forward context 始终 linear shape；FA/global tensor 仅为 backend 临时视图，exit 后写回 linear output | 2026-07-25 |
| C34 | `HybridPCPLayout` 不重复保存 `linear_positions`；唯一数据源为 `linear_batch.positions` | 2026-07-25 |
| C35 | 零长度线性段执行 state identity，最终状态取最后一个有效 rank，不能无条件取物理末 rank | 2026-07-25 |
| C36 | `CacheWritePlan` 按 kv-cache group 持有并分 decode/prefill segment；mixed batch 明确为 `LOCAL_REPLICATED + ALREADY_GLOBAL`，不存在含糊的单一 distribution | 2026-07-25 |
| C37 | `HybridPreparedStep` 带单调 `step_id` 并一次性 bind/consume；ModelState 在正常/异常路径清绑定并把只读 view 投影进 metadata；dummy/profile 走同一契约 | 2026-07-25 |
| C38 | legacy block/slot 只是 per-group 输入的同-buffer 兼容投影，不允许形成第二权威数据源 | 2026-07-25 |
| C39 | capability 协议位于中立接口层；Manager 只依赖枚举/协议，不 import MLA/GQA/GDN 实现 | 2026-07-25 |
| C40 | mixed batch 只通信 prefill 段；decode 数、请求和 positions 必须跨 PCP rank 一致，否则 collective 前 fail-fast | 2026-07-25 |
| C41 | MoE 的 linear count/valid mask 通过 `HybridAttentionMetadataMap → AscendPlatform.set_additional_forward_context → HybridPCPForwardView` 传递；不访问 Manager/ModelState，也不新增模块全局状态 | 2026-07-25 |
| C42 | Hybrid PCP 的 dummy/profile/capture 走同一 prepared-step 契约；cache writes 全禁，MoE dummy valid rows 保留；现有 `skip_attn=True` warmup 必须在 profile hook 中关闭或 fail-fast | 2026-07-25 |
| C43 | 模块依赖单向：capability → Runner descriptors → Manager/builder → prepared step → ModelState metadata → backend/ops；禁止任何反向访问 | 2026-07-25 |
| C44 | 所有运行时门禁在 payload collective 前做 rank-consistent 汇总，所有 PCP rank 同步通过或同步失败，避免 HCCL hang | 2026-07-25 |
| C45 | sample restore 复用 prefill 的 `hybrid_linear_ag_restore_idx`；pure decode 不 AG，mixed batch 将本地 decode 前缀与 AG/restore 后的 prefill 拼回 global batch，不新增第四个索引 | 2026-07-25 |
| C46 | kv-cache group 内所有 builder 的 capability 必须兼容；不能只取第一个，异构能力需拆组或 fail-fast | 2026-07-25 |
| C47 | dynamic EPLB 延续当前 MRv2 限制并在第一阶段 fail-fast；valid-mask 过滤 hook/UT 先完成，供后续 EP/EPLB 使用 | 2026-07-25 |

---

## 5.1 方案敲定状态总览

### 第一版原则

| 主题 | 结论 |
|------|------|
| 范围 | KV-attn + GDN；切分不区分 GQA/MLA |
| 宿主 | 上游 MRv2 `PCPManager`；高内聚低耦合（C8/C20） |
| Token 切分 | V1 双布局：线性给 GDN，FA 进出 DualChunk 桥 |
| FA KV | Hybrid 每 FA 层单次 entry QKV/latent payload AG 后每卡写完整 cache；plan 的 `PREFILL/ALREADY_GLOBAL` segment 禁止 backend 二次 AG；不做 DCP |
| Hidden | 层间不 AG；sample 前 restore |
| GDN state / 接缝 | 跟 V1：不分片 + ops 内 AG 接缝（C13） |
| 数据流 | C12/C29：线性主路径 + FA DualChunk 旁路；pure decode 不做 bridge |
| 生命周期 | `prepare_attn()` finalize/bind `HybridPreparedStep` 并保持旧返回签名；不改写 `execute_model()` |
| attention | `PCPGroupCapability` 驱动；MLA 先落地，GQA 复用切分并用 cache-first backend adapter |
| 第一版叠加 | PCP+TP；M0～M5 eager 必达，M6 piecewise 必达；不做 chunked/continued prefill |
| 第一版模型 | MLA/GQA + GDN（收窄）；dense + MoE |
| 第一版精度/语义 | BF16/FP16、普通因果 attention、`mamba_cache_mode=none`、`enable_prefix_caching=False` |
| EP | 第一阶段关闭并 fail-fast；MoE 的 valid-token 契约预留后续 EP/EPLB 扩展 |
| Hybrid 主干 | 语义移植 `mrv2`，先 `pcp=1` 再 PCP |
| 门禁 | 目标门禁均可删；quant/sliding/prefix hit 等未支持组合按 capability fail-fast |
| 验收 | 真机自行验收（C18） |
| 线性切分 | L1 = V1 连续线性（C28） |
| Decode 写盘 | 各卡写本地 FA KV / GDN state；write_mask≠仅 rank0（C27） |

选择题 `1A、2B、3A、4A、5A、6A、7A` 已全部收敛，**架构分叉已关闭**。  
实施前仍建议补齐的「非分叉、但易漏」清单见 **§5.3**。后续新问题按“是否改变上述公共契约”判断：若不改变，作为 backend 或实现细节处理；若需要修改 Manager/Runner 主边界、单次 AG 契约或外层 linear shape invariant，必须重新评审。

### 5.3 实施前易漏清单（非架构开放题）

| # | 项 | 说明 | 建议落点 |
|---|----|------|----------|
| R1 | **三 bridge idx 的构造算法** | 文档定了 shape/语义（§4.5），未逐步写出如何从 L1 线性段 × 上游 DualChunk segment 生成三个 idx（应对齐 V1 enter/exit 公式） | M1 UT + Manager 注释/私有 helper |
| R2 | **`state_indices` 与 decode-first** | linear 保持原 `num_reqs`；mixed batch decode 在前时，GDN `state_indices` / `query_start_loc` 与 FA virtual rows 的对应关系需有一份示例表 | M2 UT |
| R3 | **产品主路径是 GQA** | `qwen3_next/3.5*` FA=GQA；M3 MLA 只验证公共契约，**真机主验收在 M5** | 排期/验收知情 |
| R4 | **兼容字段 `max_tokens_across_pcp`** | 若写入 forward context，必须 = `linear_num_tokens_padded`，禁止 DualChunk max | §4.6 / platform 接线 |
| R5 | **TP×PCP 进程组** | 假定沿用现有 `get_pcp_group()` / TP group；文档未画拓扑，实施时勿新建平行 PCP 组 | M3/M4 联调 |
| R6 | **V1 接缝「末卡」→「末有效 rank」** | C35 已定；移植 `gdn.py`/`chunk.py` 时必须改，不能原样抄 V1 无条件 `[-1]` | M3 GDN |
| R7 | **Hybrid vs 非 Hybrid MLA 写 cache** | Hybrid：`ALREADY_GLOBAL` + 跳过二次 AG；非 Hybrid：上游 `LOCAL_FA`+gathered AG。同一 adapter 分发处勿混 | M3 |

---

## 5.2 补充材料：Ascend V1「Q/KV 都切了，FA 怎么得到正确结果」

问题：PCP 下本卡只有局部 Q、KV 也按 rank 交错落盘（`slot_mapping` 非本卡为 `-1`），单卡算不了完整因果 softmax。

V1 的解法是 **两套机制**，不要混成一种：

### 机制 1：当前 chunk 的 Prefill FA —— **先把 KV（hybrid 则 QKV）凑全，再算本地 DualChunk Q**

入口：`attention_cp.py` → `reshape_and_cache` / `_gather_and_restore_pcp_qkv`，再进 `_forward_prefill_cp`。

| 路径 | 通信 | 本卡拿到的 | 再算什么 |
|------|------|------------|----------|
| 普通 PCP | `all_gather(KV)` + `pcp_allgather_restore_idx` | **全量当前 chunk 的 K/V**（原序） | 本卡 DualChunk **局部 Q** × 全量 KV |
| hybrid PCP | `all_gather(QKV)` + `enter_fa` / `fa_padding` / `pcp_fa_query_idx` | 全量工作区 K/V + 抽出本卡 DualChunk Q | 同上 DualChunk FA |
| 出 FA（hybrid） | `all_gather(attn_out)` + `exit_fa` scatter | 本卡 **线性布局** 上的完整 FA 输出 | 给 GDN/runner |

因果细节：`_forward_prefill_cp` 把本卡 Q 拆成 **head / tail**，对 KV 再切 **mask / nomask** 两段 `npu_fused_infer_attention_score`，用 `npu_attention_update` / `_update_out_and_lse` 合成（DualChunk 因果正确性）。

要点：**当前 prefill 计算并不是「Q 分片 × KV 分片」硬算**；存储可以分片，但算 FA 前会把 **本轮 token 的 KV 通信成全量**，所以本卡 Q 上直接得到正确注意力输出（对该 Q 集合而言已是全量结果）。

### 机制 2：只能看到本地 cache 时 —— **局部 (O, LSE) + 跨 rank 合并**

适用：**decode**、以及 **chunked prefill 读历史 context**（KV 已按 PCP 交错存在各卡）。

1. 本卡用本地 KV 算 `attn_out` + `softmax_lse`  
2. `_process_attn_out_lse`：`cat(out, lse)` 后 **PCP `all_gather`**（DCP 时还有 all_to_all）  
3. `_npu_attention_update` / `_update_out_and_lse`：  
   `LSE_final = logsumexp(LSE_i)`，`O_final = Σ exp(LSE_i - LSE_final) · O_i`  
   → 与「Q 对全量 KV 做一次 softmax」数学等价。

### 一句话对照

```text
存储：KV 按 PCP 交错分片（非本卡 slot=-1）
当前 prefill 算子前：AG 还原本轮全量 KV（hybrid 还 AG QKV 做布局桥）
对 cache 的 Q×分片KV：局部 out+lse → AG → npu_attention_update 合成全量
```

上游 MRv2（不分片 KV）则通常 **不需要** 这套「分片 KV + LSE 合并」；那是 Ascend V1「PCP 切 KV」带来的代价。

### 补充：上游「不分片 KV」时，本卡全量 KV 从哪来？（已确认）

**不是**「只存在某一张卡、别人远程读」；而是 **每张 PCP 卡各持有一份完整 KV cache 副本**。

形成方式（见 `vllm/model_executor/layers/attention/pcp.py` → `_gather_prefill_cache_inputs`）：

```text
各卡 DualChunk 只算本地 token 的 K/V
  → prefill 段：PCP all_gather(K/V) 拼成全量
  → 用「按 global positions 算好的 slot_mapping」（gathered 布局）
  → 每卡把全量 K/V reshape_and_cache 写进【自己本地】的 KV cache
decode：各卡复制算同一 decode token；各卡写入【自己的】本地全量 KV 副本
（write_mask 只整理 gathered slot 表，见 §4.8.2）
```

Scheduler 侧 `pcp_world_size=1` 进 KV manager：按「每卡存全序列」做块/显存预算，**不做 PCP 交错分片**。  
代价：**KV 显存 ≈ 单卡全量 × pcp_size（相对 V1 分片）**；换来的是 decode/context 不必再做跨 rank Out+LSE 合并（PCP 维）。

本方案采用两者的组合语义，但不叠加两次通信：

```text
Hybrid prefill：
linear QKV/latent
  → bridge 内一次 PCP AG
  → 本卡 DualChunk Q + global K/V（或 latent）+ global slot plan
  → 每卡先写完整本地 KV cache
  → virtual-row Q 通过 block table 读本地完整 cache

Hybrid pure decode：
复制 token 已在每卡存在
  → 不进 bridge、不做 PCP AG
  → 每卡直接更新自己的完整 KV cache 副本
```

因此，V1 是本方案「单次 QKV/latent AG + 进出 FA 布局桥」的语义来源；上游 MRv2 是「每卡完整 KV cache + virtual-batch 因果计算」的语义来源。`CacheWritePlan` 是两者之间的显式所有权边界，禁止 adapter 再隐式 gather。

---

## 6. 实施拆分

建议按可独立验证、可回滚的提交组织：

### M0：Hybrid MRv2 基线

- 语义移植 `58d1b6502`、`291658e90` 的相关内容。  
- 增加 cache allocate/reshape、ModelState 选择、state preprocess/postprocess UT。  
- 加入第一阶段 capability 门禁：BF16/FP16、普通因果 attention、`mamba_cache_mode=none`、`enable_chunked_prefill=False`、`enable_prefix_caching=False`；其它组合 fail-fast。  
- 真机验证 Hybrid `pcp=1` prefill/decode。

### M1：双视图 Manager

- 新增私有 `HybridStepBuilder`、只读 `HybridPreparedStep`、`HybridPCPLayout` 和独立 linear/FA/bridge buffers。  
- 上游 `super().partition_batch()` 产出 FA 视图；V1 线性算法产出 linear 视图。  
- `prepare_attn()` 完成 finalize/bind，并合并为旧签名的 per-group block tables/二维 slot mappings；不改写 `execute_model()`。  
- 完成不等长 pad、三个 bridge idx、零长度 identity、容量 fail-fast、sample restore 和 step 清理 UT。

### M2：HybridModelState 按 group 构建 metadata

- 注册并校验 `PCPGroupCapability`，按 capability 选择 `fa_batch` 或 `linear_batch`。  
- MLA/GQA group 使用 `PreparedGroupInputs(input_layout=DUAL_CHUNK_VIRTUAL)`。  
- GDN/Mamba group 使用 `PreparedGroupInputs(input_layout=CONTIGUOUS_CAUSAL_STATE)`。  
- block tables/slot mappings 分组准备并按 layer name 合并；用 `HybridAttentionMetadataMap` 把只读 `HybridPCPForwardView` 接到现有 Ascend additional-forward-context 扩展点。  
- 覆盖双 group、零长度 virtual row、不同 group 行数、dummy/profile 和 stale-step 防护 UT。

### M3：MLA + GDN eager

- MLA adapter 接公共 bridge；收到 plan 的 `PREFILL/ALREADY_GLOBAL` segment 时跳过上游第二次 KV/latent AG，非 Hybrid 路径保持上游行为。  
- pure decode 旁路 bridge；prefill 的 Q/KV/slot 对齐及 cache-first 写入只发生一次。  
- GDN 继续使用 V1 conv/SSM 接缝。  
- 完成单请求、多请求、不等长、短序列/空 rank、decode、PCP+TP；外层 linear shape invariant 全程断言。

### M4：MoE（pad/unpad + valid-token；EP 关闭）

- 参考 V1 prepare/finalize，在 PCP collective 前统一 pad hidden/router logits/辅助 tensor 并显式传播 `local_valid_mask`；topk/dispatch 前 compact valid token，或使用有严格 skip 语义的 invalid id。  
- 验证 padding 不参与有效 routing、capacity 竞争和 EPLB 统计，不影响真实 token。  
- 覆盖 dense/MoE 对照与 PCP+TP；第一阶段 EP 配置必须 fail-fast，valid-token 契约仅作为后续 EP 扩展点。

### M5：GQA backend

- 新增标准 Attention 的 `PCPGroupCapability`、metadata 与 impl adapter。  
- 复用 M1 单次 entry QKV payload AG 的 global K/V 与 `PREFILL/ALREADY_GLOBAL` segment，不增加第二次 K/V gather。  
- 采用 cache-first：每卡先写完整 KV cache，virtual-row Q 再按 block table 读取；不保留 `PrefillNoCache` 的 KV 复制兜底路径。  
- 覆盖 TP head/KV-head 形状、RoPE/辅助 tensor 布局及公共 bridge。  
- 完成 GQA+GDN dense/MoE 验收。

### M6：Piecewise graph（必做里程碑）

- 通信和 pad/unpad 保持 graph break。  
- 捕获 MLA/GQA/GDN/MoE 计算主体。  
- 使用预分配双视图、索引和临时输出 buffer，并按 FA bucket 单独建图；外层 BatchDescriptor 保持 linear。  
- 重跑 M3～M5 的核心场景；M6 完成后第一版才满足 6A。

## 7. 验收参考清单（非门禁；负责人真机自行验收）

可选参考顺序：

1. Hybrid `pcp=1`，验证主干与 state 生命周期。  
2. PCP=2，单请求完整 prefill。  
3. PCP=2，多请求、不等长 prefill。  
4. 普通 decode，验证复制 token 和 KV/state 更新。  
5. PCP+TP。  
6. 序列长度小于 PCP size、某 rank/segment 为零长度。  
7. GDN conv/SSM 最终 state 与 PCP=1 对齐。  
8. MLA/GQA KV 在每个 PCP rank 上都是完整副本，slot mapping 正确。  
9. dense Hybrid。  
10. MoE Hybrid：padding 不产生有效路由、不占真实 capacity、不污染 EPLB 统计；EP 配置 fail-fast。  
11. sample restore、prompt logprobs、dummy/profile。  
12. 通信 trace/计数证明：每个 FA 层的 Hybrid prefill 只有一次 entry QKV/latent payload PCP AG（exit output AG 另计），backend 没有第二次 KV AG；pure decode 没有 bridge AG。  
13. 验证 outer model、BatchDescriptor、forward context 和 attention 输出始终为 linear token 数；只有 adapter 临时区使用 FA/global token 数。  
14. M6 piecewise graph 重跑 M3～M5 核心回归；通信/pad/unpad graph break 符合预期。  
15. 对 `enable_chunked_prefill=True`/chunked/continued prefill、`enable_prefix_caching=True`/prefix-cache hit、`mamba_cache_mode=align/all`、quant、sliding window、DCP、EP、dynamic EPLB、spec、PP、MM、LoRA、PD/KV connector、full graph 做 fail-fast。
16. 故障注入：构造 rank-local token/decode hash/capacity 不一致，确认所有 rank 在 payload collective 前同步报错而非 HCCL hang。

精度比较至少包含：

- PCP=1 与 PCP>1 的最终 logits/hidden 对齐。  
- MLA/GQA attention 层输出对齐。  
- GDN 每层及最终递推 state 对齐。  
- dense/MoE、单请求/多请求、prefill/decode 分别覆盖。

## 8. 后续扩展项

以下能力通过 `PCPGroupCapability` / backend adapter 扩展。原则上不改 Manager/Runner 生命周期和外层 linear shape invariant；若实现证明必须改变 `HybridPreparedStep`、单次 AG 或 `CacheWritePlan` 公共契约，应单独重新评审，不能在 backend 中隐式穿透：

- chunked/continued prefill、prefix-cache hit 与历史 context。  
- quant、sliding window 与其它 attention/cache 语义。  
- `mamba_cache_mode=align/all`。  
- EP 进程组与 EPLB 的完整产品化联调（复用第一版 valid-token 契约；见 §4.8.1）。  
- DCP 与 PCP 组合。  
- speculative decoding。  
- PD / KV connector。  
- MM、PP、LoRA。  
- full graph。  
- 经典 Mamba1/2、非 GDN 线性层及其它 Hybrid 结构。
