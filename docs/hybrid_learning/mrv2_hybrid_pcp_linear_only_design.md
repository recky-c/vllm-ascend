# Ascend MRv2 Hybrid PCP — 单视图线性切分方案（简化版）

> 状态：草案，供评审对齐；与双视图方案并行探索，互不替代。  
> 工作分支：`feat/mrv2-hybrid-pcp-linear-only`  
> 代码基线：`c8b480b7f`（Hybrid MRv2 主干 + PCP `validate_config` 放宽 MLA-only；**尚未**接入双视图 PCP 生产端）  
> 对照文档：同目录 `mrv2_hybrid_pcp_design.md`（双视图 / DualChunk FA 旁路方案）  
> 一句话：**全网只保留一套线性局部视图；不做 DualChunk、不做双视图桥接、不考虑 FA 负载均衡。**

---

## 0. 为什么再开一套简化方案

双视图方案（`mrv2_hybrid_pcp_design.md`）的核心是：

- GDN 友好 → **线性主路径**
- FA 跟上游 → **DualChunk 旁路**
- 进/出 FA → **三套 bridge idx + 单次 AG**

这套完整，但也重：生产端、消费端、backend adapter、idx 正确性都耦合在「两套布局」上。

本方案刻意收窄目标：

| 目标 | 本方案 | 双视图方案 |
|------|--------|------------|
| 视图数量 | **1（仅线性）** | 2（linear + FA DualChunk） |
| FA 布局 | 与 GDN/MLP **同一套** L1 连续段 | 上游 DualChunk / virtual-batch |
| 负载均衡 | **不做**（接受 prefill 段长度按名次不均） | DualChunk 改善 FA 负载 |
| 布局转换 | **无** bridge idx | 三个 hybrid bridge idx |
| 复杂度 | 低，优先跑通正确性 | 高，优先 FA 性能与上游对齐 |

适用场景：先验证「Hybrid + PCP 在线性因果切分下能否端到端正确」，再决定是否回到双视图做 FA 性能优化。

---

## 1. 从双视图方案继承什么 / 明确丢掉什么

### 1.1 继承（可取部分）

1. **宿主仍是上游 MRv2 框架**：不重写 `execute_model()`；Runner 路径保持 `prepare_inputs → prepare_attn → forward → restore`。  
2. **线性切分语义（V1 L1）**：prefill 按请求做**因果连续**分段；decode **各卡复制**。  
3. **GDN 接缝留在 ops**：`conv last_width AG`、`initial_state_mode`、SSM `final_state` AG 修正；Manager 只保证「本卡 token 是因果连续段」。  
4. **Mamba/GDN state 不分片**：请求级 state 可复制；接缝靠通信，不靠按 token 切 state。  
5. **FA KV 本阶段不分片**（与上游 PCP 精神一致；不做 DCP）。  
6. **配置边界**：不做 chunked prefill / spec / PD / MM / PP / LoRA / DCP；第一阶段 EP 关闭；`mamba_cache_mode=none`、`enable_prefix_caching=False`。  
7. **capability fail-fast**：不用 `model_type` 白名单卡死；未支持组合直接报错。  
8. **MoE 走 linear pad + `linear_valid_mask`**：与「主路径即线性」天然一致（双视图里 MoE 也走这条）。  
9. **嵌套 step bag**：用一个只读 step 对象挂在 Manager 上，避免字段散落（可比照 `HybridPreparedStep`，但字段更少）。

### 1.2 丢掉（相对双视图）

1. **不上游 `partition_batch()` 作为主路径**（那是 DualChunk）。  
2. **无 `fa_batch` / virtual-batch**。  
3. **无** `hybrid_linear_ag_restore_idx` / `hybrid_global_to_fa_idx` / `hybrid_fa_to_linear_idx` 三桥。  
4. **无** FA enter/exit 布局变换；无「AG → 原序 → 再取 DualChunk」链路。  
5. **不考虑** DualChunk 式 FA 负载均衡。  
6. **不引入** 为双视图服务的 `CacheWritePlan(PREFILL/ALREADY_GLOBAL)` 复杂分支（若 FA 需要 AG KV，用更直接的「本卡 Q + 已聚集/本地 KV」契约即可，见 §4.3）。

---

## 2. 问题定义与范围

### 2.1 要解决什么

在 MRv2 上，让 **Hybrid（KV-attention + GDN）** 在 `pcp_size > 1` 下，用**单一线性切分**完成 prefill/decode 正确性闭环。

### 2.2 第一版范围

| 类别 | 第一版 | 说明 |
|------|--------|------|
| PCP + TP | ✅ | 主叠加 |
| MLA/GQA + GDN | ✅ | FA 与 GDN **共用线性 batch** |
| dense / MoE Hybrid | ✅ | MoE 认 linear pad + valid mask |
| FA 负载均衡 | ❌ | 明确不做 |
| 双视图 / bridge | ❌ | 明确不做 |
| EP / DCP / chunked / spec / PD / MM / PP / LoRA | ❌ | 同双视图第一阶段 |
| eager | ✅ 必达 | 本方案主验收 |
| piecewise | 后置 | 先正确性，后图模式 |

### 2.3 成功标准（本方案）

1. `pcp_size=1`：与现有 Hybrid 主干行为一致（无 PCP 税）。  
2. `pcp_size>1`：prefill/decode 数值正确；GDN 接缝正确；FA 在线性局部 Q 上可算。  
3. 不支持组合 fail-fast。  
4. 真机验收由负责人自行安排；本文不绑定矩阵门禁。

---

## 3. 核心决策

### C1. 唯一视图 = 线性局部 batch

```text
global InputBatch（真实 req，建议 decode-first）
        │
        ▼
AscendPCPManager.partition_linear()   # 唯一 partition
        │
        ▼
local_batch = 线性局部 InputBatch
  - 保持原 num_reqs（不 virtualize）
  - prefill：每卡一段因果连续 token
  - decode：每卡复制完整 decode tokens
  - 层间 hidden 始终在此布局
```

- **GDN / MLP / MoE / 默认算子**：直接吃 `local_batch`。  
- **FA（MLA/GQA）**：也吃同一 `local_batch` 的 Q 侧 token；**不再**换成 DualChunk rows。

### C2. 切分算法（L1，不考虑均衡）

对每个 prefill 请求长度为 `N`、`pcp_world_size = W`、本卡 rank `r`：

```text
base, rem = divmod(N, W)
len_r = base + (1 if r < rem else 0)
off_r = base * r + min(r, rem)
本卡拿走 global[off_r : off_r + len_r]
```

- decode：`len_r = N`，`off_r = 0`（复制）。  
- 不等长时对 AG 做 **linear pad** 到 `max(各卡 local_num_tokens)`；`linear_valid_mask = arange(L) < local_num_tokens`。  
- **不做** DualChunk、**不做** `cp_interleave` 交错（第一版）；若后续要与某条 V1 路径对齐再单开决策。

### C3. FA 在单视图下如何正确（必须说清）

线性切分后，rank `r` 只有序列的一段。因果 FA 需要看到**更前缀**的 K/V，因此 FA **仍需要通信**，只是通信对象不再是「换布局」，而是「补全上下文」：

```text
本卡 linear Q（及本卡产出的 K/V 或 MLA latent）
        │
        ├─ pure decode：各卡 token 相同，旁路集体通信（与双视图一致精神）
        └─ 含 prefill：
              对 prefill 段 pad → PCP all_gather（QKV 或 KV/latent，后端定）
              → 得到全局（或前缀所需）K/V
              → 本卡只算自己的 Q rows
              → 写本地 KV cache（FA KV 不分片：写全量或按既有 cache 契约）
              → 输出仍留在本卡 linear 布局（无需 scatter 回另一套视图）
```

要点：

- **没有**「linear ↔ DualChunk」变换，故**没有**三 bridge idx。  
- 若 AG 后需要还原「按 rank 拼接 → 全局原序」，只保留**一个**与 sample/FA 共用的  
  `linear_ag_restore_idx`（语义 ≈ 双视图里的 ①，但不再接 ②③）。  
- FA 各卡算力随分段长度变化 → **接受不均衡**；这是本方案相对 DualChunk 的明确代价。

> 实施时 FA backend 以「input_layout = CONTIGUOUS_CAUSAL」消费 linear batch；  
> 禁止再假设 virtual-row / DualChunk metadata。

### C4. Sample restore

- pure decode：各卡 hidden 已是完整 decode 前缀，本地即可。  
- 含 prefill：对 prefill 段 AG → `linear_ag_restore_idx` 还原全局原序 → 与 decode 前缀拼回 `global_batch` 再 sample。  
- **禁止**依赖上游 DualChunk 的 `_hidden_restore_idx` 作为 hybrid 主路径（那条对应 DualChunk gather）。

### C5. 与上游 `PCPManager` 的关系

| 能力 | 本方案 |
|------|--------|
| `super().partition_batch()` | **主路径不调用**（避免 DualChunk 污染） |
| 线性 partition | Ascend 自研 / 复用 V1 L1 算法，落在 `hybrid_pcp` 或 Manager 私有方法 |
| `prepare_attn` / slot / block table | 全部按 **linear local_batch** 的原 `num_reqs` 行准备 |
| 上游 restore_for_sampling | Hybrid 路径覆盖为线性 AG restore |

非 hybrid 模型：仍可走现有 Ascend/上游 DualChunk PCP（若已接通），与本方案互不强制合流。

---

## 4. 架构与数据流

### 4.1 一 step 时序

```text
global InputBatch
        │
        ▼
partition_linear_only()
  ├─ L1 segments（各 rank）
  ├─ materialize linear local_batch
  ├─ linear_num_tokens / padded / valid_mask
  └─ linear_ag_restore_idx（仅 AG 还原原序；无 FA 桥）
        │
        ▼
prepare_attn(local_batch)
  ├─ 所有 attn group（FA + GDN）共用 linear batch + 对应 block/slot
  └─ 发布 LinearPCPStep（只读）并 bind → ModelState
        │
        ▼
forward（hidden 始终线性）
  ├─ GDN：本地算 + V1 接缝
  ├─ FA：§3 C3 的 AG-KV（或等价）+ 本地 Q；输出不换布局
  └─ MoE：pad / valid_mask
        │
        ▼
restore_for_sampling：§3 C4
```

### 4.2 建议结构体（精简）

```text
AscendPCPManager
  └── linear_step: LinearPCPStep | None

LinearPCPStep
  ├── step_id
  ├── global_batch
  ├── local_batch          # 唯一视图
  └── layout: LinearPCPLayout

LinearPCPLayout
  ├── num_decode_tokens
  ├── linear_num_tokens
  ├── linear_num_tokens_padded
  ├── linear_ag_restore_idx   # AG(linear pad by rank) → global real tokens（或仅 prefill）
  └── linear_valid_mask
```

相对双视图 `HybridPreparedStep`：

- 删除 `fa_batch`  
- 删除 `hybrid_global_to_fa_idx` / `hybrid_fa_to_linear_idx`  
- 保留（并改名）单一 AG restore + valid mask  

### 4.3 职责划分

| 组件 | 职责 |
|------|------|
| `AscendPCPManager` | 线性 partition、layout、sample restore；**不**维护 FA 第二视图 |
| Runner | `maybe_partition` 在 hybrid 时走线性入口；`prepare_attn` bind step |
| GDN | 只认线性 + 现有接缝 |
| FA backend | 认线性 batch；自行（或经薄包装）做跨 rank KV/latent 集合；**不做布局桥** |
| MoE | linear pad + valid mask |
| ModelState | 一次性消费 `LinearPCPStep`；FA/GDN 都从 `local_batch` 取视图 |

### 4.4 文件落点（建议）

```text
vllm_ascend/worker/v2/
  pcp_manager.py          # AscendPCPManager：接线、读上游私有态（若需要）、restore
  hybrid_pcp.py           # L1 segments、linear materialize、LinearPCPLayout/Step、AG restore
  model_runner.py         # hybrid 时调用线性 partition 入口
```

- 算法进 `hybrid_pcp.py`；Manager 只做编排与（必要时）上游私有字段外递。  
- **不要**为了「跟上游长得像」去调用 DualChunk `partition_batch` 再丢掉结果。

---

## 5. 关键风险与明确接受的代价

| 风险 / 代价 | 说明 | 态度 |
|-------------|------|------|
| FA 负载不均 | 前段 rank 与后段 rank token 数可差 1，长尾请求更明显 | **接受**（本方案前提） |
| FA 集体通信量 | 每层 FA 仍可能 AG KV/QKV；无 DualChunk 时难以复用上游 MLA PCP 捷径 | 用最简正确实现；性能不是本方案第一目标 |
| 与上游 PCP 元数据不兼容 | 上游大量假设 DualChunk virtual rows | Hybrid 路径独立，不强行复用 |
| 与双视图代码分叉 | 两套方案并存一段时间 | 分支隔离；公共能力（L1、GDN 接缝、valid mask）可后续抽取 |

---

## 6. 实施拆分（建议）

### L0：基线确认

- 分支基于 `c8b480b7f`；`pcp=1` Hybrid 可跑（已有 M0）。  
- 本文评审通过后再写代码。

### L1：线性 partition 生产端

- `get_linear_rank_segments` + materialize `local_batch`  
- `LinearPCPLayout`（restore idx + valid mask）  
- Runner 入口切换；**不**调用 DualChunk partition  
- UT：切分、pad、decode-first、pure decode、restore 排列

### L2：prepare_attn + GDN

- 全 group 走 linear block/slot  
- 接上现有 GDN 接缝（`pcp_size>1`）  
- UT / 小规模真机：纯 GDN 敏感路径（若有）或 hybrid 中 GDN 层

### L3：FA on linear

- MLA 或 GQA 之一先打通「本地 Q + AG KV/latent」  
- decode 旁路  
- 正确性优先，不做均衡

### L4：MoE valid-token + sample restore 收口

- pad/unpad 与 `linear_valid_mask`  
- sample 与 L1 restore 共用 idx

### L5（可选）：piecewise

- 通信保持 graph break；与双视图 M6 同类，本方案不阻塞 L1–L4。

---

## 7. 与双视图方案的关系（决策备忘）

| 问题 | 建议 |
|------|------|
| 哪条是主线？ | **未定**；本分支用于快速验证单视图可行性 |
| 能否合并？ | 生产端（L1 partition、GDN 接缝、MoE mask）可共用；FA 路径与 step bag **不要**过早揉成一套抽象 |
| 若单视图正确但 FA 慢？ | 再评估是否切回双视图做 FA 旁路 |
| 若单视图连正确性都难（FA 集合契约）？ | 记录阻塞点，优先回到双视图或收窄只验 GDN+假 FA |

---

## 8. 待评审确认项

请在开工前确认：

1. **FA 集合策略**：prefill 默认「AG KV/latent + 本地 Q」是否可接受（相对整段 AG QKV）？  
2. **`linear_ag_restore_idx` 域**：整段 token 还原，还是 decode-first 下仅 prefill 段（与双视图 ① 对齐）？建议：**仅 prefill，decode 旁路**。  
3. **非 hybrid PCP**：是否完全不动，继续 DualChunk？建议：**不动**。  
4. **命名**：`LinearPCPStep` / `LinearPCPLayout` 是否可采用（避免再叫 Hybrid 造成与双视图混淆）？  
5. **第一刀代码**：是否按 §6 只做 L1 生产端 + UT，暂不接 FA backend？

---

## 9. 附录：和双视图字段对照

| 双视图 | 本方案 |
|--------|--------|
| `linear_batch` + `fa_batch` | 仅 `local_batch`（线性） |
| `hybrid_linear_ag_restore_idx` | `linear_ag_restore_idx`（保留语义） |
| `hybrid_global_to_fa_idx` | **删除** |
| `hybrid_fa_to_linear_idx` | **删除** |
| `linear_valid_mask` | 保留 |
| `CacheWritePlan` 双视图分支 | 不做；FA 用更直接的 linear 契约 |
| 主路径返回值 | 仍是线性 `local_batch` |
