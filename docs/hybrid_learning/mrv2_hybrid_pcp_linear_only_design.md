# Ascend MRv2 Hybrid PCP — 单视图顺序切分方案（简化版）

> 状态：草案（已按「一套切分走到底 + 层后同步」修订）。  
> 工作分支：`feat/mrv2-hybrid-pcp-linear-only`  
> 代码基线：`c8b480b7f`（Hybrid MRv2 主干 + PCP `validate_config` 放宽 MLA-only；**尚未**接入双视图 PCP 生产端）  
> 对照文档：同目录 `mrv2_hybrid_pcp_design.md`（双视图 / DualChunk FA 旁路方案）  
> 一句话：**FA 与 GDN 共用同一套顺序（因果连续）切分；层间/层后同步 KV 或 SSM；不做双视图、不做布局桥、不做 FA 负载均衡，也不再单独搞一套 V1「接缝」体系。**

---

## 0. 方案定位

双视图方案要同时伺候两套布局（线性给 GDN、DualChunk 给 FA），再用 bridge 来回变。  
本方案更简单：

```text
一套顺序切分 ──► 所有层（FA / GDN / MLP / MoE）同一 local_batch
        │
        ▼
每层算完本卡局部结果后，按层类型同步必要状态：
  · FA 层 → 同步 KV（或 latent KV）
  · GDN 层 → 同步 SSM / conv 等递推状态
        │
        ▼
下一层继续用同一套 token 布局往下算
```

| 目标 | 本方案 | 双视图方案 |
|------|--------|------------|
| 切分 | **一套顺序切分走到底** | 线性 + DualChunk 两套 |
| FA / GDN | **同一布局** | 布局不同，靠 bridge |
| 跨卡依赖 | **  KV / SSM** | FA 靠 bridge+AG；GDN 靠 V1 接缝套件 |
| 负载均衡 | **不做** | DualChunk 改善 FA 负载 |
| V1 GDN「接缝」专项 | **不单独引入** | 明确复用 V1 接缝 |

---

## 1. 继承与丢掉

### 1.1 仍可借用

1. 不重写 `execute_model()`；Runner 仍是 `prepare_inputs → prepare_attn → forward → restore`。  
2. **顺序切分语义**：prefill 按请求做因果连续分段；decode 各卡复制。  
3. FA KV / GDN state **都不按 PCP 做存储分片**（同步后各卡持有算下一步所需的副本或前缀）。  
4. 配置边界：不做 chunked / spec / PD / MM / PP / LoRA / DCP；第一阶段关 EP；`mamba_cache_mode=none`、`enable_prefix_caching=False`。  
5. capability fail-fast；MoE 用 local batch 的 pad / `is_padding`。  
6. 精简 step bag，避免字段散落。

### 1.2 明确丢掉

1. DualChunk / `fa_batch` / virtual-batch。  
2. 三套 bridge idx 与 FA enter/exit 换布局。  
3. FA 负载均衡。  
4. **不再把 V1「接缝」（last_width AG + SSM 重跑修正等）当作本方案必选子系统**——跨卡因果依赖统一收成「同步 SSM / 同步 KV」。  
5. 主路径不调用上游 `partition_batch()`（DualChunk）。

---

## 2. 范围与成功标准

| 类别 | 第一版 |
|------|--------|
| PCP + TP；MLA/GQA + GDN 同布局 | ✅ |
| dense / MoE Hybrid | ✅ |
| 双视图 / bridge / FA 均衡 / V1 接缝专项 | ❌ |
| eager 正确性 | ✅ 必达 |
| piecewise | 后置 |

成功标准：`pcp=1` 行为不回归；`pcp>1` 下 FA/GDN 在同一顺序切分上正确；不支持组合 fail-fast。

---

## 3. 核心决策

### C1. 一套切分，全层共用

```text
global InputBatch
        │
        ▼
partition_sequential()     # 唯一 partition
        │
        ▼
local_batch（顺序切分后的局部 token）
  · 保持原 num_reqs
  · prefill：每卡一段因果连续区间
  · decode：每卡复制
  · FA、GDN、MLP、MoE 全部只认这一份
```

### C2. 切分公式（顺序 / 不考虑均衡）

对 prefill 长度 `N`、world `W`、rank `r`：

```text
base, rem = divmod(N, W)
len_r = base + (1 if r < rem else 0)
off_r = base * r + min(r, rem)
本卡：global[off_r : off_r + len_r]
```

decode：整段复制。同步前若长度不齐则 pad；有效位用 `InputBatch.is_padding` / `num_tokens`。

### C3. 跨卡怎么对齐：层后同步，而不是「接缝体系」

顺序切分后，本卡只有一段序列。下一层或本层收尾时，需要把**别的卡算出来的状态**对齐过来。统一模型：

```text
本卡用 local_batch 做完当前层局部计算
        │
        ├─ FA 层：同步 KV（或 MLA latent KV）
        │         → 各卡具备后续注意力所需的前缀/全量 KV 视图
        │
        └─ GDN 层：同步 SSM / conv 等递推状态
                  → 后段 rank 拿到前段 rank 递推到的状态，或各卡对齐末状态供 decode
        │
        ▼
hidden 仍留在同一套 local 布局，进入下一层（无需换视图）
```

说明：

- **FA**：同步的是 **KV**；本卡 Q 仍是本卡那段 token，不算 DualChunk row。  
- **GDN**：同步的是 **SSM（及必要的 conv 状态）**；不另立「接缝」模块名，实现上可以是一次 AG/P2P 状态，而不是 V1 那套 extract → AG → 注入 → 重跑 h 的完整剧本（第一版以正确、简单为准）。  
- **pure decode**：token 各卡已复制，可按层类型决定是否旁路同步。  
- 层间 **hidden 不要求整网 AG**；只同步该层类型需要的 KV 或 SSM。

> 因果事实仍然在：后段 rank 的 GDN 依赖前段状态。  
> 本方案把它看成 **「同步 SSM」**，而不是单独产品化的「接缝子系统」。

### C4. Sample restore

- decode：本地即可。  
- 含 prefill：对 local hidden 的 prefill 段 AG → `_hidden_restore_idx`（线性语义）还原全局序 → sample。  
- 不使用上游 DualChunk `_hidden_restore_idx`。

### C5. 与上游 PCPManager

| 项 | 本方案 |
|----|--------|
| `super().partition_batch()` | 主路径不用 |
| partition | 自研顺序切分 |
| prepare_attn / block / slot | 全按 `local_batch` |
| restore | 线性 AG restore |

非 hybrid：可继续 DualChunk；与本方案分流。

---

## 4. 数据流与结构

### 4.1 一 step

```text
global InputBatch
        │
        ▼
partition_sequential()
  → local_batch；写入 `_global_batch` / `_hidden_restore_idx`
        │
        ▼
prepare_attn(local_batch)   # FA 与 GDN 同一 batch
        │
        ▼
forward
  · 每层：本地算 →（FA 同步 KV / GDN 同步 SSM）→ 同布局进入下层
  · MoE：pad + is_padding / num_tokens
        │
        ▼
restore_for_sampling
```

### 4.2 状态存放（复用上游字段，不新增 step bag）

```text
Runner.input_batch              ← partition 返回的 local_batch（唯一视图）
AscendPCPManager._global_batch  ← 本 step 的 global InputBatch
AscendPCPManager._hidden_restore_idx
        ← 顺序切分语义：AG(linear_prefill pad) → global_prefill（shape T-D）
          （与 DualChunk 写入同一字段名，hybrid 路径语义不同；restore 时走 hybrid 分支）

pad / 有效 token：
  local_batch.num_tokens / num_tokens_after_padding / is_padding
  （不再另存 linear_valid_mask）
```

不引入 `LinearPCPStep` / `LinearPCPLayout`。

### 4.3 职责

| 组件 | 职责 |
|------|------|
| Manager | 顺序 partition、layout、sample restore |
| Runner | hybrid 走顺序切分入口 |
| FA | 同布局局部算 + **同步 KV** |
| GDN | 同布局局部算 + **同步 SSM**（不引入独立接缝产品层） |
| MoE | `is_padding` / `num_tokens` |
| ModelState | 读 Runner `input_batch`（即 local）+ Manager 上复用字段 |

---

## 5. 代价与风险

| 项 | 态度 |
|----|------|
| FA / GDN 段长度不均 | 接受 |
| 每层同步 KV 或 SSM 的通信量 | 接受；正确性优先 |
| 不复用上游 DualChunk / V1 接缝实现 | 故意简化；可后补优化 |
| 与双视图分叉 | 分支隔离 |

---

## 6. 实施拆分

1. **L1** 顺序 partition + layout + Runner 入口 + UT  
2. **L2** prepare_attn 全走 local_batch；GDN：**同步 SSM** 的最小正确实现  
3. **L3** FA：**同步 KV** + 本地 Q  
4. **L4** MoE valid-mask + sample restore  
5. **L5**（可选）piecewise  

---

## 7. 与双视图的关系

- 本分支验证「一套切分 + 层后同步」能否端到端正确。  
- 公共可抽：顺序 partition、valid_mask、restore idx。  
- FA 慢或同步过重时，再评估是否回到双视图。  
- **不要**把 V1 接缝与双视图 bridge 偷运回本方案「为了好像更完整」。

---

## 8. 待确认

1. GDN 同步粒度：段边界 P2P（前卡 → 后卡）还是全 rank AG 状态？第一版建议 **能正确即可，优先简单 AG**。  
2. FA 同步：AG 全量 KV，还是只收集前缀？第一版建议 **AG 全量 KV/latent（实现简单）**。  
3. `_hidden_restore_idx` 仅覆盖 prefill 段？（建议是）  
4. 非 hybrid 是否不动 DualChunk？（建议不动）  
5. 是否先只做 L1 生产端再接同步逻辑？

---

## 9. 与双视图字段对照

| 双视图 | 本方案 |
|--------|--------|
| linear + fa 双 batch | 仅 Runner `input_batch`（local） |
| `HybridPreparedStep` / layout | **不建**；复用 `_global_batch` + `_hidden_restore_idx` |
| 三 bridge idx | 无 |
| V1 GDN 接缝专项 | 改为层后/边界同步 SSM |
| FA enter/exit 换布局 | 改为层后同步 KV |
| DualChunk 均衡 | 不做 |
