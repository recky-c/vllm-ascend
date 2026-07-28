# feat/mrv2-hybrid-pcp-linear-only 远端回归验证说明

> 给远端 NPU 机器上的执行 agent / 同学用。  
> 只测本分支；环境只装一次；不测其他分支；不 rebase / force push。

## 1. 仓库与提交点

| 项 | 值 |
|----|-----|
| Remote | `https://github.com/recky-c/vllm-ascend.git` |
| 分支 | `feat/mrv2-hybrid-pcp-linear-only` |
| 分支页 | https://github.com/recky-c/vllm-ascend/tree/feat/mrv2-hybrid-pcp-linear-only |

同分支历史中的关键提交（勿混用其他分支）：

| 角色 | SHA | 说明 |
|------|-----|------|
| **Hybrid 基线（顺序 PCP 之前）** | `c2f67e717` | 含 Hybrid 主干搬运；尚无 sequential partition / GDN PCP / local-slot gather。**此点 Hybrid 明确不支持 `pcp>1`，禁止测 pcp=2。** |
| **GQA local-slot 修改前** | `e4cd42744` | 已有顺序切分 + GDN conv/SSM；尚无 `6fa4095f5` |
| **当前 tip** | `8e1efd2fd`（以 `origin/feat/mrv2-hybrid-pcp-linear-only` 为准） | local-slot gather + sample restore 全量 AG + GQA/hybrid validate |

配套 `vllm` 使用机器上现有可跑 MRv2 PCP 的版本。已有环境则 **只 `git fetch` + checkout，不重装**。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

## 2. 模型（固定）

| 场景 | 路径 |
|------|------|
| Hybrid | `/mnt/weight/Qwen3.5-35B-A3B/` |
| 纯 GQA | `/mnt/weight/Qwen3-0.6B/` |

## 3. 并行与配置

- `pcp=1` 作对照；需要 PCP 时用 `pcp=2`（至少 **2 卡** 同 PCP group）
- 第一轮：`TP=1`、`DP=1`、`PP=1`、`DCP=1`、`EP=1`；**eager**（关 full cudagraph）
- Hybrid：`mamba_cache_mode=none`；关闭 prefix caching、chunked prefill、LoRA、speculative decoding
- 每个可跑点至少：
  1. 短 prefill + 若干 decode
  2. 较长 prefill（token 数 > pcp）再 decode
  3. 可选：mixed decode+prefill batch
- **通过标准**：进程不挂、能返回 token；同一 tip 内同 prompt **greedy** 下 `pcp=2` vs `pcp=1` 尽量一致（差很大记 FAIL 并保留日志）

服务参数骨架（按团队 CLI 微调，语义保持）：

```text
--model <MODEL>
--trust-remote-code
--dtype bfloat16
--max-model-len 4096
--max-num-seqs 4
--gpu-memory-utilization 0.85
# pcp=2 时增加：
--prefill-context-parallel-size 2
# Hybrid：mamba_cache_mode=none（配置项名以当前代码为准）
# 关闭：enable_prefix_caching / enable_chunked_prefill / speculative
```

## 4. 执行顺序（严格）

### Phase 0 — 本分支 Hybrid 基线（必须先做，仅 pcp=1）

**目的**：Hybrid 是搬运进本分支的，不是干净 cherry-pick。先确认基线 forward 正常，避免后面失败时分不清是「搬运坏了」还是「新逻辑坏了」。

```bash
git fetch origin
git checkout feat/mrv2-hybrid-pcp-linear-only
git reset --hard c2f67e717
git rev-parse --short HEAD   # 期望 c2f67e717
```

- 模型：`/mnt/weight/Qwen3.5-35B-A3B/`
- **0a：`pcp=1` smoke（必过）**
- **禁止跑 pcp=2**（此提交明确不支持 Hybrid+PCP）

**0a 失败 → 停止**，报告「本分支 Hybrid 基线搬运有问题」，不要继续 Phase 1/2。

### Phase 1 — GQA 修改前（`e4cd42744`）

```bash
git reset --hard e4cd42744
git rev-parse --short HEAD   # 期望 e4cd42744
```

- 模型：`/mnt/weight/Qwen3-0.6B/`
- **1a：`pcp=1`**
- **1b：`pcp=2`**，与 1a greedy 对照

### Phase 2 — Tip（`8e1efd2fd`）

```bash
git reset --hard origin/feat/mrv2-hybrid-pcp-linear-only
git rev-parse --short HEAD   # 期望 8e1efd2fd（若 tip 前进以远端为准并记入报告）
```

**纯 GQA** — `/mnt/weight/Qwen3-0.6B/`

- **2a：`pcp=1`**
- **2b：`pcp=2`**，对照 2a（实现相对 Phase 1 已变，greedy 以 2a 对齐）

**Hybrid（顺序单视图）** — `/mnt/weight/Qwen3.5-35B-A3B/`

- **2c：`pcp=1`**（对照 Phase 0 的 0a）
- **2d：`pcp=2`**，对照 2c；建议再跑长 prefill、mixed batch

### Phase 3 — 单元测试（同一 tip）

```bash
# cwd = vllm-ascend 仓库根目录；按机器 pytest 入口调整
pytest tests/ut/worker/test_pcp_manager_v2.py -q
pytest tests/ut/worker/test_hybrid_pcp_restore.py -q
pytest tests/ut/attention/a2/test_attention_cp.py -q
```

评估是否需要新增 UT（只建议或补最小用例）：

- sequential partition / `get_linear_rank_segments`
- `build_linear_hidden_restore_idx`（已有 `test_hybrid_pcp_restore.py`）
- local-slot KV gather（`test_attention_cp.py`）

**不要**为 MoE compact、SFA 旁路加 UT / 改代码。

## 5. 明确不要做

- 不测其他分支
- Phase 0 **禁止** `pcp=2`
- 不重装环境（除非证明当前环境不可用）
- 不改 SFA；不加 MoE `is_padding` compact；不开 EP / DCP / PP
- 不整理 commit、不 `push --force`

## 6. 交付报告（必须）

1. 各 Phase 实际 SHA、是否跳过安装、机器 / CANN / torch_npu 简要信息  
2. 结果表：

| Case | SHA | 模型 | pcp | 命令摘要 | PASS/FAIL | 日志路径 | greedy 对照 |
|------|-----|------|-----|----------|-----------|----------|-------------|
| 0a | c2f67e717 | Hybrid | 1 | | | | |
| 1a | e4cd42744 | GQA | 1 | | | | |
| 1b | e4cd42744 | GQA | 2 | | | | |
| 2a | tip | GQA | 1 | | | | |
| 2b | tip | GQA | 2 | | | | |
| 2c | tip | Hybrid | 1 | | | | |
| 2d | tip | Hybrid | 2 | | | | |

3. pytest 命令与结果；UT 缺口建议（若有）  
4. 三句结论：Hybrid 基线搬运是否 OK？GQA 修改前/后？tip 上顺序 Hybrid PCP？  
5. 失败时：完整 traceback + 复现命令 + checkout SHA  

**先完成 Phase 0，再 Phase 1 → 2 → 3。** 每完成一个 Phase 落盘报告，避免中断丢失。

## 7. 背景摘要（判读日志用，勿再做大改）

本分支目标：FA 与 GDN **同一套顺序（因果连续）切分**；层后同步 KV / SSM；GQA KV 使用 local slots + `AG(K/V/slots)`；sample restore 与 DualChunk 同款全量 `AG(hidden)[idx]`。  
MoE 跟 DualChunk：切分时 pad 齐即可，不做 valid-token compact。  
设计文档：`docs/hybrid_learning/mrv2_hybrid_pcp_linear_only_design.md`。
