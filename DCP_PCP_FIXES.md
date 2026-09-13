# Mooncake V1 与 Model Runner V2 的 DCP/PCP 修复说明

本分支包含独立 indexer metadata 修复，以及 MooncakeConnectorV1 的不对称 DCP、MRV2 PCP 适配。这里只讨论 Model Runner V2，不新增旧 Model Runner V1 PCP 的兼容逻辑。

基础版本：vLLM-Ascend `12be3d34a1510be8e19542f577e732a12c47a1d1`，配套 vLLM `b2f685834a6456197e7033966fdef52a23f1abcd`。分支包含已有的 indexer 修复提交 `d6dee705d`，后续提交叠加传输和 PCP 修复。

## 正常的数据关系

- Attention KV 按 DCP 分片；indexer KV 保持完整副本。
- MRV2 的 DCP 组已经包含 PCP 维度，不能再按 `PCP × DCP` 计算 KV 分片数。DCP rank 按 PCP 优先、TP 随后的顺序排列。
- Indexer 独立构建完整地址视图，复用公共计算方法，不读取 SFA builder 的 metadata 或缓冲区。
- PCP 本地 query 片段的顺序和 KV 写入 gather 顺序是两个不同的问题：block table 行与本地 query 片段对应，写入 slot 与 gather 后的 token 对应。

## 修复的问题

| 问题 | 原因与影响 | 最终修改 |
|---|---|---|
| Indexer 使用了被 DCP 屏蔽的 slot | Attention 的非本卡 slot 被置为 `-1`，不能用于完整 indexer KV；分片 block table 也不能直接作为完整地址 | Indexer 用自己的 spec 和 buffer 构建 replicated block table、完整 slot mapping。该问题非 PD 场景也存在 |
| 无 PCP、P 不开 DCP 而 D 开 DCP 的布局转换 | P 保存完整 attention KV，D 只拥有部分逻辑 block，源地址与目的地址不能用同一索引 | 按 D rank 选择全局 block，分别计算 P 的完整地址和 D 的本地地址；该路径要求两端 block size 相等 |
| 两端 CP 大小不等时选错源分片或 block | 原有检查只允许 P 是 D 的倍数；反方向的端口和 block 选择没有完整覆盖 | 保留正整数 CP 约束，允许任一方向整除；按两端 CP 关系选择端口，并同步过滤 D 不拥有的远端、本地 block |
| Indexer 被当成 attention 分片重复传输 | 完整 indexer KV 跟随多个 attention 端口的 group pull，可能覆盖完整副本传输 | 从选定源端口传输完整 indexer KV，其他端口移除 indexer group pull |
| PCP 被重复算入 CP 大小 | MRV2 DCP 已覆盖 PCP，旧的 `PCP × DCP` 会重复计算 | 使用实际 DCP size 和 DCP rank 计算 KV 分片 |
| PCP 的 rank/端口对应及 PP 偏移不正确 | 逻辑 DCP rank 顺序与物理端口顺序不同，PCP 端口偏移还可能被解释为 PP rank | 按 PCP 优先的 DCP 顺序映射到 PCP-major 端口；PP 偏移除以 `TP × PCP` |
| PCP 下 indexer 的写入顺序错误 | 独立 indexer 未接收 PCP context，跳过完整视图路径，写入 slot 未匹配 gather 顺序 | MRV2 向 indexer 传入 PCP context；从全局地址构建 slot，再按 gather 索引重排并屏蔽 padding。原有计算方法移入 `common_cp.py` 复用 |
| PCP 图捕获启动失败 | Indexer 图捕获入口不接受 `pcp_context` 参数 | 补齐图捕获入口，转发参数到同一 build 路径 |
| PCP 并发 metadata 行数不足 | 每个 prefill 拆成两个本地 query 片段；配置最大 8 请求时只预留 9 行，实际 12 行触发越界并导致 HTTP 500 | PCP 下预留 `2 * max_num_seqs + 1` 行，包含 FIA padding；只增加 metadata 行容量，不增加 KV 存储或传输量 |

其中 PCP 写入顺序错误曾在 PD 和直接请求 P 端两种方式下复现，说明属于计算侧问题。图捕获和行数不足也是适配过程中暴露的 metadata 问题，不应全部归因于 Mooncake 传输。

## 代码位置

- `vllm_ascend/attention/indexer.py`：独立 indexer 地址、PCP context、图捕获入口、行容量。
- `vllm_ascend/attention/context_parallel/common_cp.py`：完整 KV 地址及 PCP gather 顺序的共享计算方法。
- `vllm_ascend/attention/context_parallel/sfa_cp.py`：复用上述公共方法，删除对应重复实现。
- `vllm_ascend/worker/v2/attn_utils.py`：将 PCP context 传给 indexer builder。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`：CP 约束、rank/端口、block 映射、完整 indexer 传输。
- `tests/ut/attention/test_indexer.py`、`tests/ut/worker/test_attn_utils_v2.py`、`tests/ut/kv_offload/test_mooncake_connector.py`：对应回归覆盖。

没有修改上游 `compute_slot_mappings` 接口、`initialize_kv_cache` 或 KV cache 分配规则；没有叠加 TP8→TP16 专项修改。

## 硬件验证

模型 GLM-5.2-w4a8，MooncakeConnectorV1、Model Runner V2，两端 TP8，D 端 PCP1，EP 与异步调度开启。Block size/interleave 均为 128，最大长度 32768、batch tokens 4096、最大请求数 8；开启 chunked prefill，关闭 prefix caching，使用 FULL_DECODE_ONLY 图模式。未启用 DSA-CP、SFA/indexer C8 或 FlashComm1。

每组 28 条：19 条串行边界和长上下文请求、8 条并发请求、1 条并发后检查。正确性包含答案校验，不只检查 HTTP 200。

| P PCP | P DCP | D DCP | 结果 | 日期 / 运行标签 |
|---|---|---|---|---|
| 1 | 8 | 8 | 28/28 | 2026-09-13 / `matrix0913-pcp1-p8-d8` |
| 1 | 1 | 8 | 28/28 | 2026-09-13 / `matrix0913r2-pcp1-p1-d8` |
| 1 | 8 | 1 | 28/28 | 2026-09-13 / `matrix0913r2-pcp1-p8-d1` |
| 2 | 1 | 8 | 28/28 | 2026-09-13 / `matrix0913r2-pcp2-p1-d8` |
| 2 | 2 | 1 | 28/28 | 2026-09-13 / `matrix0913r2-pcp2-p2-d1` |
| 2 | 16 | 1 | 28/28 | 2026-09-13 / `matrix0913r2-pcp2-p16-d1` |
| 2 | 16 | 8 | 28/28 | 2026-09-11 / `pcp2-cp16-d8-r5` |
| 2 | 2 | 8 | 28/28 | 2026-09-11 / `pcp2-cp2-d8-r1` |

共 224/224。9 月 13 日新增六组均核对了 D 端 87,767 个外部 KV token，以及 28 条请求各只预填充 1 token，确认实际复用了传输 KV。旧 Mooncake V2 的结果未计入。

相关单元测试 174 项通过：

```bash
python -m pytest -q \
  tests/ut/attention/test_indexer.py \
  tests/ut/attention/test_sfa_cp.py \
  tests/ut/worker/test_attn_utils_v2.py \
  tests/ut/kv_offload/test_mooncake_connector.py
```

## 限制与未声明的覆盖

- 传输 CP 大小须满足一方是另一方的整数倍；这不是允许所有整除组合的模型配置。
- GLM SFA replicated indexer 在开启 DCP 时要求 DCP 等于 PCP 或 TP×PCP。因此 TP8/PCP1 的 DCP2 在配置校验阶段被拒绝。无 PCP 的 2→8、8→2 不属于支持组合，未绕过限制。
- 结果限定在上述模型、配套版本、等 block size 和拓扑。不同 TP、不同 block size、其他模型、D 端 PCP>1、prefix caching 开启、PP、DSA-CP 和性能指标不由这些用例证明。
- 验证脚本的 PowerShell 变量冲突与 TIME_WAIT 端口检查问题已修正，仅属测试基础设施，不计入产品修复；测试服务已停止并释放本次使用的卡。
