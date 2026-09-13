# 基于 PR #16325 的 Mooncake V1 DCP/PCP 适配

## 组合方式

本分支从 [PR #16325](https://github.com/vllm-project/vllm-ascend/pull/16325) 的 `b43f34281887e689ae978c54c0b7ff7f246f078d` 开始，使用该 PR 的完整源码，包括其 indexer、MoE 和 speculative metadata 修改。面向 Model Runner V2，配套 vLLM 为 `b2f685834a6456197e7033966fdef52a23f1abcd`。

不再叠加我们原来的 `d6dee705d` 独立 indexer 实现，也不带入后续对 `common_cp.py`、`sfa_cp.py`、`worker/v2/attn_utils.py` 的重复改动。完整 slot、block table、PCP gather 顺序及 metadata 行容量均使用 #16325 的实现，indexer 不依赖 SFA metadata。

## 仍需保留的适配

以下生产修改全部位于 `distributed/kv_transfer/kv_p2p/mooncake_connector.py`，从原验证分支提取后干净应用：

1. 无 PCP、P 不开 DCP 而 D 开 DCP 时，将完整 attention KV 映射到 D 各卡的本地 block；该分支要求两端 block size 相等。
2. 使用 MRV2 的实际 DCP size/rank，不重复乘 PCP；CP 大小允许任一方向整除。
3. 按 PCP 优先的 DCP 顺序映射物理端口，避免把 PCP 偏移当成 PP rank。
4. 不对称 CP 下匹配远端和本地 block，过滤本卡不拥有的 block。
5. 完整 indexer KV 只由选定源端口传输，其余 attention 分片端口移除 indexer group pull。

另外在 `attention/indexer.py` 增加一个 5 行的 `build_for_cudagraph_capture` 转发入口。MRV2 调用该名字，而 #16325 实现的是 `build_for_graph_capture`；配套 vLLM 父类的前者不接受 `pcp_context`。该入口转发到 PR 自己的图捕获实现，不另写 slot 算法。

测试保留原 Mooncake 传输回归，并将 PR 的 PCP+DCP 地址测试扩展为同时调用普通 build 和实际 MRV2 图捕获入口。

## 本组合的验证状态

- 针对 indexer、SFA CP、MRV2 metadata plumbing、Mooncake 的四文件单测：**182 passed**。
- 修改的四个 Python 文件 Ruff check 通过；`git diff --check HEAD` 通过。
- 单测在已有容器的独立源码目录运行，不覆盖原部署或启动模型服务。
- **这次替换后的组合尚未重新做硬件端到端验证。** 原实现的 8 组 224/224 只能作为回归基准，不能标为本分支结果。

待重新验证的原基准：两端 TP8，D 端 PCP1，GLM-5.2-w4a8，EP 与异步调度开启，等 block size/interleave 128。

| P PCP | P DCP | D DCP |
|---|---|---|
| 1 | 8 | 8 |
| 1 | 1 | 8 |
| 1 | 8 | 1 |
| 2 | 1 | 8 |
| 2 | 2 | 1 |
| 2 | 16 | 1 |
| 2 | 16 | 8 |
| 2 | 2 | 8 |

本分支不增加旧 Model Runner V1 PCP 兼容、不包含 TP8→TP16 专项适配。GLM SFA replicated indexer 的模型约束仍保留：开启 DCP 时须等于 PCP 或 TP×PCP；因此无 PCP 的 TP8/DCP2 组合不支持。不同 block size、其他模型、D 端 PCP>1、prefix caching、PP 和性能结果不由上述基准证明。
