# Ascend MRv2 Hybrid PCP — 全层 DualChunk 头尾切分

> 基线：`d7abe43299171c168aa666257df78af18a365460`
>
> 分支：`feat/mrv2-hybrid-pcp-dual-chunk-all-layers`
> 状态：第一版实现；eager 正确性优先，待 NPU 多卡验证与性能优化。

## 1. 结论

Hybrid 模型只保留一份上游 PCP DualChunk local batch：GQA、GDN、MLP、
MoE 都消费同一个头尾布局。GQA 继续使用已有的 DualChunk attention 路径；
GDN 在算子内部按全局 segment 因果顺序修正 conv/SSM 状态。

不引入顺序 partition、第二份 batch 或 GQA/GDN 布局 bridge。

## 2. DualChunk 因果顺序

PCP=4 时，一个请求被切成八个 chunk：

```text
global:  0   1   2   3   4   5   6   7
rank 0:  0                           7
rank 1:      1                   6
rank 2:          2           5
rank 3:              3   4
```

GDN 状态必须按以下顺序传播：

```text
r0.head -> r1.head -> r2.head -> r3.head
        -> r3.tail -> r2.tail -> r1.tail -> r0.tail
```

因此，物理 rank 顺序不能直接代表 GDN 的因果顺序。

## 3. Segment ID 契约

Manager 为每个 rank-local prefill row 生成稳定 ID：

```text
segment_id = global_request_index * (2 * pcp_size) + chunk_index
```

- ID 与本地 row 重排无关。
- decode row 使用 `-1`。
- 每个 rank 的 collective payload 固定 pad 到
  `2 * global_num_requests` 行。
- gather 后根据 ID scatter 到 `[request, 2 * pcp_size]` 的规范布局。

字段通过以下路径传递：

```text
AscendPCPManager
  -> AscendInputBatch
  -> AscendCommonAttentionMetadata
  -> GDNChunkedPrefillMetadata
  -> gdn_pcp_conv / gdn_pcp_ssm
```

## 4. GDN conv

同一请求的 head/tail 虚拟 row 共享真实 `conv_state` cache slot。如果直接把
两个 row 同时交给 causal-conv 算子，会出现不同 initial history 争用同一个
cache row 的问题。

第一版采用临时 segment cache：

1. 每卡提取本地 head/tail 的最后 `state_len` 个原始输入。
2. AG tails 与 `(segment_id, valid_len)`；`state_index` 保持 rank-local，不参与通信。
3. 按规范 chunk 顺序折叠 history，得到每个 segment 的前驱 history。
4. 为每个本地 segment 分配独立临时 conv-state row。
5. prefill causal-conv 使用临时 cache；mixed batch 中 decode 单独使用真实 cache。
6. 只把每个请求最后有效 segment 的 history 写回真实 conv cache。

短于 conv width 的 segment 通过前驱 history 补齐，不要求本地 segment 长度
大于卷积核宽度。

## 5. GDN SSM

每个 segment 的状态转移写成：

```text
s_out = p_i + Phi_i @ s_in
```

当前阶段禁用 prefix caching 和 continued prefill，因此 provisional local run
统一从零状态开始，`p_i` 就是 local final state。

流程：

1. 每卡并行计算本地 segment 的 provisional `fwd_h`。
2. 计算每个 segment 的 `Phi_i`。
3. AG `(segment_id, p_i, Phi_i)` 并恢复规范布局。
4. 按 `chunk 0 .. 2P-1` 执行状态 scan。
5. 用正确的 segment initial state 重跑本地 `fwd_h`。
6. 同一请求的所有本地虚拟 row 返回相同的全局 final state，再写入真实
   SSM cache；重复 state index 的写入值完全一致。

第一版为降低实现风险，所有本地 prefill segment 都重跑 `fwd_h`。后续可只
重跑除第一个全局 segment 以外的 token，降低 GDN 额外计算量。

## 6. 其它层

- GQA：保持已有 DualChunk Q 与完整 KV cache 路径。
- MLP：token-wise，直接使用 local DualChunk token。
- MoE：沿用 PCP AG/RS；`max_tokens_across_pcp` 使用 Manager 已 pad 的本地
  token width。
- sample restore：继续使用上游 `restore_hidden_states` 和
  `_hidden_restore_idx`。
- decode：各卡复制计算；prefill 结束后各卡持有一致的 conv/SSM state。

## 7. 第一阶段限制

- Hybrid prefill 的每个请求至少包含 `pcp_size` 个 token，保证所有 rank
  同步进入 GDN collectives；不满足时在 partition 阶段 rank-consistent
  fail-fast。
- 不支持 DCP、PP、MM、LoRA、spec decode、prefix caching、scheduler
  chunked prefill。
- 以 eager 为首要验收路径；piecewise graph 需要在动态临时 buffer 消除后
  单独验证。

## 8. 验收

CPU UT：

- Manager 的 local row 与 canonical segment ID 对齐。
- conv history 按全局 chunk 顺序折叠。
- 缺失 tail chunk 作为 identity 跳过。
- SSM 仿射 scan 的 head/tail 顺序正确。
- 短 prefill fail-fast。

NPU 多卡：

- PCP=1 与 PCP=2/4 的最终 token、逐层 GDN 输出对齐。
- mixed prefill/decode 下 conv/SSM cache 对齐。
- 不等长多请求与缺失 tail chunk。
- GQA KV cache、MoE AG/RS、sample restore 不回归。
- 统计每 GDN 层 collective 次数、临时显存和 `fwd_h` 重算耗时。

## 9. 后续优化

1. 将临时 conv-state 和 canonical gather tensor 改为初始化时预分配。
2. 合并 segment metadata collectives，避免 conv/SSM 重复 gather ID。
3. 只重跑 initial state 被修正的 SSM segment。
4. 支持某些 rank 完全没有 prefill segment 的短请求，移除
   `query_len >= pcp_size` 限制。
5. 完成 piecewise ACL graph 捕获与回放验证。
