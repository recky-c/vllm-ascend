# GLM-5.2 直接远端 KV 读取：64K / 128K 在线 TTFT

日期：2026-09-08；机器：192.168.13.157。

## 本轮结果

客户端 TTFT，单位秒；每个长度预热 1 次，正式测量 3 次取均值；正百分比表示直接远端读更慢。

| 总输入长度 | 激活页 Broadcast | 直接远端 KV 读取 | TTFT 变化 |
|---|---:|---:|---:|
| 64K | 8.602045 | 59.245771 | +588.74% |
| 128K | 18.402349 | 129.452814 | +603.46% |

直接远端读取的 TTFT 为本轮激活页 Broadcast 的约 6.89 倍（64K）和 7.03 倍（128K）。当前完整实现不适合所测长 prefill 场景；该结论包含本实现逐层同步开销，尚未通过 profiler 拆分归因。

全部 16 个请求成功，实际输入长度均为 65536 / 131072 token，输出均为 1 token，无失败请求或零 TTFT 混入。对应正式请求的首 token 文本一致，并与前一轮 Broadcast 对照一致。这是最小输出对照，不是完整精度验收。

## 直接读取的具体含义

候选代码独立保存在 remote-read-product / remote-read-runtime。通过 enable_kvpp=true、kvpp_transport=ipc_pull、kvpp_remote_read=true 选择；ipc_pull 在这里复用 IPC 配置与初始化，实际选择 IpcRemoteReadKVPPTransport，而不是执行历史 KV 拉取。

每个 rank 按原路径把当前 chunk 的 KV 写入本地 cache；等待各 rank 当前层写入完成后，主 SFA、indexer 和 scale 的读取参数替换为 owner 的 IPC 映射视图。非 owner 的 attention/indexer 算子直接使用远端地址。没有显式历史 KV payload copy、pack/scatter 或 Broadcast；跨卡读取本身仍产生链路流量。

该实现保留本地 scratch 分配，以隔离“远端读”这一变量；不是最小显存实现。采用保守的逐层写完成/读完成同步，时间包含在 TTFT 中。原有异步调度配置保持开启，但不能把结果当作消除了所有同步后的性能上限。

父类初始化仍加载已有 copy 库并建立 IPC 映射，热路径不调用 copy-prefetch。两种模式均为完整部署路径测试，不是昨天的合成算子耗时外推。

## 配置与限制

- GLM-5.2-w4a8c8；A3 8 张物理卡 / 16 芯片，TP16、DP1、EP 开启。
- DSA-CP、异步调度开启；max_num_batched_tokens=32768；max_model_len=131200；max_num_seqs=1；num_gpu_blocks_override=1026。
- 并发 1；eager；关闭 prefix caching；随机输入 seed=0；每个请求生成 1 token。
- HCCL_BUFFSIZE=1024，不强制 HCCL_OP_EXPANSION_MODE=AIV。
- 旧 enable_flashcomm1 参数保留，但 DP1 下当前部署版本的 FlashComm/SP 路径未生效。
- 同一候选代码与运行环境；只更改 kvpp_remote_read 与对应的 kvpp_transport 选择。固定顺序先 direct 后 broadcast，尚无交替顺序多轮统计。
- 64K / 128K 是请求总输入长度，不是额外加上 32K 新输入的历史 KV 场景。

## 验证和交付

38 项单元测试通过。2-rank 和 16-rank 使用真实 KVPPScheduler、IPC allocation、混合 int8 / BF16 / FP16 张量检查远端指针、NPU 读取、owner 数据更新、scratch 复用、延迟读者和清理；合计 3552 项数据检查通过。非 owner 本地 sentinel 未变化，排除了暗中搬运到 scratch 再读的测试路径。

原始结果：evidence/remote-read-{direct,broadcast}-result.json。
汇总与逐次样本：evidence/remote-read-summary.json。
源代码增量与哈希：remote-read-delta.tar.gz。
精确两次服务运行目录、生命周期日志、最终空闲状态：artifacts/remote-read-runtime-evidence.tar.gz，内置 SHA256 清单。原始请求结果保存在上述 result.json 中，随完整交付包提供。
完整交付：remote-read-delivery.tar.gz。测试结束后自有服务退出，16 芯片空闲。

## 整层 Broadcast 分配方案

按用户确认：持久化层按各层实际大小分配，scratch 容量能容纳任意层；每次仅传当前层实际区域。这里记录设计约束，本轮没有把该 allocator 改造混入远端读 TTFT 比较。
