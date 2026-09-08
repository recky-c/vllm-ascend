# 全页直接 Broadcast 与激活页 Broadcast：64K / 128K TTFT

日期：2026-09-08；机器：192.168.13.157；本轮已完成。

## 实测结果

下表为客户端测得 TTFT，单位秒；每项预热 1 次后测量 3 次，列出算术平均值。变化为全页版相对激活页版，正值表示更慢。

| 总输入长度 | 激活页 Broadcast | 全页直接 Broadcast | TTFT 变化 |
|---|---:|---:|---:|
| 64K | 8.629355 | 8.545382 | -0.973% |
| 128K | 18.405439 | 18.357835 | -0.259% |

本轮全页版在 64K 降低 TTFT 约 0.97%（84 ms），128K 降低约 0.26%（48 ms）。收益很小，特别是 128K 的样本有重叠；目前不能认定稳定优于激活页版，也没有依据宣称明显加速。省掉 pack/scatter 的同时增加了无效页传输，这可能抵消部分收益，具体归因尚未做 profiler 验证。

两种模式分别成功完成 8 个请求，共 16 个，零失败。逐项核对实际输入为 65536 / 131072 token、输出为 1 token，计时值均大于零。对应测量请求的首 token 文本一致；这不代表完整模型精度验收。

## 配置与比较范围

- GLM-5.2-w4a8c8，A3 的 8 张物理卡 / 16 个芯片，TP16、DP1，EP 与异步调度开启。
- DSA-CP 全程开启；max_num_batched_tokens=32768；max_model_len=131200；max_num_seqs=1；并发 1；eager；关闭 prefix caching；num_gpu_blocks_override=1026。
- HCCL_BUFFSIZE=1024；不强制 HCCL_OP_EXPANSION_MODE=AIV。
- 两个命名 preset 的运行环境、设备、服务参数、benchmark 参数已程序化比对，只有 kvpp_broadcast_full_pages 开关不同。
- 旧 enable_flashcomm1 参数保留，但当前 DP1 不满足部署版本的 sequence-parallel MoE 路径条件，不能将结果称为实际 FlashComm/SP 已生效。
- 64K / 128K 指本次请求的总输入长度，未额外追加一个 32K chunk，也不是“外部预填充 128K 历史再输入 32K”的测试。
- 固定顺序：先全页版，后激活页版；各 3 个有效样本。小差异不足以证明稳定的性能收益。

## 实现与验证

全页版位于独立 fullpages-product / fullpages-runtime，原 product / product-runtime 未修改。通过 additional_config.kvpp_broadcast_full_pages=true 选择；原激活页路径保持 false。

全页版直接对当前 bundle 内各 KV 张量的完整连续物理页区间做 HCCL Broadcast，包含未激活页，接收缓冲直接别名到 attention cache，省掉 pack、wire buffer 和 scatter。它不是全模型所有 KV 只调用一次 Broadcast；多个缓存张量仍分别调用 HCCL。非连续布局会被显式拒绝。

继承原有 source-ready、scratch last-use、generation、reader ACK 生命周期保护。父类初始化仍依赖原 IPC / copy 库，不能宣称已完全移除自定义库依赖；本次传输热路径不调用 pack/scatter 自定义算子。

远端 43 项单元测试通过；2-rank 与 16-rank 生命周期测试共 10800 次 NPU 数据检查通过，包含全部页、非默认流、延迟读者、scratch 复用与清理。真实模型两种模式均通过短请求 readiness 后再计时。

## 证据与复现

- evidence/fullpages-summary.json：均值、标准差、每次计时、首 token 与原始结果路径。
- evidence/fullpages-{full,active}-result.json：官方 benchmark 结果与解析后的配置。
- artifacts/fullpages-runtime-evidence.tar.gz：两次服务的精确运行目录日志、请求结果、生命周期检查与最终空闲状态；内置 SHA256 清单。
- fullpages-delta.tar.gz：候选代码与代码 SHA256 清单。
- 命名 preset：glm52-fullpages-full / glm52-fullpages-active；使用工作区 benchmark skill，--runs 4 --warmup-runs 1 --input-lengths 65536,131072。

完整交付包为 fullpages-delivery.tar.gz。测试后自有服务已退出，所有 16 个芯片已复核无运行进程。
