# GLM-5.2 KVPP 各方案 TTFT 汇总

截至 2026-09-08，机器 157，GLM-5.2-w4a8c8，8 张 A3 物理卡 / 16 芯片。
共同配置：TP16、DP1、EP、DSA-CP、Chunk32K、异步调度、eager；单请求，生成 1 token；每项预热 1 次后测量 3 次取均值。64K / 128K 均指请求总输入长度。

旧 FlashComm 参数虽然设置为 true，但当前部署版本在 DP1 下对应 SP 路径未生效，配置不能写成 TP16+EP+SP+DSA-CP。

| 方案 | 64K TTFT / 秒 | 128K TTFT / 秒 | 状态 |
|---|---:|---:|---|
| KVPP 关闭基线 | 8.412 | 17.201 | 之前轮次实测 |
| 直接远端读取主 KV / indexer / scale | 59.246 | 129.453 | 本轮实测，包含保守同步 |
| 激活页打包 Broadcast | 8.602 | 18.402 | 最新重测 |
| 逐张量全页直接 Broadcast | 8.545 | 18.358 | 前一轮实测，不是整层一次发送 |
| 自定义 MTE Pull | 8.610 | 18.947 | 之前轮次实测 |
| 每层连续内存、一次 Broadcast | 待测 | 待测 | 分配方案已确认，尚未实现验证 |

以上包含不同轮次，不应利用不到 1% 的差值宣称稳定优势。全页对照原轮次的激活页结果为 8.629 / 18.405 秒；更早 Off/Pull/Broadcast 同轮次中的 Broadcast 为 8.552 / 18.279 秒。对应详细报告保留逐次样本。

直接远端读在本轮是激活页 Broadcast TTFT 的 6.89 / 7.03 倍，当前实现不适合所测长 prefill 场景。不可直接将所有差距归因于链路带宽，尚未拆解读取与同步开销。

激活页 Broadcast 传的数据少，但有 pack/scatter 和描述符开销。逐张量全页版去掉 pack/scatter，却仍逐张量调用 HCCL，一层可能 1～3 次，且父类初始化仍依赖已有 copy 库；不能表述成已经实现了整层单次广播或完全去除了算子库依赖。

下一版整层 Broadcast 按用户确认：每个持久化层按实际大小分配，scratch 容量能容纳任意层，主 KV / indexer / scale 使用固定偏移视图，每层每轮仅广播当前层实际字节范围。

Pull 使用自定义 MTE copy，消耗 AIV/MTE、HBM 与跨卡互连资源；不调用 HCCL Broadcast 不等于不占通信资源。已测多 rank 的 Pull/Push 微基准支持在自定义 copy 路线中优先 Pull，但不代表整模型 TTFT 优于 Broadcast。

“不继续沿用旧算子实现”属于路线选择或讨论结论，不是这组 TTFT 单独证明的事实。当前数据支持优先推进整层连续 Broadcast，并保留 Pull 作为自定义 copy 对照。

参考：RESULTS_ONLINE_64K128K.md、RESULTS_FULLPAGES_64K128K.md、RESULTS_REMOTE_READ_64K128K.md。最新测试已退出，所有 16 芯片空闲。
