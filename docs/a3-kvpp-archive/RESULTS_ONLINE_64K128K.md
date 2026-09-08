# 157：64K / 128K 在线 TTFT

测试模型：GLM-5.2-w4a8c8。8张A3物理卡、全部16芯片，TP16、DP1、EP开启、异步调度开启、Chunked Prefill预算32768，DSA-CP开启。

**FlashComm限制：旧参数enable_flashcomm1=true已设置，但部署版本的use_sequence_parallel_moe要求DP>1；本轮DP1，该SP路径未生效。因此不能声称完整覆盖了实际FlashComm开启的组合。DSA-CP有独立开关，不受这个DP条件控制。**

三组共同设置：不强制HCCL_OP_EXPANSION_MODE，HCCL_BUFFSIZE=1024；eager；单请求、输出1 token；无prefix cache；max_model_len=131200、blocks=1026；服务端口30001。每种输入预热1次、正式3次。同一长度除KVPP后端/必要copy参数外，其余服务、环境和压测参数经脚本核验一致。

## 正式结果

| 输入token | KVPP关闭 | Pull 8核 | Broadcast 32核 | Pull相对关闭 | Broadcast相对关闭 |
|---|---:|---:|---:|---:|---:|
| 65,536 | 8.412s | 8.610s | 8.552s | +2.35% | +1.67% |
| 131,072 | 17.201s | 18.947s | 18.279s | +10.15% | +6.27% |

数值是流式在线API的客户端TTFT，来自vllm bench serve，不是模型加载时间或完整服务启动耗时。输入是总长度64K/128K，而非额外加128K历史。此前160K离线、TP8结果不能用于直接计算这轮的加速比。

所有24个请求（含6个预热请求）均成功：每次输入长度正确、输出1 token，失败数为0。对应正式请求首token文本跨后端一致；未做完整logits或业务精度验收。固定执行顺序Broadcast→Off→Pull，样本量小，结论限于本轮单请求设置。

## 正式样本（秒）

| 后端 | 输入 | 三个TTFT样本 | 样本标准差 |
|---|---:|---|---:|
| off | 65536 | 8.400923, 8.440074, 8.395241 | 0.024410 |
| off | 131072 | 17.207910, 17.201681, 17.193320 | 0.007321 |
| ipc_pull | 65536 | 8.606395, 8.604902, 8.617472 | 0.006867 |
| ipc_pull | 131072 | 18.996185, 18.901314, 18.944262 | 0.047506 |
| ipc_broadcast | 65536 | 8.548486, 8.574506, 8.533434 | 0.020779 |
| ipc_broadcast | 131072 | 18.316315, 18.213826, 18.307817 | 0.056878 |

## 失败和诊断记录

- TP8（每物理卡一个芯片）：Off在128K缓存初始化OOM；Pull在64K预热时OOM。这些不是有效TTFT。之后统一采用skill原有TP16拓扑。
- 强制HCCL_OP_EXPANSION_MODE=AIV的TP16组：Off/Pull成功，Broadcast在Hello、max_tokens=8的真实就绪检查中停顿，出现持续通信等待，未进入64K/128K计时。清理后移除该强制项，Broadcast完整成功。这个结果支持避免该配置组合，尚未定位底层停顿的精确根因。
- 为避免混用环境，Off/Pull也按移除强制AIV后的环境重新完整测试。最终表只使用这组一致环境。
- Off首次结果打印发生Windows GBK编码错误；UTF-8结果文件此前已落盘完整。后续统一PYTHONIOENCODING=utf-8。

## 复现与证据

使用D:/code/vllm-ascend-workspace/.agents/skills/vllm-ascend-benchmark/scripts/bench_run.py。三个preset位于该skill的presets目录，名称为glm52-ipc-online-default-{off,ipc_pull,ipc_broadcast}。参数：--session-id kvpp-ttft-157-20260906 --model /mnt/weight/GLM-5.2-w4a8c8 --port 30001 --health-timeout 900 --runs 4 --warmup-runs 1 --input-lengths 65536,131072 --skip-parity。

skip-parity用于保留任务隔离的product-runtime；其539个Python文件已独立核验与phase2交付代码一致。没有修改模型或传输产品代码。真实运行源位于/home/recky/a3-kv-transfer-20260907/phase2/product-runtime；实际vLLM来自/mnt/share/recky/vllm_6e448d0_pp2_20260906。

evidence/online-default-{off,pull,broadcast}-result.json包含全部原始请求结果与配置；evidence/online-default-summary.json为经过成功率、输入长度及配置一致性验证的汇总。summarize_online.py --family default可重建。artifacts/online-runtime-evidence.tar.gz保留原始服务日志和结束时空闲证据。
