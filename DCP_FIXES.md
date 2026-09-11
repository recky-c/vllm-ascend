# DCP：复用完整 block table 和 slot 转换

本版按讨论将地址处理放回 metadata 构建阶段。代码待用户确认，本轮不执行测试、格式检查或硬件验证。

## indexer 的插入位置

在 AscendSFAIndexerMetadataBuilder.build() 中，构造返回 metadata 之前：

```python
block_table = self._build_block_table_replicated_view(
    self._get_dcp_local_block_table(block_table, num_reqs),
    common_attn_metadata.seq_lens,
)
slot_mapping = self._build_slot_mapping_replicated_view(
    common_attn_metadata, block_table
)
```

仅在 PCP=1、indexer cache spec 的 sfa_dcp_replicated_indexer_size>1 时执行。其余路径沿用原有行为。

第一个调用生成 indexer 读取完整缓存所需的表；第二个根据全局 position 和该表计算写入 slot，避免沿用 DCP 本地 slot 的 -1 及本地偏移。结果只放入 indexer 的返回 metadata，不修改传入的 common metadata。

## 如何复用现有方法

将 sfa_cp.py 中现有的完整地址转换方法和相关缓冲初始化提取到 现有 attention/context_parallel/common_cp.py 的 ReplicatedKVMetadataMixin。包括 _build_block_table_replicated_view、_build_slot_mapping_replicated_view，以及它们使用的表切片、缓冲检查方法。

SFA builder 和 indexer builder 分别继承该 mixin，使用自己的 cache spec 和持久缓冲。两者共享算法代码，不共享 metadata、缓冲或执行顺序。SFA 的 DCP 本地序列长度、通信顺序、临时 metadata 替换等处理仍留在原类中。

文件差异中较大的删除和新增主要是把现有方法移到公共文件，不是增加另一套地址算法。indexer 自身的新增逻辑是初始化所需缓冲，以及 build() 内调用上述两个转换方法。

## 已撤销上一版改动

以下文件恢复到修改前基线 12be3d34a：

- worker/v2/model_runner.py
- worker/v2/block_table.py
- ops/triton/v2/block_table/compute_slot_mappings.py
- 上一版新增地址层回归所在的 test_compute_slot_mapping.py

不修改 initialize_kv_cache、KV tensor 分配、原始 block table 创建或上游 compute_slot_mappings/gather_block_tables 的接口。因此不再引入上一版通过扩大 gather 返回表造成的 out 宽度问题。

## 保留的 PD 修复

连接器和对应单元测试保留 e37de1b32 的两处修改：仅 P 开 DCP 时避免完整 indexer 被其他端口覆盖；仅 D 开 DCP 时完成 attention 分片映射。P TP >= D TP 限制保留，不包含 P TP8 → D TP16 扩展。

## 验证状态

新增 indexer 回归用例描述完整 block table、完整 slot、padding、common metadata 不被改写及缓冲地址稳定性。原有 SFA 转换用例保留。上述测试本轮均未运行，等待用户确认后再统一验证。

历史版本的单卡或模型通过结果不能作为本版通过的依据。完整 GLM-5.2、图重放、PD 矩阵及性能均未在本轮验证。
