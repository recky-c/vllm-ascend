# SFA C8 与 KV Offload 兼容适配方案

## 1. 背景与基线

本文面向 `feat/sfa-offload-layerwise-reuse` 分支，分析其与已合入主线的 SFA C8 改动如何统一，并给出可执行的适配方案。

基线信息：

- 主线：`upstream/main`，当前观察到最新提交为 `3e34542d6`。
- SFA C8 合入提交：`587ac6aa8 [Feature][Refactor] Support SFA C8 on A3 with unified packed KV cache layout (#11228)`。
- OFFLOAD 分支：`ader47/feat/sfa-offload-layerwise-reuse`，当前观察到最新提交为 `c3a271f8d`。
- 关键冲突文件：`vllm_ascend/attention/sfa_v1.py`、`vllm_ascend/device/device_op.py`、`vllm_ascend/core/kv_cache_interface.py`、`vllm_ascend/worker/model_runner_v1.py`。

目标：

1. 让 SFA KV Offload 在 `enable_sparse_c8=true` 时与主线 #11228 的 C8 语义保持一致。
2. 保留 GLM5.2 1M 长序列场景下的 OFFLOAD 能力：main MLA KV 可下沉 CPU，indexer cache 常驻/可通过 PD 拉取，decode TopK 后按需 H2D。
3. 避免 OFFLOAD 分支继续维护一套与主线不同的 LIC8 cache layout。

推荐原则：

- 以主线 #11228 的 packed C8 layout 为准。
- OFFLOAD 只扩展 resident / CPU staging，不重新定义 SFA C8 的主 cache 语义。
- 所有 tuple 下标必须收敛到常量/布局 helper，禁止在 `sfa_v1.py`、`device_op.py`、PD read thread 中继续裸写 `[2]`、`[5]` 这类下标。

## 2. 主线 SFA C8 做了什么

### 2.1 C8 语义被拆成两层

主线在 `vllm_ascend/attention/sfa_v1.py` 将 SFA C8 拆成两个概念：

- `use_sparse_c8_indexer`：indexer `k_li` / `k_li_scale` 走 C8。
- `use_sparse_c8_sfa`：SFA 主 attention 的 main KV 走 packed C8，用于 `npu_kv_quant_sparse_flash_attention`。

关键位置：

- `upstream/main:vllm_ascend/attention/sfa_v1.py:613`
- `upstream/main:vllm_ascend/attention/sfa_v1.py:614`
- `upstream/main:vllm_ascend/attention/sfa_v1.py:632`

含义：

- 有 indexer 的 sparse C8 层：`use_sparse_c8_indexer=true`，同时 `use_sparse_c8_sfa=true`。
- 没有 indexer、但复用 topk 的层：可能 `use_sparse_c8_indexer=false`，但仍需要 `use_sparse_c8_sfa=true`，因为 main KV 仍要 packed C8。
- PD decode consumer 上，packed C8 会启用 `enable_sfa_prolog_v3`，通过 Prolog V3 直接写 packed cache。

### 2.2 主 KV 从分离布局变成 packed 布局

#11228 后，SFA C8 不再只是把 indexer key 变成 int8/fp8。它还将 main MLA KV packed 到 `kv_cache[0]`。

主线 C8 no-offload 典型布局：

```text
C8 packed SFA, has indexer:
  kv_cache[0] = packed main KV
  kv_cache[1] = indexer_k
  kv_cache[2] = indexer_scale

C8 packed SFA, no indexer / skip_topk layer:
  kv_cache[0] = packed main KV
```

非 C8 仍保持旧语义：

```text
non-C8 SFA:
  kv_cache[0] = kv_lora / k_nope
  kv_cache[1] = k_rope
  kv_cache[2] = indexer_k, if layer has indexer
```

关键位置：

- `upstream/main:vllm_ascend/core/kv_cache_interface.py:28`
- `upstream/main:vllm_ascend/core/kv_cache_interface.py:68`
- `upstream/main:vllm_ascend/core/kv_cache_interface.py:74`
- `upstream/main:vllm_ascend/worker/model_runner_v1.py:4285`
- `upstream/main:vllm_ascend/worker/model_runner_v1.py:4289`
- `upstream/main:vllm_ascend/worker/model_runner_v1.py:5031`
- `upstream/main:vllm_ascend/worker/model_runner_v1.py:5063`

`AscendMLAAttentionSpec.cache_sparse_c8=true` 时，主线要求 `sparse_head_dim` 里的 `qk_rope_head_dim == 0`，用它表示 packed layout。

### 2.3 page size / split ratio 按 packed 字节重新计算

主线 `AscendMLAAttentionSpec.page_size_bytes` 对 C8 packed layout 重新计数：

- `ckv_bytes`：packed main KV 占用。
- `qli_bytes`：indexer key 占用。
- `qli_scale_bytes`：indexer scale 占用，且要考虑 `sfa_dcp_replicated_indexer_size`。

关键位置：

- `upstream/main:vllm_ascend/core/kv_cache_interface.py:64`
- `upstream/main:vllm_ascend/core/kv_cache_interface.py:79`
- `upstream/main:vllm_ascend/core/kv_cache_interface.py:92`
- `upstream/main:vllm_ascend/core/kv_cache_interface.py:120`

这意味着 OFFLOAD 不能继续按旧的 `[kv_lora, k_rope, indexer_k, indexer_scale]` 线性比例直接切分同一个 paged tensor。

### 2.4 写 cache 的路径发生变化

主线新增/统一了几条 C8 写入路径：

- `custom_kv_rmsnorm_rope`：native path 下对 main KV 做 RMSNorm、RoPE、block quant，再拼成 packed cache。
- `execute_sfa_mla_prolog_v3`：decode consumer / Prolog V3 path 下写 packed cache。
- A5 与 A3 都通过 `use_sparse_c8_sfa` 判断是否使用 packed cache。

关键位置：

- `upstream/main:vllm_ascend/attention/sfa_v1.py:1165`
- `upstream/main:vllm_ascend/attention/sfa_v1.py:1533`
- `upstream/main:vllm_ascend/attention/sfa_v1.py:1867`
- `upstream/main:vllm_ascend/device/device_op.py:429`
- `upstream/main:vllm_ascend/device/device_op.py:447`
- `upstream/main:vllm_ascend/device/device_op.py:480`

### 2.5 读 cache 的 attention op 发生变化

C8 packed SFA 的主 attention 不再使用普通 `npu_sparse_flash_attention` 的 `key=kv_cache[0]`、`value=kv_cache[0]` 分离语义，而是走 quant sparse attention：

- `torch_npu.npu_kv_quant_sparse_flash_attention`
- `kv_quant_mode`
- `tile_size`
- `rope_head_dim`

关键位置：

- `upstream/main:vllm_ascend/device/device_op.py:687`

### 2.6 indexer 下标也随 packed layout 改变

主线 `device_op.py` 根据 `packed_kv_cache` 解析 indexer 下标：

```text
packed_kv_cache = use_sparse_c8_sfa
indexer_cache_idx = 1 if packed_kv_cache else 2
indexer_scale_cache_idx = 2 if packed_kv_cache else 3
```

关键位置：

- `upstream/main:vllm_ascend/device/device_op.py:567`
- `upstream/main:vllm_ascend/device/device_op.py:568`
- `upstream/main:vllm_ascend/device/device_op.py:569`
- `upstream/main:vllm_ascend/device/device_op.py:572`

这正是 OFFLOAD 分支现有六元组方案最大的冲突点。

## 3. OFFLOAD 分支现状

### 3.1 单机 OFFLOAD 现有语义

当前 OFFLOAD 分支通过 `SFAKVOffloadConnector` / `SFAKVOffloadWorker` 支持：

- main MLA KV 下沉 CPU。
- decode TopK 后，按 `num_offloaded_blocks` 将 TopK token 分成 NPU resident / CPU offloaded 两部分。
- CPU miss 通过 LRU resident buffer 异步拉回 HBM。
- attention 先算 NPU 侧，再算 CPU resident 侧，最后合并 LSE/out。

关键位置：

- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/__init__.py:90`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py:198`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/sfa_v1.py:1908`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/sfa_v1.py:1984`

### 3.2 现有 OFFLOAD tuple contract

当前分支定义：

```text
non-C8 offload:
  [0] = main_k / k_nope
  [1] = main_v / k_rope
  [2] = indexer_k
  [3] = resident_k
  [4] = resident_v

C8 + offload, 当前分支:
  [0] = main_k / k_nope
  [1] = main_v / k_rope
  [2] = indexer_k
  [3] = resident_k
  [4] = resident_v
  [5] = indexer_scale
```

关键位置：

- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_kv_offload/offload_kv_cache_layout.py:13`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_kv_offload/offload_kv_cache_layout.py:14`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_kv_offload/offload_kv_cache_layout.py:16`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_kv_offload/offload_kv_cache_layout.py:19`

这个设计的好处是 `[3]/[4]` resident 不因 C8 改动下标；问题是它已经与 #11228 packed C8 layout 不一致。

### 3.3 OFFLOAD metadata 现有扩展

分支新增了 OFFLOAD 所需 metadata：

- `indexer_block_table_tensor`
- `indexer_slot_mapping`
- `num_offloaded_blocks`
- `req_ids_tensor`
- `token_to_req`

关键位置：

- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/sfa_v1.py:239`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/sfa_v1.py:361`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/sfa_v1.py:507`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/utils.py:237`

这些 metadata 仍然是正确方向，适配时应该保留。

### 3.4 PD CPU Offload 现有语义

PD D 侧组合：

- `SFAKVOffloadWorker`：本地 CPU pool + LRU resident H2D。
- `MembPullReadThread`：从 P 拉 main/indexer/scale。

当前 PD 假设：

- group 0 = indexer。
- group 1 = main MLA。
- D 侧 indexer tensor 来自 `main_tuple[2]`。
- D 侧 LIC8 scale 来自 `main_tuple[5]`。
- P 侧 scale 来自 `p_base_addrs[3]`。

关键位置：

- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py:73`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py:374`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py:375`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/read_thread.py:245`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/read_thread.py:269`

这部分在 #11228 后必须重做下标和 block length 计算。

## 4. 现有差距

### 4.1 最大差距：main KV layout 不一致

主线 C8：

```text
[0] = packed main KV
[1] = indexer_k
[2] = indexer_scale
```

OFFLOAD 分支 C8：

```text
[0] = main_k
[1] = main_v
[2] = indexer_k
[3] = resident_k
[4] = resident_v
[5] = indexer_scale
```

结果：

- 主线 `device_op.py` 会在 C8 packed 下读 `[1]` 作为 indexer key，但 OFFLOAD 分支 `[1]` 是 main_v。
- 主线 `sfa_v1.py` 会在 C8 packed 下把 indexer scale 写到 `[2]`，但 OFFLOAD 分支 `[2]` 是 indexer key。
- 主线 `npu_kv_quant_sparse_flash_attention` 期待 packed main KV，OFFLOAD 分支 resident 是分离的 `[3]/[4]`。

### 4.2 page size / split ratio 不一致

OFFLOAD 分支已有 `compute_offload_sparse_c8_layout`、`page_bytes_per_token`、`OffloadMLAAttentionSpec` 等辅助逻辑，但它建立在旧的 offload 双 group + 六元组设定上。

主线 #11228 后：

- `cache_sparse_c8=true` 要用 packed main KV。
- `sparse_head_dim` 使用 `qk_rope_head_dim=0` 表示 packed。
- `sfa_dcp_replicated_indexer_size` 会影响 scale bytes。

需要重新推导 OFFLOAD 两个 group 的 page size：

- main group：packed main KV page。
- indexer group：indexer key + indexer scale page。
- non-C8 group：保持当前分离 main/indexer 逻辑。

### 4.3 resident buffer 与 attention op 不一致

当前 `_get_topk_buffer` 硬编码：

- `topk_buffer_k = kv_cache[3]`
- `topk_buffer_v = kv_cache[4]`
- reshape 为 `[*, block_size, 1, 512]` 与 `[*, block_size, 1, 64]`
- CPU resident attention 走普通 `npu_sparse_flash_attention`

关键位置：

- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/sfa_v1.py:1924`
- `ader47/feat/sfa-offload-layerwise-reuse:vllm_ascend/attention/sfa_v1.py:1995`

如果采用主线 packed C8，resident 也应保存 packed main KV，并走 `npu_kv_quant_sparse_flash_attention` 或等价的 packed sparse attention 路径。

### 4.4 PD MembPull 下标完全不匹配

P 侧 no-offload C8 主线 layout 是 `[0]=packed_main, [1]=indexer_k, [2]=indexer_scale`。

当前 PD read thread 假设：

- `p_base_addrs[2]` 是 indexer key。
- `p_base_addrs[3]` 是 scale。

#11228 后正确读取应按 layout helper：

- packed C8 P producer：`p_base_addrs[1]` 是 indexer key，`p_base_addrs[2]` 是 scale。
- non-C8 P producer：`p_base_addrs[2]` 是 indexer key，无 scale。

### 4.5 per-layer indexer 与全局 C8 假设冲突

主线允许：

- 某些层有 indexer。
- 某些层 skip topk / 复用 topk，没有 indexer，但仍需要 packed main KV。

当前 OFFLOAD worker 对 mixed 五/六元组做了禁止，且 `use_sparse_c8_indexer` 偏全局化。适配后要重新定义：

- 是否允许 C8 packed main 全层开启，但 indexer tensor 只存在于有 indexer 的层。
- OFFLOAD worker 注册层时不能仅靠 tuple length 判断 C8 / non-C8。
- PD indexer group 不能假设每个 main layer 都有对应 indexer tensor。

### 4.6 C++ sparse attention binding 冲突

`git merge-tree upstream/main ader47/feat/sfa-offload-layerwise-reuse` 显示以下 C++/binding 文件有冲突或重叠：

- `csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp`
- `csrc/torch_binding_meta.cpp`
- `csrc/torch_binding.cpp`
- `csrc/attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h`

这些文件影响离散 TopK、quant sparse attention、metadata shape，不能只做 Python 层适配。

## 5. 推荐目标设计

### 5.1 推荐 layout：OFFLOAD 跟随 packed C8

建议把 OFFLOAD C8 tuple 改成以下语义：

```text
non-C8 offload, 保持当前:
  [0] = main_k / k_nope
  [1] = main_v / k_rope
  [2] = indexer_k, if layer has indexer
  [3] = resident_k
  [4] = resident_v

C8 packed offload, 推荐:
  [0] = packed main KV
  [1] = indexer_k, if layer has indexer
  [2] = indexer_scale, if layer has indexer
  [3] = resident packed main KV
```

如果需要支持 no-indexer / skip_topk 层：

```text
C8 packed offload, no indexer layer:
  [0] = packed main KV
  [1] = resident packed main KV
```

也可以为了统一实现保留 placeholder：

```text
C8 packed offload, no indexer layer, with placeholders:
  [0] = packed main KV
  [1] = None / empty indexer_k
  [2] = None / empty indexer_scale
  [3] = resident packed main KV
```

推荐使用显式 layout object / dataclass，不推荐靠 tuple length 推断。

### 5.2 备选短期方案，不推荐长期使用

短期为了快速跑通，也可以在 `use_offload=true` 时强制：

```text
use_sparse_c8_sfa = false
use_sparse_c8_indexer = true
```

即只保留 indexer LIC8，main MLA 继续旧的 `[0]/[1]` 分离布局，保留 `[5]=indexer_scale`。

优点：

- 改动小。
- 更接近当前 OFFLOAD 分支。

缺点：

- 与 #11228 的 SFA C8/c8 统一目标相反。
- 后续主线继续围绕 packed C8 演进时，OFFLOAD 会再次偏离。
- `enable_sparse_c8` 的用户语义变成 OFFLOAD 与 no-OFFLOAD 不一致。

本文后续步骤按推荐方案展开。

## 6. 文件级适配步骤

### 6.1 第一步：新增/重写 OFFLOAD layout helper

文件：

- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/offload_kv_cache_layout.py`

工作：

1. 保留 non-C8 layout 常量。
2. 新增 C8 packed layout 常量。
3. 提供 helper 函数，所有调用方都走 helper。

建议接口：

```python
class SFAOffloadLayout(Enum):
    NON_C8 = "non_c8"
    C8_PACKED_WITH_INDEXER = "c8_packed_with_indexer"
    C8_PACKED_NO_INDEXER = "c8_packed_no_indexer"

def resolve_sfa_offload_layout(kv_cache: tuple, *, use_sparse_c8_sfa: bool, has_indexer: bool) -> SFAOffloadLayout:
    ...

def get_main_cache(kv_cache, layout):
    ...

def get_indexer_cache(kv_cache, layout):
    ...

def get_indexer_scale_cache(kv_cache, layout):
    ...

def get_resident_cache(kv_cache, layout):
    ...
```

验收：

- `device_op.py`、`sfa_v1.py`、`sfa_kv_offload_worker.py`、PD read thread 不再裸写 C8 tuple 下标。
- 新增单测覆盖三种 layout。

### 6.2 第二步：适配 `kv_cache_interface.py`

文件：

- `vllm_ascend/core/kv_cache_interface.py`

工作：

1. 将 OFFLOAD 分支新增的 `OffloadMLAAttentionSpec` 与 `OffloadMLAAttentionManager` 注册迁移到主线结构。
2. 不覆盖主线 `AscendMLAAttentionSpec.cache_sparse_c8` packed 语义。
3. 为 OFFLOAD main group 新增 packed C8 page 计算：
   - 输入：`kv_lora_rank`、`qk_rope_head_dim`、`tile_size`、`c8_k_cache_dtype`。
   - 输出：`packed_kv_head_dim` 与 `page_bytes_per_token`。
4. 为 OFFLOAD indexer group 新增 indexer key + scale page 计算：
   - key dtype：A3 int8 / A5 fp8。
   - scale dtype：A3 fp16 / A5 fp32。
   - DCP replicated indexer 时考虑 `sfa_dcp_replicated_indexer_size`。
5. 保留 non-C8 OFFLOAD 原计算逻辑。

注意：

- 不要让 OFFLOAD C8 spec 使用旧的 `qk_rope_head_dim != 0` C8 语义。
- C8 main group page 与 indexer group page 是否需要 8:1 unify，要重新按 packed C8 实际 bytes 验算，不沿用旧六元组文档里的固定结论。

验收：

- `tests/ut/core/test_kv_cache_interface.py` 增加：
  - non-C8 offload page size。
  - C8 packed main page size。
  - C8 indexer key/scale page size。
  - DCP replicated indexer scale bytes。

### 6.3 第三步：适配 `model_runner_v1.py`

文件：

- `vllm_ascend/worker/model_runner_v1.py`

工作点一：`get_kv_cache_spec`

1. 合并 OFFLOAD 分支中 `use_sparse and use_offload` 的双 group spec 逻辑。
2. 使用主线的 per-layer 判断：
   - `impl.use_sparse_c8_sfa`
   - `impl.use_sparse_c8_indexer`
   - `impl.has_indexer`
3. main group：
   - non-C8：沿用 `OffloadMLAAttentionSpec` 分离 main。
   - C8：使用 packed main spec。
4. indexer group：
   - 只为 `has_indexer` 的层创建。
   - C8：包含 indexer key + scale。
   - non-C8：只包含 indexer key。

工作点二：`_allocate_kv_cache_tensors`

1. non-C8 OFFLOAD 保持当前分离分配。
2. C8 OFFLOAD main：
   - 分配 packed main HBM tail。
   - 分配 resident packed buffer。
   - 不再分配 resident_k/resident_v 两个 buffer。
3. C8 OFFLOAD indexer：
   - 分配 indexer_k。
   - 分配 indexer_scale。
   - 注意 scale dtype alignment。
4. 删除/隔离旧 OFFLOAD C8 `[5]` scale 分配。

工作点三：`_reshape_kv_cache_tensors`

1. non-C8 tuple 维持 `[0..4]`。
2. C8 tuple 按推荐 layout reshape：
   - `[0] packed main`
   - `[1] indexer_k`
   - `[2] indexer_scale`
   - `[3] resident packed main`
3. no-indexer 层要么返回 `[0], [3]`，要么返回 placeholder layout，必须由 helper 统一解析。

工作点四：`may_reinitialize_input_batch`

1. 保留 OFFLOAD 双 group 的 `kernel_block_sizes` 逻辑。
2. C8 indexer block size 需重新确认：
   - main block 仍为 128。
   - indexer block 是否为 `128 * page_block_multiplier`，必须与 `indexer_block_table` 和 `indexer_slot_mapping` 一致。
3. C8 packed main 不应影响 main group logical block size。

验收：

- 打印/断言每个 layer 的 layout、page size、tuple shape。
- dummy profile 不出现 `assert qk_rope_head_dim == 0` 失败。
- `kv_cache_config.kv_cache_groups` 顺序稳定，PD group index 不靠魔法数字。

### 6.4 第四步：适配 `attention/utils.py` 和 common metadata

文件：

- `vllm_ascend/attention/utils.py`

工作：

1. 将 OFFLOAD 分支的字段合入主线 `AscendCommonAttentionMetadata`：
   - `indexer_block_table_tensor`
   - `indexer_slot_mapping`
   - `num_offloaded_blocks`
   - `req_ids_tensor`
   - `token_to_req`
2. 确认 ubatch slicing 时这些字段同步切片。
3. 保证 spec decode / MTP 下 `token_to_req` 正确。
4. 保留 `maybe_prepare_lru_resident_and_load_graph`，但参数从 `resident_k/resident_v` 模型升级到 layout-aware resident。

验收：

- decode-only、mixed prefill/decode、spec decode metadata 都能 build。
- `num_offloaded_blocks` 与 `block_table` / `indexer_block_table` 行数一致。

### 6.5 第五步：适配 `attention/sfa_v1.py`

文件：

- `vllm_ascend/attention/sfa_v1.py`

工作点一：metadata builder

1. 合入 OFFLOAD metadata 字段。
2. `indexer_block_table` 用于 indexer TopK。
3. `block_table` 用于 main packed sparse attention。
4. `indexer_slot_mapping` 用于 indexer key/scale scatter。
5. `slot_mapping` 用于 main packed cache scatter。

工作点二：cache write

1. C8 packed main 写入完全沿用主线：
   - native path：`custom_kv_rmsnorm_rope`
   - decode consumer：`execute_sfa_mla_prolog_v3`
2. indexer key/scale scatter：
   - no-offload：主线 `[1]/[2]`。
   - offload C8：通过 layout helper 解析 `[1]/[2]`。
   - offload non-C8：旧 `[2]`。
3. scale scatter 必须使用 `indexer_slot_mapping`，不能用 main `slot_mapping`。

工作点三：TopK 与 CPU/NPU 分流

1. `indexer_select_post_process` 的 block table：
   - OFFLOAD：`indexer_block_table_tensor`
   - no-OFFLOAD：`block_table`
2. `_get_topk_buffer` 改成 layout-aware：
   - non-C8：返回 `resident_k` / `resident_v`，走普通 sparse attention。
   - C8 packed：返回 `resident_packed_kv`，走 quant sparse attention。
3. NPU resident 与 CPU resident 两段 attention 需要使用同一 packed quant attention kernel，输出 LSE 后合并。

验收：

- 不再硬编码 `512`、`64` 做 C8 resident reshape。
- `use_sparse_c8_sfa` 与 `use_offload` 同时为真时，不触发主线 `assert len(kv_cache) == (3 if self.use_sparse_c8_sfa else 4)`。
- skip_topk 层能使用上一层 topk，不因没有 indexer tensor 崩溃。

### 6.6 第六步：适配 `device/device_op.py`

文件：

- `vllm_ascend/device/device_op.py`

工作：

1. `indexer_select_post_process` 改成 layout-aware，不再只看 `packed_kv_cache` 决定下标。
2. A3 与 A5 两个 adaptor 都要同步。
3. `execute_sparse_flash_attention_process` 或新增 helper：
   - non-C8：普通 `npu_sparse_flash_attention`。
   - C8 packed：`npu_kv_quant_sparse_flash_attention`。
4. Prolog V3 path 在 OFFLOAD C8 下写 `[0]=packed main`，不能写旧 `[0]/[1]`。

验收：

- A3/A5 adaptor 都通过静态 import。
- C8 packed indexer 的 `key_dequant_scale` 指向正确 scale tensor。
- block table 参数可由调用方传入，支持 `indexer_block_table` 与 main `block_table` 分离。

### 6.7 第七步：适配 `sfa_kv_offload_worker.py`

文件：

- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py`

工作点一：register kv cache

1. 用 layout helper 注册 layer，不再靠 tuple length 推断 C8。
2. C8 packed：
   - main CPU pool 存 packed main blocks。
   - resident buffer 是单 packed tensor。
   - indexer key/scale 不进入 main CPU offload pool，仍按 indexer 常驻/PD 拉取逻辑处理。
3. no-indexer 层不注册 indexer tensor，但 main packed resident 仍注册。

工作点二：CPU pool / save

1. non-C8：继续保存 k/v 两路。
2. C8 packed：保存 packed main 一路。
3. `save_kv_layer`、`prepare_lru_resident_and_load`、`offload.sparse_copy` 的 size/address 计算按 layout 分支。

工作点三：LRU resident H2D

1. `cpu_sparse_attn.compute_lru_resident_addrs` 目前按 K/V 两路计算，需要支持一路 packed tensor。
2. atomic transfer 数：
   - non-C8：`2 * n_main`
   - C8 packed：`1 * n_main`
3. workspace 中 `gvas/addr/size` buffer 维度要支持一路/两路。

验收：

- `tests/ut/distributed/kv_transfer/sfa_kv_offload/test_sfa_kv_offload_register.py` 覆盖 C8 packed tuple。
- `tests/ut/ops/test_cpu_lru_resident_compact.py` 增加 packed resident case。

### 6.8 第八步：适配 `sfa_kv_offload/cpu_sparse_attn.cpp`

文件：

- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/cpu_sparse_attn.cpp`

工作：

1. `compute_lru_resident_addrs` 支持 `num_legs=1/2`。
2. `token_size_bytes_k/v` 改成 layout-aware：
   - non-C8：K/V 两套 bytes。
   - C8 packed：packed bytes。
3. 输出地址数组的 layout 与 worker 侧一致。
4. 保持原 non-C8 单测不变。

验收：

- packed case 不再访问 `addr_v_base`。
- OpenMP row-level compact 逻辑不变，只改变 miss load 地址生成。

### 6.9 第九步：适配 PD CPU Offload

文件：

- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/read_thread.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/protocol.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/send_thread.py`

工作点一：D 侧注册

1. 不再写死 `main_tuple[2]` 是 indexer、`main_tuple[5]` 是 scale。
2. 使用 layout helper 获取：
   - D indexer key tensor。
   - D indexer scale tensor。
   - D packed main tensor。
   - D resident packed tensor。
3. group index 不再写死 `_INDEXER_GROUP_IDX=0`，至少增加断言验证 group name。

工作点二：P 侧 metadata

1. P no-offload C8 layout 是主线 packed tuple：
   - `[0] packed main`
   - `[1] indexer_k`
   - `[2] indexer_scale`
2. `MembPullReadThread` 需要根据 P/D layout 映射 base addrs：
   - C8：indexer key `p_base_addrs[1]`，scale `p_base_addrs[2]`。
   - non-C8：indexer key `p_base_addrs[2]`，无 scale。
3. main pull：
   - non-C8：K/V 两路。
   - C8：packed main 一路。

工作点三：block_len / block_size_scale

1. C8 scale 的 block_len 以 scale tensor shape 计算，不再假设 `p_block_len[3]`。
2. C8 packed main 的 block_len 以 packed main tensor shape 计算。
3. 对 P/D block size 不一致时继续保留 refuse-to-transfer 防护，避免 stale scale。

验收：

- `tests/ut/kv_offload/test_sfa_pd_cpu_offload_single_rank.py` 增加：
  - C8 packed P -> C8 packed D。
  - no-indexer layer。
  - P/D LIC8 配置不一致时明确报错。

### 6.10 第十步：适配 C++ sparse attention 与 binding

文件：

- `csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp`
- `csrc/torch_binding.cpp`
- `csrc/torch_binding_meta.cpp`
- `csrc/attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h`

工作：

1. 先以主线 #11228 为底，保留 quant sparse attention / packed C8 所需参数。
2. 再叠加 OFFLOAD 的 discrete sparse indices、CPU resident sparse attention 扩展。
3. 对 meta function 增加 packed resident shape case。
4. 不要回退主线 quant sparse attention 参数。

验收：

- C++ 编译通过。
- Python import `torch.ops._C_ascend` 成功。
- sparse attention op meta 测试覆盖 non-C8 / C8 packed。

### 6.11 第十一步：配置、示例、文档

文件：

- `examples/disaggregated_prefill_v1/run_sfa_pd_decode.sh`
- `examples/disaggregated_prefill_v1/run_sfa_pd_prefill.sh`
- `docs/source/developer_guide/Design_Documents/sparse_c8_kv_offload_guide.md`
- `docs/source/user_guide/configuration/additional_config.md`

工作：

1. 更新启动说明：
   - P：`use_offload=false`，但可 `enable_sparse_c8=true`。
   - D：`use_offload=true`，`enable_sparse_c8=true`。
2. 明确 C8 OFFLOAD 使用 packed layout，不再是六元组 `[5]=scale`。
3. 删除或改写旧文档中 “C8 offload 六元组” 的描述。
4. 增加日志观测项：
   - `use_sparse_c8_sfa`
   - tuple layout
   - packed main page bytes
   - indexer page bytes
   - resident packed buffer shape

## 7. 建议拆分 PR / commit 顺序

### PR 1：rebase 与 layout helper

内容：

- rebase 到 `upstream/main`。
- 解决纯文本/低风险冲突。
- 新增 `offload_kv_cache_layout.py` packed C8 helper。
- 单测覆盖 layout 解析。

验收：

- 不改变 runtime 行为。
- 所有裸下标调用方可以逐步迁移。

### PR 2：KV cache spec 与 model runner

内容：

- `kv_cache_interface.py`
- `model_runner_v1.py`
- C8 packed offload spec。
- C8 packed offload allocation / reshape。

验收：

- dummy run 可以创建 KV cache。
- 日志输出 C8 packed tuple shape。
- `tests/ut/core/test_kv_cache_interface.py` 通过。

### PR 3：SFA forward 与 device op

内容：

- `attention/utils.py`
- `attention/sfa_v1.py`
- `device/device_op.py`

验收：

- no-offload C8 UT 不回退。
- offload C8 单机 decode 小 case 可跑。
- indexer block table / slot mapping 正确。

### PR 4：SFAKVOffloadWorker 与 CPU sparse helper

内容：

- `sfa_kv_offload_worker.py`
- `sfa_kv_offload/cpu_sparse_attn.cpp`
- local CPU pool packed main support。

验收：

- 单机 offload non-C8 不回退。
- 单机 offload C8 packed 能正确 load resident。

### PR 5：PD MembPull

内容：

- `sfa_pd_cpu_offload/*`
- P/D metadata base addr 映射。
- C8 packed main 与 indexer scale transfer。

验收：

- 单 rank PD UT。
- 双进程/双机 smoke。
- P/D 配置不一致时报错清晰。

### PR 6：C++ binding、文档与 e2e

内容：

- C++ sparse attention binding 冲突收敛。
- examples / docs。
- E2E 配置。

验收：

- 编译通过。
- 小模型或 mock e2e 通过。
- GLM5.2 目标配置启动通过。

## 8. 高风险点与困难点

| 风险点 | 影响 | 建议处理 |
| --- | --- | --- |
| packed resident attention kernel 不支持当前离散 TopK / resident block_table 组合 | C8 OFFLOAD 核心路径不可用 | 优先验证 `npu_kv_quant_sparse_flash_attention` 是否支持 resident packed buffer + discrete indices；不支持则需要新增/扩展 C++ op |
| tuple layout 混用 | silent corruption，尤其 indexer key/scale 读错 | 所有 C8 下标走 helper；启动时打印每层 layout；注册时强断言 tensor dtype/shape |
| P/D base addr 映射错误 | PD 拉错 tensor，结果异常或 stale scale | read_thread 用 layout 显式映射，不用 `p_base_addrs[3]` 这类魔法下标 |
| scale dtype/alignment | A3 fp16 / A5 fp32 对齐失败或数值错 | 分配 scale raw buffer 时按 dtype 对齐；单测 data_ptr alignment |
| no-indexer / skip_topk 层 | 某些 GLM5.2 层没有 indexer，但仍需 packed main KV | model_runner 按 `impl.has_indexer` 分层处理；worker 不靠 tuple length 判断 |
| DCP replicated indexer | scale bytes 与 block_table 扩展不一致 | spec page size、MembPull block_len 均纳入 `sfa_dcp_replicated_indexer_size` |
| LRU resident H2D 从两路变一路 | worker/C++/transfer buffer 都受影响 | C++ helper 支持 `num_legs`；先做单机 packed resident UT |
| graph / Prolog V3 capture | decode graph 下 cache_index、stream sync、shape 固化 | 先 eager/small case，再开 graph；graph path 单独 smoke |
| non-C8 回归 | 已有 OFFLOAD 路径被改坏 | non-C8 layout 保持旧常量，所有新逻辑分支化 |
| C++ op 合并冲突 | 编译阻断 | 先以 upstream/main 为基底，再 cherry-pick OFFLOAD discrete/resident 变更 |

## 9. 验证计划

### 9.1 静态检查

```bash
git diff --check
python -m compileall vllm_ascend
```

### 9.2 单测

建议最小集合：

```bash
pytest -sv tests/ut/core/test_kv_cache_interface.py
pytest -sv tests/ut/attention/a2/test_sfa_v1.py
pytest -sv tests/ut/worker/a2/test_model_runner_v1.py
pytest -sv tests/ut/distributed/kv_transfer/sfa_kv_offload/
pytest -sv tests/ut/kv_offload/test_sfa_pd_cpu_offload_single_rank.py
pytest -sv tests/ut/ops/test_cpu_lru_resident_compact.py
```

### 9.3 smoke case

矩阵：

| 场景 | `enable_sparse_c8` | `use_offload` | 期望 |
| --- | --- | --- | --- |
| no-offload baseline | false | false | 主线 SFA 正常 |
| no-offload C8 | true | false | #11228 UT/e2e 不回退 |
| single-node OFFLOAD non-C8 | false | true | 当前 OFFLOAD 能力不回退 |
| single-node OFFLOAD C8 packed | true | true | packed resident H2D + attention 正常 |
| PD OFFLOAD non-C8 | false | D true / P false | 当前 PD 能力不回退 |
| PD OFFLOAD C8 packed | true | D true / P false | P packed -> D packed transfer 正常 |

### 9.4 日志观测

启动时建议输出：

- 每层 `layer_name`、`has_indexer`、`use_sparse_c8_sfa`、`use_sparse_c8_indexer`。
- 每层 tuple layout 名称。
- main/indexer group page bytes。
- main/indexer block size。
- resident buffer shape。
- PD P/D base addr mapping。

运行时建议观察：

- `num_offloaded_blocks`
- CPU/NPU TopK split 数量。
- LRU resident hit/miss。
- H2D transfer block 数。
- indexer scale transfer block 数。

## 10. 完成标准

适配完成应满足：

1. `enable_sparse_c8=true` 时，OFFLOAD 与 no-OFFLOAD 都使用同一套主线 SFA C8 语义。
2. C8 OFFLOAD 不再使用旧 `[5]=indexer_scale` 六元组作为主路径。
3. non-C8 OFFLOAD 路径不回退。
4. P/D C8 配置一致时能完成 MembPull；不一致时明确报错。
5. GLM5.2 目标场景可启动，并能观察到 main packed KV 从 CPU 按 TopK 拉回 resident。
6. 文档、示例、UT 覆盖新的 packed C8 OFFLOAD layout。

## 11. 最小实施清单

按文件列出必须改动：

```text
vllm_ascend/core/kv_cache_interface.py
  - 合入 OffloadMLAAttentionSpec / manager 注册
  - 新增 packed C8 offload page size helper
  - 保留主线 AscendMLAAttentionSpec packed 语义

vllm_ascend/worker/model_runner_v1.py
  - get_kv_cache_spec 支持 use_offload + use_sparse_c8_sfa
  - C8 packed main/indexer group spec
  - C8 packed allocation / reshape
  - kernel_block_sizes 与 input_batch reinit

vllm_ascend/attention/utils.py
  - 合入 offload metadata 字段
  - ubatch slicing
  - layout-aware maybe_prepare_lru_resident_and_load_graph

vllm_ascend/attention/sfa_v1.py
  - indexer_block_table / indexer_slot_mapping
  - C8 packed cache write
  - layout-aware indexer scatter
  - layout-aware _get_topk_buffer
  - C8 packed resident attention

vllm_ascend/device/device_op.py
  - layout-aware indexer key/scale selection
  - C8 packed sparse attention op
  - A3/A5 同步

vllm_ascend/distributed/kv_transfer/sfa_kv_offload/offload_kv_cache_layout.py
  - 新 layout helper

vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py
  - C8 packed CPU pool
  - C8 packed resident buffer
  - save/load H2D 单路 packed transfer

vllm_ascend/distributed/kv_transfer/sfa_kv_offload/cpu_sparse_attn.cpp
  - compute_lru_resident_addrs 支持一路 packed tensor

vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/*
  - P/D layout-aware base addr mapping
  - C8 packed main/indexer/scale transfer

csrc/attention/sparse_flash_attention/*
csrc/torch_binding.cpp
csrc/torch_binding_meta.cpp
  - 合并 quant sparse attention 与 offload discrete/resident 扩展

tests/ut/core/test_kv_cache_interface.py
tests/ut/attention/a2/test_sfa_v1.py
tests/ut/worker/a2/test_model_runner_v1.py
tests/ut/distributed/kv_transfer/sfa_kv_offload/*
tests/ut/kv_offload/test_sfa_pd_cpu_offload_single_rank.py
tests/ut/ops/test_cpu_lru_resident_compact.py
  - 覆盖 packed C8 offload
```

## 12. 结论

当前 OFFLOAD 分支的 LIC8 适配已经解决了 “indexer key/scale 如何在 OFFLOAD 六元组里保存” 的问题，但 #11228 将 SFA C8 的主线语义推进到了 “main KV packed + indexer C8” 的统一模型。

因此真正兼容 #11228 的适配，不应继续维护旧六元组，而应把 OFFLOAD 迁移为 packed C8 的扩展：

```text
主线负责定义 packed C8 cache 语义；
OFFLOAD 负责定义 packed main 如何 CPU pool / resident / PD transfer。
```

这是改动较大但长期风险最低的路线。

## 13. 当前分支已落地的适配状态

基于 `feat/sfa-offload-c8-unified` 当前代码，已经完成以下最小可运行适配：

1. 新增 `sfa_kv_offload/offload_kv_cache_layout.py`，统一 OFFLOAD tuple 常量：
   - non-C8：`[0]=main_k, [1]=main_v, [2]=indexer_k, [3]=resident_k, [4]=resident_v`
   - C8：`[0]=packed_main_kv, [1]=indexer_k, [2]=indexer_scale, [3]=resident_packed_main_kv`
2. `sfa_v1.py` / `device_op.py` 已按 `use_sparse_c8_sfa` 和 `use_offload` 切换 indexer slot，C8 OFFLOAD 不再从 `[5]` 读取 scale。
3. `sfa_kv_offload_worker.py` 已支持 C8 packed 单路 main KV：
   - NPU main K/V 指向同一个 packed tensor；
   - resident K/V 指向同一个 packed resident tensor；
   - CPU pool 以 `token_size_bytes_v=0` 表示 packed 单路拷贝。
4. `cpu_sparse_attn.cpp` 已支持 `token_size_bytes_v == 0` 的单路 packed resident 地址生成。
5. `sfa_pd_cpu_offload/worker.py` 已识别 D 侧四元组 C8 layout：
   - D 侧 main HBM/CPU 目的端按 packed 单 tensor 注册；
   - indexer / scale 改为 `[1]` / `[2]`；
   - non-C8 仍保持五元组路径。
6. `sfa_pd_cpu_offload/read_thread.py` 已识别 P 侧 packed C8 三元组：
   - P 侧 main 只读 `[0]` 一路 packed tensor；
   - P 侧 indexer / scale 改为 `[1]` / `[2]`；
   - partial HBM block 与 full CPU block 都按 packed/non-packed 分支生成 memfabric descriptor。
7. `model_runner_v1.py` 已恢复 OFFLOAD 分支的 `is_profiling` 透传，避免 `initialize_kv_cache(..., is_profiling=...)` 调用初始化 attention backend 时直接报错。

已执行的本地检查：

```bash
C:\Users\程\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile \
  vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py \
  vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/read_thread.py \
  vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/attention/sfa_v1.py \
  vllm_ascend/device/device_op.py \
  vllm_ascend/distributed/kv_transfer/sfa_kv_offload/sfa_kv_offload_worker.py
```

检查结果：通过。

仍需在 Ascend 环境重点验证：

1. C8 packed main cache 当前是独立 NPU tensor，由 SFA prolog / scatter 写入；需要在真实 GLM5.2 C8 OFFLOAD 场景确认它与 block table、PD 注册、CPU offload 保存路径完全一致。
2. `torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention(..., return_softmax_lse=True)` 的返回值顺序和可用 namespace 需要在带算子包环境验证。
3. PD C8 MembPull 需要至少跑一次 P no-offload C8 三元组 -> D offload C8 四元组，确认 full CPU block、partial HBM block、indexer scale 三类 descriptor 都正确。
4. 当前本地网络 fetch 失败，尚未从远端重新拉取最新提交；后续需在网络可用后重新 `git fetch upstream main` 和 `git fetch ader47 feat/sfa-offload-layerwise-reuse` 复核。

## 14. SFA C8 下 OFFLOAD 内存切分方案

### 14.1 当前结论

当前代码应收敛到 **packed CKV + indexer 独立物理分配 + 4:1 page accounting**：

```text
main group:
  block_size = 128 tokens
  physical bytes/token = 656
  page_bytes = 128 * 656 = 83,968

indexer group:
  kernel block_size = 512 tokens
  accounting bytes/token = 164
  page_bytes = 512 * 164 = 83,968

page_block_multiplier = 4
kernel_block_sizes = [[512], [128]]
```

这里的 `kernel_block_sizes = [[512], [128]]` 遵循当前代码里的 KV cache group 顺序：indexer group 在前，main group 在后。概念上仍然是 main 128 tokens、indexer 512 tokens 的 4:1 token 映射。

### 14.2 三层内存语义

SFA C8 OFFLOAD 需要分清三层：

| 层级 | 负责内容 | C8 OFFLOAD 结论 |
| --- | --- | --- |
| KVCacheManager / block table | 两个 KV cache group 的 page size unify | main page 与 indexer page 都是 83,968 bytes；token 映射为 4:1 |
| Main 物理层 | attention 真正消费的 main KV | `kv_cache[0]` 是 packed CKV，656 bytes/token，不加 main pad |
| Indexer 物理层 | TopK 粗筛用的 indexer cache | `kv_cache[1]` 是 indexer_k，`kv_cache[2]` 是 indexer_scale；34 bytes/token pad 只存在于 page accounting |

关键点：**main 侧不再为了 page unify 分配 bf16 raw_k/raw_v/pad，也不再额外分配一份 packed CKV**。main paged pool 本身就是 packed CKV。

### 14.3 CKV 定义

CKV 是 SFA C8 下 packed main KV，存在 `kv_cache[0]`。A3 GLM5.2 的 packed head dim 来自 `get_sfa_qsfa_packed_head_dim`：

```text
packed_head_dim = kv_lora_rank
                + qk_rope_head_dim * sizeof(bf16)
                + (kv_lora_rank / tile_size) * sizeof(fp32)

packed_head_dim = 512 + 64 * 2 + (512 / 128) * 4
                = 656
```

单 token 内部 layout：

```text
kv_cache[0][..., 0:656]

| 0..511 | 512..639 | 640..655 |
| k_nope | k_rope   | scale    |
| int8   | bf16 bytes | fp32 bytes |
```

写入时需要按 byte-packed 语义拼接，不是混 dtype 直接 `torch.cat`。attention 读取时仍按 packed tensor 传给 C8 SFA 算子：

```text
npu_kv_quant_sparse_flash_attention(key=kv_cache[0], value=kv_cache[0])
```

### 14.4 Page 计算

main page：

```text
main_bytes_per_token = 656
main_block_size = 128
main_page_bytes = 128 * 656 = 83,968
```

indexer 真实 payload：

```text
indexer_k bytes/token = 128 * sizeof(int8) = 128
indexer_scale bytes/token = 1 * sizeof(fp16) = 2
indexer_payload_bytes/token = 130
```

indexer page accounting：

```text
page_block_multiplier = 4
indexer_block_size = 128 * 4 = 512
indexer_accounting_bytes/token = 83,968 / 512 = 164
indexer_accounting_pad/token = 164 - 130 = 34
```

最终 page 对齐：

| Group | block tokens | physical tensor | physical bytes/token | accounting pad/token | accounting bytes/token | page bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| main | 128 | packed CKV | 656 | 0 | 656 | 83,968 |
| indexer | 512 | indexer_k + indexer_scale | 130 | 34 | 164 | 83,968 |

### 14.5 四元组 layout

每层 C8 OFFLOAD `kv_cache` 固定为四元组：

```text
kv_cache[0] = packed_main_ckv
kv_cache[1] = indexer_k
kv_cache[2] = indexer_scale
kv_cache[3] = resident_packed_ckv
```

推荐 shape：

| 下标 | 名称 | shape | 说明 |
| --- | --- | --- | --- |
| `[0]` | `packed_main_ckv` | `[main_num_blocks, 128, 1, 656] int8` | HBM main paged pool；CPU 前缀保存同 layout |
| `[1]` | `indexer_k` | `[indexer_num_blocks, 512, 1, 128] int8` | indexer TopK key，常驻 HBM |
| `[2]` | `indexer_scale` | `[indexer_num_blocks, 512, 1, 1] fp16` | indexer key 的量化 scale |
| `[3]` | `resident_packed_ckv` | `[topk_rows, resident_capacity, 1, 656] int8` | CPU TopK 命中后 H2D 拉回的 LRU resident buffer |

表里的 `main_num_blocks` 和 `indexer_num_blocks` 不应混为同一个概念。由于 block token 数不同，同样 token 容量下 indexer block 数约为 main block 数的 1/4；当前实现为了 page unify 和 group 管理，会按各自 group 的 block size 解释 block table / slot mapping。

### 14.6 ModelRunner 分配逻辑

当前代码需要按以下方式分配 C8 OFFLOAD：

```text
packed_main_raw:
  num_blocks * 128 * 1 * 656 * sizeof(int8)

indexer_k_raw:
  num_blocks * 512 * 1 * 128 * sizeof(int8)

indexer_scale_raw:
  num_blocks * 512 * 1 * 1 * sizeof(fp16)

resident_packed_ckv:
  topk_rows * resident_capacity * 1 * 656 * sizeof(int8)
```

也就是说，C8 OFFLOAD 不再走历史 LIC8 的三段 main raw split：

```text
不再使用：
  raw_k_tensor:     [num_blocks, 128, 1, 512] bf16
  raw_v_tensor:     [num_blocks, 128, 1, 64]  bf16
  raw_scale/pad:    [num_blocks, 128, 1, 8]   bf16
  packed_main_ckv:  额外 torch.zeros(...)

改为：
  raw main pool 直接 view 成 packed_main_ckv
```

对应代码落点：

| 文件 | 需要保证的行为 |
| --- | --- |
| `vllm_ascend/core/kv_cache_interface.py` | `compute_offload_sparse_c8_layout` 返回 4:1：`packed_head_dim=656`, `main_bytes_per_token=656`, `indexer_bytes_per_token=164`, `indexer_pad_dim=34` |
| `vllm_ascend/worker/model_runner_v1.py` | C8 OFFLOAD main raw tensor 直接分配 packed CKV；indexer_k/indexer_scale 单独分配；reshape 时不再额外创建 packed main zeros |
| `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/offload_kv_cache_layout.py` | C8 四元组下标固定为 `[0] main_ckv, [1] indexer_k, [2] indexer_scale, [3] resident` |

### 14.7 resident 与 CPU pool

`resident_packed_ckv` 不是另一份持久 main KV，而是 decode 阶段 CPU TopK 命中后的 LRU 驻留缓冲区：

```text
CPU pool:
  [cpu_blocks, 128, 1, 656] int8

H2D resident target:
  [topk_rows, resident_capacity, 1, 656] int8

H2D descriptor:
  token_size_bytes_k = 656
  token_size_bytes_v = 0
```

`token_size_bytes_v = 0` 表示 C8 main KV 是 packed 单路，不存在单独 V tensor。

### 14.8 与历史 LIC8 8:1 的区别

历史 LIC8 方案：

```text
main accounting = (512 + 64 + 8) * sizeof(bf16) = 1168 bytes/token
indexer accounting = 128 + 16 + 2 = 146 bytes/token
ratio = 1168 / 146 = 8
indexer block = 1024 tokens
```

这套方案现在只作为历史背景，不再作为 C8 OFFLOAD 目标。新方案的核心变化是：

```text
main accounting = packed CKV = 656 bytes/token
indexer accounting = 164 bytes/token
ratio = 656 / 164 = 4
indexer block = 512 tokens
```

因此旧的 `kv_pad_dim_bf16_slots=8`、`indexer_pad_dim=16`、`page_block_multiplier=8` 都不应继续用于 SFA C8 OFFLOAD。

### 14.9 验收数字

```text
packed_head_dim = 656
main_bytes_per_token = 656
indexer_payload_bytes_per_token = 130
indexer_pad_bytes_per_token = 34
indexer_bytes_per_token = 164
page_block_multiplier = 4

128 * 656 = 83,968
512 * 164 = 83,968
```
