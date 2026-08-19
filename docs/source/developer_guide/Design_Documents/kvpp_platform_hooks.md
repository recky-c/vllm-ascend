# KVPP 平台钩子函数说明

## `get_kv_cache_groups_for_worker`

### 入参

| 参数 | 类型 | 含义 | 举例 |
|---|---|---|---|
| `vllm_config` | `VllmConfig` | 全局 vLLM 配置，从中读取 `KVPPConfig.size` 和 MTP 相关模型配置 | `kvpp_size=4`，4 层 Transformer（每层 main+indexer）+ 1 层 MTP |
| `global_groups` | `list[KVCacheGroupSpec]` | 全局视角的 KV cache group 列表（未按 worker 切分前） | 1 个 group，含 9 个层名（4 层 × main+indexer + MTP） |
| `worker_spec` | `dict[str, KVCacheSpec]` | 当前 worker 实际持有的层名 → KV cache spec 映射。`KVCacheSpec` 描述每层 KV cache 的格式（`block_size`、`num_kv_heads`、`head_size`、`dtype` 等），决定如何分配物理 tensor | `{"model.layers.0.self_attn": AttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=bfloat16), ..., "model.layers.4.mtp_attn": <spec>}` |
| `worker_index` | `int` | 当前 worker 的全局索引，`worker_index % kvpp_size` 即该 worker 的 KVPP rank | `0` → KVPP rank 0（own layer 0） |

### 返回

| 字段 | 类型 | 含义 | 举例 |
|---|---|---|---|
| 返回值 | `list[KVCacheGroupSpec]` | 该 worker 实际需要分配的 KV cache group 列表。每个 `KVCacheGroupSpec` 含 `layer_names`（共享同一 block table 的层名列表）、`kv_cache_spec`（组的 cache 格式）、`is_eagle_group`（是否是 EAGLE/MTP draft 层） | rank 0 返回 1 个 group，`layer_names=["model.layers.0.self_attn", "model.layers.0.self_attn.index", "model.layers.1.self_attn", "model.layers.2.self_attn", "model.layers.1.self_attn.index", "model.layers.2.self_attn.index", "model.layers.4.mtp_attn"]`。其中 layers.0.* 是 own 的持久化层，layers.1.*/layers.2.* 是作为 scratch buffer 复用的层名（同 layout group 前 2 个），layers.3.* 不分配（由 rank 3 own），layers.4 是 MTP 副本 |

### 主要工作

1. 调用 `get_kvpp_layer_owners` 算出每层 KV cache 的 owner rank（MTP 层被排除，全量复制）。
2. 调用 `_get_allocation_groups`，按当前 worker 的 KVPP rank 生成**分配视图**：
   - owner 是本 rank 的层 → 保留为持久化 KV cache。
   - owner 是其他 rank 的层（非 MTP）→ 替换成 2 个 scratch buffer（双缓冲）。
   - MTP 层 → 全量复制保留。
3. 返回投影后的 allocation groups，让 vLLM 据此为该 worker 分配物理 tensor。

rank 0 的返回结果（省略 spec 细节）：

| 层名 | 处理方式 |
|---|---|
| `model.layers.0.self_attn` | owner=本 rank，持久化 KV cache |
| `model.layers.0.self_attn.index` | owner=本 rank，持久化 KV cache |
| `model.layers.1.self_attn` | owner=rank 1，作为 scratch buffer 0 复用 |
| `model.layers.2.self_attn` | owner=rank 2，作为 scratch buffer 1 复用 |
| `model.layers.1.self_attn.index` | owner=rank 1，作为 scratch buffer 0 复用 |
| `model.layers.2.self_attn.index` | owner=rank 2，作为 scratch buffer 1 复用 |
| `model.layers.3.self_attn` | owner=rank 3，不分配（不在 allocation_names 里） |
| `model.layers.3.self_attn.index` | owner=rank 3，不分配 |
| `model.layers.4.mtp_attn` | MTP 层，全量复制 |

---

## `finalize_kv_cache_config`

### 入参

| 参数 | 类型 | 含义 | 举例 |
|---|---|---|---|
| `vllm_config` | `VllmConfig` | 全局 vLLM 配置 | `kvpp_size=4`，4 层 Transformer（每层 main+indexer）+ 1 层 MTP |
| `kv_cache_config` | `KVCacheConfig` | vLLM 已经初步规划好的 KV cache 配置，**会被本函数原地修改** | `kv_cache_tensors=[KVCacheTensor(shared_by=["model.layers.1.self_attn", "model.layers.2.self_attn", "model.layers.3.self_attn"], ...), ...]`（vLLM 按 scratch_aliases 预分组） |
| `global_groups` | `list[KVCacheGroupSpec]` | 全局视角的 KV cache group 列表 | 同 `get_kv_cache_groups_for_worker` 的入参 |
| `worker_spec` | `dict[str, KVCacheSpec]` | 当前 worker 实际持有的层名 → KV cache spec | 同 `get_kv_cache_groups_for_worker` 的入参 |
| `worker_index` | `int` | 当前 worker 的全局索引，`worker_index % kvpp_size` 即 KVPP rank | `0` → KVPP rank 0 |

### 返回

| 字段 | 类型 | 含义 | 举例 |
|---|---|---|---|
| 返回值 | `None` | 原地修改 `kv_cache_config`：展开 `kv_cache_tensors[*].shared_by` 的 scratch 层名，重置 `kv_cache_groups` 为 worker 投影视图 | rank 0 调用后，`kv_cache_tensors[0].shared_by` 从 `["model.layers.1.self_attn", "model.layers.2.self_attn", "model.layers.3.self_attn"]` 展开成 `["model.layers.1.self_attn", "model.layers.2.self_attn"]`（只含本 rank 分配的 2 个 scratch buffer，layers.3 不在本 rank）；`kv_cache_groups` 只含 rank 0 持有的逻辑层名 |

### 主要工作

1. 重新算一次 `owners` 和 `_get_allocation_groups`，这次只取 `scratch_aliases`（层名 → 它实际共享的 scratch buffer 名列表）。
2. 遍历 `kv_cache_config.kv_cache_tensors`，把每个 tensor 的 `shared_by` 里属于 scratch 的层名**展开**成对应的双 buffer 名，让 vLLM 知道这块物理内存被哪些逻辑层共享。
3. 把 `kv_cache_config.kv_cache_groups` 重置为按 worker 投影后的 group 列表（只含本 worker 实际持有的层）。

rank 0 的 `scratch_aliases` 实例（`kvpp_size=4`，main 的 layout group = `[layers.1, layers.2, layers.3]`，取前 2 个作 scratch buffer）：

```python
{
    "model.layers.1.self_attn": ["model.layers.1.self_attn", "model.layers.2.self_attn"],  # scratch buffer 0/1
    "model.layers.2.self_attn": ["model.layers.1.self_attn", "model.layers.2.self_attn"],
    "model.layers.3.self_attn": ["model.layers.1.self_attn", "model.layers.2.self_attn"],
    "model.layers.1.self_attn.index": ["model.layers.1.self_attn.index", "model.layers.2.self_attn.index"],
    "model.layers.2.self_attn.index": ["model.layers.1.self_attn.index", "model.layers.2.self_attn.index"],
    "model.layers.3.self_attn.index": ["model.layers.1.self_attn.index", "model.layers.2.self_attn.index"],
}
```

vLLM 初步规划出的 `kv_cache_tensors` 项（按 scratch_aliases 预分组）：

```python
KVCacheTensor(shared_by=["model.layers.1.self_attn", "model.layers.2.self_attn", "model.layers.3.self_attn"], ...)
```

`finalize_kv_cache_config` 展开后（rank 0 视角，只保留本 rank 分配的 2 个 scratch buffer）：

```python
KVCacheTensor(shared_by=["model.layers.1.self_attn", "model.layers.2.self_attn"], ...)
```

这样 vLLM 为 main 的 scratch 分配**两块**物理 buffer（layers.1 和 layers.2 共享同一对 buffer），layer N 读其中一块时，layer N+1 的 prefetch 可以写另一块。`layers.3` 不在 rank 0 的 `shared_by` 里——它由 rank 3 自己分配。

---

## `get_kvpp_layer_owners`

### 入参

| 参数 | 类型 | 含义 | 举例 |
|---|---|---|---|
| `vllm_config` | `VllmConfig` | 全局 vLLM 配置，从中读取 `KVPPConfig.size` 和 MTP 相关模型配置（`num_hidden_layers`、`num_nextn_predict_layers`） | `kvpp_size=4`，`num_hidden_layers=4`，`num_nextn_predict_layers=1` |
| `layer_names` | `Iterable[str]` | 当前 worker 持有的所有 KV cache 层名（可能来自 set 或无序 dict，函数内部会按 `(layer_index, name)` 排序保证所有 rank 顺序一致） | `{"model.layers.0.self_attn", "model.layers.0.self_attn.index", ..., "model.layers.4.mtp_attn"}`（9 个层名，4 层 × main+indexer + MTP） |

### 返回

| 字段 | 类型 | 含义 | 举例 |
|---|---|---|---|
| 返回值 | `dict[str, int]` | 层名 → owner rank 的映射。**MTP 层不会出现在返回值里**（它们被排除，走全量复制路径） | `{"model.layers.0.self_attn": 0, "model.layers.0.self_attn.index": 0, "model.layers.1.self_attn": 1, "model.layers.1.self_attn.index": 1, "model.layers.2.self_attn": 2, "model.layers.2.self_attn.index": 2, "model.layers.3.self_attn": 3, "model.layers.3.self_attn.index": 3}`（layers.0→rank0, layers.1→rank1, layers.2→rank2, layers.3→rank3, layers.4.mtp_attn 不在） |

### 主要工作

1. 按 `(extract_layer_index(name), name)` 排序 `layer_names`，保证所有 rank 的 owner 分配顺序一致。
2. 调用 `_get_replicated_mtp_layers` 识别 MTP 层（`num_hidden_layers <= layer_index < num_hidden_layers + num_nextn_predict_layers`），这些层会被排除出 owner 分配。
3. 按 `layer_index` 把非 MTP 层分桶（同一个 transformer layer 的 main + indexer 等 cache 归到同一个桶）。
4. 把桶按 KVPP size 均分给各 rank（`divmod` 处理余数，前 `remainder` 个 rank 多分一个桶）。
5. 返回每层对应的 owner rank。

`kvpp_size=4` 时 `divmod(4, 4) = (1, 0)`，每个 rank 分 1 个桶：rank 0 → index 0, rank 1 → index 1, rank 2 → index 2, rank 3 → index 3。

---

## `_get_allocation_groups`

### 入参

| 参数 | 类型 | 含义 |
|---|---|---|
| `logical_groups` | `list[KVCacheGroupSpec]` | 全局视角的 KV cache group 列表（未切分前） |
| `worker_spec` | `dict[str, KVCacheSpec]` | 当前 worker 实际持有的层名 → KV cache spec 映射 |
| `owners` | `dict[str, int]` | `get_kvpp_layer_owners` 的返回值，层名 → owner rank |
| `kvpp_rank` | `int` | 当前 worker 的 KVPP rank（`worker_index % kvpp_size`） |

### 返回

`tuple[list[KVCacheGroupSpec], dict[str, list[str]]]`：

- **第一个元素**：该 worker 实际要分配的 KV cache group 列表（allocation_groups），用于物理 tensor 分配。
- **第二个元素**：`scratch_aliases`，层名 → 它实际共享的 scratch buffer 名列表，用于 `finalize_kv_cache_config` 展开 `shared_by`。

### 主要工作

遍历每个 logical group，把 group 里的层分成三类：

1. **本 rank own 的层**（`owners[name] == kvpp_rank`）→ 进 `allocation_names`，作为持久化 KV cache 分配。
2. **其他 rank own 的层**（`owners[name] != kvpp_rank`，非 MTP）→ 替换成 2 个 scratch buffer（双缓冲），并记录到 `scratch_aliases`。
3. **MTP 层**（不在 `owners` 里）→ 进 `allocation_names`，全量复制。

对第 2 类的 scratch 分组：按 `worker_spec` 相等性把同 layout 的层归到同一个 `scratch_layout_groups`，每个 layout group 取前 2 个 scratch 名（`scratch_0`、`scratch_1`），并把所有同 layout 的层映射到这 2 个 buffer 上。

### 举例

接上例，`kvpp_rank=0`，`owners` 同上。假设 `logical_groups` 只有一个 group，包含所有 9 个层（含 MTP）。

输入：

```python
logical_groups = [KVCacheGroupSpec(
    layer_names=[
        "model.layers.0.self_attn", "model.layers.0.self_attn.index",
        "model.layers.1.self_attn", "model.layers.1.self_attn.index",
        "model.layers.2.self_attn", "model.layers.2.self_attn.index",
        "model.layers.3.self_attn", "model.layers.3.self_attn.index",
        "model.layers.4.mtp_attn",
    ],
    kv_cache_spec=UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={...}),
)]
worker_spec = {<上述 9 个层名>: <对应 KVCacheSpec>}
owners = {<同上例，不含 mtp_attn>}
kvpp_rank = 0
```

处理过程（对该 group）：

1. `local_names` = group 里在 `worker_spec` 中的层（全部 9 个）。
2. `managed_names` = `local_names` 中在 `owners` 里的层（8 个，排除 MTP）。
3. `allocation_names` 初始值 = 本 rank own 的层 + MTP 层：
   - `model.layers.0.self_attn`、`model.layers.0.self_attn.index`（owner=0）
   - `model.layers.1.self_attn`、`model.layers.1.self_attn.index`（owner=0）
   - `model.layers.4.mtp_attn`（MTP，不在 owners）
4. 对 `managed_names` 里 owner≠本 rank 的层（即 `layers.2`、`layers.3` 的 4 个层），按 `worker_spec` 相等性分组：
   - 假设 main 和 indexer 的 spec 不同，则分 2 个 layout group：
     - `[layers.2.self_attn, layers.3.self_attn]`（main 同 layout）
     - `[layers.2.self_attn.index, layers.3.self_attn.index]`（indexer 同 layout）
   - 每个 layout group 取前 2 个作为 scratch buffer 名。
5. 把 scratch buffer 名加入 `allocation_names`，并填充 `scratch_aliases`。

返回的 `allocation_groups`（第一个元素）对应的 `allocation_spec` 包含：

| 层名 | 类型 |
|---|---|
| `model.layers.0.self_attn` | 持久化 KV cache |
| `model.layers.0.self_attn.index` | 持久化 KV cache |
| `model.layers.1.self_attn` | 持久化 KV cache |
| `model.layers.1.self_attn.index` | 持久化 KV cache |
| `model.layers.2.self_attn.scratch_0` | scratch buffer 0 |
| `model.layers.2.self_attn.scratch_1` | scratch buffer 1 |
| `model.layers.2.self_attn.index.scratch_0` | scratch buffer 0 |
| `model.layers.2.self_attn.index.scratch_1` | scratch buffer 1 |
| `model.layers.3.self_attn.scratch_0` | scratch buffer 0 |
| `model.layers.3.self_attn.scratch_1` | scratch buffer 1 |
| `model.layers.3.self_attn.index.scratch_0` | scratch buffer 0 |
| `model.layers.3.self_attn.index.scratch_1` | scratch buffer 1 |
| `model.layers.4.mtp_attn` | MTP 全量复制 |

返回的 `scratch_aliases`（第二个元素）：

```python
{
    "model.layers.2.self_attn": ["model.layers.2.self_attn.scratch_0", "model.layers.3.self_attn.scratch_0"],
    "model.layers.3.self_attn": ["model.layers.2.self_attn.scratch_1", "model.layers.3.self_attn.scratch_1"],
    "model.layers.2.self_attn.index": ["model.layers.2.self_attn.index.scratch_0", "model.layers.3.self_attn.index.scratch_0"],
    "model.layers.3.self_attn.index": ["model.layers.2.self_attn.index.scratch_1", "model.layers.3.self_attn.index.scratch_1"],
}
```

**关键点**：`scratch_aliases` 的映射不是"一个逻辑层 → 它自己的两个 buffer"，而是"一个逻辑层 → 同 layout group 里所有层共享的两个 buffer 的某一列"。这样 `layers.2.self_attn` 和 `layers.3.self_attn` 共用同一对 scratch buffer（通过 `shared_by` 共享物理内存），双缓冲交替使用。

---

## `project_kv_cache_groups_to_worker`

### 入参

| 参数 | 类型 | 含义 |
|---|---|---|
| `global_groups` | `list[KVCacheGroupSpec]` | 全局视角的 KV cache group 列表（未切分前） |
| `worker_spec` | `dict[str, KVCacheSpec]` | 当前 worker 实际持有的层名 → KV cache spec 映射（决定哪些层保留） |

### 返回

`list[KVCacheGroupSpec]` —— 投影到该 worker 后的 group 列表。每个 group 的 `layer_names` 只保留该 worker 实际持有的层，`kv_cache_spec` 也相应收缩到这些层。

### 主要工作

遍历每个 global group：

1. 用 `worker_spec` 过滤 `group.layer_names`，只保留当前 worker 持有的层 → `worker_layer_names`。
2. 如果 group 的 spec 是 `UniformTypeKVCacheSpecs`，重建一个只含 `worker_layer_names` 的 spec（保留 `block_size`，收缩 `kv_cache_specs` 字典）。
3. `is_eagle_group` 只有在 `worker_layer_names` 非空时才继承原值，否则置 `False`。
4. 即使 `worker_layer_names` 为空也保留该 group（保持 group 数量和顺序稳定）。

这是一个**纯过滤 + 收缩**操作，不改变层的归属语义，只把全局视图裁剪成 worker 视图。

### 举例

假设全局有 2 个 group，`worker_spec` 只持有部分层：

```python
global_groups = [
    KVCacheGroupSpec(
        layer_names=["model.layers.0.self_attn", "model.layers.1.self_attn",
                     "model.layers.2.self_attn", "model.layers.3.self_attn"],
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs={
                "model.layers.0.self_attn": <KVCacheSpec A>,
                "model.layers.1.self_attn": <KVCacheSpec A>,
                "model.layers.2.self_attn": <KVCacheSpec A>,
                "model.layers.3.self_attn": <KVCacheSpec A>,
            },
        ),
        is_eagle_group=False,
    ),
    KVCacheGroupSpec(
        layer_names=["model.layers.4.mtp_attn"],
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs={"model.layers.4.mtp_attn": <KVCacheSpec B>},
        ),
        is_eagle_group=True,
    ),
]

worker_spec = {
    "model.layers.0.self_attn": <KVCacheSpec A>,   # 只持有 0 和 MTP
    "model.layers.4.mtp_attn": <KVCacheSpec B>,
}
```

返回：

```python
[
    KVCacheGroupSpec(
        layer_names=["model.layers.0.self_attn"],   # 只剩 worker 持有的层
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs={"model.layers.0.self_attn": <KVCacheSpec A>},
        ),
        is_eagle_group=False,                        # 原值 False 且 worker_layer_names 非空
    ),
    KVCacheGroupSpec(
        layer_names=["model.layers.4.mtp_attn"],
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs={"model.layers.4.mtp_attn": <KVCacheSpec B>},
        ),
        is_eagle_group=True,                         # 原值 True 且 worker_layer_names 非空
    ),
]
```

### 两个调用场景的区别

`project_kv_cache_groups_to_worker` 在代码里被调用两次，语义略有不同：

| 调用点 | `worker_spec` 内容 | 作用 |
|---|---|---|
| `get_kv_cache_groups_for_worker` → `_get_allocation_groups` 内部 | `allocation_spec`（含 scratch buffer 名） | 决定**物理 tensor 分配**：哪些层要分配内存、分配成持久化还是 scratch |
| `finalize_kv_cache_config` 末尾 | 原始 `worker_spec`（只含真实逻辑层名） | 决定**调度视图**：vLLM 调度器看到的 group 只含本 worker 真实持有的逻辑层，不含 scratch buffer 名 |

第一个场景让分配器知道要分配 scratch buffer；第二个场景让调度器只看到真实层名，scratch buffer 是底层实现细节，不暴露给调度逻辑。

---

## `_get_replicated_mtp_layers`

### 入参

| 参数 | 类型 | 含义 |
|---|---|---|
| `vllm_config` | `VllmConfig` | 全局 vLLM 配置，从中读取 `speculative_config`（判断是否启用 MTP）和 `model_config.hf_config`（读取 `num_hidden_layers` 和 `num_nextn_predict_layers`） |
| `layer_names` | `Iterable[str]` | 当前 worker 持有的所有 KV cache 层名（函数会从中筛选出 MTP 层） |

### 返回

`set[str]` —— 需要全量复制（不参与 KVPP owner 分配）的 MTP 层名集合。

特殊情况：
- 未启用 speculative decoding 或 method 不是 `mtp` → 返回空集。
- 启用了 MTP 但找不到任何 MTP 层 → 抛 `ValueError`（KVPP 要求至少有一层 MTP KV cache 可复制）。
- `hf_config` 缺少 `num_hidden_layers` 或 `num_nextn_predict_layers` → 抛 `ValueError`。

### 主要工作

1. 从 `vllm_config.speculative_config` 判断是否启用 MTP；未启用或非 MTP 方法直接返回空集。
2. 从 `hf_config` 读 `num_hidden_layers`（作为 MTP 层的起始 index）和 `num_nextn_predict_layers`（MTP 层数）。
3. 计算 MTP 层的 index 区间：`[num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)`。
4. 遍历 `layer_names`，用 `extract_layer_index` 提取每层 index，落在上述区间内的即为 MTP 层。
5. 如果筛选结果为空，抛 `ValueError`（说明配置声明了 MTP 但实际没有对应的 KV cache 层）。
6. 返回 MTP 层名集合。

### 为什么 MTP 层要全量复制

MTP（Multi-Token Prediction）投机解码的 speculator 层在每个 rank 上都要**本地**做 attention 来生成 speculative token。它的 KV cache 不能跨 rank 分布——否则每次 speculative forward 都要触发跨 rank 传输，延迟代价远大于收益。所以 MTP 层被排除出 KVPP 的 owner 分配，每个 rank 都完整保留一份副本。

### 举例

假设 GLM-5.2 模型：4 层 Transformer + 1 层 MTP，启用 MTP 投机解码。

输入：

```python
vllm_config  # speculative_config.method="mtp"
             # model_config.hf_config.num_hidden_layers=4
             # model_config.hf_config.num_nextn_predict_layers=1

layer_names = [
    "model.layers.0.self_attn",       # index 0
    "model.layers.1.self_attn",       # index 1
    "model.layers.2.self_attn",       # index 2
    "model.layers.3.self_attn",       # index 3
    "model.layers.4.mtp_attn",        # index 4
]
```

处理过程：

1. `speculative_config.method == "mtp"` → 继续。
2. `mtp_start = num_hidden_layers = 4`，`num_mtp_layers = 1`。
3. MTP index 区间：`[4, 5)`。
4. 遍历 `layer_names`：
   - `index 0, 1, 2, 3` → 不在 `[4, 5)`，跳过。
   - `index 4` → 落在 `[4, 5)`，加入结果集。

返回：

```python
{"model.layers.4.mtp_attn"}
```

这个集合会被 `get_kvpp_layer_owners` 用来排除 MTP 层——MTP 层不进入 `owners` 字典，进而在 `_get_allocation_groups` 里走"全量复制"路径（每个 rank 都分配一份完整的 MTP KV cache），而不是被切成 owner + scratch。

---

## `_initialize_kvpp_scheduler`

### 位置

`NPUModelRunner._initialize_kvpp_scheduler`，在 `initialize_kv_cache` 末尾被调用（KV cache tensor 分配完成之后）。

### 主要做的事情

按执行顺序：

1. **收集层名**：从 `kv_cache_config.kv_cache_groups` 汇总所有层名（用 `dict.fromkeys` 保序去重）。

2. **算 owner**：调用 `get_kvpp_layer_owners` 得到每层 KV cache 的 owner rank。

3. **建分布式 group**：调用 `get_kvpp_group()` 拿到 KVPP 的 `GroupCoordinator`（跨 rank 通信用）。

4. **定位 KVPP 管理的 group 索引**：调用 `get_kvpp_managed_group_index` 找到 KVPP 管理的 cache group 在 `kv_cache_groups` 里的下标 `kvpp_cache_group_index`，存到 `self.kvpp_cache_group_index`。

5. **MTP speculator 校验**：如果存在 speculator，检查 speculator 的 draft attention 层名不能出现在 `layer_owners` 里——MTP 层必须全量复制，不能被 KVPP 管理。违反则抛 `RuntimeError`。

6. **读 block 参数**：从 `self.block_tables` 按 `kvpp_cache_group_index` 读 `blocks_per_kv_block` 和 `kernel_block_sizes`，算出 `num_kernel_blocks`（物理 block 总数）和 `block_size`（每 block 的 token 数）。

7. **收集 attention impl**：遍历 `static_forward_context`，找出在 `layer_owners` 里且带有 `layerwise_kv_cache_hook` 属性的 attention impl（MLA/SFA），组成 `kvpp_impls`。找不到则抛 `RuntimeError`。

8. **建 transport**：`MemFabricMTEKVPPTransport(kvpp_group, layer_owners, num_kernel_blocks)`。

9. **建 scheduler**：`KVPPScheduler(group, layer_owners, num_blocks, block_size, transport, execution_layers=tuple(kvpp_impls))`。`execution_layers` 传的是 attention impl 的层名顺序，决定 prefetch 的执行顺序。

10. **校验并喂 KV cache tensor**：检查 `self._kvpp_kv_caches`（由 `_on_kv_caches_initialized` 钩子暂存）非空，从中提取 `layer_owners` 里各层对应的 tensor，组成 `managed_kv_caches`。

11. **初始化 transport**：`kvpp_scheduler.initialize_transport(managed_kv_caches)`——这里会建 comm stream、thread pool executor、校验跨 rank plan 一致性。

12. **清空暂存**：`self._kvpp_kv_caches = None`（用完即弃）。

13. **挂载 scheduler**：`self.kvpp_scheduler = kvpp_scheduler`。

14. **注入 hook**：把 `kvpp_scheduler` 赋给每个 attention impl 的 `layerwise_kv_cache_hook`，让 attention forward 时能回调 `enter_layer`/`wait_for_layer`/`leave_layer`。

### 主要变化的字段实例

以 GLM-5.2（4 层 Transformer + 1 层 MTP，每层 main + indexer）、`kvpp_size=4`、`worker_index=0`（rank 0）为例。调用前后 `NPUModelRunner` 上关键字段的变化：

| 字段 | 调用前 | 调用后 |
|---|---|---|
| `self.kvpp_cache_group_index` | `None` | `0`（假设 KVPP group 是第 0 个） |
| `self._kvpp_kv_caches` | `{"model.layers.0.self_attn": <tensor>, ..., "model.layers.4.mtp_attn": <tensor>}`（由 `_on_kv_caches_initialized` 暂存的全部层） | `None`（已消费并清空） |
| `self.kvpp_scheduler` | `None` | `KVPPScheduler(group=kvpp_group, layer_owners={...}, num_blocks=<N>, block_size=128, transport=<MemFabricMTEKVPPTransport>, execution_layers=("model.layers.0.self_attn", "model.layers.1.self_attn", "model.layers.2.self_attn", "model.layers.3.self_attn"))` |

`layer_owners` 实例（rank 0 视角，MTP 层不在内）：

```python
{
    "model.layers.0.self_attn": 0,
    "model.layers.0.self_attn.index": 0,
    "model.layers.1.self_attn": 0,
    "model.layers.1.self_attn.index": 0,
    "model.layers.2.self_attn": 1,
    "model.layers.2.self_attn.index": 1,
    "model.layers.3.self_attn": 1,
    "model.layers.3.self_attn.index": 1,
}
```

`managed_kv_caches` 实例（喂给 transport 的，只含 owner 层的 tensor）：

```python
{
    "model.layers.0.self_attn": <KV tensor>,
    "model.layers.0.self_attn.index": <KV tensor>,
    "model.layers.1.self_attn": <KV tensor>,
    "model.layers.1.self_attn.index": <KV tensor>,
    "model.layers.2.self_attn": <KV tensor>,
    "model.layers.2.self_attn.index": <KV tensor>,
    "model.layers.3.self_attn": <KV tensor>,
    "model.layers.3.self_attn.index": <KV tensor>,
}
```

> 注意：`managed_kv_caches` 包含**所有** owner 层（含其他 rank own 的层），因为 transport 需要知道所有 owner 层的 tensor 地址——本 rank own 的用于 push，其他 rank own 的用于 receive 时写入对应的 scratch buffer。

`kvpp_impls` 实例（被注入 hook 的 attention impl）：

```python
{
    "model.layers.0.self_attn": <MLA/SFA impl>,
    "model.layers.1.self_attn": <MLA/SFA impl>,
    "model.layers.2.self_attn": <MLA/SFA impl>,
    "model.layers.3.self_attn": <MLA/SFA impl>,
}
```

每个 impl 调用后的 `layerwise_kv_cache_hook`：

```python
impl.layerwise_kv_cache_hook = <KVPPScheduler 实例>
```

这样 attention forward 走到每层时，impl 会回调 `kvpp_scheduler.enter_layer(layer_name)` / `wait_for_layer(layer_name)` / `leave_layer(layer_name)`，驱动 KV page 的跨 rank prefetch。

---

## `KVPPExecutionPlan.build`

### 位置

`vllm_ascend/worker/v2/kvpp.py`，是 `KVPPScheduler.__init__` 里通过 `KVPPExecutionPlan.build(self.layer_owners, execution_layers)` 调用的类方法。

### 入参

| 参数 | 类型 | 含义 |
|---|---|---|
| `layer_owners` | `dict[str, int]` | `get_kvpp_layer_owners` 的返回值，层名 → owner rank。**键的集合代表所有有 KV cache 的层**（含 main 和 indexer），顺序可能无序。 |
| `execution_layers` | `tuple[str, ...] \| None` | 实际执行的 attention 层名顺序（来自 `kvpp_impls` 的键）。`None` 表示不区分 main/indexer，按 `layer_owners` 的键构造。 |

### 返回

`KVPPExecutionPlan`（frozen dataclass，只有一个字段 `cache_bundles: dict[str, tuple[str, ...]]`）：

- **键**：执行层名（attention 实际调用的层，通常是 main attention 层如 `model.layers.N.self_attn`）。
- **值**：该执行层对应的 cache bundle——一个 tuple，包含同 transformer layer 下所有需要一起 prefetch 的 KV cache 层名（如 main + indexer）。

`KVPPExecutionPlan.layers` property 返回 `tuple(cache_bundles)`，即执行层的顺序，决定了 prefetch 的执行顺序。

### 主要工作

1. `layers = tuple(execution_layers or layer_owners)`：确定执行层顺序。优先用 `execution_layers`，为 `None` 时用 `layer_owners` 的键。
2. 如果 `execution_layers is None`（简单模式）：每个执行层自己就是一个 bundle，`cache_bundles = {layer: (layer,)}`。
3. 如果 `execution_layers` 非 `None`（bundle 模式，实际生产路径）：
   - 按 `(layer_index, name)` 排序 `layer_owners` 的键，把同 index 的 cache 层分桶 → `cache_layers_by_index`。
   - 遍历每个执行层，找到它对应 index 的桶，整桶作为它的 bundle。
   - 校验：每个 index 只能有一个执行层（否则一个 transformer layer 有多个 main attention，冲突）。
   - 校验：所有 cache 层必须被某个执行层 claim（否则有 cache 层无人 prefetch，漏数据）。
4. 统一校验：每个 bundle 内所有层的 owner 必须相同（一个 bundle 不能跨 owner，否则 push/receive 逻辑混乱）。

### 为什么需要 bundle

SFA/MLA 的一个 transformer layer 可能有两个 KV cache tensor：main（主 attention）和 indexer（用于-rope 重组等辅助计算）。它们必须**一起** prefetch——否则 attention forward 时 indexer 读不到数据。`execution_layers` 只列 main attention（实际被 attention hook 回调的层），但 bundle 把同 index 的 main + indexer 一起打包，让 transport 一次传输整个 bundle。

### 举例

假设 GLM-5.2，4 层 Transformer，每层有 main + indexer 两个 KV cache，`kvpp_size=4`。

`layer_owners`：

```python
{
    "model.layers.0.self_attn": 0,        # main, index 0
    "model.layers.0.self_attn.index": 0,  # indexer, index 0
    "model.layers.1.self_attn": 0,        # main, index 1
    "model.layers.1.self_attn.index": 0,  # indexer, index 1
    "model.layers.2.self_attn": 1,        # main, index 2
    "model.layers.2.self_attn.index": 1,  # indexer, index 2
    "model.layers.3.self_attn": 1,        # main, index 3
    "model.layers.3.self_attn.index": 1,  # indexer, index 3
}
```

`execution_layers`（来自 `kvpp_impls` 的键，只含 main attention）：

```python
("model.layers.0.self_attn", "model.layers.1.self_attn",
 "model.layers.2.self_attn", "model.layers.3.self_attn")
```

#### 步骤 3 处理过程

1. 按 `(index, name)` 排序 `layer_owners` 的键，分桶：

```python
cache_layers_by_index = {
    0: ["model.layers.0.self_attn", "model.layers.0.self_attn.index"],
    1: ["model.layers.1.self_attn", "model.layers.1.self_attn.index"],
    2: ["model.layers.2.self_attn", "model.layers.2.self_attn.index"],
    3: ["model.layers.3.self_attn", "model.layers.3.self_attn.index"],
}
```

2. 遍历 `execution_layers`，每个执行层 claim 自己 index 的桶：

| 执行层 | index | claim 的桶 | bundle |
|---|---|---|---|
| `model.layers.0.self_attn` | 0 | `cache_layers_by_index[0]` | `("model.layers.0.self_attn", "model.layers.0.self_attn.index")` |
| `model.layers.1.self_attn` | 1 | `cache_layers_by_index[1]` | `("model.layers.1.self_attn", "model.layers.1.self_attn.index")` |
| `model.layers.2.self_attn` | 2 | `cache_layers_by_index[2]` | `("model.layers.2.self_attn", "model.layers.2.self_attn.index")` |
| `model.layers.3.self_attn` | 3 | `cache_layers_by_index[3]` | `("model.layers.3.self_attn", "model.layers.3.self_attn.index")` |

3. 校验 `claimed_indices = {0, 1, 2, 3}`，`unclaimed` 为空，通过。
4. 每个 bundle 内 owner 都相同，通过。

#### 返回

```python
KVPPExecutionPlan(
    cache_bundles={
        "model.layers.0.self_attn": ("model.layers.0.self_attn", "model.layers.0.self_attn.index"),
        "model.layers.1.self_attn": ("model.layers.1.self_attn", "model.layers.1.self_attn.index"),
        "model.layers.2.self_attn": ("model.layers.2.self_attn", "model.layers.2.self_attn.index"),
        "model.layers.3.self_attn": ("model.layers.3.self_attn", "model.layers.3.self_attn.index"),
    }
)
```

`plan.layers` 返回：

```python
("model.layers.0.self_attn", "model.layers.1.self_attn",
 "model.layers.2.self_attn", "model.layers.3.self_attn")
```

这个顺序就是 KVPPScheduler 在 `enter_layer` / `wait_for_layer` 里校验和驱动的 prefetch 顺序——attention forward 必须按这个顺序进入每一层，否则 scheduler 会抛 `RuntimeError`。

### 校验失败的场景

- **执行层重复 claim 同一个 index**：比如 `execution_layers` 里有 `model.layers.0.self_attn` 和 `model.layers.0.cross_attn`（都是 index 0），抛 "multiple executable attention layers for transformer layer 0"。
- **有 cache 层无人 claim**：比如 `layer_owners` 里有 `model.layers.2.cross_attn`（index 2）但 `execution_layers` 里没有 index 2 的执行层，抛 "KVPP cache layers have no executable attention owner"。
- **bundle 跨 owner**：比如 index 0 的 main 是 owner 0，indexer 是 owner 1，抛 "KVPP cache bundle for ... spans owners [0, 1]"。

---

## `_active_pages`

### 位置

`vllm_ascend/worker/v2/kvpp.py`，是模块级函数，被 `KVPPScheduler.begin_forward` 调用，把 vLLM 的 `block_table` 和 `seq_lens` 转换成 transport 能消费的 `KVPPActivePages`。

### 入参

| 参数 | 类型 | 含义 |
|---|---|---|
| `block_table` | `torch.Tensor` | vLLM 的 block table，shape `[num_requests, max_blocks_per_request]`，元素是物理 block ID（int32）。可能在 device 上。 |
| `seq_lens` | `Any` | 每个请求的序列长度。**必须在 CPU 上**——如果在 device 上会抛 `ValueError`（避免 device→host sync）。 |
| `block_size` | `int` | 每个 block 包含的 token 数（如 128）。 |
| `num_blocks` | `int` | 物理 block 总数（`num_kernel_blocks`），用于校验 block ID 有效性。 |

### 返回

`KVPPActivePages`（frozen dataclass）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `page_ids` | `torch.Tensor` | shape `[N]`（N = `num_requests * max_blocks_per_request`），int64，在 device 上。**已排序**，无效位置填 `num_blocks`（sentinel）。 |
| `valid_mask` | `torch.Tensor` | shape `[N]`，bool，在 device 上。`True` 表示对应 `page_ids` 是有效的 active page。 |
| `count_upper_bound` | `int` | CPU 上的标量，active page 数量的上界。用于 transport 预分配 staging buffer 容量。 |

### 设计要点：为什么用固定形状 + mask 而不是紧凑列表

`page_ids` 是**固定形状**（铺平整个 block table），不是紧凑的 `[2, 3, 5, 8]`。原因：

1. **避免 device→CPU sync**：算"有效 page 数 = 4"需要 `valid_mask.sum().item()`，这是 device→CPU 同步，会阻塞 hot path。固定形状让 transport 不需要知道精确数量，直接遍历整个 tensor 用 mask 跳过无效位置。
2. **图捕获友好**（虽然当前分支已删图模式，但设计保留了这一点）：固定形状的 tensor 可以被图捕获，动态形状不行。
3. **`count_upper_bound` 是 CPU 标量**：用 host 上的 `seq_lens` 算出，不需要 device sync。transport 用它做容量校验（确保 staging buffer 够大），但不用它来决定遍历范围。

### 主要工作

1. **校验 seq_lens 在 CPU**：如果在 device 上，抛 `ValueError`。
2. **算 host 上的 page 数上界**：用 CPU 上的 `seq_lens` 算每个请求的 page 数（`ceil(seq_len / block_size)`），clamp 到 `[0, table_columns]`，求和得到 `count_upper_bound`。这里有一次 `.item()` 但在 host tensor 上，不涉及 device sync。
3. **构造 covered mask**：把 `seq_lens` 拷到 device，算每个请求实际覆盖的列数，`columns < pages_per_request` 的位置为 `True`。
4. **构造 valid mask**：`covered & (table >= 0) & (table < num_blocks)`——同时排除未覆盖列、padding 列（-1）、越界 ID。
5. **填充 sentinel + 排序**：把无效位置填成 `num_blocks`（大于任何合法 ID），flatten 后排序。这样所有无效 page 都堆到末尾。
6. **去重 mask**：`sorted_pages[1:] != sorted_pages[:-1]` 标记重复位置，第一个出现的是 `True`，后续重复是 `False`。
7. **最终 valid_mask**：`unique & (sorted_pages < num_blocks)`——既去重又排除 sentinel。

### 举例

假设 `block_size=128`，`num_blocks=10`（block ID 范围 0~9），2 个请求。

输入：

```python
block_table = torch.tensor([
    [2, 3, 5, -1],   # 请求 0: seq_len=300 → 覆盖 3 个 block (2,3,5)，第 4 列是 padding (-1)
    [8, 0, 0, 0],    # 请求 1: seq_len=128 → 覆盖 1 个 block (8)，后 3 列虽是 0 但未覆盖
], dtype=torch.int32)  # shape [2, 4]

seq_lens = torch.tensor([300, 128])  # 在 CPU 上
block_size = 128
num_blocks = 10
```

#### 步骤 2：host 上算 count_upper_bound

```
pages_per_request_host = ceil([300, 128] / 128) = [3, 1]
sum = 4
count_upper_bound = min(10, 4) = 4
```

#### 步骤 3-4：构造 valid mask

```
lengths (device) = [300, 128]
pages_per_request (device) = [3, 1]
columns = [[0, 1, 2, 3], [0, 1, 2, 3]]
covered = [[T, T, T, F], [T, F, F, F]]   # columns < pages_per_request

table (int64) = [[2, 3, 5, -1], [8, 0, 0, 0]]
valid = covered & (table >= 0) & (table < 10)
      = [[T, T, T, F], [T, F, F, F]]
```

> 注意请求 1 的第 2~4 列虽然 table 值是 0（合法 ID），但 `covered` 是 `False`，所以 `valid` 也是 `False`。

#### 步骤 5：填充 sentinel + 排序

```
sentinel = 10
where(valid, table, sentinel) = [[2, 3, 5, 10], [8, 10, 10, 10]]
flatten = [2, 3, 5, 10, 8, 10, 10, 10]
sorted_pages = [2, 3, 5, 8, 10, 10, 10, 10]
```

#### 步骤 6-7：去重 + 最终 mask

```
unique = [T, T, T, T, T, F, F, F]   # 后三个 10 是重复
sorted_pages < num_blocks = [T, T, T, T, F, F, F, F]
valid_mask = unique & (sorted_pages < 10) = [T, T, T, T, F, F, F, F]
```

#### 返回

```python
KVPPActivePages(
    page_ids=torch.tensor([2, 3, 5, 8, 10, 10, 10, 10], dtype=torch.int64, device="npu"),
    valid_mask=torch.tensor([True, True, True, True, False, False, False, False], device="npu"),
    count_upper_bound=4,
)
```

### transport 如何消费

transport 拿到 `KVPPActivePages` 后：

- 用 `count_upper_bound` 校验 staging buffer 容量够不够（`count_upper_bound <= staging_capacity`）。
- 遍历 `page_ids` 的前 `count_upper_bound` 个位置（或整个 tensor，用 `valid_mask` 跳过），对每个有效 page ID 执行 MTE 拷贝。
- 因为 `page_ids` 已排序去重，transport 不会重复拷贝同一个 block——即使 vLLM 的 block table 里有重复 ID（比如两个请求共享同一个 prefix block）。

### 为什么不直接返回 `[2, 3, 5, 8]`

紧凑列表 `[2, 3, 5, 8]` 需要知道"有效 page 数 = 4"。算这个数有两种方式：

1. `valid_mask.sum().item()` —— **device→CPU sync**，hot path 上不可接受。
2. 在 host 上用 `seq_lens` 算 —— 这就是 `count_upper_bound` 的来源。

但 `count_upper_bound` 是**上界**不是精确值（host 算的时候没去重，也没排除 padding）。如果 transport 按 `count_upper_bound` 切片 `page_ids[:4]`，可能切到 sentinel 或重复值。所以设计成"固定形状 tensor + mask"，transport 遍历整个 tensor 用 mask 过滤——既不需要精确数量，也不引入 sync。

---

## `KVPPScheduler.__init__` 字段说明

### 位置

`vllm_ascend/worker/v2/kvpp.py`，由 `_initialize_kvpp_scheduler` 构造。

### 入参（构造时传入）

| 参数 | 类型 | 含义 |
|---|---|---|
| `group` | `GroupCoordinator` | KVPP 分布式 group，封装跨 rank 通信（`cpu_group`、`device_group`、`ranks`、`rank_in_group`、`world_size`）。 |
| `layer_owners` | `dict[str, int]` | `get_kvpp_layer_owners` 的返回值，层名 → owner rank。决定每层是本地持久化还是需要跨 rank 传输。 |
| `num_blocks` | `int` | 物理 block 总数（`num_kernel_blocks = kv_cache_config.num_blocks * blocks_per_kv_block`）。用于 `_active_pages` 校验 block ID 有效性。 |
| `block_size` | `int` | 每个 block 的 token 数（如 128）。用于 `_active_pages` 把 `seq_lens` 换算成 page 数。 |
| `transport` | `MemFabricMTEKVPPTransport` | 数据平面，负责实际的 MTE 跨 rank page 拷贝。 |
| `execution_layers` | `tuple[str, ...] \| None` | attention 实际执行的层名顺序（来自 `kvpp_impls` 的键）。决定 prefetch 顺序和 bundle 划分。 |

### 实例字段（`__init__` 里赋值）

#### 从入参直接赋值

| 字段 | 类型 | 含义 |
|---|---|---|
| `self.group` | `GroupCoordinator` | 同入参 `group`。 |
| `self.layer_owners` | `dict[str, int]` | 同入参 `layer_owners`。 |
| `self.num_blocks` | `int` | 同入参 `num_blocks`。 |
| `self.block_size` | `int` | 同入参 `block_size`。 |
| `self.transport` | `MemFabricMTEKVPPTransport` | 同入参 `transport`。 |

#### 从入参派生

| 字段 | 类型 | 含义 |
|---|---|---|
| `self.plan` | `KVPPExecutionPlan` | 由 `KVPPExecutionPlan.build(layer_owners, execution_layers)` 构造。包含 `cache_bundles`（执行层 → bundle 的映射）和 `layers` property（执行层顺序）。 |

#### 状态机字段（初始值，后续由生命周期方法修改）

| 字段 | 初始值 | 类型 | 含义 |
|---|---|---|---|
| `self._phase` | `KVPPPhase.IDLE` | `KVPPPhase` | 当前 scheduler 所处阶段。状态机：`IDLE → FORWARD_ACTIVE → LAYER_ENTERED → LAYER_WAITED → FORWARD_ACTIVE → ... → IDLE`。 |
| `self._next_layer_index` | `0` | `int` | 下一个要进入的执行层在 `plan.layers` 中的下标。`enter_layer` 时校验 `layer_name == plan.layers[next_layer_index]`，然后递增。 |
| `self._selected_pages` | `None` | `KVPPActivePages \| None` | 当前 forward 的 active page 描述符。`begin_forward` 时由 `_active_pages` 计算并暂存，`enter_layer` 时取用传给 prefetch，`finish_forward`/`abort_batch` 时清空。 |
| `self._comm_stream` | `None` | `Any \| None` | KVPP 专用通信 stream。`initialize_transport` 时创建（`torch.npu.Stream()`），`_run_prefetch` 里用 `torch.npu.stream(self._comm_stream)` 把 MTE 拷贝切到这个 stream 上，和 compute stream 并行。 |
| `self._current_layer` | `None` | `str \| None` | 当前正在执行的层名。`enter_layer` 时设置，`leave_layer` 时清空。用于校验 `wait_for_layer`/`leave_layer` 的层身份匹配。 |
| `self._executor` | `None` | `ThreadPoolExecutor \| None` | 单线程池（`max_workers=1`），用于异步执行 `_run_prefetch`。`initialize_transport` 时创建，`close` 时 shutdown。 |
| `self._transfer_future` | `None` | `Future[None] \| None` | 当前在途的 prefetch future。`_start_prefetch` 时由 `executor.submit` 返回，`wait_for_layer` 时 `result()` 阻塞等待完成，`abort_batch` 时 drain。 |
| `self._pending_layer` | `None` | `str \| None` | 当前正在 prefetch 的层名。`_start_prefetch` 时设置，`wait_for_layer` 时清空。用于防重入（不能同时 prefetch 两层）和提前命中判断（如果 `enter_layer` 时 `_pending_layer` 已是该层，说明前一层结束时已提前 prefetch）。 |
| `self._device_id` | `None` | `int \| None` | 当前 NPU device ID。`initialize_transport` 时记录（`torch.npu.current_device()`），`_run_prefetch` worker 线程开头 `torch.npu.set_device(self._device_id)` 确保 worker 线程绑定到正确 device。 |

### 举例

以 GLM-5.2（4 层 Transformer，每层 main + indexer，无 MTP 简化）、`kvpp_size=4`、`worker_index=0`（rank 0）为例。

#### 构造时的入参实例

```python
KVPPScheduler(
    group=<GroupCoordinator: rank_in_group=0, world_size=2, ranks=[0, 8]>,
    layer_owners={
        "model.layers.0.self_attn": 0,
        "model.layers.0.self_attn.index": 0,
        "model.layers.1.self_attn": 0,
        "model.layers.1.self_attn.index": 0,
        "model.layers.2.self_attn": 1,
        "model.layers.2.self_attn.index": 1,
        "model.layers.3.self_attn": 1,
        "model.layers.3.self_attn.index": 1,
    },
    num_blocks=1024,   # kv_cache_config.num_blocks * blocks_per_kv_block
    block_size=128,
    transport=<MemFabricMTEKVPPTransport>,
    execution_layers=(
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
        "model.layers.3.self_attn",
    ),
)
```

#### `__init__` 执行后的字段实例

```python
# 从入参直接赋值
self.group = <GroupCoordinator>
self.layer_owners = {<同上>}
self.num_blocks = 1024
self.block_size = 128
self.transport = <MemFabricMTEKVPPTransport>

# 从入参派生
self.plan = KVPPExecutionPlan(
    cache_bundles={
        "model.layers.0.self_attn": ("model.layers.0.self_attn", "model.layers.0.self_attn.index"),
        "model.layers.1.self_attn": ("model.layers.1.self_attn", "model.layers.1.self_attn.index"),
        "model.layers.2.self_attn": ("model.layers.2.self_attn", "model.layers.2.self_attn.index"),
        "model.layers.3.self_attn": ("model.layers.3.self_attn", "model.layers.3.self_attn.index"),
    }
)
# plan.layers = ("model.layers.0.self_attn", "model.layers.1.self_attn",
#                "model.layers.2.self_attn", "model.layers.3.self_attn")

# 状态机字段（初始值）
self._phase = KVPPPhase.IDLE
self._next_layer_index = 0
self._selected_pages = None
self._comm_stream = None        # initialize_transport 后才创建
self._current_layer = None
self._executor = None           # initialize_transport 后才创建
self._transfer_future = None
self._pending_layer = None
self._device_id = None          # initialize_transport 后才记录
```

#### `initialize_transport(kv_caches)` 调用后的字段变化

```python
self._phase = KVPPPhase.IDLE                    # 不变
self._comm_stream = <torch.npu.Stream()>        # 新创建
self._executor = <ThreadPoolExecutor(max_workers=1)>  # 新创建
self._device_id = 0                             # torch.npu.current_device()
```

其他状态字段仍为初始值，等 `begin_forward` 被调用时才开始流转。

### 字段在生命周期中的流转

一次完整的 forward（4 层），关键字段变化：

| 时刻 | `_phase` | `_next_layer_index` | `_current_layer` | `_pending_layer` | `_transfer_future` | `_selected_pages` |
|---|---|---|---|---|---|---|
| `begin_forward` 后 | `FORWARD_ACTIVE` | 0 | None | None | None | `<KVPPActivePages>` |
| `enter_layer("...0")` 后 | `LAYER_ENTERED` | 1 | `"...0"` | `"...0"` | `<Future>` | 同上 |
| `wait_for_layer("...0")` 后 | `LAYER_WAITED` | 1 | `"...0"` | `"...1"` (提前 prefetch) | `<Future>` (layer 1 的) | 同上 |
| `leave_layer("...0")` 后 | `FORWARD_ACTIVE` | 1 | None | `"...1"` | `<Future>` | 同上 |
| `enter_layer("...1")` 后 | `LAYER_ENTERED` | 2 | `"...1"` | `"...1"` (命中已 pending) | 同上 | 同上 |
| ... | ... | ... | ... | ... | ... | ... |
| `finish_forward` 后 | `IDLE` | 0 | None | None | None | None |

关键流转点：

- `wait_for_layer` 结束时立即 `_start_prefetch` 下一层——这就是"layer N attention 执行时，layer N+1 的 KV page 已经在传"的流水线核心。
- `enter_layer` 时如果 `_pending_layer` 已经是当前层（前一层 `wait_for_layer` 时提前 prefetch 了），直接命中，不再重复启动 prefetch。
- `_transfer_future` 在 `wait_for_layer` 时 `result()` 阻塞，但只阻塞"残留传输时间"——因为 prefetch 是在上一层 attention 执行时并行启动的，大部分时间已经被 attention 计算掩盖了。







