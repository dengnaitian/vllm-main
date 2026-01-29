# query_start_loc 详细解释

## 概述

`query_start_loc` 是 vLLM 中一个**关键的索引数据结构**，用于在**批处理场景**下快速定位每个请求的 token 在扁平化（flattened）张量中的起始和结束位置。它是实现高效注意力计算的核心机制。

**文件位置**: `vllm/v1/worker/gpu_model_runner.py:1411-1417`

## 什么是 query_start_loc？

### 定义

`query_start_loc` 是一个**累积和数组（cumulative sum array）**，记录了每个请求在批次中的起始索引。

### 形状

- **形状**: `[num_reqs + 1]`
- **类型**: `torch.Tensor` (int32)
- **设备**: GPU

### 数据结构示例

假设有 3 个请求，分别调度了 2、5、3 个 tokens：

```python
num_scheduled_tokens = [2, 5, 3]

# query_start_loc 的计算过程：
# 初始化：query_start_loc[0] = 0
# 累积和：
#   query_start_loc[1] = 2
#   query_start_loc[2] = 2 + 5 = 7
#   query_start_loc[3] = 2 + 5 + 3 = 10

query_start_loc = [0, 2, 7, 10]
```

### 可视化表示

```
扁平化的 token 数组 (索引 0-9):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
└─请求0─┘ └─────────请求1────────┘ └──请求2──┘

query_start_loc = [0, 2, 7, 10]
                  │  │  │  │
                  │  │  │  └─ 请求2的结束位置 (总token数)
                  │  │  └──── 请求2的起始位置
                  │  └─────── 请求1的起始位置
                  └────────── 请求0的起始位置
```

## 代码实现

### 计算过程（gpu_model_runner.py:1411-1417）

```python
# 1. 初始化第一个元素为 0
self.query_start_loc.np[0] = 0

# 2. 计算累积和
# cu_num_tokens 是预先计算的累积和数组
# 例如：cu_num_tokens = [2, 7, 10]
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens

# 3. 填充剩余元素（用于 CUDA Graph）
# FlashAttention 等内核要求 query_start_loc 是非递减的
self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])

# 4. 复制到 GPU
self.query_start_loc.copy_to_gpu()

# 5. 提取有效部分
query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
```

### 关键点解释

#### 1. **初始化为 0**
```python
self.query_start_loc.np[0] = 0
```
- 第一个请求总是从索引 0 开始

#### 2. **累积和填充**
```python
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
```
- `cu_num_tokens` 是通过 `np.cumsum(num_scheduled_tokens)` 计算的
- 例如：`[2, 5, 3]` → `[2, 7, 10]`

#### 3. **填充未使用的位置**
```python
self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
```
- **为什么需要填充？**
  - CUDA Graph 要求张量形状固定
  - FlashAttention 等内核要求数组非递减
  - 填充为最大值确保不会越界

#### 4. **异步复制到 GPU**
```python
self.query_start_loc.copy_to_gpu()
```
- 使用异步传输，与 CPU 计算重叠

## query_start_loc 的实际使用场景

基于代码搜索结果，`query_start_loc` 在以下位置被实际使用：

### 1. **计算 logits_indices（用于采样）**

**文件**: `vllm/v1/worker/gpu_model_runner.py:1466`

```python
# 常规情况：每个请求的最后一个 token
logits_indices = query_start_loc[1:] - 1
# 例如：[2, 7, 10] - 1 = [1, 6, 9]
```

**作用**: 从每个请求的 tokens 中提取最后一个 token 的索引，用于采样下一个 token。

---

### 2. **构建注意力元数据**

**文件**: `vllm/v1/worker/gpu_model_runner.py:1591-1592`

```python
common_attn_metadata = CommonAttentionMetadata(
    query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
    # ... 其他参数
)
```

**文件**: `vllm/v1/worker/gpu/attn_utils.py:147-169`

```python
def build_attn_metadata(
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,  # ← 接收参数
    query_start_loc_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    # ... 其他参数
) -> dict[str, Any]:
    max_query_len = int(query_start_loc_cpu.max())  # ← 使用：计算最大查询长度

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,  # ← 传递到注意力元数据
        query_start_loc_cpu=query_start_loc_cpu,
        # ... 其他参数
    )
```

**作用**: 将 `query_start_loc` 传递给注意力机制内核，用于确定每个请求的查询范围。

---

### 3. **计算 Slot Mapping（KV 缓存映射）**

**文件**: `vllm/v1/worker/gpu/block_table.py:161-183`

```python
def compute_slot_mappings(
    self,
    query_start_loc: torch.Tensor,  # ← 接收参数
    positions: torch.Tensor,
) -> torch.Tensor:
    num_reqs = query_start_loc.shape[0] - 1  # ← 使用：计算请求数量
    num_tokens = positions.shape[0]

    # 调用 Triton 内核计算 slot mapping
    _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
        num_tokens,
        self.max_num_batched_tokens,
        query_start_loc,  # ← 传递到内核
        positions,
        self.input_block_table_ptrs,
        # ... 其他参数
    )
    return self.slot_mappings[:, :num_tokens]
```

**作用**: 计算 KV 缓存的物理块映射，每个 token 需要知道存储在哪个物理块中。`query_start_loc` 用于确定每个请求的 token 范围。

---

### 4. **Prefill Token 准备（Triton 内核）**

**文件**: `vllm/v1/worker/gpu/input_batch.py:164-174`

```python
@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,  # ← 接收 query_start_loc 指针
    prefill_token_ids_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    # ... 其他参数
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    # ← 使用 query_start_loc 确定 token 写入位置
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    # 将 prefill tokens 写入 input_ids
    prefill_ptr = prefill_token_ids_ptr + req_state_idx * prefill_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(prefill_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)
```

**作用**: 在 Triton 内核中，使用 `query_start_loc` 将 prefill tokens 复制到正确的位置。

---

### 5. **位置和序列长度计算**

**文件**: `vllm/v1/worker/gpu/input_batch.py:227-239`

```python
@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,  # ← 接收 query_start_loc 指针
    num_computed_tokens_ptr,
    # ... 其他参数
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    # ← 使用 query_start_loc 计算查询长度
    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    # 计算并存储位置信息
    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)
```

**作用**: 使用 `query_start_loc` 计算每个请求的序列长度和位置信息。

---

### 6. **Prompt Logprobs 计算**

**文件**: `vllm/v1/worker/gpu/model_runner.py:686-722`

```python
def compute_prompt_logprobs(
    self,
    hidden_states: torch.Tensor,
    input_batch: InputBatch,
) -> dict[str, LogprobsTensors]:
    # ... 前置逻辑 ...

    query_start_loc = self.input_buffers.query_start_loc.np  # ← 获取 query_start_loc

    for i, req_id in enumerate(input_batch.req_ids):
        if not needs_prompt_logprobs[i]:
            continue

        # ← 使用 query_start_loc 提取每个请求的 tokens
        start_idx = query_start_loc[i]
        end_idx = query_start_loc[i + 1]
        assert start_idx < end_idx, (
            f"start_idx ({start_idx}) >= end_idx ({end_idx})"
        )

        # 提取请求 i 的 logprobs
        logprobs = LogprobsTensors(
            logprob_token_ids=prompt_token_ids[start_idx:end_idx],
            logprobs=prompt_logprobs[start_idx:end_idx],
            selected_token_ranks=prompt_ranks[start_idx:end_idx],
        )
```

**作用**: 使用 `query_start_loc` 从扁平化的张量中提取每个请求的 logprobs。

---

### 7. **CUDA Graph 捕获准备**

**文件**: `vllm/v1/worker/gpu/cudagraph_utils.py:231-251`

```python
def prepare_inputs_to_capture(
    input_buffers,
    num_reqs: int,
    num_tokens_per_req: int,
    num_tokens: int,
):
    query_start_loc = input_buffers.query_start_loc  # ← 获取 query_start_loc

    # 为 CUDA Graph 设置固定的 query_start_loc
    query_start_loc.np[: num_reqs + 1] = np.arange(num_reqs + 1) * num_tokens_per_req
    query_start_loc.np[num_reqs:] = num_tokens  # ← 填充
    query_start_loc.copy_to_gpu()

    # 构建注意力元数据
    attn_metadata = build_attn_metadata(
        attn_metadata_builders=attn_metadata_builders,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc.gpu[: num_reqs + 1],  # ← 传递
        query_start_loc_cpu=query_start_loc.cpu[: num_reqs + 1],
        # ... 其他参数
    )
```

**作用**: 为 CUDA Graph 捕获准备固定的 `query_start_loc`，确保每次执行的张量形状相同。

---

### 8. **EAGLE 投机解码**

**文件**: `vllm/v1/worker/gpu/spec_decode/eagle.py:142-171`

```python
class EAGLE:
    def prepare_inputs_padded(
        self,
        input_batch: InputBatch,
    ) -> None:
        query_start_loc = self.input_buffers.query_start_loc.gpu[: num_reqs + 1]

        # 使用 query_start_loc 计算 slot mappings
        if self.block_tables is not None:
            self.block_tables.compute_slot_mappings(query_start_loc, pos)
```

**作用**: 在 EAGLE 投机解码中使用 `query_start_loc` 进行 KV 缓存管理。

---

## query_start_loc 的关键使用模式总结

基于实际代码分析，`query_start_loc` 的主要用途包括：

| 使用场景 | 文件位置 | 作用 |
|---------|---------|------|
| **采样索引计算** | `gpu_model_runner.py:1466` | 提取每个请求的最后一个 token 索引 |
| **注意力元数据构建** | `gpu_model_runner.py:1591`<br>`attn_utils.py:168` | 传递给注意力内核 |
| **KV 缓存映射** | `block_table.py:166` | 计算 slot mapping |
| **Prefill Token 复制** | `input_batch.py:165-166` | Triton 内核中确定写入位置 |
| **位置计算** | `input_batch.py:228-229` | 计算序列长度和位置 |
| **Prompt Logprobs** | `model_runner.py:713-714` | 提取每个请求的 logprobs |
| **CUDA Graph 准备** | `cudagraph_utils.py:232` | 设置固定形状 |
| **投机解码** | `eagle.py:143` | EAGLE 算法中的 KV 缓存管理 |

## 与其他数据结构的关系

### 与 seq_lens 的配合

```python
# seq_lens: 每个请求的序列长度
seq_lens = [102, 105, 103]

# query_start_loc: 每个请求的起始位置
query_start_loc = [0, 2, 7, 10]

# 关系：
# seq_lens[i] = query_start_loc[i + 1] - query_start_loc[i]
# 例如：seq_lens[1] = 7 - 2 = 5 ✓
```

### 与 cu_num_tokens 的关系

```python
# cu_num_tokens 是 query_start_loc 的来源
cu_num_tokens = np.cumsum(num_scheduled_tokens)

query_start_loc = [0] + cu_num_tokens.tolist()
```

## 性能优化意义

### 1. **O(1) 查找复杂度**

```python
# 直接索引查找，无需遍历
start = query_start_loc[i]
end = query_start_loc[i + 1]
```

### 2. **支持向量化操作**

```python
# 一次性获取所有请求的最后一个 token
logits_indices = query_start_loc[1:] - 1
```

### 3. **CUDA Graph 兼容**

```python
# 填充确保形状固定
padded_query_start_loc = [
    0, 2, 7, 10,  # 实际数据
    10, 10, 10, 10  # 填充数据
]
```

## 注意事项

### 1. **非递减要求**

```python
# FlashAttention 要求非递减
# 正确：[0, 2, 7, 10, 10, 10]  ✓
# 错误：[0, 2, 7, 10, 0, 0]    ✗

# 因此需要填充：
self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
```

### 2. **边界检查**

```python
# 提取时确保不越界
num_reqs = len(query_start_loc) - 1
for i in range(num_reqs):
    start = query_start_loc[i]
    end = query_start_loc[i + 1]
    assert start <= end, f"Invalid range: [{start}, {end})"
```

### 3. **CUDA Graph 模式**

```python
# 在 CUDA Graph 模式下，使用填充后的版本
if cudagraph_mode == CUDAGraphMode.FULL:
    # 使用填充的 query_start_loc
    query_start_loc_padded = self.query_start_loc.gpu[:max_num_reqs + 1]
else:
    # 使用实际的 query_start_loc
    query_start_loc_padded = self.query_start_loc.gpu[:num_reqs + 1]
```

## 总结

`query_start_loc` 是 vLLM 批处理推理的**核心数据结构**之一：

### 核心作用

1. **快速定位**: O(1) 时间复杂度定位每个请求的 token
2. **批处理支持**: 将多个请求的 tokens 组织在一个扁平化数组中
3. **性能优化**: 支持向量化操作和 CUDA Graph
4. **内存效率**: 避免动态形状计算和内存分配

### 关键特性

- ✅ **累积和数组**: 通过 `np.cumsum` 高效计算
- ✅ **固定形状**: 填充支持 CUDA Graph
- ✅ **非递减**: 满足 FlashAttention 等内核要求
- ✅ **GPU 友好**: 异步传输，向量化操作

### 在 vLLM 中的地位

`query_start_loc` 与 `positions`、`seq_lens`、`slot_mapping` 一起构成了 vLLM 推理的四大核心数据结构，是实现高性能 LLM 推理的基础设施。

```python
# vLLM 推理的核心数据结构
核心数据结构 = {
    "positions": token位置信息,          # 用于 RoPE
    "query_start_loc": 请求起始位置,     # 用于批处理
    "seq_lens": 序列长度,               # 用于注意力
    "slot_mapping": KV缓存映射          # 用于 PagedAttention
}
```

理解 `query_start_loc` 是理解 vLLM 如何高效处理动态批处理请求的关键！
