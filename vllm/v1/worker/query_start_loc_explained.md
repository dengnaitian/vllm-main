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

### 计算过程（行 1411-1417）

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

## 使用场景

### 1. **提取每个请求的 token**

```python
# 提取请求 i 的 tokens
i = 1  # 第二个请求
start = query_start_loc[i]       # 2
end = query_start_loc[i + 1]     # 7

request_tokens = input_ids[start:end]  # tokens[2:7]
```

### 2. **计算 logits_indices（用于采样）**

```python
# 常规情况：每个请求的最后一个 token
logits_indices = query_start_loc[1:] - 1
# 例如：[2, 7, 10] - 1 = [1, 6, 9]

# 这些索引用于从 logits 中采样
sampled_logits = logits[logits_indices]
```

**代码位置**: `gpu_model_runner.py:1466`

### 3. **注意力计算中的使用**

在 FlashAttention 等高效注意力实现中，`query_start_loc` 用于：

```python
# FlashAttention 需要：
# 1. query_start_loc: 每个请求的起始位置
# 2. seq_lens: 每个请求的序列长度
# 3. 扁平化的 Q, K, V 张量

# 伪代码示例
for i in range(num_reqs):
    start = query_start_loc[i]
    end = query_start_loc[i + 1]

    # 处理请求 i 的 queries
    q_i = q[start:end]
    k_i = k[start:end]
    v_i = v[start:end]

    # 计算注意力
    attn_output[i] = attention(q_i, k_i, v_i)
```

### 4. **KV 缓存管理**

```python
# 为每个请求分配 KV 缓存空间
for i in range(num_reqs):
    start = query_start_loc[i]
    end = query_start_loc[i + 1]
    num_tokens = end - start

    # 分配物理缓存块
    alloc_kv_cache(request_id[i], num_tokens)
```

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

### 1. **避免动态形状计算**

```python
# 不好的做法：每次都计算
for i in range(num_reqs):
    if i == 0:
        start = 0
    else:
        start = sum(num_scheduled_tokens[:i])
    end = start + num_scheduled_tokens[i]

# 好的做法：使用预计算的 query_start_loc
for i in range(num_reqs):
    start = query_start_loc[i]
    end = query_start_loc[i + 1]
```

### 2. **支持向量化操作**

```python
# 一次性获取所有请求的最后一个 token
last_tokens = input_ids[query_start_loc[1:] - 1]

# 向量化计算每个请求的平均表示
# (伪代码)
mean_hidden = []
for i in range(num_reqs):
    start, end = query_start_loc[i], query_start_loc[i + 1]
    mean_hidden.append(hidden_states[start:end].mean(0))
```

### 3. **CUDA Graph 兼容**

```python
# 填充确保形状固定
padded_query_start_loc = [
    0, 2, 7, 10,  # 实际数据
    10, 10, 10, 10  # 填充数据
]

# 这样可以在 CUDA Graph 中重用相同形状的张量
```

## 实际应用示例

### 示例 1：批处理中的请求提取

```python
# 场景：3 个用户同时发送请求
requests = [
    {"user": "Alice", "text": "Hello", "tokens": 2},
    {"user": "Bob", "text": "How are you?", "tokens": 5},
    {"user": "Charlie", "text": "Hi!", "tokens": 3}
]

# 计算后的 query_start_loc
query_start_loc = [0, 2, 7, 10]

# 提取 Bob 的请求
bob_start = query_start_loc[1]  # 2
bob_end = query_start_loc[2]    # 7
bob_tokens = input_ids[2:7]     # Bob 的 5 个 tokens
```

### 示例 2：分块 Prefill

```python
# 场景：长 prompt 被分成多块处理

# 第1次迭代：处理前 10 个 tokens
num_scheduled_tokens = [10, 0, 0]
query_start_loc = [0, 10, 10, 10]

# 第2次迭代：处理后续 15 个 tokens
num_scheduled_tokens = [15, 0, 0]
query_start_loc = [0, 15, 15, 15]

# 使用 query_start_loc 正确拼接结果
full_tokens = torch.cat([
    input_ids_1[0:10],
    input_ids_2[0:15]
])
```

### 示例 3：动态批处理

```python
# 场景：不同时间到达的请求

# 第1轮：2 个请求
num_scheduled_tokens = [5, 3]
query_start_loc = [0, 5, 8]

# 第2轮：1 个新请求 + 1 个继续生成的请求
num_scheduled_tokens = [1, 4]  # 第1个请求生成1个token，新请求4个tokens
query_start_loc = [0, 1, 5]

# query_start_loc 使我们能够高效处理这种动态变化
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
