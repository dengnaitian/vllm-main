# vLLM Model Runner `prepare_inputs` 函数分析

## 概述

`prepare_inputs` 函数是 vLLM 模型推理引擎中的关键预处理函数，负责在每次模型前向传播之前准备所有必要的输入张量和元数据。该函数位于 `vllm/v1/worker/gpu_model_runner.py` 中的 `GpuModelRunner` 类。

**文件位置**: `vllm/v1/worker/gpu_model_runner.py:1294-1515`

## 主要功能

`_prepare_inputs` 函数的核心职责包括：

1. **输入 Token 准备** - 从调度输出中提取和组织输入 token IDs
2. **位置编码计算** - 计算每个 token 的位置信息
3. **注意力元数据构建** - 准备注意力机制所需的元数据
4. **KV 缓存管理** - 计算和提交槽位映射（slot mapping）
5. **投机解码支持** - 处理投机解码的 draft tokens
6. **LoRA 支持** - 处理 LoRA 模型的热切换

## 函数签名

```python
def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[
    torch.Tensor,
    SpecDecodeMetadata | None,
]:
```

### 参数说明

- **scheduler_output**: 调度器的输出，包含当前批次中所有请求的调度信息
- **num_scheduled_tokens**: NumPy 数组，表示每个请求调度的 token 数量

### 返回值

返回一个元组，包含：
- **logits_indices**: 需要采样的 token 的索引（用于从 logits 中采样）
- **spec_decode_metadata**: 投机解码的元数据（如果没有使用投机解码则为 None）

## 详细执行流程

### 1. 初始化和优化（行 1307-1314）

```python
total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
assert total_num_scheduled_tokens > 0
num_reqs = self.input_batch.num_reqs
assert num_reqs > 0

# 优化：尽早开始复制块表，以便与后续 CPU 操作重叠
self.input_batch.block_table.commit_block_table(num_reqs)
```

**关键点**：
- 提前提交块表复制操作，利用异步传输与 CPU 计算重叠
- 这是性能优化的重要手段，减少 GPU 等待时间

### 2. 计算请求索引和累积令牌数（行 1316-1322）

```python
# 获取请求索引
# 例如：[2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

# cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
# arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
```

**作用**：
- `req_indices`: 将每个请求的 token 展平到一维数组，标记每个 token 属于哪个请求
- `cu_num_tokens`: 累积和数组，用于确定每个请求在批次中的起始和结束位置
- `arange`: 每个请求内部的相对位置索引

### 3. 计算位置信息（行 1324-1340）

```python
# 获取位置
positions_np = self.positions.np[:total_num_scheduled_tokens]
np.add(
    self.input_batch.num_computed_tokens_cpu[req_indices],
    arange,
    out=positions_np,
)

# 计算 M-RoPE 位置（仅对使用 M-RoPE 的模型，如 Qwen2-VL）
if self.uses_mrope:
    self._calc_mrope_positions(scheduler_output)

# 计算 XD-RoPE 位置（仅对使用 XD-RoPE 的模型，如 HunYuan-VL）
if self.uses_xdrope_dim > 0:
    self._calc_xdrope_positions(scheduler_output)
```

**关键点**：
- 位置 = 已计算的 token 数 + 相对位置
- 支持多种位置编码（RoPE、M-RoPE、XD-RoPE）
- M-RoPE（Multimodal Rotary Positional Embedding）用于视觉-语言模型
- XD-RoPE（Cross-dimensional RoPE）用于某些特定的多模态模型

### 4. 获取 Token IDs（行 1342-1405）

```python
# 获取 token 索引
token_indices = (
    positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
)
token_indices_tensor = torch.from_numpy(token_indices)

# 使用 torch.index_select 提取 token IDs
torch.index_select(
    self.input_batch.token_ids_cpu_tensor.flatten(),
    0,
    token_indices_tensor,
    out=self.input_ids.cpu[:total_num_scheduled_tokens],
)
```

**作用**：
- 根据 token 在全局缓冲区中的位置提取实际的 token IDs
- 使用 `torch.index_select` 而不是 `np.take`，因为在大张量上更快

**支持 Prompt Embeddings**（行 1360-1405）：
```python
if self.enable_prompt_embeds:
    is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
    torch.index_select(
        is_token_ids,
        0,
        token_indices_tensor,
        out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
    )

# 复制 prompt embeddings 到预分配的张量中
if self.input_batch.req_prompt_embeds:
    # 为每个请求复制相应的 embeddings
    output_idx = 0
    for req_idx in range(num_reqs):
        num_sched = num_scheduled_tokens[req_idx]
        # ... 复制逻辑
```

### 5. 计算槽位映射（行 1407-1408）

```python
self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
```

**关键点**：
- 计算每个 token 在 KV 缓存中的物理存储位置
- 这是 PagedAttention 的核心机制，将逻辑位置映射到物理缓存块

### 6. 准备注意力元数据（行 1410-1434）

```python
# 准备注意力元数据
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
# 填充以使其非递减（FlashAttention 等内核要求）
self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
self.query_start_loc.copy_to_gpu()
query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

self.seq_lens.np[:num_reqs] = (
    self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
)
# 填充未使用的请求（用于 CUDA Graph）
self.seq_lens.np[num_reqs:].fill(0)
self.seq_lens.copy_to_gpu()

# 记录哪些请求不应该被采样（分块 prefills 的情况）
self.discard_request_mask.np[:num_reqs] = (
    self.seq_lens.np[:num_reqs] < num_tokens_np
)
self.discard_request_mask.copy_to_gpu(num_reqs)
```

**关键数据结构**：
- **query_start_loc**: 每个请求在批次中的起始位置（累积和）
- **seq_lens**: 每个请求的序列长度
- **discard_request_mask**: 标记哪些请求是分块 prefill，不应该采样

### 7. 准备输入 IDs（行 1436-1457）

```python
# 调用 _prepare_input_ids 处理异步调度的情况
self._prepare_input_ids(
    scheduler_output,
    total_num_scheduled_tokens,
    cu_num_tokens,
)

# 复制位置信息到 GPU
if self.uses_mrope:
    self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
        self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
        non_blocking=True,
    )
elif self.uses_xdrope_dim > 0:
    self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
        self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
        non_blocking=True,
    )
else:
    # 常见情况（一维位置）
    self.positions.copy_to_gpu(total_num_scheduled_tokens)
```

**关键点**：
- `_prepare_input_ids` 处理异步调度的特殊情况
- 使用 `non_blocking=True` 实现异步传输，提高性能

### 8. 处理投机解码（行 1459-1500）

```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
if not use_spec_decode:
    # 常规情况（无投机解码）
    logits_indices = query_start_loc[1:] - 1
    num_draft_tokens = None
    spec_decode_metadata = None
    num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
else:
    # 投机解码情况
    num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
    num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
    for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
        req_idx = self.input_batch.req_id_to_index[req_id]
        num_draft_tokens[req_idx] = len(draft_token_ids)
        # ... 处理 draft tokens

    spec_decode_metadata = self._calc_spec_decode_metadata(
        num_draft_tokens, cu_num_tokens
    )
    logits_indices = spec_decode_metadata.logits_indices
    num_sampled_tokens = num_draft_tokens + 1
```

**投机解码说明**：
- 投机解码是一种性能优化技术，使用小模型快速生成多个候选 token
- 然后使用大模型并行验证这些候选 token
- 显著提高推理吞吐量

### 9. LoRA 模型热切换（行 1502-1510）

```python
# Hot-Swap lora 模型
if self.lora_config:
    assert (
        np.sum(num_sampled_tokens)
        <= self.vllm_config.scheduler_config.max_num_batched_tokens
    )
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

**LoRA 说明**：
- LoRA（Low-Rank Adaptation）是一种参数高效的微调方法
- 支持在同一批处理中混合多个不同的 LoRA 适配器
- 热切换机制允许动态切换不同的 LoRA 模型

## 辅助函数：`_prepare_input_ids`

该函数处理输入 IDs 的准备，特别是异步调度的情况（行 1138-1258）。

### 主要逻辑

1. **正常调度情况**（行 1150-1156）：
```python
if self.input_batch.prev_sampled_token_ids is None:
    # 正常调度情况
    self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
    if self.enable_prompt_embeds:
        self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
        self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
    return
```

2. **异步调度情况**（行 1158-1258）：
   - 处理来自上一次迭代的采样 token
   - 使用 scatter 操作将 token 放到正确的位置
   - 处理 draft tokens 的散布

**关键优化**（行 1210-1221）：
```python
if indices_match and max_flattened_index == (num_commmon_tokens - 1):
    # 常见情况优化：批次未改变且没有重排序
    self.input_ids.gpu[:num_commmon_tokens].copy_(
        self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0],
        non_blocking=True,
    )
    return
```

## 在 `execute_model` 中的调用

在 `execute_model` 函数中（行 2958-3154），`_prepare_inputs` 在预处理阶段被调用：

```python
with record_function_or_nullcontext("gpu_model_runner: preprocess"):
    with self.synchronize_input_prep():
        # 更新持久化批次状态
        self._update_states(scheduler_output)

        # 准备输入
        (
            logits_indices,
            spec_decode_metadata,
        ) = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens_np,
        )

        # ... 构建注意力元数据
```

## 性能优化要点

1. **异步传输重叠**：
   - 尽早启动块表复制，与 CPU 计算重叠
   - 使用 `non_blocking=True` 进行 GPU 传输

2. **内存预分配**：
   - 使用预分配的缓冲区避免动态内存分配
   - 减少 GPU 内存分配开销

3. **批量操作**：
   - 使用 NumPy 和 PyTorch 的向量化操作
   - 避免 Python 循环

4. **CUDA Graph 支持**：
   - 填充数据以支持 CUDA Graph
   - 确保张量形状不变

5. **投机解码优化**：
   - 支持并行验证多个 draft tokens
   - 提高吞吐量而不增加太多计算开销

## 总结

`_prepare_inputs` 函数是 vLLM 高性能推理的关键组件之一，它：

1. **高效组织输入数据**：将调度输出转换为模型所需的格式
2. **支持多种高级特性**：投机解码、LoRA、多模态输入等
3. **优化性能**：通过异步传输、内存预分配、批量操作等技术
4. **灵活适配**：支持不同的位置编码、注意力机制和调度策略

该函数的设计体现了 vLLM 在性能和灵活性方面的平衡，是实现高吞吐量 LLM 推理的重要基础。
