AI Infra 公开课设计
一、思路
你现在的结构是：
公域讲 vLLM → 建立专业度 → 引流 → 付费课做一个 vLLM 推理优化项目
方向是对的。
因为：
- 公域解决“信任”
- 付费课解决“能力”
- 项目解决“面试转化”
这是一个标准的专家型知识产品路径。

---
🎯 课程核心项目定义
基于 vLLM 真实源码进行性能分析与改造，并设计可扩展优化方案。
更具体一点：
完成一个 LLM 推理系统性能优化与架构升级项目。
这个项目最终要形成：
- 一份可运行的优化代码
- 一份性能对比报告
- 一份系统设计升级方案
- 一份面试讲稿结构
二、课程主线结构（项目驱动）
课程不再按“知识章节”讲。
改为按“项目推进阶段”讲。

---
阶段 1：核心机制拆解（基础知识）
目标：
理解 vLLM 的关键模块
内容：
- BlockManager 机制
- PagedAttention 设计
- Continuous batching 调度流程
- PD 分离路径
输出：
- 一张完整系统架构图
- 一次完整请求生命周期 tracing
面试映射：
“vLLM 系统的核心流程组件”

---
阶段 2：建立 Baseline（问题发现阶段）
目标：
明确推理系统真正的性能瓶颈
内容：
- 跑 vLLM baseline
- 分析 prefill / decode 性能差异
- 使用 profiler 定位瓶颈
- 分析 KV cache 占用
输出：
- 一份 baseline 性能报告，例如时延过高，吞吐太低，内存占用过多。PD传输太慢。动态上车问题。（理论分析和实际的差距）
- 学生理解真实瓶颈
面试映射：
“你是如何定位系统瓶颈的？”

---
阶段 3：真实优化改造
目标：
对 vLLM 进行实际改造
可以选择几个改造点：
- 调度参数优化
- Block 分配策略优化
- Decode 阶段 micro-batching 优化
- KV eviction 策略实验
输出：
- 优化代码版本
- 性能对比（吞吐 / 延迟 / 显存）
面试映射：
“你做过哪些优化？效果如何？”

---
阶段 4：架构升级设计
目标：
提出比 vLLM 更可扩展的设计
例如：
- 超长上下文支持
- MoE 模型优化
- 多 GPU KV 管理改进
- PD 分离瓶颈分析
这部分不一定全部实现，但必须设计。
输出：
- 一份架构改进文档
- 设计权衡分析
面试映射：
“如果规模扩大 10 倍怎么办？”

---
阶段 5：表达重构
目标：
把项目转化为面试竞争力
内容：
- 如何讲项目背景
- 如何讲瓶颈
- 如何讲优化路径
- 如何讲 trade-off
- 如何回答追问
输出：
- 一份项目讲稿结构模板
- 常见追问清单

---
三、课程最终结构可以是这样
模块 1：推理系统整体认知
模块 2：vLLM 关键机制
模块 3：性能分析方法论
模块 4：优化项目池（任选）
模块 5：架构升级设计
模块 6：面试表达训练
这就是一个完整产品。

---
四、课程最终交付物（必须清晰）
1. GitHub 项目仓库
2. baseline 与优化版本
3. profiling 报告模板
4. 架构设计文档
5. 面试讲稿模板
这才是“高转化产品”。

---
五、公开课的设计
记住一个核心原则：
公开课 ≠ 讲知识
 公开课 = 制造差距 + 展示高度 + 引出解决方案
公开课不是教完内容，而是：
- 让人意识到自己不够强
- 让人看到你真的懂
- 让人相信”这门课能让我涨薪”

---

## 公开课主题选择（三个候选）

### 方案 A：《从源码剖析：为什么你的 vLLM 吞吐只有理论值的 30%？》
**核心钩子**：大部分人跑 vLLM 都没跑出官方性能

**优势**：
- 痛点直接（性能问题）
- 数据有冲击力
- 展示深度分析能力

### 方案 B：《大厂面试必问：vLLM 的 PagedAttention 到底解决了什么问题？》
**核心钩子**：面试必问，但大部分人答不深

**优势**：
- 面试焦虑驱动
- 展示对原理的深度理解
- 转化路径清晰

### 方案 C：《一个 Bug 的发现：从 vLLM 源码中找到的性能优化机会》
**核心钩子**：真实发现的优化点

**优势**：
- 故事性强
- 展示实战能力
- 有 uniqueness

**推荐**：方案 A + C 结合，既有数据冲击，又有故事性

---

## 公开课完整流程设计（90 分钟）

### 第一部分：开场钩子（10 分钟）

**1. 数据冲击（5 分钟）**
```
“大家看这张图：
- vLLM 官方宣称：Llama-2-13B 可达 3000+ tokens/s
- 但大部分公司实际跑：只有 800-1200 tokens/s
- 为什么差了 3 倍？”
```

**展示内容**：
- 一张性能对比图（官方 vs 实际）
- 列举 3 个真实案例（可匿名）
- 抛出问题：”是你的模型问题？还是你的配置问题？还是 vLLM 本身的问题？”

**2. 自我介绍（3 分钟）**
```
“我是 xx，做过 xx 项目的推理优化
把某司的吞吐从 500 提到 2000+
这个过程踩了很多坑，也读懂了 vLLM 的源码
今天把这些发现分享给大家”
```
- 用数据说话
- 不吹牛，讲具体结果
- 暗示”我有实战经验”

**3. 公开课承诺（2 分钟）**
```
“今天我会带你：
1. 用 5 分钟看懂 vLLM 的核心机制
2. 用一个真实案例展示性能瓶颈的定位过程
3. 给你一套可以自己用的分析方法论

听完这节课，你会：
- 知道自己的系统慢在哪里
- 能用 profiler 分析瓶颈
- 知道优化的方向”
```

---

### 第二部分：制造差距（20 分钟）

**目标**：让听众意识到”我以为我懂，但其实我不懂”

**1. 快速测试（5 分钟）**
```
“在开始之前，我先问 3 个问题：
1. vLLM 的 continuous batching，一次调度最多能上多少个 request？
2. PagedAttention 解决的核心问题是什么？不是'省内存'
3. 为什么 decode 阶段 batch size 太大会反而变慢？

答对的同学打 1，答不出来的打 2”
```
- 大部分人答不出来
- 制造”我不会”的焦虑

**2. 答案 + 深度解析（15 分钟）**

**问题 1 的深度解析**：
```
“大部分人会答：受 GPU 显存限制
错！真正限制的是：
- scheduler 的调度延迟
- block manager 的分配效率
- KV cache 的碎片化程度

我看过真实案例：
- 显存只用了 60%
- 但因为 scheduler latency，batch size 上不去
- 这个瓶颈用 nsys 一测就能看到”

[展示一张 nsys 的火焰图]
```

**问题 2 的深度解析**：
```
“PagedAttention 不是'省内存'
是'让内存可以动态分配'

传统方法：
- 每个 sequence 预分配连续 KV
- 导致 60%+ 内存浪费
- 无法动态上下车

PagedAttention：
- KV 分块存储（像操作系统分页）
- 可以上下车的关键在这里
- 但代价是：多了 memory copy 的开销”

[展示一张架构对比图]
```

**问题 3 的深度解析**：
```
“Decode 阶段 batch size 太大反而慢？
原因：
1. kernel 启动开销 vs 计算密度的 trade-off
2. memory bandwidth 限制
3. KV cache 读取的 cache locality

我测过真实数据：
- batch size = 1: 120 tokens/s/GPU
- batch size = 16: 1800 tokens/s/GPU
- batch size = 64: 1500 tokens/s/GPU  ← 反而下降！

瓶颈在哪？
- 不是 compute
- 是 memory bandwidth + kernel overhead”

[展示一张性能曲线图]
```

**这一部分的关键**：
- 展示你知道的”更深”
- 用真实数据支撑
- 让听众意识到”表面的理解不够”

---

### 第三部分：展示高度（30 分钟）

**目标**：展示真实的分析能力 + 源码级理解

**1. 真实案例：从 800 到 2000 的优化过程（15 分钟）**

**案例背景**：
```
“某公司找我咨询：
- 场景：Llama-2-13B 在线推理
- 配置：4x A100 (80G)
- 当前性能：800 tokens/s
- 目标：2000+ tokens/s

他们已经试过：
- 调大 batch size → 没用
- 升级 vLLM 版本 → 没用
- 换 GPU → 太贵

问题在哪？”
```

**分析过程（展示专业度）**：
```
“Step 1: 先跑 baseline（10 分钟）
```
[展示命令]
```bash
vllm serve llama-2-13b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

“用 nsys profile：
```
[展示火焰图，重点标注]
- 红：prefill 阶段（占用 60% 时间）
- 黄：scheduler latency（占用 20% 时间）
- 绿：decode kernel（占用 20% 时间）

问题发现了：
- decode kernel 没吃满
- scheduler 太慢
- prefill 太频繁（请求太多小请求）
```

**Step 2: 定位瓶颈（5 分钟）**
```
“为什么 scheduler 这么慢？
看源码（展示代码）：
```
[展示 vllm/core/scheduler.py 关键代码]
```python
def schedule(self):
    # 问题：每次调度都要遍历所有 running requests
    for req in self.running:
        if req.completed():
            self.free(req.blocks)  # ← 这里有锁竞争！

    # 问题：block 分配是线性的
    for req in waiting_requests:
        blocks = self.block_manager.allocate(req.num_blocks)
        # ← 这里是 O(n) 的
```

“瓶颈找到了：
1. block_manager.allocate 有锁竞争
2. scheduler 是单线程的
3. 没有优先级调度（长请求占着资源）”
```

**Step 3: 优化方案（10 分钟）**
```
“方案 1：调整调度策略（不改代码）
```
[展示配置]
```python
--max-num-batched-tokens 4096  # 限制一次上车的 tokens
--max-num-seqs 128             # 限制并发序列数
```

“效果：1100 tokens/s（+37%）”

“方案 2：改源码 - 优化 block 分配（简单改）
```
[展示代码改动]
- 用 freelist 管理 free blocks
- 分配从 O(n) 降到 O(1)

“效果：1400 tokens/s（+75%）”

“方案 3：架构调整 - PD 分离（大改）
```
[展示架构图]
- prefill 用独立的 GPU pool
- decode 用独立的 GPU pool
- 中间用 KV cache 传输

“效果：2200 tokens/s（+175%）”
```

**这一部分的关键**：
- 展示完整的分析流程
- 展示源码级理解
- 展示渐进式优化路径
- 数据说话

---

### 第四部分：引出解决方案（15 分钟）

**1. 总结方法论（5 分钟）**
```
“今天我们看到的不是孤立的技巧
是一套完整的推理系统优化方法论：

第一步：建立 baseline
- 用标准 workload
- 记录各项指标

第二步：定位瓶颈
- 用 profiler（nsys / pytorch profiler）
- 看 flame graph
- 找到 top 3 耗时点

第三步：设计优化
- 从配置开始（成本低）
- 再改代码（中等成本）
- 最后改架构（高成本）

第四步：验证效果
- 对比 baseline
- 分析 trade-off
- 文档化”

[展示一张流程图]
```

**2. 现场演示（5 分钟）**
```
“现在我现场演示：
- 打开 vLLM 源码
- 找到 scheduler 的代码
- 用 30 秒定位到瓶颈位置
- 展示 profiler 的使用”

[实际操作，展示熟练度]
```

**3. 转化钩子（5 分钟）**
```
“今天我只讲了案例的 20%
完整的优化过程还包括：
- KV cache 的 eviction 策略
- Multi-GPU 的通信优化
- 不同模型的针对性调优
- 生产环境的稳定性保障

更重要的：
- 这些内容怎么转化成面试亮点
- 怎么讲这个故事
- 怎么应对追问

如果你想要：
- 一个完整的 vLLM 优化项目
- 可以写在简历上的实战经验
- 一套能复用的方法论

可以考虑我的系统课：
- 6 周时间
- 从源码到优化到架构设计
- 最后给你一个能讲的项目

今天到场的同学，有优惠：
- 原价 xxx
- 现在报名 xxx
- 送一对一简历修改”
```

---

### 第五部分：Q&A（15 分钟）

**设计好的问题（从学员中安排）**：
```
Q1: “老师，请问 vLLM 和 TensorRT-LLM 有什么区别？”
A: 展示对不同框架的理解

Q2: “老师，我是做后端的，转 AI Infra 难吗？”
A: 展示对职业路径的理解，给信心

Q3: “老师，你的系统课需要什么基础？”
A: 降低门槛，说明可学习性
```

**真实问题应对**：
- 准备 10 个常见问题的标准答案
- 展示专业度 + 服务意识
- 每个回答都悄悄展示”我很懂”

---

## 公开课的获客转化设计

### 转化漏斗

```
1000 人看公开课宣传
    ↓
500 人报名公开课
    ↓
300 人实际参加
    ↓
50 人咨询系统课
    ↓
20 人最终报名
```

### 转化率提升策略

**1. 报名阶段 - 提高报名率**

**策略 A：制造稀缺性**
```
“本次公开课限额 500 人
- 已报名 342 人
- 仅剩 158 个名额
- 先到先得”
```

**策略 B：承诺价值**
```
“参加公开课，你将获得：
- vLLM 性能分析 checklist（PDF）
- profiler 使用模板（Jupyter）
- 系统课 500 元优惠券”
```

**策略 C：降低门槛**
```
“只需要：
- 会 Python
- 了解基本的深度学习概念
- 不需要 vLLM 经验”
```

**2. 参加阶段 - 提高完课率**

**策略 A：提前发资料**
```
“提前发给大家：
- vLLM 源码导航图
- 常用命令清单
- 这样大家可以边听边看”
```

**策略 B：互动设计**
```
“中途提问：
- '这个懂的打 1'
- '有疑问的打 ?'
- 保持参与感”
```

**策略 C：抽奖环节**
```
“结束前抽奖：
- 一等奖：系统课半价（1 名）
- 二等奖：简历一对一修改（3 名）
- 三等奖：vLLM 学习资料包（10 名）

条件：全程参与 + 填写反馈表”
```

**3. 转化阶段 - 提高咨询率**

**策略 A：制造紧迫感**
```
“系统课优惠：
- 公开课结束后 24 小时内有效
- 仅限今天参加的同学
- 过期恢复原价”
```

**策略 B：降低决策风险**
```
“承诺：
- 第 1 节课不满意，全额退款
- 不满意可以随时退
- 剩下课时按比例退款”
```

**策略 C：展示成功案例**
```
“上期学员：
- A 同学：学完后拿到 xx 公司 offer
- B 同学：薪资从 20k 涨到 35k
- C 同学：从 Java 转到 AI Infra”
```

**4. 成交阶段 - 提高成交率**

**策略 A：一对一咨询**
```
“公开课后，我会留 30 分钟
- 可以单独聊你的情况
- 给你针对性建议
- 帮你判断课程是否适合”
```

**策略 B：分期付款**
```
“觉得一次付太多？
- 支持分期付款
- 首付 xxx
- 剩下的分 3 期”
```

**策略 C：早鸟优惠**
```
“前 10 名报名：
- 额外送简历修改
- 额外送模拟面试
- 额外送内推机会”
```

---

## 公开课的物料准备清单

### 必须准备的

**1. PPT（50-60 页）**
- 开场钩子（5 页）
- 差距制造（15 页）
- 案例展示（20 页）
- 方法论总结（10 页）
- 转化页面（5 页）

**2. 演示环境**
- vLLM 源码（提前打开）
- nsys / pytorch profiler（提前安装）
- 一个可以跑 demo 的环境

**3. 数据素材**
- 性能对比图
- 火焰图
- 架构图
- 代码对比图

**4. 赠送资料**
- vLLM 性能分析 checklist（PDF）
- profiler 使用模板（Jupyter）
- 源码导航图（PDF）

**5. 转化物料**
- 系统课介绍页
- 报名链接
- 优惠券码
- 成功案例整理

### 可选准备

**1. 录播**
- 提前录一个版本
- 防止现场出问题
- 可以二次传播

**2. 问卷**
- 课前问卷（了解背景）
- 课后问卷（收集反馈）

**3. 社群**
- 建公开课群
- 持续答疑
- 后续转化

---

## 公开课的宣传文案

### 标题方案（5 个）

1. 《从源码剖析：为什么你的 vLLM 吞吐只有理论值的 30%？》
2. 《大厂面试必问：vLLM 的 PagedAttention 到底解决了什么问题？》
3. 《一个真实案例：把 vLLM 性能从 800 提到 2000+ 的全过程》
4. 《90 分钟，带你读懂 vLLM 的核心机制 + 性能优化方法论》
5. 《AI Infra 工程师必会：vLLM 源码分析与性能调优实战》

### 宣传渠道

**1. 公众号 / 知乎**
- 发布深度文章（展示专业度）
- 文末插入公开课报名

**2. 掘金 / CSDN**
- 发布技术干货
- 吸引工程师群体

**3. 职场社群**
- 目标群：AI / 后端 / 架构师
- 痛点：涨薪 / 转岗 / 面试

**4. 朋友圈海报**
- 简洁有力
- 突出价值
- 明确时间

### 海报文案

```
【公开课】vLLM 源码分析与性能优化

🎯 你将学到：
✓ vLLM 核心机制（PagedAttention / Continuous Batching）
✓ 性能瓶颈定位方法（Profiler + 源码分析）
✓ 真实优化案例（800 → 2000+ tokens/s）

👤 适合人群：
- AI Infra 工程师
- 想转 AI Infra 的后端工程师
- 准备大模型面试的同学

⏰ 时间：xx 月 xx 日 20:00
📍 形式：线上直播
💰 费用：免费（限额 500 人）

扫码报名，领 vLLM 学习资料包
[二维码]
```

---

## 公开课后的二次传播

**1. 录播剪辑**
- 剪 3 个 5 分钟精华片段
- 发到 B 站 / 抖音 / 视频号
- 引导关注公众号

**2. 文章整理**
- 把公开课内容整理成文章
- 发到知乎 / 掘金
- 引导加微信

**3. 资料包传播**
- 公开课资料包可以分享
- 但要加”转发可领取”
- 扩散到更多社群

**4. 学员见证**
- 收集学员反馈
- 截图发朋友圈
- 增加信任度

---

## 六、vLLM 关键模块梳理（核心内容）

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         vLLM 推理框架架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       │
│  │   API 层      │      │   Engine 层   │      │  Executor 层  │       │
│  │              │      │              │      │              │       │
│  │ OpenAI API   │────→│ LLMEngine   │────→│ ModelRunner  │       │
│  │              │      │              │      │              │       │
│  └──────────────┘      └──────┬───────┘      └──────┬───────┘       │
│                               │                     │                │
│                        ┌──────▼───────┐     ┌──────▼───────┐        │
│                        │  Scheduler   │     │   Worker     │        │
│                        │              │     │              │        │
│                        │ 请求调度      │     │ 模型执行      │        │
│                        │ KV 管理       │     │ GPU 计算      │        │
│                        └──────────────┘     └──────────────┘        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 核心模块详解

#### 1. API 层 (vllm/entrypoints/)

**作用**：提供外部接口，兼容 OpenAI API

**关键文件**：
- `api_server.py` - 基础 API 服务器
- `openai/api_server.py` - OpenAI 兼容服务器
- `openai/serving_engine.py` - 请求路由引擎

**核心功能**：
```
用户请求 → OpenAI API Server
    ↓
解析请求（prompt, parameters）
    ↓
转发给 LLMEngine
    ↓
返回流式/非流式响应
```

**面试要点**：
- 如何支持流式响应？（SSE / Server-Sent Events）
- 如何处理并发请求？（AsyncLLMEngine）
- 请求队列如何管理？

---

#### 2. Engine 层 (vllm/engine/)

**作用**：推理引擎的核心，协调调度和执行

**关键文件**：
- `llm_engine.py` - 同步引擎主类
- `async_llm_engine.py` - 异步引擎封装
- `protocol.py` - 内部通信协议

**LLMEngine 核心方法**：
```python
class LLMEngine:
    def __init__(self):
        self.model_config = ...      # 模型配置
        self.cache_config = ...       # KV Cache 配置
        self.scheduler_config = ...   # 调度配置
        self.parallel_config = ...    # 并行配置

        # 核心组件
        self.scheduler = Scheduler(...)           # 调度器
        self.cache_engine = CacheEngine(...)      # 缓存引擎
        self.model_executor = ModelExecutor(...)  # 模型执行器
        self.block_manager = BlockManager(...)    # Block 管理器

    def step(self) -> List[RequestOutput]:
        """单步执行：调度 → 执行 → 更新"""
        # 1. 调度阶段
        scheduled_requests = self.scheduler.schedule()

        # 2. 准备输入
        model_input = self._prepare_model_input(scheduled_requests)

        # 3. 执行模型
        outputs = self.model_executor.execute_model(model_input)

        # 4. 更新 KV Cache
        self.scheduler.update(outputs)

        return outputs
```

**AsyncLLMEngine 包装**：
```
同步请求 → AsyncLLMEngine
    ↓
加入请求队列（RequestTracker）
    ↓
后台线程执行 LLMEngine.step()
    ↓
通过 Future 返回结果
```

**面试要点**：
- 为什么需要异步层？（避免阻塞 API 线程）
- 请求如何流转？（API → Engine → Executor → GPU）
- step() 方法的执行流程是什么？

---

#### 3. Scheduler 模块

**作用**：请求调度、KV Cache 管理、连续批处理

**关键概念**：

**a) Continuous Batching（连续批处理）**
```
传统 Static Batching：
┌─────────────────────────────────────┐
│ Batch 1: [Req1, Req2, Req3, Req4]   │  等待全部完成
│ Batch 2: [Req5, Req6, Req7, Req8]   │  等待全部完成
└─────────────────────────────────────┘
问题：Req1 完成，但要等 Req2/3/4 才能下一批

Continuous Batching：
┌─────────────────────────────────────┐
│ t1: [Req1, Req2, Req3, Req4]        │  Req1 完成
│ t2: [Req2, Req3, Req4, Req5]        │  Req2 完成，加入 Req5
│ t3: [Req3, Req4, Req5, Req6]        │  动态上下车
└─────────────────────────────────────┘
优势：请求可以随时上下车，GPU 利用率更高
```

**b) 请求状态管理**
```python
class Scheduler:
    def __init__(self):
        self.waiting = []      # 等待队列
        self.running = []      # 运行中
        self.completed = []    # 已完成

    def schedule(self) -> SchedulerOutputs:
        """调度核心逻辑"""
        # 1. 从 waiting 尝试加入 running
        #    - 检查是否有足够的 KV Cache blocks
        #    - 检查是否超过 max_num_seqs

        # 2. 从 running 中移除已完成的
        #    - 释放 KV Cache blocks
        #    - 移到 completed

        # 3. 决定本次 batch 的组成
        #    - prefill requests（新请求）
        #    - decode requests（进行中）

        return SchedulerOutputs(
            selected_requests=[...],
            blocks_to_copy=[...],
            blocks_to_swap=[...],
        )
```

**c) PagedAttention 的 Block 管理**
```python
class BlockAllocator:
    """KV Cache Block 分配器（类似操作系统分页）"""

    def allocate(self, seq: Sequence) -> List[Block]:
        """为 sequence 分配 blocks"""
        # 每个 sequence 的 KV cache 被切成固定大小的 blocks
        # blocks 可以不连续存储

    def free(self, seq: Sequence):
        """释放 sequence 的 blocks"""
        # blocks 回收到 freelist，供其他 sequence 使用
```

**面试要点**：
- Continuous Batching vs Static Batching 的区别？
- 如何决定一个请求能否上车？（available blocks + max_num_seqs）
- PagedAttention 为什么能提高显存利用率？

---

#### 4. CacheEngine & BlockManager

**作用**：管理 KV Cache 的显存分配

**关键数据结构**：
```python
# Block 是 KV Cache 的基本分配单位
Block = {
    'block_id': int,
    'physical_block_ids': List[int],  # 实际存储位置
    'ref_count': int,                  # 引用计数
}

# Sequence 的 KV Cache 由多个 Blocks 组成
Sequence = {
    'seq_id': int,
    'status': 'running' | 'completed',
    'blocks': List[Block],  # KV Cache blocks
    'num_tokens': int,
}
```

**BlockManager 核心功能**：
```python
class BlockManager:
    def allocate_context(self, seq: Sequence):
        """为新 sequence 分配 KV blocks"""
        num_blocks = seq.get_num_required_blocks()
        physical_blocks = self.allocator.allocate(num_blocks)
        seq.block_table.append(physical_blocks)

    def free_context(self, seq: Sequence):
        """释放 sequence 的 KV blocks"""
        for block in seq.block_table:
            self.allocator.free(block)

    def can_allocate(self, num_tokens: int) -> bool:
        """检查是否有足够的 blocks"""
        required_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        return self.allocator.get_free_blocks() >= required_blocks
```

**面试要点**：
- Block size 如何选择？（太小：管理开销大；太大：碎片化）
- 如何避免内存碎片？（freelist + 分配策略）
- KV Cache 占用多少显存？（2 * num_layers * num_heads * dim * seq_len * batch_size）

---

#### 5. ModelExecutor / ModelRunner

**作用**：执行模型前向传播

**关键文件**：
- `model_executor/model_executor.py` - 执行器接口
- `worker/model_runner.py` - 具体执行逻辑

**执行流程**：
```python
class ModelRunner:
    def execute_model(self, model_input: ModelInput) -> List[SamplerOutput]:
        """执行单步推理"""

        # 1. 准备 input_ids 和 positions
        input_ids = model_input.input_ids
        positions = model_input.positions  # 每个 token 的位置

        # 2. 准备 KV Cache
        kv_cache = self.cache_engine.get_kv_cache(model_input.seq_ids)

        # 3. 执行 Attention
        #    - PagedAttention kernel
        #    - 根据 block_table 读取 KV
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_cache,
            block_tables=model_input.block_tables,
        )

        # 4. 采样（获取下一个 token）
        sampler_output = self.sampler(
            hidden_states=hidden_states,
            sampling_params=model_input.sampling_params,
        )

        return sampler_output
```

**PagedAttention Kernel**：
```cpp
// 伪代码：PagedAttention 的核心
__global__ void paged_attention_kernel(
    const float* Q,           // Query [batch_size, num_heads, head_dim]
    const float* block_table, // 每个 seq 的 KV block 映射
    const float** KV_cache,   // KV Cache blocks
    float* output
) {
    // 1. 根据 block_table 找到对应的 KV blocks
    // 2. 从 blocks 中加载 K, V
    // 3. 计算 attention
    // 4. 写入 output

    // 关键：支持动态 block 访问，不需要连续内存
}
```

**面试要点**：
- prefill 和 decode 的区别？（prefill：一次处理多个 tokens；decode：每次处理 1 个 token）
- block_tables 如何传递给 CUDA kernel？（通过模型输入）
- 如何实现 Speculative Decoding？

---

#### 6. 采样模块 (Sampling)

**作用**：从 logits 中采样下一个 token

**关键文件**：
- `logits_process.py` - logits 后处理
- `sampling_params.py` - 采样参数

**采样流程**：
```python
def sample(logits, sampling_params):
    # 1. logits 处理
    logits = apply_temperature(logits, sampling_params.temperature)
    logits = apply_top_p(logits, sampling_params.top_p)
    logits = apply_top_k(logits, sampling_params.top_k)

    # 2. 采样
    if sampling_params.temperature == 0:
        # Greedy decoding
        next_token = argmax(logits)
    else:
        # Multinomial sampling
        probs = softmax(logits)
        next_token = multinomial(probs)

    # 3. 处理 stop sequences
    if should_stop(next_token, sampling_params.stop_sequences):
        return finish_token

    return next_token
```

**面试要点**：
- temperature、top_p、top_k 的作用？
- 如何处理 stop sequences？
- beam search 的实现？

---

#### 7. 分布式执行 (Multi-GPU)

**作用**：多 GPU 并行推理

**并行模式**：

**a) Tensor Parallelism（张量并行）**
```
┌─────────────┐     ┌─────────────┐
│   GPU 0     │     │   GPU 1     │
│             │     │             │
│  Q, K_0, V_0│     │  K_1, V_1   │  ← K, V 按头切分
│             │     │             │
└──────┬──────┘     └──────┬──────┘
       │ allreduce         │
       ←──────────────────→
```

**b) Pipeline Parallelism（流水线并行）**
```
Stage 0: Layer 0-11    Stage 1: Layer 12-23
┌─────────────┐        ┌─────────────┐
│   GPU 0     │  →     │   GPU 1     │
│             │  send  │             │
└─────────────┘        └─────────────┘
```

**c) Data Parallelism（数据并行，vLLM 不常用）**
```
每个 GPU 完整模型，处理不同请求
┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │
│ Req 1-4     │  │ Req 5-8     │
└─────────────┘  └─────────────┘
```

**Worker 通信**：
```python
class Worker:
    def execute_model(self, model_input):
        # 1. 如果是 tensor parallel，需要 allreduce attention output
        if self.parallel_config.tensor_parallel_size > 1:
            output = all_reduce(output)

        # 2. 如果是 pipeline parallel，需要发送给下一 stage
        if self.parallel_config.pipeline_parallel_size > 1:
            send_to_next_stage(output)

        return output
```

**面试要点**：
- Tensor Parallel 和 Pipeline Parallel 的区别？
- vLLM 默认使用哪种并行？（Tensor Parallel）
- 多 GPU 通信的开销如何优化？

---

### 关键数据流

**完整请求处理流程**：
```
1. 用户请求
   ↓
2. OpenAI API Server
   ↓
3. AsyncLLMEngine.add_request()
   ↓
4. RequestTracker（请求队列）
   ↓
5. LLMEngine.step()
   ├─→ Scheduler.schedule()          # 调度决策
   │   ├─ 检查哪些请求可以上车
   │   ├─ 分配 KV blocks
   │   └─ 返回 scheduled_requests
   │
   ├─→ CacheEngine.get_kv_cache()    # 准备 KV
   │   └─ 根据 block_table 组装 KV
   │
   ├─→ ModelExecutor.execute_model() # 执行模型
   │   ├─ PagedAttention forward
   │   ├─ 采样
   │   └─ 返回 outputs
   │
   └─→ Scheduler.update()            # 更新状态
       ├─ 更新 sequence 状态
       ├─ 释放完成的请求
       └─ 返回 RequestOutputs
   ↓
6. AsyncLLMEngine 返回结果
   ↓
7. API Server 返回给用户
```

---

### 性能优化要点

**1. 调度优化**
- 限制 max_num_seqs（避免 batch 太大）
- 限制 max_num_batched_tokens（避免 prefill 太长）
- 优先级调度（让短请求先完成）

**2. KV Cache 优化**
- 合适的 block size（通常 16）
- KV cache eviction（LRU / 其他策略）
- Prefix Caching（复用相同 prompt 的 KV）

**3. Kernel 优化**
- FlashAttention / FlashInfer
- PagedAttention kernel（减少 memory copy）
- CUDA Graph（减少 kernel launch 开销）

**4. 架构优化**
- PD 分离（Prefill / Decode 分离）
- KV cache offload（CPU / NVMe）
- Multi-Lora serving

---

### 面试高频问题清单

**基础问题**：
1. vLLM 的核心优化是什么？
2. PagedAttention 的原理？
3. Continuous Batching 是如何实现的？

**进阶问题**：
1. 如何定位 vLLM 的性能瓶颈？
2. KV Cache 的 block size 如何选择？
3. Multi-GPU 通信的开销如何优化？

**项目问题**：
1. 你做过哪些 vLLM 优化？
2. 优化效果如何？有什么 trade-off？
3. 如果吞吐还是不够，你会怎么做？

---

### 源码阅读路径

**入门路径**（按顺序）：
1. `vllm/engine/llm_engine.py` - 理解整体流程
2. `vllm/worker/model_runner.py` - 理解模型执行
3. `vllm/attention/ops/paged_attn.py` - 理解 PagedAttention
4. `vllm/scheduler.py` - 理解调度逻辑

**进阶路径**：
1. `vllm/v1/engine/llm_engine.py` - 新版架构
2. `vllm/v1/core/sched/scheduler.py` - 新版调度器
3. `vllm/distributed/` - 分布式通信
4. `vllm/model_executor/` - 模型实现细节