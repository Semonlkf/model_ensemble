# Step Ensemble 参数说明文档

本文档详细介绍 `run.py` 中所有命令行参数的含义和用法。

## 目录

- [模型配置参数](#模型配置参数)
- [任务相关参数](#任务相关参数)
- [生成与采样参数](#生成与采样参数)
- [评估方法参数](#评估方法参数)
- [搜索算法参数](#搜索算法参数)
- [资源配置参数](#资源配置参数)
- [其他参数](#其他参数)

---

## 模型配置参数

### `--model_pool_config`
- **类型**: `str`
- **默认值**: `"configs/3model.yaml"`
- **说明**: 模型池配置文件的路径。该配置文件定义了可用的生成模型及其配置信息，系统会根据此配置注册和管理多个模型。

### `--backend`
- **类型**: `str`
- **默认值**: `"gpt-4o"`
- **说明**: 主要的生成模型名称。用于执行文本生成任务（如推理步骤生成）。支持的模型包括 OpenAI GPT 系列、LLaMA、Qwen 等。

### `--backend_prm`
- **类型**: `str`
- **默认值**: `"gpt-4o"`
- **可选值**: 
  - `gpt-4` - OpenAI GPT-4 模型
  - `gpt-3.5-turbo` - OpenAI GPT-3.5 模型
  - `gpt-4o` - OpenAI GPT-4o 模型
  - `llama-3.1-405b` - Meta LLaMA 3.1 405B 参数模型
  - `llama-3.1-70b` - Meta LLaMA 3.1 70B 参数模型
  - `llama-3.1-8b` - Meta LLaMA 3.1 8B 参数模型
  - `Qwen2.5-7B` - 阿里通义千问 2.5 7B 模型
  - `Qwen2.5-72B` - 阿里通义千问 2.5 72B 模型
  - `Qwen2.5-0.5B` - 阿里通义千问 2.5 0.5B 模型
  - `Mistral-7B-Instruct-v0.3` - Mistral 7B 指令微调模型
  - `internlm2_5-step-prover-critic` - InternLM 步骤证明评价模型
  - `internlm2-1_8b-reward` - InternLM 1.8B 奖励模型
  - `QwQ-32B-Preview` - QwQ 32B 预览版模型
- **说明**: 过程奖励模型（Process Reward Model）的后端。用于评估推理步骤的质量，为搜索算法提供价值估计。

### `--port`
- **类型**: `int`
- **默认值**: `8001`
- **说明**: FastAPI 服务的端口号，用于连接本地部署的奖励模型服务。

---

## 任务相关参数

### `--task`
- **类型**: `str`
- **必需**: 是
- **可选值**:
  | 任务名称 | 说明 |
  |---------|------|
  | `game24` | 24点游戏：使用四则运算将4个数字组合成24 |
  | `text` | 文本生成任务 |
  | `crosswords` | 填字游戏任务 |
  | `bamboogle` | Bamboogle 问答数据集 |
  | `strategyqa` | StrategyQA 多跳推理问答 |
  | `hotpotqa` | HotpotQA 多跳问答数据集 |
  | `gsm8k` | GSM8K 小学数学应用题 |
  | `gsm8k_perb` | GSM8K 扰动版本 |
  | `gsm_hard` | GSM-Hard 困难数学题 |
  | `MATH500` | MATH 数据集（500题子集） |
  | `fever` | FEVER 事实验证数据集 |
  | `prontoqa` | ProntoQA 逻辑推理数据集 |
  | `humaneval` | HumanEval 代码生成评测 |

### `--task_start_index`
- **类型**: `int`
- **默认值**: `900`
- **说明**: 任务数据的起始索引。用于指定从数据集的哪个位置开始处理。

### `--task_end_index`
- **类型**: `int`
- **默认值**: `1000`
- **说明**: 任务数据的结束索引。与 `task_start_index` 配合使用，可以处理数据集的特定子集。

---

## 生成与采样参数

### `--temperature`
- **类型**: `float`
- **默认值**: `0.7`
- **说明**: 生成温度参数。控制生成文本的随机性：
  - 较低的值（如 0.1）：生成更确定、保守的输出
  - 较高的值（如 1.0）：生成更多样、有创意的输出

### `--top_p`
- **类型**: `float`
- **默认值**: `0.9`
- **说明**: 核采样（Nucleus Sampling）参数。只从累积概率达到 `top_p` 的最高概率 token 中采样。

### `--prompt_sample`
- **类型**: `str`
- **可选值**:
  | 值 | 说明 |
  |---|------|
  | `standard` | 标准提示：直接让模型回答问题，不要求展示推理过程 |
  | `cot` | 链式思维（Chain-of-Thought）提示：要求模型逐步展示推理过程 |
  | `reflect_cot` | 反思式链式思维：在 CoT 基础上增加自我反思和验证步骤 |

### `--method_generate`
- **类型**: `str`
- **可选值**:
  | 值 | 说明 |
  |---|------|
  | `sample` | 采样生成：使用温度采样生成多个候选答案 |
  | `propose` | 提议生成：生成结构化的解决方案提议 |

### `--n_generate_sample`
- **类型**: `int`
- **默认值**: `1`
- **说明**: 每次生成的候选样本数量。在 Best-of-N、Beam Search 等方法中，此参数控制每一步生成的候选数量。

### `--n_select_sample`
- **类型**: `int`
- **默认值**: `1`
- **说明**: 从候选中选择的样本数量。在 Beam Search 中对应 beam width（束宽）。

### `--max_tokens`
- **类型**: `int`
- **默认值**: `2048`
- **说明**: 单次生成的最大 token 数量限制。

---

## 评估方法参数

### `--method_evaluate`
- **类型**: `str`
- **可选值**:

| 值 | 说明 | 使用场景 |
|---|------|---------|
| `value` | 自评估价值函数：让生成模型自己评估答案的正确性 | 无需额外奖励模型 |
| `vote` | 投票评估：多个答案通过投票机制选出最佳 | 多样本场景 |
| `random` | 随机评估：随机分配 0-1 之间的分数 | 基线对比 |
| `self_process_value` | 自我过程评估：模型评估推理过程的正确性 | 过程监督 |
| `self_result_value` | 自我结果评估：模型仅评估最终答案 | 结果监督 |
| `llm_as_binary` | LLM 二元评估：使用 LLM 判断答案对错（0或1） | 简单评判 |
| `llm_as_process_reward` | LLM 过程奖励：使用 LLM 评估每个推理步骤 | 细粒度过程监督 |
| `llm_as_reuslt_reward` | LLM 结果奖励：使用 LLM 评估最终结果 | 结果导向评估 |
| `llm_as_reward_value` | LLM 奖励值：使用专门的奖励模型（如 InternLM）计算分数 | 专业奖励模型 |
| `llm_as_critic_value` | LLM 评论值：使用 LLM 作为评论家进行评估 | 批判性评估 |
| `qwq_as_process_reward` | QwQ 过程奖励：使用 QwQ 模型评估推理过程 | QwQ 专用 |

### `--n_evaluate_sample`
- **类型**: `int`
- **默认值**: `1`
- **说明**: 评估时的采样次数。多次评估取平均可以获得更稳定的评估结果。

### `--method_select`
- **类型**: `str`
- **默认值**: `"greedy"`
- **可选值**:
  | 值 | 说明 |
  |---|------|
  | `sample` | 按概率采样选择候选 |
  | `greedy` | 贪婪选择：始终选择评分最高的候选 |

### `--score_criterion`
- **类型**: `str`
- **默认值**: `"max"`
- **说明**: 评分准则。在 MCTS 和 Beam Search 中用于决定如何选择最佳路径：
  - `max`：选择最大分数
  - `min`：选择最小分数（某些奖励模型输出的是错误概率）

---

## 搜索算法参数

### `--baseline`
- **类型**: `str`
- **说明**: 基线方法类型。决定使用哪种推理时计算方法。

| 值 | 说明 | 适用场景 |
|---|------|---------|
| `naive` | 朴素方法：直接生成一个答案，不进行搜索 | 快速基线 |
| `greedy` | 贪婪搜索：每步选择当前最优的候选继续扩展 | 简单搜索 |
| `majority` | 多数投票：生成多个答案，选择出现最多的答案 | 自一致性 |
| `weighted_majority` | 加权多数投票：结合奖励分数的加权投票 | 增强一致性 |
| `best_of_n` | N 选一：生成 N 个候选，选择奖励分数最高的 | 简单有效 |
| `mcts` | 蒙特卡洛树搜索：基于 AlphaZero 的树搜索算法 | 复杂推理 |
| `beam_search` | 束搜索：保持多个候选路径同时扩展 | 平衡效率与质量 |
| `ToT_dfs` | 思维树深度优先搜索：Tree-of-Thoughts 的 DFS 实现 | 探索性搜索 |
| `self_refine` | 自我精炼：迭代生成、反馈、修改的循环 | 迭代改进 |

### `--trick_type`
- **类型**: `str`
- **说明**: 技巧类型标识，用于日志文件命名和实验区分。

### `--single_agent_method`
- **类型**: `str`
- **默认值**: `"naive"`
- **可选值**:
  | 值 | 说明 |
  |---|------|
  | `naive` | 朴素单次生成 |
  | `greedy` | 贪婪逐步生成 |
  | `mcts` | MCTS 搜索 |
  | `majority` | 多数投票 |
  | `beamsearch` | 束搜索 |
  | `selfconsistency` | 自一致性方法 |

---

## 树搜索专用参数

### `--max_depth`
- **类型**: `int`
- **默认值**: `8`
- **说明**: 搜索树的最大深度限制。适用于 MCTS、Beam Search、DFS 等算法。

### `--value_thresh`
- **类型**: `float`
- **默认值**: `-10`
- **说明**: 价值阈值，用于 DFS 搜索中的剪枝。低于此阈值的节点将被剪除。

### `--prune_ratio`
- **类型**: `float`
- **默认值**: `0.4`
- **说明**: 剪枝比例，用于 DFS 搜索。保留分数最高的 `(1 - prune_ratio)` 比例的子节点。

### `--num_paths`
- **类型**: `int`
- **默认值**: `3`
- **说明**: DFS 搜索中探索的完整路径数量。

### `--num_simulation`
- **类型**: `int`
- **默认值**: `5`
- **说明**: MCTS 中每个节点的模拟次数。更多的模拟通常能获得更准确的价值估计。

### `--sample_action`
- **类型**: `bool`
- **默认值**: `False`
- **说明**: 是否在 MCTS 中按访问次数的概率分布采样动作。`False` 时选择访问次数最多的动作。

### `--c_base`
- **类型**: `int`
- **默认值**: `19652`
- **说明**: MCTS UCB 公式中的基数参数。用于动态调整探索-利用平衡。

### `--c_puct`
- **类型**: `float`
- **默认值**: `1.25`
- **说明**: MCTS PUCT 公式中的探索常数。较大的值鼓励更多探索，较小的值偏向利用已知好的路径。

### `--num_iteration`
- **类型**: `int`
- **默认值**: `3`
- **说明**: Self-Refine 方法中的迭代次数。每次迭代包括反馈生成和答案修改。

---

## 资源配置参数

### `--inference_gpu_memory_utilization`
- **类型**: `float`
- **默认值**: `0.9`
- **说明**: 推理模型的 GPU 显存利用率（0-1）。vLLM 会根据此参数预分配显存。

### `--reward_gpu_memory_utilization`
- **类型**: `float`
- **默认值**: `0.9`
- **说明**: 奖励模型的 GPU 显存利用率（0-1）。

---

## 其他参数

### `--naive_run`
- **类型**: `bool`（flag）
- **默认值**: `False`
- **说明**: 启用朴素运行模式的标志。

### `--agent_framework_version`
- **类型**: `str`
- **默认值**: `"v1.0"`
- **说明**: 代理框架版本号，用于实验追踪和日志命名。

### `--debug`
- **类型**: `bool`（flag）
- **默认值**: `False`
- **说明**: 调试模式。启用后会开启 `debugpy` 远程调试，监听 `0.0.0.0:5678` 端口等待调试器连接。

---

## 使用示例

### 基础运行（Naive）
```bash
python run.py --task gsm8k --baseline naive --prompt_sample cot --backend gpt-4o
```

### Best-of-N 搜索
```bash
python run.py --task MATH500 --baseline best_of_n \
    --prompt_sample cot \
    --n_generate_sample 8 \
    --method_evaluate llm_as_reward_value \
    --backend_prm internlm2-1_8b-reward
```

### MCTS 搜索
```bash
python run.py --task gsm8k --baseline mcts \
    --prompt_sample cot \
    --n_generate_sample 5 \
    --num_simulation 10 \
    --max_depth 8 \
    --method_evaluate llm_as_reward_value \
    --backend_prm internlm2-1_8b-reward
```

### Beam Search
```bash
python run.py --task hotpotqa --baseline beam_search \
    --prompt_sample cot \
    --n_generate_sample 5 \
    --n_select_sample 3 \
    --max_depth 10 \
    --method_evaluate llm_as_process_reward
```

### Self-Refine
```bash
python run.py --task gsm8k --baseline self_refine \
    --prompt_sample cot \
    --num_iteration 3
```

### 多数投票（Self-Consistency）
```bash
python run.py --task strategyqa --baseline majority \
    --prompt_sample cot \
    --n_generate_sample 16
```

---

## 方法对比

| 方法 | 计算成本 | 适用场景 | 关键参数 |
|-----|---------|---------|---------|
| `naive` | 低 | 快速基线测试 | - |
| `majority` | 中 | 简单问题、多样答案 | `n_generate_sample` |
| `best_of_n` | 中 | 需要奖励模型指导 | `n_generate_sample`, `method_evaluate` |
| `weighted_majority` | 中 | 结合投票和奖励 | `n_generate_sample`, `method_evaluate` |
| `greedy` | 中-高 | 逐步推理任务 | `n_generate_sample`, `method_evaluate` |
| `beam_search` | 高 | 需要保持多路径探索 | `n_select_sample`, `max_depth` |
| `ToT_dfs` | 高 | 深度探索、回溯 | `max_depth`, `prune_ratio`, `num_paths` |
| `mcts` | 最高 | 复杂推理、需要精确搜索 | `num_simulation`, `c_puct`, `max_depth` |
| `self_refine` | 中 | 迭代改进答案 | `num_iteration` |

---

## 评估方法选择指南

1. **无额外模型**：使用 `value`, `self_process_value`, `self_result_value`
2. **有奖励模型服务**：使用 `llm_as_reward_value`, `llm_as_critic_value`
3. **使用 InternLM 奖励模型**：必须使用 `llm_as_reward_value` 或 `llm_as_critic_value`
4. **快速实验**：使用 `random` 作为基线对比
5. **二元判断**：使用 `llm_as_binary`

