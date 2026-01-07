import random
import numpy as np
import torch
import math

# 假设你的 base.py 中有 MCTSNode 定义
from .base import MCTSNode
from methods.method_utils.str_utils import extract_last_question, extract_last_answer

class LEMCTSSolverMixin:
    """
    LE-MCTS Implementation based on:
    'Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning'
    """

    def solve_lemcts(self, x, idx, to_print=True):
        # 1. 初始化 Prompt
        # 论文中提到将步骤定义为以换行符结束的句子 
        if self.args.prompt_sample == 'standard':
            initial_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            initial_prompt = self.task.cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")
        
        root = MCTSNode(state=initial_prompt, depth=0)
        
        # 2. 从参数中获取配置
        # 论文指出 C 值对不同难度任务有很大影响 (Math/MQA 推荐 0.5, 简单任务推荐 1.0+) 
        c_uct = getattr(self.args, 'c_uct', 0.5) 
        max_depth = getattr(self.args, 'max_depth', 8)
        num_simulation = getattr(self.args, 'num_simulation', 20) # n_iter
        
        # 3. 开始搜索循环
        best_node = self.lemcts_search(
            root=root,
            num_simulation=num_simulation,
            c_uct=c_uct,
            max_depth=max_depth,
            to_print=to_print
        )

        # 4. 提取最终答案
        # 论文提到选择 reward 最高的 trajectory [cite: 229]
        solution = best_node.state
        ys = extract_last_answer(solution)
        return [ys], {'steps': [], 'info': []}

    def lemcts_search(self, root: MCTSNode, num_simulation: int, c_uct: float, max_depth: int, to_print: bool):
        """
        LE-MCTS 主循环，对应论文 Algorithm 1 [cite: 153]
        """
        for i in range(num_simulation):
            node = root
            
            # --- Phase 1: Selection (选择) ---
            # 使用 UCT 选择最有潜力的未完成节点 [cite: 110-111]
            while not node.is_leaf() and not node.is_terminal():
                node = self.lemcts_select_child(node, c_uct)

            # --- Phase 2: Expansion (扩展) ---
            # 如果未达到最大深度且非终止节点，则扩展 [cite: 119]
            if node.depth < max_depth and not node.is_terminal():
                # 随机选择一个 LLM 进行生成 
                llm_names = self.lm_names
                selected_model = random.choice(llm_names)
                new_node = self.lemcts_expand(node, selected_model)
                
                # --- Phase 3: Evaluation (评估) ---
                # 不做 Rollout，直接用 PRM 打分 [cite: 134-136]
                reward = self.lemcts_evaluate(new_node)
                new_node.MC_estimate = reward
                new_node.visits += 1
                new_node.total_value += reward # 初始化自身价值
                
                # --- Phase 4: Backpropagation (反向传播) ---
                # 使用乐观反向传播策略 [cite: 147]
                self.lemcts_backpropagate_optimistic(new_node)

        # 返回访问次数最多或者价值最高的子节点作为下一步，或者直接返回最优路径
        return self.get_best_trajectory(root)

    def lemcts_select_child(self, node: MCTSNode, c_uct: float):
        """
        Selection Phase: 使用标准 UCT 公式 [cite: 112]
        U(s) = v_s + C * sqrt(ln(N_parent) / N_s)
        """
        best_score = -float('inf')
        best_child = None
        
        # 论文中还提到了剪枝策略：如果子节点价值太低则不选 (Threshold epsilon) [cite: 128]
        # 这里简化实现标准 UCT
        for child in node.children:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                # v_s 是平均价值 (在乐观策略下，它是基于子节点最大值更新的)
                v_s = child.total_value / child.visits 
                exploration = c_uct * np.sqrt(np.log(node.visits + 1) / child.visits)
                uct_score = v_s + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        
        return best_child

    def lemcts_expand(self, node: MCTSNode, model_name: str):
        """
        Expansion Phase: 使用选定的模型生成下一个推理步骤 [cite: 121-123]
        """
        # 这里模拟生成 "Step-by-step"，即生成直到换行符 \n
        # 实际调用时，需要你的 generate_sentences 支持 stop='\n'
        prompt = node.state
        
        # 生成 1 个样本 (Greedy or Sampling)
        new_steps = self.generate_sentences(model_name=model_name, prompt=prompt, n_samples=1, stop="\n")
        action = new_steps[0]
        
        next_state = node.state + " " + action
        # 创建新节点
        child_node = MCTSNode(state=next_state, parent=node, action=action, depth=node.depth + 1)
        node.children.append(child_node)
        
        return child_node

    def lemcts_evaluate(self, node: MCTSNode):
        """
        Evaluation Phase: 使用 PRM (Math Shepherd) 计算奖励 [cite: 135-136]
        """
        # 提取问题和当前的推理步骤
        x = extract_last_question(node.state)
        # 这里假设 get_values 是你的 PRM 接口，返回是一个 list of scores
        # 论文中 r_k = phi(q, p_k)
        rewards = self.get_values(model_name=self.args.backend_prm, x=x, ys=[node.action])
        
        if len(rewards) > 0:
            return rewards[0]
        return 0.0

    def lemcts_backpropagate_optimistic(self, node: MCTSNode):
        """
        Value Backpropagation: 乐观更新策略 
        公式 (4): v_s = ((N_s - 1) * v_s + max(v_child)) / N_s
        """
        current = node.parent 
        while current is not None:
            current.visits += 1
            
            # 获取所有子节点中的最大价值
            # 注意：论文中的公式 (4) 是更新 v_s (即平均值)
            # 但为了保持和标准 MCTS 兼容，我们通常存储 total_value。
            # 这里我们直接复现论文逻辑：根据子节点的最大值来更新父节点对路径的估值。
            
            max_child_value = -float('inf')
            for child in current.children:
                # 计算子节点的当前价值 Q(s')
                if child.visits > 0:
                    child_v = child.total_value / child.visits
                    if child_v > max_child_value:
                        max_child_value = child_v
            
            if max_child_value == -float('inf'):
                max_child_value = 0 # Fallback
            
            # 更新逻辑：旧的总分 + 最优子节点的分数 (而不是当前路径的反馈)
            # 这使得父节点的值倾向于它最好的那条路
            # 这里的实现方式是将 "max_child_value" 视为本次模拟的反向传播值
            
            # 按照论文公式 (4) 的变体实现：
            # v_new = [ (N-1)*v_old + max_child_val ] / N
            # 转化为 total_value 的更新:
            # total_new = (N-1) * (total_old / (N-1)) + max_child_val = total_old + max_child_val
            
            # 注意：首次访问时 N=1, previous total 这里的处理需小心。
            # 简单实现：将 max_child_value 加入 total_value
            current.total_value += max_child_value
            
            current = current.parent

    def get_best_trajectory(self, root):
        """
        从树中提取最优路径
        """
        # 简单策略：贪婪地选择价值最高的子节点直到叶子
        node = root
        while not node.is_leaf():
            # 按照平均价值选择
            best_child = max(node.children, key=lambda c: c.total_value / (c.visits + 1e-6))
            node = best_child
        return node