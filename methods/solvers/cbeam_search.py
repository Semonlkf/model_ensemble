import numpy as np
import math
from methods.method_utils.str_utils import extract_last_answer

class CollaborativeBeamSearchSolverMixin:
    """
    Collaborative Beam Search (CBS) Solver Mixin
    Paper: Collaborative Beam Search Enhancing LLM Reasoning via Collective Consensus (Xu et al., 2025)
    """

    def solve_cbs(self, x, idx, to_print=True):
        # 1. 参数设置
        # llm_pool: 参与集成的模型列表 (对应论文中的 N LLMs)
        llm_names = getattr(self.args, 'llm_names', self.model_names)
        if not llm_names:
            raise ValueError("No LLM names defined for CBS. Please set args.llm_names or self.model_names.")

        beam_width = getattr(self.args, 'beam_width', 4)  # B
        # 每一步生成的样本总数 (对应论文中的 K，或理解为每个 Beam 的总生成量)
        # 这里将其定义为每个 Beam 分配的总采样数
        samples_per_beam = getattr(self.args, 'n_generate_sample', 8) 
        max_depth = getattr(self.args, 'max_depth', 8)    # T
        temperature = getattr(self.args, 'temperature', 0.1) # Softmax 温度
        
        # 初始化 Prompt
        if self.args.prompt_sample == 'standard':
            initial_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            initial_prompt = self.task.cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")
        
        # 强制模型逐步生成 (Step-by-step)
        initial_prompt += "Answer: Step"

        # 初始化 Beams
        # 结构: {'state': 当前文本, 'step_scores': 每步得分历史, 'history_models': 记录每步是谁生成的}
        beams = [{
            'state': initial_prompt,
            'step_scores': [],
            'history_models': [] 
        }]
        
        # 初始化模型权重 (均匀分布)
        # 对应论文 Line 6: a_i <- K / (B*N) 的初始状态
        n_models = len(llm_names)
        model_probs = {m: 1.0 / n_models for m in llm_names}
        
        completed_beams = []

        # --- 主循环 (Step-level Beam Search) ---
        for depth in range(max_depth):
            if not beams:
                break
                
            # --- 1. 动态配额分配 (Differential Quota Allocation) ---
            # 根据当前 model_probs 计算每个模型在本轮应生成的样本数
            # 对应论文 Line 22 & Section 2.3
            quotas = self._cbs_allocate_quotas(model_probs, samples_per_beam)
            
            if to_print:
                print(f"\n--- Depth {depth+1} ---")
                print(f"Model Probs: { {k: round(v, 3) for k, v in model_probs.items()} }")
                print(f"Quotas: {quotas}")

            # --- 2. 多样化生成 (Diverse Candidates Generation) ---
            candidates_map = {} # 用于去重: Key=生成的完整文本
            
            for beam in beams:
                if self._cbs_is_terminal(beam['state']):
                    completed_beams.append(beam)
                    continue

                prefix = beam['state']
                
                for model_name, count in quotas.items():
                    if count <= 0:
                        continue
                    
                    try:
                        # 调用模型生成，设置 stop='\n' 以生成单个推理步骤
                        generations = self.generate_sentences(
                            model_name=model_name, 
                            prompt=prefix, 
                            n_samples=count, 
                            stop="\n"
                        )
                    except Exception as e:
                        print(f"Generation failed for {model_name}: {e}")
                        generations = []
                    
                    for gen in generations:
                        if not gen.strip():
                            continue
                        
                        gen_step = gen.strip()
                        new_state = prefix + " " + gen_step
                        
                        # 记录生成该候选的模型 (用于后续配额更新)
                        # 论文提到：如果多个模型生成了相同的步骤，它们都算作贡献者
                        if new_state not in candidates_map:
                            candidates_map[new_state] = {
                                'step': gen_step,
                                'parent_beam': beam,
                                'generating_models': set()
                            }
                        candidates_map[new_state]['generating_models'].add(model_name)
            
            if not candidates_map:
                break
            
            # --- 3. 集体共识验证 (Collective Consensus Verification) ---
            scored_candidates = []
            
            for new_state, info in candidates_map.items():
                # 计算步骤得分: 所有模型的负困惑度 (Negative Perplexity) 的平均值
                # 对应论文 Eq (1) & (2)
                ppl_sum = 0.0
                valid_models = 0
                
                for evaluator_model in llm_names:
                    try:
                        # 调用你提供的 get_ppl 接口
                        ppl = self.get_ppl(model_name=evaluator_model, prompt=new_state)
                        ppl_sum += (-ppl) # 论文中使用负 PPL 作为 Reward
                        valid_models += 1
                    except Exception as e:
                        pass
                
                if valid_models > 0:
                    step_score = ppl_sum / valid_models
                else:
                    step_score = -1e9 # 惩罚无法计算分数的项
                
                # 路径得分: 历史所有步骤得分的平均值
                prev_scores = info['parent_beam']['step_scores']
                new_step_scores = prev_scores + [step_score]
                path_score = sum(new_step_scores) / len(new_step_scores)
                
                new_beam = {
                    'state': new_state,
                    'step_scores': new_step_scores,
                    'history_models': info['parent_beam']['history_models'] + [info['generating_models']],
                    'path_score': path_score
                }
                scored_candidates.append(new_beam)
            
            # --- 4. 选择 (Selection) ---
            # 根据路径得分排序，保留 Top-B
            scored_candidates.sort(key=lambda x: x['path_score'], reverse=True)
            beams = scored_candidates[:beam_width]
            
            if to_print:
                print(f"Top Beam Score: {beams[0]['path_score']:.4f}")
                print(f"Top Beam State: ...{beams[0]['state'][-50:]}")

            # --- 5. 更新模型配额 (Update Probabilities) ---
            # 统计被选中的 Beam 的最后一步是由哪些模型生成的
            model_counts = {m: 0 for m in llm_names}
            for beam in beams:
                if beam['history_models']:
                    # 获取生成最后一步的模型集合
                    last_step_models = beam['history_models'][-1]
                    for m in last_step_models:
                        if m in model_counts:
                            model_counts[m] += 1
            
            # Softmax 归一化更新概率
            # prob_i = exp(count_i / tau) / sum(...)
            exp_vals = {}
            sum_exp = 0.0
            for m in llm_names:
                val = np.exp(model_counts[m] / temperature)
                exp_vals[m] = val
                sum_exp += val
            
            for m in llm_names:
                model_probs[m] = exp_vals[m] / sum_exp

        # --- 输出最终结果 ---
        all_candidates = beams + completed_beams
        if not all_candidates:
            return ["Error: No solution found"], {'steps': []}
            
        # 选择路径得分最高的作为最终答案
        best_beam = max(all_candidates, key=lambda x: sum(x['step_scores'])/len(x['step_scores']) if x['step_scores'] else -1e9)
        
        solution = best_beam['state']
        ys = extract_last_answer(solution)
        
        if to_print:
            print(f"Final Solution:\n{solution}")

        return [ys], {'steps': []}

    def _cbs_allocate_quotas(self, probs, total_samples):
        """
        根据概率分配整数配额 (Largest Remainder Method)
        确保 sum(quotas) == total_samples
        """
        quotas = {}
        remainders = {}
        
        current_sum = 0
        for m, p in probs.items():
            count = p * total_samples
            int_part = int(math.floor(count))
            quotas[m] = int_part
            remainders[m] = count - int_part
            current_sum += int_part
        
        deficit = int(total_samples - current_sum)
        
        # 将剩余配额分配给小数部分最大的模型
        sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(deficit):
            model = sorted_remainders[i][0]
            quotas[model] += 1
            
        return quotas

    def _cbs_is_terminal(self, text):
        # 判断是否生成结束
        return "\\boxed{" in text or "End of answer." in text or "final answer is" in text