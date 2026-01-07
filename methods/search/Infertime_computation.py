"""
InferTimeComputation - Main class for inference-time computation strategies.

This class uses a Mixin pattern to combine various solving strategies.
Each strategy is implemented in a separate file under methods/solvers/.
"""

import re
import random
import numpy as np
from functools import partial
import yaml
from typing import List, Optional
from models.llm_pool import LLMModelPool
from methods.method_utils.str_utils import extract_last_question, extract_last_answer, add_step_tags
from prompts.binary_evaluate import binary_evaluate, binary_evaluate_unwrap

# Import all solver Mixins
from methods.solvers import (
    MCTSNode,
    NaiveSolverMixin,
    GreedySolverMixin,
    MajoritySolverMixin,
    BestOfNSolverMixin,
    WeightedMajoritySolverMixin,
    BeamSearchSolverMixin,
    MCTSSolverMixin,
    DFSSolverMixin,
    SelfRefineSolverMixin,
    LEMCTSSolverMixin,
    )


class InferTimeComputation(
    NaiveSolverMixin,
    GreedySolverMixin,
    MajoritySolverMixin,
    BestOfNSolverMixin,
    WeightedMajoritySolverMixin,
    BeamSearchSolverMixin,
    MCTSSolverMixin,
    DFSSolverMixin,
    SelfRefineSolverMixin,
    LEMCTSSolverMixin,
):
    """
    Main class for inference-time computation.
    
    Combines multiple solving strategies through Mixin inheritance:
    - NaiveSolverMixin: Direct generation without search
    - GreedySolverMixin: Step-by-step greedy search
    - MajoritySolverMixin: Majority voting
    - BestOfNSolverMixin: Best-of-N selection
    - WeightedMajoritySolverMixin: Weighted majority voting
    - BeamSearchSolverMixin: Beam search
    - MCTSSolverMixin: Monte Carlo Tree Search
    - DFSSolverMixin: Depth-First Search (Tree-of-Thought)
    - SelfRefineSolverMixin: Self-refinement with feedback
    - LEMCTSSolverMixin: LE-MCTS
    """
    
    def __init__(self, task, args):
        self.task = task
        self.args = args
        
        if self.args.baseline == 'naive':
            self.args.n_generate_sample = 1

        self.value_cache = {}
        self.memory = []
        self.args.score_criterion = 'min'  # or 'min' or 'max'
        self.args.max_node_depth = 8 
        self.max_memory_size = args.max_memory_size if hasattr(args, 'max_memory_size') else 100
        self.stop = "Step"
        
        # Initialize LLM pool
        self.llm_pool = LLMModelPool()
        
        # Load model configuration
        with open(self.args.model_pool_config, 'r') as f:
            model_pool_config = yaml.load(f, Loader=yaml.FullLoader)
        
        for model in model_pool_config['models']:
            self.llm_pool.register_model(model['name'], model['config'])
        
        self.model_names = self.llm_pool.get_all_model_names()
        self.lm_names = [model_name for model_name in self.model_names if not self.llm_pool.instances[model_name].is_reward_model]
        self.rm_names = [model_name for model_name in self.model_names if self.llm_pool.instances[model_name].is_reward_model]
        
        # Setup reward model if needed
        if args.method_evaluate not in ['value', 'vote', 'random', 'self_process_value', 'self_result_value']:
            self.prm = partial(self.llm_pool.get_reward_score, model_name=args.backend_prm)
    
    def reset(self):
        """Reset internal state between tasks to avoid cache pollution."""
        self.value_cache = {}
        self.memory = []
        
    def generate_sentences(self,model_name: str, prompt: str, n_samples: int, stop: Optional[str] = None) -> List[str]:
        """Generate n_samples sentences from the given prompt."""
        if model_name not in self.model_names:
            raise NameError(f"Unknown model: {model_name}")
        generated = self.llm_pool.generate(prompt, model_name=model_name, n=n_samples, stop=stop)
        return [g.strip() for g in generated]
    
    def prm(self, model_name: str, prompt: str, return_step_scores: bool = False) -> List[float]:
        """Get reward score from PRM model."""
        if model_name not in self.model_names:
            raise NameError(f"Unknown model: {model_name}")
        value_outputs = self.llm_pool.get_reward_score(prompt, model_name=model_name, return_step_scores=return_step_scores)
        return value_outputs
    
    def get_values(self, model_name: str, x: str, ys: List[str]) -> List[float]:
        """
        Evaluate candidate solutions and return their values.
        
        Args:
            x: The input problem (will extract last question)
            ys: List of candidate solutions
            
        Returns:
            List of values corresponding to each candidate
        """
        # Extract the last question, avoiding the few-shot examples
        x = extract_last_question(x)

        # 自我评估
        if self.args.method_evaluate == "value":
            values = []
            for y in ys:
                value_prompt = self.task.value_prompt_wrap(x, y)
                if value_prompt in self.value_cache:
                    value = self.value_cache[value_prompt]
                else:
                    value_outputs = self.generate_sentences(model_name, value_prompt, self.args.n_evaluate_sample, "End of answer.")
                    value = self.task.value_outputs_unwrap(x, y, value_outputs)
                    self.value_cache[value_prompt] = value
                values.append(value)
                
        # 自我过程评估：模型评估推理过程的正确性
        elif self.args.method_evaluate == "self_process_value":
            values = []
            for y in ys:
                value_prompt = self.task.self_process_value_prompt_wrap(x, y)
                if value_prompt in self.value_cache:
                    value = self.value_cache[value_prompt]
                else:
                    value_outputs = self.generate_sentences(model_name, value_prompt, self.args.n_evaluate_sample, "End of answer.")
                    value = self.task.value_outputs_unwrap(x, y, value_outputs)
                    self.value_cache[value_prompt] = value
                values.append(value)
                
        # 自我结果评估：模型评估推理结果的正确性
        elif self.args.method_evaluate == "self_result_value":
            values = []
            for y in ys:
                y = self.task.extract_answer(y).replace(': ', '').strip()
                value_prompt = self.task.self_result_value_prompt_wrap(x, y)
                if value_prompt in self.value_cache:
                    value = self.value_cache[value_prompt]
                else:
                    value_outputs = self.generate_sentences(model_name, value_prompt, self.args.n_evaluate_sample, "End of answer.")
                    value = self.task.value_outputs_unwrap(x, y, value_outputs)
                    self.value_cache[value_prompt] = value
                values.append(value)
                
        # 随机评估
        elif self.args.method_evaluate == "random":
            values = [random.uniform(0, 1) for _ in ys]

        # 奖励模型判断答案对错
        elif self.args.method_evaluate == "llm_as_binary":
            values = []
            for y in ys:
                value_prompt = binary_evaluate + x + '\nThought Process: ' + y + '\nEvaluation Process:\n'
                if value_prompt in self.value_cache:
                    value = self.value_cache[value_prompt]
                else:
                    value_outputs = self.generate_sentences(model_name, value_prompt, self.args.n_evaluate_sample, "End of answer.")
                    value = binary_evaluate_unwrap(value_outputs=value_outputs)
                    self.value_cache[value_prompt] = value
                values.append(value)
                
        # PRM模型判断推理过程的正确性 (返回每个步骤的分数)
        elif self.args.method_evaluate == "prm_as_process_reward":
            if self.args.backend_prm.startswith("math_shepherd"):
                values = []
                for y in ys:
                    # 为每个推理步骤后加入ки标记
                    y_with_step_tag = add_step_tags(y)
                    input_for_prm = f"{x} {y_with_step_tag}"
                    value_outputs = self.llm_pool.get_reward_score(input_for_prm, model_name=self.args.backend_prm, return_step_scores=True)
                    value = value_outputs['step_scores']  # List of step scores
                    values.append(value)
            else:
                raise NameError(f"Unknown backend model: {self.args.backend_prm}")
                
        # RM模型判断整个答案的分数
        elif self.args.method_evaluate == "prm_as_result_reward":
            if self.args.backend_prm.startswith("math_shepherd"):
                values = []
                for y in ys:
                    # 为每个推理步骤后加入ки标记
                    y_with_step_tag = add_step_tags(y)
                    input_for_prm = f"{x} {y_with_step_tag}"
                    value_outputs = self.llm_pool.get_reward_score(input_for_prm, model_name=self.args.backend_prm, return_step_scores=False)
                    value = value_outputs
                    values.append(value)
            else:
                raise NameError(f"Unknown backend model: {self.args.backend_prm}")
                
        # LLM PPL评分
        elif self.args.method_evaluate == "llm_ppl_score":
            values = []
            for y in ys:
                input_for_ppl = f"{x} {y}"
                value = self.llm_pool.calculate_ppl(text=input_for_ppl, model_name=model_name)
                values.append(value)
        else:
            raise NameError(f"Unknown evaluation method: {self.args.method_evaluate}")

        return values

    def solve(self, x, idx, to_print=True):
        """
        Main entry point for solving a problem.
        
        Dispatches to the appropriate solving method based on args.baseline.
        
        Args:
            x: The input problem
            idx: The problem index
            to_print: Whether to print intermediate results
            
        Returns:
            tuple: (solutions, info_dict)
        """
        if self.args.baseline == 'naive':
            return self.solve_naive(x, idx, to_print)
        elif self.args.baseline == 'greedy':
            return self.solve_greedy(x, idx, to_print)
        elif self.args.baseline == 'majority':
            return self.solve_majority(x, idx, to_print)
        elif self.args.baseline == 'weighted_majority':
            return self.solve_best_of_n_with_weighted_voting(x, idx, to_print)
        elif self.args.baseline == 'best_of_n':
            return self.solve_best_of_n(x, idx, to_print)
        elif self.args.baseline == 'mcts':
            return self.solve_mcts(x, idx, to_print)
        elif self.args.baseline == 'beam_search':
            return self.solve_beam_search(x, idx, to_print)
        elif self.args.baseline == 'ToT_dfs':
            return self.solve_dfs(x, idx, to_print)
        elif self.args.baseline == 'self_refine':
            return self.solve_self_refine(x, idx, to_print)
        elif self.args.baseline == 'lemcts':
            return self.solve_lemcts(x, idx, to_print)
        else:
            raise ValueError(f"Unknown baseline method: {self.args.baseline}")
