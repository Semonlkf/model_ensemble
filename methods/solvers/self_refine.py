"""
Self-refine solver - Iteratively refine solution based on feedback.
"""

from prompts.self_refine import feedback_prompt, refine_prompt


class SelfRefineSolverMixin:
    """
    Mixin class providing self-refinement strategy.
    Generates an initial solution, then iteratively refines it based on self-feedback.
    """
    
    def solve_self_refine(self, x, idx, to_print=True):
        if self.args.prompt_sample == 'standard':
            initial_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            initial_prompt = self.task.cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")
        
        initial_prompt += "Answer: Step"
        num_iteration = getattr(self.args, 'num_iteration', 3)
        current_ans = self.generate_sentences(model_name=self.args.backend, prompt=initial_prompt, n_samples=1, stop=self.stop)[0].strip()
        step = [{"iteration": 0, "answer": current_ans, "feedback": None}]
        info = []
        info.append(step[0])
        
        for i in range(num_iteration):
            feedback_ans = self.generate_sentences(
                model_name=self.args.backend,
                prompt=feedback_prompt.format(question=x, solution=current_ans),
                n_samples=1,
                stop=self.stop
            )[0].strip()
            
            if 'No error' in feedback_ans:
                break
            
            current_ans = self.generate_sentences(
                model_name=self.args.backend,
                prompt=refine_prompt.format(question=x, solution=current_ans, feedback=feedback_ans),
                n_samples=1,
                stop=self.stop
            )[0].strip()
    
            info.append({
                "iteration": i + 1,
                "answer": current_ans, 
                "feedback": feedback_ans, 
                "revised_answer": current_ans
            })
            
            if to_print:
                print(f"--- Iteration {i + 1} ---")
                print(f"Feedback: {feedback_ans}")
                print(f"Revised version: {current_ans}")
                print("-------------------------")

        return [current_ans], {'info': info}

