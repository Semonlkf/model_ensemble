"""
Naive solver - Direct generation without any search or refinement.
"""


class NaiveSolverMixin:
    """
    Mixin class providing naive solving strategy.
    Generates a single answer directly from the prompt.
    """
    
    def solve_naive(self, x, idx, to_print=True):
        if self.args.prompt_sample == 'standard':
            current_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            current_prompt = self.task.cot_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'reflect_cot':
            current_prompt = self.task.reflect_cot_prompt_wrap(x, "")
        else:
            raise NameError(f"Unknown prompt_sample: {self.args.prompt_sample}")
        
        step_data = []
        candidates = self.generate_sentences(model_name=self.args.backend, prompt=current_prompt, n_samples=self.args.n_generate_sample, stop="End of answer.")
        final_answer = candidates[0].strip()
        
        if to_print:
            print(final_answer)
            
        return [final_answer], {'steps': step_data}

