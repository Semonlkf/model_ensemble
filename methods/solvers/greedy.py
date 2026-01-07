"""
Greedy solver - Step-by-step greedy search selecting best candidate at each step.
"""

import numpy as np


class GreedySolverMixin:
    """
    Mixin class providing greedy solving strategy.
    At each step, generates multiple candidates and selects the best one based on value function.
    """
    
    def solve_greedy(self, x, idx, to_print=True):
        if self.args.prompt_sample == 'standard':
            current_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            current_prompt = self.task.cot_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'reflect_cot':
            current_prompt = self.task.reflect_cot_prompt_wrap(x, "")
        else:
            raise NameError(f"Unknown prompt_sample: {self.args.prompt_sample}")
            
        step_data = []
        step_index = 0
        current_solution = ""
        
        while True:
            # Generate N candidate sentences in batch
            retry_count = 0
            max_retry = 2
            
            while retry_count <= max_retry:
                candidates = self.generate_sentences(model_name=self.args.backend, prompt=current_prompt, n_samples=self.args.n_generate_sample, stop='Step')
                valid_candidates = []
                for candidate in candidates:
                    if "Question" in candidate or candidate.count("Answer:") > 2:
                        continue
                    valid_candidates.append(candidate)
                if valid_candidates:
                    candidates = valid_candidates
                    break
                retry_count += 1
                
            if retry_count > max_retry:
                candidates = [candidate.split("Question")[0] for candidate in candidates]

            values = self.get_values(model_name=self.args.backend, x=x, ys=[current_solution + " " + candidate for candidate in candidates])
            best_idx = np.argmax(np.array(values))
            best_candidate = candidates[best_idx]
            best_score = values[best_idx]

            # Append to current prompt
            current_prompt += " " + best_candidate
            current_solution += " " + best_candidate
            
            # Record step information
            step_info = {
                "step": step_index,
                "candidates": [{"candidate": c, "score": v} for c, v in zip(candidates, values)],
                "best_candidate": best_candidate,
                "best_solution": current_solution,
                "best_score": best_score
            }
            step_data.append(step_info)

            if to_print:
                print(f"\n--- Step {step_index} ---")
                print("Best Candidate:")
                print(best_candidate)
                print("\nCurrent Solution:")
                print(current_solution)
                print("-" * 40)
         
            # Check for stopping condition
            if ("answer is" in best_candidate or 
                "final answer is" in best_candidate or 
                "End of answer." in best_candidate or 
                step_index >= 20):
                break

            step_index += 1

        # Process the final answer
        final_answer = current_solution.strip()
        return [final_answer], {'steps': step_data}

