"""
Beam search solver - Maintains multiple solution paths and prunes based on value.
"""

from methods.method_utils.str_utils import extract_last_answer


class BeamSearchSolverMixin:
    """
    Mixin class providing beam search strategy.
    Maintains multiple solution paths (beams) and prunes based on value function.
    """
    
    def solve_beam_search(self, x, idx, to_print=True):
        # Generate the initial prompt
        if self.args.prompt_sample == 'standard':
            initial_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            initial_prompt = self.task.cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")
        
        initial_prompt += "Answer: Step"
        temp_dict = {}
        beam_width = self.args.n_select_sample  # Beam width
        max_depth = getattr(self.args, 'max_node_depth', 20)  # Max steps
        n_generate_sample = getattr(self.args, 'n_generate_sample', 5)  # Candidates per beam

        beams = [(initial_prompt, "", [0])]  # Each beam is (prompt, candidate, cumulative_score)
        completed_beams = []
        step_data = []

        for depth in range(max_depth):
            if len(beams) < 1:
                break
            all_candidates = []
            
            for prompt, cand, score in beams:
                # Generate possible continuations
                new_prompt = prompt + " " + cand
                candidates = self.generate_sentences(model_name=self.args.backend, prompt=new_prompt, n_samples=n_generate_sample, stop=self.stop)
                current_solutions = [extract_last_answer(new_prompt + " " + candidate) for candidate in candidates]

                values = self.get_values(model_name=self.args.backend, x=x, ys=current_solutions)
                # Combine current score with candidate values
                for candidate, value in zip(candidates, values):
                    if candidate in temp_dict:
                        continue  # Skip if this candidate has already been generated
                    temp_dict[new_prompt + " " + candidate] = True  # Mark candidate as generated
                    new_score = score + [value]
                    all_candidates.append((new_prompt, candidate, new_score))
            
            # Keep the top beam_width beams
            if self.args.score_criterion == 'max':
                all_candidates.sort(key=lambda x: -x[2][-1])
            else:
                all_candidates.sort(key=lambda x: -max(x[2]))
            beams = all_candidates[:beam_width]
            
            # Record step information
            step_info = {
                "step": depth,
                "beams": [
                    {"prompt": beam[0], "cand": beam[1], "score": beam[2]} for beam in beams
                ]
            }
            step_data.append(step_info)
            
            # Check for completion
            remaining_beams = []
            for prompt, cand, score in beams:
                if "\\boxed{" in cand or "\\\\boxed{" in cand or "answer is" in cand or "final answer is" in cand or "End of answer." in cand:
                    completed_beams.append((prompt, cand, score))
                else:
                    remaining_beams.append((prompt, cand, score))
            
            beams = remaining_beams  # Only keep unfinished beams

            if to_print:
                print(f"\n--- Step {depth} ---")
                for beam in beams:
                    print(f"PROMPT: {beam[0]}\nCANDIDATE: {beam[1]}\nSCORE: {beam[2]}")
                    print("=" * 40)
                print("-" * 40)

        # Select the best completed beam
        if completed_beams:
            if self.args.score_criterion == 'max':
                best_prompt, candid, best_score = max(completed_beams, key=lambda x: x[2][-1])
            else:
                best_prompt, candid, best_score = max(completed_beams, key=lambda x: max(x[2]))
        else:
            beams = step_info['beams']
            # No completed beam, take the best current beam
            if self.args.score_criterion == 'max':
                ans = max(beams, key=lambda x: max(x["score"]))
                best_prompt, candid, best_score = ans['prompt'], ans['cand'], ans['score']
            else:
                ans = max(beams, key=lambda x: max(x["score"]))
                best_prompt, candid, best_score = ans['prompt'], ans['cand'], ans['score']
        
        if to_print:
            print(f"Best beam with score {best_score}:\n{best_prompt} {candid}")
        
        ys = extract_last_answer(best_prompt + " " + candid)
        return [ys], {'steps': step_data}

