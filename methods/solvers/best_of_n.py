"""
Best-of-N solvers - Generate N candidates and select the best one(s) based on value function.
"""

import numpy as np


class BestOfNSolverMixin:
    """
    Mixin class providing best-of-N selection strategy.
    Generates N candidates and selects the one with highest value.
    """
    
    def solve_best_of_n(self, x, idx, to_print=True):
        # Generate the prompt based on the selected mode
        if self.args.prompt_sample == 'standard':
            current_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            current_prompt = self.task.cot_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'reflect_cot':
            current_prompt = self.task.reflect_cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")

        # Generate N candidate solutions
        N = self.args.n_generate_sample  # Number of candidates
        candidates = self.generate_sentences(model_name=self.args.backend, prompt=current_prompt, n_samples=N, stop="End of answer.")

        # Evaluate candidates
        values = self.get_values(model_name=self.args.backend, x=x, ys=candidates)

        # Select the best candidate
        best_idx = np.argmax(np.array(values))

        best_candidate = candidates[best_idx]
        best_value = values[best_idx]

        # Prepare the info dictionary to store each candidate and its value
        info = []
        for candidate, value in zip(candidates, values):
            info.append({"x": x, "candidate": candidate, "value": value})

        if to_print:
            print(f"Best candidate with value {best_value}:\n{best_candidate}")
            # Print all candidates and their corresponding values
            print("\nAll candidates and their values:")
            for candidate_value in info:
                value = candidate_value.get("value")
                candidate = candidate_value.get("candidate")
                print(f"Value: {value} -> Candidate: {candidate}")

        # Return the best candidate and the info dictionary
        return [best_candidate], {'info': info}


class WeightedMajoritySolverMixin:
    """
    Mixin class providing weighted majority voting strategy.
    Combines best-of-N selection with weighted voting based on value function.
    """
    
    def solve_best_of_n_with_weighted_voting(self, x, idx, to_print=True):
        # Generate the prompt based on the selected mode
        if self.args.prompt_sample == 'standard':
            current_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            current_prompt = self.task.cot_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'reflect_cot':
            current_prompt = self.task.reflect_cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")

        # Generate N candidate solutions
        N = self.args.n_generate_sample  # Number of candidates
        candidates = self.generate_sentences(model_name=self.args.backend, prompt=current_prompt, n_samples=N, stop="End of answer.")

        # Extract answers from candidates
        extracted_answers = [
            self.task.extract_answer(candidate).replace(': ', '').strip()
            for candidate in candidates
        ]

        # Evaluate candidates
        values = self.get_values(model_name=self.args.backend, x=x, ys=candidates)

        # If all values are the same, avoid division by zero and handle it
        min_value = np.min(values)
        max_value = np.max(values)

        if max_value == min_value:
            # All values are the same, so we assign equal weight to all candidates
            normalized_values = [1.0 for _ in values]
        else:
            # Normalize the values between 0 and 1
            normalized_values = [(v - min_value) / (max_value - min_value) for v in values]

        # Aggregate votes with normalized weights
        weighted_votes = {}
        for i, answer in enumerate(extracted_answers):
            weight = normalized_values[i]  # Get the normalized weight of the candidate
            if answer in weighted_votes:
                weighted_votes[answer] += weight  # Add weight to the total for this answer
            else:
                weighted_votes[answer] = weight  # Initialize weight for this answer

        # Select the answer with the highest total weighted vote
        most_voted_answer = max(weighted_votes, key=weighted_votes.get)

        # Filter candidates that match the most voted answer
        matching_candidates = [
            candidates[i] for i, ans in enumerate(extracted_answers) if ans == most_voted_answer
        ]
        
        # Evaluate these matching candidates and select the one with the best value
        matching_values = [values[i] for i, ans in enumerate(extracted_answers) if ans == most_voted_answer]
        best_idx = np.argmax(np.array(matching_values))

        best_candidate = matching_candidates[best_idx]
        best_value = matching_values[best_idx]

        # Prepare the info dictionary to store each candidate, its value, and extracted answer
        info = []
        for candidate, value, extracted_answer in zip(candidates, values, extracted_answers):
            info.append({
                "x": x, 
                "candidate": candidate, 
                "value": value, 
                "extracted_answer": extracted_answer
            })

        if to_print:
            print(f"Best candidate with value {best_value} and extracted answer '{most_voted_answer}':\n{best_candidate}")
            # Print all candidates and their corresponding values
            print("\nAll candidates and their values:")
            for candidate_value in info:
                value = candidate_value.get("value")
                candidate = candidate_value.get("candidate")
                extracted_answer = candidate_value.get("extracted_answer")
                print(f"Value: {value} -> Candidate: {candidate} (Extracted answer: {extracted_answer})")

        # Return the best candidate and the info dictionary
        return [best_candidate], {'info': info}

