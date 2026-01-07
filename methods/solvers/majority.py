"""
Majority voting solver - Generates multiple candidates and selects by majority vote.
"""

from collections import Counter


class MajoritySolverMixin:
    """
    Mixin class providing majority voting strategy.
    Generates multiple candidates and selects the answer that appears most frequently.
    """
    
    def solve_majority(self, x, idx, to_print=True, max_retry=5):
        retry_count = 0

        while retry_count < max_retry:
            # Generate the prompt based on the selected mode
            if self.args.prompt_sample == 'standard':
                current_prompt = self.task.standard_prompt_wrap(x, "")
            elif self.args.prompt_sample == 'cot':
                current_prompt = self.task.cot_prompt_wrap(x, "")
            elif self.args.prompt_sample == 'reflect_cot':
                current_prompt = self.task.reflect_cot_prompt_wrap(x, "")
            else:
                raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")

            # Generate candidate answers
            candidates = self.generate_sentences(
                model_name=self.args.backend,
                prompt=current_prompt,
                n_samples=self.args.n_generate_sample,
                stop="End of answer."
            )

            # Prepare the info dictionary to store each candidate and its value
            info = []
            for candidate in candidates:
                info.append({"prompt": x, "candidate": candidate})

            # Extract and clean answers from the candidates
            extracted_answers = [
                self.task.extract_answer(candidate).replace(': ', '').strip()
                for candidate in candidates
            ]

            # Count occurrences of each extracted answer
            answer_counts = Counter(extracted_answers)

            # Ensure no empty answers ("") are selected as the final answer
            sorted_answers = answer_counts.most_common()
            most_common_answer = None

            for answer, _ in sorted_answers:
                if answer != "":  # Skip empty answers
                    most_common_answer = answer
                    break

            if most_common_answer:
                # Find the first candidate corresponding to the most common extracted answer
                matching_candidate = next(
                    candidate for candidate, extracted_answer in zip(candidates, extracted_answers)
                    if extracted_answer == most_common_answer
                )

                # Optional: Print the matching candidate
                if to_print:
                    print(matching_candidate)

                return [matching_candidate], {'info': info}

            # If no valid answer is found, increment retry count and try again
            retry_count += 1
            print(f"Retrying... ({retry_count}/{max_retry})")

        # If retries are exhausted, return the final answer string
        print("Maximum retries reached. Returning final answer.")
        return ["the final answer is "], {'info': info}

