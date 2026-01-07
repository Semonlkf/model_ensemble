"""
Monte Carlo Tree Search (MCTS) solver - Uses MCTS algorithm for solution search.
"""

import random
import numpy as np
import torch.nn as nn
import torch

from .base import MCTSNode
from methods.method_utils.str_utils import extract_last_question, extract_last_answer


class MCTSSolverMixin:
    """
    Mixin class providing MCTS solving strategy.
    Follows the paper: AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training
    """
    
    def solve_mcts(self, x, idx, to_print=True):
        """
        Main MCTS solving method.
        """
        if self.args.prompt_sample == 'standard':
            initial_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            initial_prompt = self.task.cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")
        
        initial_prompt += "Answer: Step"
        # Set the root node with initial state and depth
        root = MCTSNode(state=initial_prompt, depth=0)
        num_simulation = getattr(self.args, 'num_simulation', 5)
        max_depth = getattr(self.args, 'max_depth', 8)
        sample_action = getattr(self.args, 'sample_action', False)
        c_base = getattr(self.args, 'c_base', 19652)
        c_init = getattr(self.args, 'c_puct', 1.25)
        n_generate_sample = getattr(self.args, 'n_generate_sample', 5)

        # Perform MCTS search
        solution_node = self._mcts_alpha_search(
            model_name=self.args.backend,
            root=root, 
            n_samples=n_generate_sample,
            num_simulation=num_simulation,
            c_base=c_base,
            c_init=c_init,
            max_depth=max_depth,
            sample_action=sample_action,
            to_print=to_print
        )

        # Build solution from final node upward
        solution = solution_node.state
        steps = []
        info = []
        node = solution_node

        # Collect path and node info
        while node:
            child_info = []
            for child in node.children:
                q_value = child.total_value / (child.visits + 1e-6)
                child_info.append({
                    'child_action': child.action,
                    'child_state': child.state,
                    'value': child.total_value,
                    'visits': child.visits,
                    'q_value': q_value
                })
            info.insert(0, {
                'depth': node.depth,
                'action': node.action,
                'state': node.state,
                'value': node.total_value,
                'visits': node.visits,
                'children': child_info
            })
            steps.insert(0, node.action)
            node = node.parent

        if to_print:
            for i, layer in enumerate(info[::-1]):
                print("---------- DEPTH:", layer['depth'], "----------")
                print("ACTION:", layer['action'])
                print("STATE:", layer['state'])
                print("CHILDREN:")
                for c in layer['children']:
                    print("   ACTION:", c['child_action'])
                    print("   STATE:", c['child_state'])
                print("----------------------------------------")
        
        ys = extract_last_answer(solution)
        if to_print:
            print(f"FINAL SOLUTION:\n{ys}")

        return [ys], {'steps': steps, 'info': info}
    
    def _mcts_alpha_search(self, model_name: str, root: MCTSNode, n_samples: int, num_simulation: int, c_base: float, c_init: float, max_depth: int, sample_action: bool, to_print: bool):
        """
        Perform MCTS-Alpha search.
        """
        if model_name not in self.model_names:
            raise NameError(f"Unknown model: {model_name}")
        def action(node):
            temperature = 1.0
            action_visits = []
            for act in node.children:
                action_visits.append((act, act.visits))
            actions, visits = zip(*action_visits)
            action_probs = nn.functional.softmax(
                1.0 / temperature * np.log(torch.as_tensor(visits, dtype=torch.float32) + 1e-10), 
                dim=0
            ).numpy()
            if sample_action:
                selected_action = np.random.choice(actions, p=action_probs)
            else:
                selected_action = actions[np.argmax(action_probs)]
            return selected_action
        
        current_node = root
        if not root.is_fully_expanded():
            self._mcts_expand(model_name=model_name, node=current_node, n_samples=n_samples)
        
        while not current_node.is_terminal():
            # Simulate Phase
            for n in range(num_simulation):
                node = current_node
                while node.is_fully_expanded() and not node.is_terminal():
                    if node.depth >= max_depth:
                        if to_print:
                            print(f"Reached maximum depth of {max_depth}. Stopping expansion at this node.")
                        break
                    node = self._mcts_select_best_child(node, c_base, c_init)
                # Expansion Phase
                    if not node.is_terminal() and node.depth <= max_depth:
                        new_node = self._mcts_expand(model_name=model_name, node=node, n_samples=n_samples)
                        # Backpropagation Phase
                        self._mcts_backpropagate(new_node, new_node.MC_estimate)
            try:
                current_node = action(current_node)
                print("Do Action: ", current_node.action)
            except:
                print("Fail Action, The final current_node is: ", current_node.action)
                break
        
        return current_node  # Choose the best child without exploration

    def _mcts_select_best_child(self, model_name: str, node: MCTSNode, c_base: float = 19652, c_init: float = 1.25):
        """
        Select the best child using refined PUCT formula.
        The cpuct is dynamically adjusted based on the number of total visits.
        """
        if model_name not in self.model_names:
            raise NameError(f"Unknown model: {model_name}")
        best_value = -float('inf')
        best_child = None

        for child in node.children:
            q_value = child.total_value / (child.visits + 1e-6)  # Mean value of the child
            u_value = ((c_init + np.log((node.visits + c_base + 1) / c_base)) 
                       * np.sqrt(node.visits) / (1 + child.visits))
            puct_value = q_value + u_value

            if puct_value > best_value:
                best_value = puct_value
                best_child = child

        return best_child

    def _mcts_expand(self, model_name: str, node: MCTSNode, n_samples: int):
        """
        Expand the given node by sampling its children based on a temperature-scaled 
        probability distribution proportional to the value function.

        Args:
            node (MCTSNode): The current node to expand.
            n_generate_sample (int): Number of actions to generate.

        Returns:
            MCTSNode: The selected child node based on the sampling probability.
        """

        print("\n===== Starting Expand =====")
        print(f"Expanding node at depth {node.depth}")
        print(f"Generating up to {n_samples} candidate actions")

        # Generate children if none exist
        if len(node.children) == 0:
            actions = self.generate_sentences(model_name=model_name, prompt=node.state, n_samples=n_samples, stop=self.stop)
            hist_actions = {}
            # Initialize child nodes and collect their values
            for action in actions:
                if hist_actions.get(action):
                    hist_actions[action] += 1
                    continue
                else:
                    hist_actions[action] = 1

                next_state = node.state + " " + action
                child_node = MCTSNode(state=next_state, parent=node, action=action, depth=node.depth + 1)

                print("\n===== Next State =====")
                print(next_state)
                print("======================")

                x = extract_last_question(next_state)
                ys = extract_last_answer(x)
                if "Answer" in x:
                    x = x.split("Answer")[0]

                print("\n===== Extracted X =====")
                print(x)
                print("======================")

                print("\n===== Extracted YS =====")
                print(ys)
                print("======================")
               
                mc_estimate = self.get_values(model_name=model_name, x=x, ys=[ys])[0]

                print("\n===== Value Outputs =====")
                print("MC Estimate:", mc_estimate)
                print("=========================")

                child_node.MC_estimate = mc_estimate
                child_node.rollout_length = node.rollout_length + 1

                # Store child node and its value
                node.expand(action, child_node)

        return random.choice(node.children)

    def _mcts_simulate(self, model_name: str, state: str, current_depth: int, max_depth: int):
        """
        Simulate from a given state to estimate value.
        """
        if model_name not in self.model_names:
            raise NameError(f"Unknown model: {model_name}")
        current_state = state
        depth = current_depth
        total_reward = 0.0
        step_rewards = []
        action = ""

        while ("\\boxed{" not in action or "answer is" not in action or 
               "final answer is" not in action or "End of answer." not in action or 
               depth < max_depth):
            # Generate possible actions
            action = self.generate_sentences(model_name=model_name, prompt=current_state, n_samples=1, stop=self.stop)
            action = action[0]
            if not action:
                break  # No further actions possible

            x = extract_last_question(current_state + " " + action)
            ys = extract_last_answer(x)
            if "Answer" in x:
                x = x.split("Answer")[0]
            # Evaluate the current state using the reward model
            reward = self.get_values(model_name=model_name, x=x, ys=[ys])
            current_state += " " + action

            if len(reward) != 0:
                total_reward += reward[0]
            step_rewards.append(reward)
            depth += 1

        return total_reward, step_rewards, current_state

    def _mcts_backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value up the tree.
        """
        while node is not None:
            node.visits += 1
            # Update Q(s, a) incrementally
            node.total_value += value
            node.value = node.total_value / node.visits
            node = node.parent

    def _mcts_sample_action_based_on_visits(self, node: MCTSNode):
        """
        if model_name not in self.model_names:  
        Sample action based on visit counts.
        """
        probabilities = [child.visits / node.visits for child in node.children]
        return random.choices(node.children, probabilities)[0]

