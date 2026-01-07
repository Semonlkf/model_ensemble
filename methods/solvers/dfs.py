"""
Depth-First Search (DFS) solver - Tree-of-Thought style exploration.
"""

from .base import MCTSNode
from methods.method_utils.str_utils import extract_last_answer


class DFSSolverMixin:
    """
    Mixin class providing DFS solving strategy.
    Performs depth-first search with pruning based on value function.
    """
    
    def solve_dfs(self, x, idx, to_print=True):
        """
        Perform Depth-First Search (DFS) to explore possible solutions.

        Args:
            x: The input problem.
            idx: The problem index.
            to_print (bool): Whether to print intermediate results.

        Returns:
            tuple: (best_solution, info_dict)
        """
        if self.args.prompt_sample == 'standard':
            initial_prompt = self.task.standard_prompt_wrap(x, "")
        elif self.args.prompt_sample == 'cot':
            initial_prompt = self.task.cot_prompt_wrap(x, "")
        else:
            raise ValueError(f"Invalid prompt_sample mode: {self.args.prompt_sample}")

        max_depth = getattr(self.args, 'max_depth', 8)
        value_thresh = getattr(self.args, 'value_thresh', 0.3)
        prune_ratio = getattr(self.args, 'prune_ratio', 0.4)
        num_paths = getattr(self.args, 'num_paths', 3)
        root = MCTSNode(state=initial_prompt, depth=0)

        stack = [(root, 0)]
        best_solution = None
        best_score = -float('inf') if self.args.score_criterion == 'max' else float('inf')
        path = []
        info = []

        if to_print:
            print("---------- Starting DFS ----------")
            print(f"Initial Prompt:\n{initial_prompt}\n-----------------------------------")

        while stack:
            current_node, depth = stack.pop()
            
            if to_print:
                print(f"VISITING DEPTH {depth}:")
                print(f"CURRENT NODE STATE:\n{current_node.state}")
                print("-----------------------------------")

            # Check terminal or depth
            if current_node.is_terminal():
                path.append(current_node)
                if current_node.MC_estimate is not None:
                    if self.args.score_criterion == 'max':
                        if current_node.MC_estimate > best_score:
                            best_score = current_node.MC_estimate
                            best_solution = current_node
                    else:  # 'min'
                        if current_node.MC_estimate < best_score:
                            best_score = current_node.MC_estimate
                            best_solution = current_node
                continue
            
            if depth >= max_depth:
                continue
            if len(path) >= num_paths:
                continue

            # Expand current node
            if not current_node.is_fully_expanded():
                self._dfs_expand(model_name=self.args.backend, node=current_node, n_samples=self.args.n_generate_sample)

            children = current_node.children
            # Sort children for better paths first
            children.sort(key=lambda c: -c.MC_estimate)

            if to_print and children:
                print("CHILDREN (SORTED BY MC_ESTIMATE):")
                for child in children:
                    print(f"CHILD ACTION: {child.action} | ESTIMATE: {child.MC_estimate}")
                print("-----------------------------------")

            # Prune and push to stack
            keep_count = int((1 - prune_ratio) * len(children))
            for i, child in enumerate(children):
                if i > keep_count:
                    break
                if child.MC_estimate < value_thresh and len(path) != 0:
                    continue
                stack.append((child, depth + 1))

            # Record info for each depth
            info.append({
                'depth': depth,
                'state': current_node.state,
                'children': [{'action': child.action, 'estimate': child.MC_estimate} for child in children]
            })

        # Aggregate Path
        best_path = None
        best_solution_str = ''
        best_value_hist = [-float('inf')]

        for solution in path:
            steps = []
            value_hist = []
            node = solution
            while node:
                steps.insert(0, node.action)
                value_hist.insert(0, node.MC_estimate)
                node = node.parent

            if sum(best_value_hist) / len(best_value_hist) < sum(value_hist) / len(value_hist):
                best_path = steps
                best_solution_str = solution.state
                best_value_hist = value_hist

        ys = extract_last_answer(best_solution_str)

        if to_print:
            print(f"---------- DFS COMPLETE ----------")
            print(f"BEST SOLUTION FOUND WITH AVERAGE SCORE {sum(best_value_hist)/len(best_value_hist):.3f}:")
            print("SOLUTION PATH:")
            for step in best_path:
                print(f"{step}")
            print("-----------------------------------")

        return [ys], {'info': info, 'best_path': best_path}

    def _dfs_expand(self, model_name: str, node: MCTSNode, n_samples: int):
        """
        Expand a node for DFS by generating and evaluating children.
        
        Args:
            node (MCTSNode): The node to expand.
            n_samples (int): Number of candidate actions to generate.
        """
        from methods.method_utils.str_utils import extract_last_question, extract_last_answer
        
        if len(node.children) == 0:
            actions = self.generate_sentences(model_name=model_name, prompt=node.state, n_samples=n_samples, stop=self.stop)
            hist_actions = {}
            
            for action in actions:
                if hist_actions.get(action):
                    hist_actions[action] += 1
                    continue
                else:
                    hist_actions[action] = 1

                next_state = node.state + " " + action
                child_node = MCTSNode(state=next_state, parent=node, action=action, depth=node.depth + 1)

                x = extract_last_question(next_state)
                ys = extract_last_answer(x)
                if "Answer" in x:
                    x = x.split("Answer")[0]
               
                mc_estimate = self.get_values(model_name=model_name, x=x, ys=[ys])[0]
                child_node.MC_estimate = mc_estimate
                child_node.rollout_length = node.rollout_length + 1

                node.expand(action, child_node)

