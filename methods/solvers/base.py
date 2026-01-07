"""
Base classes and utilities for solvers.
"""

import numpy as np


class MCTSNode:
    """
    Node class for Monte Carlo Tree Search and DFS algorithms.
    """
    def __init__(self, state, parent=None, action="", depth=0):
        self.state = state  # The current solution (partial or complete)
        self.parent = parent
        self.action = action  # The action that led to this state (the last sentence added)
        self.children = []
        self.visits = 0
        self.mean_value = 0.0  # Cumulative value from simulations (Q-value)
        self.total_value = 0.0
        self.untried_actions = None  # Actions (sentences) that can be tried from this state
        self.depth = depth  # Depth of the node in the tree
        self.MC_estimate = 0.0  # Monte Carlo estimation of correctness (for Omega MCTS)
        self.rollout_length = 0  # Length of the rollout from this node

    def is_fully_expanded(self):
        return len(self.children) > 0  # No more actions to try

    def is_terminal(self):
        return ("\\boxed{" in self.action or 
                "answer is" in self.action or 
                "final answer is" in self.action or 
                "End of answer." in self.action)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.total_value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            if child.visits > 0 else 0
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action, child_node):
        self.children.append(child_node)

