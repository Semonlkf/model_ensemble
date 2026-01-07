"""
Solvers module - Contains various solving strategies as Mixins.

Each solver is implemented as a Mixin class that can be combined with 
InferTimeComputation to provide different problem-solving strategies.
"""

from .base import MCTSNode
from .naive import NaiveSolverMixin
from .greedy import GreedySolverMixin
from .majority import MajoritySolverMixin
from .best_of_n import BestOfNSolverMixin, WeightedMajoritySolverMixin
from .beam_search import BeamSearchSolverMixin
from .mcts import MCTSSolverMixin
from .dfs import DFSSolverMixin
from .self_refine import SelfRefineSolverMixin
from .lemcts import LEMCTSSolverMixin

__all__ = [
    'MCTSNode',
    'NaiveSolverMixin',
    'GreedySolverMixin',
    'MajoritySolverMixin',
    'BestOfNSolverMixin',
    'WeightedMajoritySolverMixin',
    'BeamSearchSolverMixin',
    'MCTSSolverMixin',
    'DFSSolverMixin',
    'SelfRefineSolverMixin',
    'LEMCTSSolverMixin',
]

