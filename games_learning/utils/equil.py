""" Methods to compute (coarse) (correlated) equilibria"""
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from games_learning.game.matrix_game import MatrixGame


def find_pure_nash_equilibrium(game: MatrixGame) -> Dict[str, List[tuple]]:
    action_profiles = generate_action_profiles(game.n_actions)
    weak_ne, strict_ne = [], []
    for a in action_profiles:
        result = check_pure_nash_equilibrium(a, game.payoff_matrix, game.n_actions)
        if result == 0:
            weak_ne.append(a)
        elif result == 1:
            strict_ne.append(a)
    return {"weak_ne": weak_ne, "strict_new": strict_ne}


def generate_action_profiles(n_actions: List[int]) -> Tuple[tuple]:
    """Compute all possible action profiles"""
    return tuple(i for i in product(*[range(n) for n in n_actions]))


def generate_deviations(action_profile: Tuple[int], agent: int, n_actions_agent: int):
    """Given a strategy profile, create all deviations of agent i"""
    for action in range(n_actions_agent):
        if action != action_profile[agent]:
            yield action_profile[:agent] + (action,) + action_profile[agent + 1 :]


def check_pure_nash_equilibrium(
    action_profile: Tuple[int], payoff_matrix: Tuple[np.ndarray], n_actions: List[int]
) -> int:
    """return -1 if not pure equilibria, 0 if weak, and 1 if strict equilibrium"""
    n_agents = len(n_actions)
    is_pure = 1
    for i in range(n_agents):
        payoff_deviations = np.array(
            tuple(
                payoff_matrix[i][d]
                for d in generate_deviations(action_profile, i, n_actions[i])
            )
        )
        if np.any(payoff_matrix[i][action_profile] < payoff_deviations):
            # one deviation yields higher payoff -> no equilibrium
            return -1
        elif np.all(payoff_matrix[i][action_profile] > payoff_deviations):
            # for this agent, action is strictly better
            is_pure *= 1
        else:
            # if for one agent it isn't strictly better, is_pure stays 0
            is_pure *= 0
    return is_pure
