""" Methods to compute (coarse) (correlated) equilibria"""
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from pulp import *

from games_learning.game.matrix_game import MatrixGame

# ------------------------- PURE NASH EQUILIBRIUM ----------------------------- #

def find_pure_nash_equilibrium(game: MatrixGame) -> Dict[str, List[tuple]]:
    action_profiles = generate_action_profiles(game.n_actions)
    weak_ne, strict_ne = [], []
    for a in action_profiles:
        result = check_pure_nash_equilibrium(a, game.payoff_matrix, game.n_actions)
        if result == 0:
            weak_ne.append(a)
        elif result == 1:
            strict_ne.append(a)
    return {"weak_ne": weak_ne, "strict_ne": strict_ne}


def generate_action_profiles(n_actions: List[int]) -> Tuple[tuple]:
    """Compute (indices for) all possible action profiles"""
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


# ------------------------- CORRELATED EQUILIBRIUM ----------------------------- #

def find_correlated_equilibrium(game: MatrixGame, coarse: bool=True, objective: np.ndarray=None) -> np.ndarray:
    """Compute one (coarse) correlated equilibrium (C)CE that maximizes some given objective function

    Args:
        game (MatrixGame): matrix game
        coarse (bool, optional): If True we compute CCE else CE. Defaults to True.
        objective (np.ndarray, optional): Objective to choose specific (C)CE. Defaults to None.

    Returns:
        np.ndarray: (coarse) correlated equilibrium
    """ 
    
    # create new problem
    lp = LpProblem("correlated_equilibrium", LpMaximize)
    
    # create variables
    action_profiles = list(generate_action_profiles(game.n_actions))
    x = LpVariable.dicts(
        "x",
        (action_profiles),
        lowBound = 0,
        upBound = 1,
    )
    
    # probability constraint
    lp += (lpSum([x[a] for a in action_profiles]) == 1)    
    
    # CCE constraints
    if coarse:
        for i in game.agents:
            for j in range(game.n_actions[i]):
                exp_util = lpSum([x[a]*game.payoff_matrix[i][a] for a in action_profiles])
                exp_util_j = lpSum([x[a]*game.payoff_matrix[i][a[:i] + (j,) + a[i+1:]] for a in action_profiles])
                lp += (exp_util >= exp_util_j)
    else:
        print("CE not implemented")
        raise NotImplementedError
    
    # optimize
    status = lp.solve(pulp.PULP_CBC_CMD(msg=False))
    if LpStatus[lp.status] != "Optimal":
        print("This should not happen. Fix this!")
        return None

    results = np.array([x[a].varValue for a in action_profiles])
    return results.reshape(game.n_actions)
                
                
                
            
                
        
        
    
    