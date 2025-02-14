""" Methods to compute (coarse) (correlated) equilibria"""
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from pulp import *

# ------------------------- PURE NASH EQUILIBRIUM ----------------------------- #


def get_pure_nash_equilibrium(
    payoff_matrix: Tuple[np.ndarray], atol: float = 1e-10
) -> Dict[str, List[tuple]]:
    """Find all pure Nash equilibria (by brute force)

    Args:
        payoff_matrix (Tuple[np.ndarray]): payoff matrices from matrix game
        atol (float, optional): tolerance to distinguish between weak and strict NE. Defaults to 1e-10.

    Returns:
        Dict[str, List[tuple]]: returns "strict_ne", "weak_ne", and all "ne"
    """
    n_actions = payoff_matrix[0].shape
    action_profiles = generate_action_profiles(n_actions)
    weak_ne, strict_ne = [], []
    for a in action_profiles:
        result = check_pure_nash_equilibrium(
            action_profile=a, payoff_matrix=payoff_matrix, atol=atol
        )
        if result == 0:
            weak_ne.append(a)
        elif result == 1:
            strict_ne.append(a)
    return {"weak_ne": weak_ne, "strict_ne": strict_ne, "ne": strict_ne + weak_ne}


def generate_action_profiles(n_actions: List[int]) -> Tuple[tuple]:
    """Compute (indices for) all possible action profiles"""
    return tuple(i for i in product(*[range(n) for n in n_actions]))


def generate_deviations(action_profile: Tuple[int], agent: int, n_actions_agent: int):
    """Given a strategy profile, create all deviations of agent i"""
    for action in range(n_actions_agent):
        if action != action_profile[agent]:
            yield action_profile[:agent] + (action,) + action_profile[agent + 1 :]


def check_pure_nash_equilibrium(
    action_profile: Tuple[int], payoff_matrix: Tuple[np.ndarray], atol: float = 1e-10
) -> int:
    """return -1 if not pure equilibria, 0 if weak, and 1 if strict equilibrium
    atol is necessary due to numerical inaccuracies.
    """
    agents = list(range(len(payoff_matrix)))
    n_actions = payoff_matrix[0].shape

    is_pure = 1
    for i in agents:
        payoff_deviations = np.array(
            tuple(
                payoff_matrix[i][d]
                for d in generate_deviations(action_profile, i, n_actions[i])
            )
        )
        if np.any(payoff_matrix[i][action_profile] < payoff_deviations - atol):
            # one deviation yields higher payoff -> no equilibrium
            return -1
        elif np.all(payoff_matrix[i][action_profile] > payoff_deviations + atol):
            # for this agent, action is strictly better
            is_pure *= 1
        else:
            # if for one agent it isn't strictly better, is_pure stays 0
            is_pure *= 0
    return is_pure


# ------------------------- CORRELATED EQUILIBRIUM ----------------------------- #


def get_correlated_equilibrium(
    payoff_matrix: Tuple[np.ndarray], coarse: bool = True, objective: np.ndarray = None
) -> np.ndarray:
    """Compute one (coarse) correlated equilibrium (C)CE that maximizes some given objective function

    Args:
        payoff_matrix (Tuple[np.ndarray]): payoff matrices from matrix game
        coarse (bool, optional): If True we compute CCE else CE. Defaults to True.
        objective (np.ndarray, optional): Objective to choose specific (C)CE. Defaults to None.

    Returns:
        np.ndarray: (coarse) correlated equilibrium
    """
    n_actions = payoff_matrix[0].shape
    # create objective
    if objective is None:
        objective = np.ones(n_actions)
    assert objective.shape == tuple(n_actions)

    # create lp
    x, lp = create_cce_lp(
        payoff_matrix=payoff_matrix, coarse=coarse, objective=objective
    )

    # optimize
    status = lp.solve(pulp.PULP_CBC_CMD(msg=False))
    if LpStatus[lp.status] != "Optimal":
        print("This should not happen. Fix this!")
        return None
    results = np.array([x[a].varValue for a in x.keys()])
    return results.reshape(n_actions)


def create_cce_lp(
    payoff_matrix: Tuple[np.ndarray], coarse: bool = True, objective: np.ndarray = None
):
    """create LP to compute (C)CE for matrix game

    Args:
        payoff_matrix ( Tuple[np.ndarray]): payoff matrix of matrix game
        coarse (bool, optional): CCE or CE. Defaults to True.
        objective (np.ndarray, optional): objective to select certain (C)CE. Defaults to None.

    Returns:
        Variables, LP (pulp)
    """
    # parameters
    n_actions = payoff_matrix[0].shape
    agents = list(range(len(payoff_matrix)))

    # create problem
    lp = LpProblem("correlated_equilibrium", LpMaximize)
    # create variables
    action_profiles = list(generate_action_profiles(n_actions))
    x = LpVariable.dicts(
        "x",
        (action_profiles),
        lowBound=0,
        upBound=1,
    )
    # objective function
    lp += lpSum(x[a] * objective[a] for a in action_profiles)
    # probability constraint
    lp += lpSum([x[a] for a in action_profiles]) == 1
    # CCE constraints
    if coarse:
        for i in agents:
            for j in range(n_actions[i]):
                exp_util = lpSum([x[a] * payoff_matrix[i][a] for a in action_profiles])
                exp_util_j = lpSum(
                    [
                        x[a] * payoff_matrix[i][a[:i] + (j,) + a[i + 1 :]]
                        for a in action_profiles
                    ]
                )
                lp += exp_util >= exp_util_j
    # CE constraints
    else:
        for i in agents:
            for j1 in range(n_actions[i]):
                for j2 in range(n_actions[i]):
                    exp_util_j1 = lpSum(
                        [
                            x[a] * payoff_matrix[i][a]
                            for a in action_profiles
                            if a[i] == j1
                        ]
                    )
                    exp_util_j2 = lpSum(
                        [
                            x[a] * payoff_matrix[i][a[:i] + (j2,) + a[i + 1 :]]
                            for a in action_profiles
                            if a[i] == j1
                        ]
                    )
                    lp += exp_util_j1 >= exp_util_j2
    return x, lp


def get_support_correlated_equilibria(
    payoff_matrix: Tuple[np.ndarray], coarse: bool = True, atol: float = 1e-10
) -> np.ndarray:
    """Returns if action_profile is part of some (C)CE

    Args:
        payoff_matrix (Tuple[np.ndarray]): payoff matrices from matrix game
        coarse (bool, optional): CCE or CE. Defaults to True.

    Returns:
        np.ndarray
    """
    n_actions = payoff_matrix[0].shape
    result = np.zeros(n_actions, dtype=bool)
    action_profiles = generate_action_profiles(n_actions)

    for a in action_profiles:
        objective = np.zeros(n_actions)
        objective[a] = 1
        x, lp = create_cce_lp(
            payoff_matrix=payoff_matrix, coarse=coarse, objective=objective
        )
        # optimize
        status = lp.solve(pulp.PULP_CBC_CMD(msg=False))
        if LpStatus[lp.status] != "Optimal":
            print("This should not happen. Fix this!")
            return None
        result[a] = x[a].varValue > atol
        del x, lp

    return result


def get_supported_actions_correlated_equilibria(
    payoff_matrix: Tuple[np.ndarray], coarse: bool = True, atol: float = 1e-10
) -> np.ndarray:
    """Returns if action_profile is part of some (C)CE

    Args:
        payoff_matrix (Tuple[np.ndarray]): payoff matrices from matrix game
        coarse (bool, optional): CCE or CE. Defaults to True.

    Returns:
        np.ndarray
    """
    agents = list(range(len(payoff_matrix)))
    n_actions = payoff_matrix[0].shape

    supp_actions = {i: [] for i in agents}
    for i in agents:
        for j in range(n_actions[i]):

            # create objective
            objective = np.zeros(n_actions)
            slices = [slice(None)] * n_agents
            slices[i] = j
            objective[tuple(slices)] = 1

            # optimize
            x, lp = create_cce_lp(game, coarse=coarse, objective=objective)
            status = lp.solve(pulp.PULP_CBC_CMD(msg=False))
            if LpStatus[lp.status] != "Optimal":
                print("This should not happen. Fix this!")
                return None
            supp_actions[i].append(value(lp.objective) > atol)
            del x, lp
    return supp_actions
