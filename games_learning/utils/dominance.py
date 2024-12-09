from copy import copy
from itertools import product
from typing import Tuple

import numpy as np
from pulp import *


def iterated_dominance_solver(
    payoff_matrix: Tuple[np.ndarray],
    dominance: str = "strict",
    atol: float = 1e-9,
    print: bool = True,
):
    """run iterated dominance solver"""
    agents, n_actions = game_parameters(payoff_matrix)
    successful_iter = True
    reduced_payoff_matrix = copy(payoff_matrix)
    removed_actions_index = {agent: [] for agent in agents}
    while successful_iter:
        counter_actions_removed = 0
        for agent in agents:
            dom_action = find_dominated_action(
                reduced_payoff_matrix, agent, dominance, atol
            )
            if dom_action is not None:
                reduced_payoff_matrix = remove_action(
                    reduced_payoff_matrix, agent, dom_action, atol
                )
                removed_actions_index[agent].append(dom_action)
                counter_actions_removed += 1
        successful_iter = counter_actions_removed > 0

    removed_actions, remaining_actions = reconstruct_removed_actions(
        payoff_matrix, removed_actions_index
    )
    if print:
        print_result(payoff_matrix, reduced_payoff_matrix, removed_actions)
    return reduced_payoff_matrix, removed_actions, remaining_actions


def remove_action(
    payoff_matrix: Tuple[np.ndarray], agent: int, action: int, atol: float = 1e-9
) -> Tuple[np.ndarray]:
    """remove entries of given action of agent in payoff matrix"""
    reduced_payoff_matrix = [
        np.delete(matrix, action, axis=agent) for matrix in payoff_matrix
    ]
    return reduced_payoff_matrix


def find_dominated_action(
    payoff_matrix: Tuple[np.ndarray],
    agent: int,
    dominance: str = "strict",
    atol: float = 1e-9,
) -> int:
    """find dominated action in game for agent"""
    n_action_agent = payoff_matrix[0].shape[agent]
    if dominance == "strict":
        for action in range(n_action_agent):
            if check_if_action_strictly_dominated(payoff_matrix, agent, action, atol):
                return action
        return None

    elif dominance == "strong":
        for action in range(n_action_agent):
            if check_if_action_strongly_dominated(payoff_matrix, agent, action, atol):
                return action
        return None
    else:
        raise ValueError


def check_if_action_strongly_dominated(
    payoff_matrix: Tuple[np.ndarray], agent: int, action: int, atol: float = 1e-9
) -> bool:
    """check if action is dominated by some pure action"""
    raise NotImplementedError


def check_if_action_strictly_dominated(
    payoff_matrix: Tuple[np.ndarray], agent: int, action: int, atol: float = 1e-9
) -> bool:
    """check if an action is dominated by some mixed strategy"""
    n_action_agent = payoff_matrix[agent].shape[agent]
    model = LpProblem("check_action_dom", LpMaximize)
    x = LpVariable.dicts("x", range(n_action_agent), 0, 1)
    y = LpVariable("y", cat="Continuous")
    # objective
    model += y
    # constraints
    for opp_action in generate_opponents_action_profiles(payoff_matrix, agent):
        model += (
            lpSum(
                payoff_matrix[agent][generate_action_profiles(opp_action, agent, i)]
                * x[i]
                - y
                for i in range(n_action_agent)
            )
            >= payoff_matrix[agent][generate_action_profiles(opp_action, agent, action)]
        )
    model += x[action] == 0
    model += lpSum(x[i] for i in range(n_action_agent)) == 1
    solver = getSolver("PULP_CBC_CMD", msg=False)
    result = model.solve(solver=solver)
    return value(model.objective) > atol


def generate_opponents_action_profiles(
    payoff_matrix: Tuple[np.ndarray], agent: int
) -> Tuple[tuple]:
    """Compute (indices for) all possible action profiles"""
    agents, n_actions = game_parameters(payoff_matrix)
    opp_n_actions = [n_actions[i] for i in agents if i != agent]
    return tuple(i for i in product(*[range(n) for n in opp_n_actions]))


def generate_action_profiles(opponent_profile: Tuple[int], agent: int, index: int):
    """Given a strategy profile, create all deviations of agent i"""
    return opponent_profile[:agent] + (index,) + opponent_profile[agent:]


def reconstruct_removed_actions(
    payoff_matrix: Tuple[np.ndarray], removed_actions_index
) -> tuple:
    """get the removed and remaining actions from the initial game"""
    agents, n_actions = game_parameters(payoff_matrix)
    removed_actions = {i: [] for i in agents}
    actions = {i: list(range(n_actions[i])) for i in agents}
    for i in agents:
        for idx in removed_actions_index[i]:
            removed_actions[i].append(actions[i].pop(idx))
    return removed_actions, actions


def print_result(payoff_matrix, reduced_payoff_matrix, removed_actions):
    agents, n_actions = game_parameters(payoff_matrix)
    _, n_actions_reduced = game_parameters(reduced_payoff_matrix)

    print(n_actions, "-->", n_actions_reduced)
    for i in agents:
        actions_str = " ".join(
            [
                str(a) if a not in removed_actions[i] else "X"
                for a in range(n_actions[i])
            ]
        )
        print(f"Actions Agent {i}: {actions_str}")


def game_parameters(payoff_matrix) -> Tuple[List, List]:
    agents = list(range(len(payoff_matrix)))
    n_actions = payoff_matrix[0].shape
    return agents, n_actions
