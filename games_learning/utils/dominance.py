from copy import deepcopy
from itertools import product

import numpy as np
from pulp import *

from games_learning.game.matrix_game import MatrixGame


def iterated_dominance_solver(
    game: MatrixGame,
    strict: bool = True,
    atol: float = 1e-9,
    print: bool = True,
):
    """run iterated dominance solver"""
    successful_iter = True
    reduced_game = deepcopy(game)
    removed_actions_index = {agent: [] for agent in game.agents}

    while successful_iter:
        counter_actions_removed = 0
        for agent in game.agents:
            dom_action = find_dominated_action(reduced_game, agent, strict, atol)
            if dom_action is not None:
                reduced_game = remove_action(
                    reduced_game, agent, dom_action, strict, atol
                )
                removed_actions_index[agent].append(dom_action)
                counter_actions_removed += 1
        successful_iter = counter_actions_removed > 0

    removed_actions, remaining_actions = reconstruct_removed_actions(
        game, removed_actions_index
    )

    if print:
        print_result(game, reduced_game, removed_actions)

    return reduced_game, removed_actions, remaining_actions


def remove_action(
    game: MatrixGame, agent: int, action: int, strict: bool = True, atol: float = 1e-9
) -> MatrixGame:
    """remove action for agent in game"""
    payoff_matrix = game.payoff_matrix
    reduced_payoff_matrix = [
        np.delete(matrix, action, axis=agent) for matrix in payoff_matrix
    ]
    reduced_game = MatrixGame(game.n_agents, reduced_payoff_matrix, game.name)
    return reduced_game


def find_dominated_action(
    game: MatrixGame, agent: int, strict: bool = True, atol: float = 1e-9
) -> int:
    """find dominated action in game for agent"""
    n_action_agent = game.n_actions[agent]
    for action in range(n_action_agent):
        if check_if_action_dominated(game, agent, action, strict, atol):
            return action
    return None


def check_if_action_dominated(
    game: MatrixGame, agent: int, action: int, strict: bool = True, atol: float = 1e-9
) -> bool:
    """check if an action is dominated by some mixed strategy"""
    n_action_agent = game.n_actions[agent]
    model = LpProblem("check_action_dom", LpMaximize)
    x = LpVariable.dicts("x", range(n_action_agent), 0, 1)
    y = LpVariable("y", cat="Continuous")
    # objective
    model += y
    # constraints
    for opp_action in generate_opponents_action_profiles(game, agent):
        model += (
            lpSum(
                game.payoff_matrix[agent][
                    generate_action_profiles(opp_action, agent, i)
                ]
                * x[i]
                - y
                for i in range(n_action_agent)
            )
            >= game.payoff_matrix[agent][
                generate_action_profiles(opp_action, agent, action)
            ]
        )
    model += x[action] == 0
    model += lpSum(x[i] for i in range(n_action_agent)) == 1
    solver = getSolver("PULP_CBC_CMD", msg=False)
    result = model.solve(solver=solver)
    return value(model.objective) > atol


def generate_opponents_action_profiles(game: MatrixGame, agent: int) -> Tuple[tuple]:
    """Compute (indices for) all possible action profiles"""
    opp_n_actions = [game.n_actions[i] for i in range(game.n_agents) if i != agent]
    return tuple(i for i in product(*[range(n) for n in opp_n_actions]))


def generate_action_profiles(opponent_profile: Tuple[int], agent: int, index: int):
    """Given a strategy profile, create all deviations of agent i"""
    return opponent_profile[:agent] + (index,) + opponent_profile[agent:]


def reconstruct_removed_actions(game: MatrixGame, removed_actions_index) -> tuple:
    """get the removed and remaining actions from the initial game"""
    removed_actions = {i: [] for i in game.agents}
    actions = {i: list(range(game.n_actions[i])) for i in game.agents}
    for i in game.agents:
        for idx in removed_actions_index[i]:
            removed_actions[i].append(actions[i].pop(idx))
    return removed_actions, actions


def print_result(game, reduced_game, removed_actions):
    print(game, "-->", reduced_game)
    for i in game.agents:
        actions_str = " ".join(
            [
                str(a) if a not in removed_actions[i] else "X"
                for a in range(game.n_actions[i])
            ]
        )
        print(f"Actions Agent {i}: {actions_str}")
