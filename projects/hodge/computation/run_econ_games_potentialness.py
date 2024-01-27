import multiprocessing
import os
import sys
from collections import deque
from datetime import datetime
from functools import partial
from itertools import chain, product
from typing import List, Tuple

import numpy as np
import pandas as pd
from decomposition.game import Game
from tqdm import tqdm

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest, EconGame
from games_learning.game.matrix_game import MatrixGame, RandomMatrixGame
from games_learning.learner.learner import MirrorAscent
from games_learning.simulation import Simulator
from games_learning.utils.equil import find_pure_nash_equilibrium
from projects.hodge.configs import *
from projects.hodge.util import save_result


def run_matrix_potentialness():
    """compute potentialness for different interesting matrix games"""
    settings = {
        "shapley_game": [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        ],
    }

    data = deque()
    for key, payoff_matrix in settings.items():
        n_agents = len(payoff_matrix)
        actions = list(payoff_matrix[0].shape)

        game = MatrixGame(len(payoff_matrix), payoff_matrix, name=key)
        pure_equil = find_pure_nash_equilibrium(game)

        hodge = Game(actions, save_load=False)
        hodge.compute_decomposition_matrix(game.payoff_matrix)
        potentialness = hodge.metric

        result = {
            "name": key,
            "actions": actions,
            "potentialness": potentialness,
            "n_weak_ne": len(pure_equil["weak_ne"]),
            "n_strict_ne": len(pure_equil["strict_ne"]),
            "weak_ne": pure_equil["weak_ne"],
            "strict_new": pure_equil["strict_ne"],
        }
        data.append(result)

    # save results
    save_result(data, "matrix_games", f"potentialness.csv", PATH_TO_DATA)


def get_valuations(setting: str, n_agents: int) -> Tuple[float]:
    if setting == "symmetric":
        return tuple([1.0] * n_agents)
    elif setting == "asymmetric":
        return tuple([0.75] + [1.0] * (n_agents - 1))
    else:
        raise ValueError(f"valuation setting '{setting}' not available")


def run_econgames_potentialness(
    list_actions: List[int],
    interval: tuple = (0.00, 0.95),
    compute_equil: bool = False,
):
    """compute potentialness for econgames"""
    data = deque()

    for actions in tqdm(list_actions):

        # prepare decomposition
        n_agents = len(actions)
        n_actions = actions[0]
        hodge = Game(actions, save_load=True, path=PATH_TO_DATA)

        for val_setting in ["symmetric", "asymmetric"]:

            # create game
            valuations = get_valuations(val_setting, n_agents)
            games = [
                FPSB(n_agents, n_actions, valuations=valuations, interval=interval),
                SPSB(n_agents, n_actions, valuations=valuations, interval=interval),
                AllPay(n_agents, n_actions, valuations=valuations, interval=interval),
                Contest(
                    n_agents,
                    n_actions,
                    valuations=valuations,
                    interval=interval,
                    csf_param=1.0,
                ),
            ]

            for game in games:
                # compute decomposition
                hodge.compute_flow_decomposition_matrix(game.payoff_matrix)
                potentialness = hodge.flow_metric
                result = {
                    "game": game.name,
                    "n_agents": n_agents,
                    "n_discr": n_actions,
                    "valuation": val_setting,
                    "interval": interval,
                    "potentialness": potentialness,
                }
                # compute pure equilibria
                if compute_equil:
                    pure_equil = find_pure_nash_equilibrium(game)
                    equilibria = {
                        "n_weak_ne": len(pure_equil["weak_ne"]),
                        "n_strict_ne": len(pure_equil["strict_ne"]),
                    }
                    result.update(equilibria)
                # log result
                result.update(
                    {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                )
                data.append(result)

    # save results
    save_result(data, "econgames", f"potentialness.csv", PATH_TO_DATA, overwrite=False)


if __name__ == "__main__":

    # compute potentialness for econcames
    list_n_discr = range(5, 26)
    list_actions = [[n_discr, n_discr] for n_discr in list_n_discr]
    run_econgames_potentialness(list_actions, interval=(0.0, 0.95), compute_equil=True)