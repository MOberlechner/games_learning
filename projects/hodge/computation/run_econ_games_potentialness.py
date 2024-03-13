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


def get_valuations(setting: str, n_agents: int) -> Tuple[float]:
    if setting == "symmetric":
        return tuple([1.0] * n_agents)
    elif setting == "asymmetric":
        return tuple([0.75] + [1.0] * (n_agents - 1))
    else:
        raise ValueError(f"valuation setting '{setting}' not available")


def run_econgames_potentialness(
    list_actions: List[int],
    action_space: tuple = (0.00, 0.95),
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
            valuation = get_valuations(val_setting, n_agents)
            games = [
                FPSB(n_agents, action_space, valuation),
                SPSB(n_agents, action_space, valuation),
                AllPay(n_agents, action_space, valuation),
                Contest(n_agents, action_space, valuation, csf_param=1.0),
            ]

            for game in games:
                # compute decomposition
                matrix_game = game.discretize(n_actions)
                hodge.compute_flow_decomposition_matrix(matrix_game.payoff_matrix)
                potentialness = hodge.flow_metric
                result = {
                    "game": matrix_game.name,
                    "n_agents": n_agents,
                    "n_discr": n_actions,
                    "valuation": val_setting,
                    "interval": action_space,
                    "potentialness": potentialness,
                }
                # compute pure equilibria
                if compute_equil:
                    pure_equil = find_pure_nash_equilibrium(matrix_game)
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
    run_econgames_potentialness(
        list_actions, action_space=(0.0, 0.95), compute_equil=True
    )
