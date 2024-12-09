import os
import sys

sys.path.append(os.path.realpath("."))

import math
from collections import deque
from datetime import datetime
from functools import partial
from itertools import chain, product
from typing import List, Tuple

import numpy as np
from games_decomposition.game import Game
from tqdm import tqdm

from games_learning.game.bayesian_game import BayesianEconGame
from games_learning.game.econ_game import (
    FPSB,
    SPSB,
    AllPay,
    Contest,
    EconGame,
    WarOfAttrition,
)
from projects.hodge.configs import *
from projects.hodge.util import save_result


def get_types(n_types: int) -> List:
    if n_types == 1:
        return np.array([1.0])
    elif n_types == 2:
        return np.array([0.5, 1])
    elif n_types == 3:
        return np.array([0.3, 0.6, 1.0])
    else:
        return None


def get_number_strategies(n_types, n_actions) -> int:
    return math.comb(n_types + n_actions - 1, n_actions - 1)


def run_econgames_bayesian_potentialness(
    list_n_types: List[int], compute_equil: bool = False
):
    """compute potentialness of bayesian version of economic games"""
    # create complete-information games
    n_agents = 2
    valuations = (1.0, 1.0)
    n_actions = 4
    interval = (0.0, 0.95)
    games = [
        FPSB(n_agents, n_actions, valuations=valuations, interval=interval),
        SPSB(n_agents, n_actions, valuations=valuations, interval=interval),
        AllPay(n_agents, n_actions, valuations=valuations, interval=interval),
        WarOfAttrition(n_agents, n_actions, valuations=valuations, interval=interval),
        Contest(
            n_agents, n_actions, valuations=valuations, interval=interval, csf_param=1.0
        ),
    ]

    data = deque()
    for n_types in list_n_types:
        types = get_types(n_types)
        n_strategies = get_number_strategies(n_types, n_actions)
        hodge = Game([n_strategies] * n_agents, save_load=True, path=PATH_TO_DATA)
        for game in games:
            bayesian_game = BayesianEconGame(
                game, types=types, distribution="uniform", monotone_strategies=True
            )
            hodge.compute_flow_decomposition_matrix(bayesian_game.payoff_matrix)
            potentialness = hodge.flow_metric
            result = {
                "game": bayesian_game.name,
                "n_agents": n_agents,
                "n_actions": n_actions,
                "n_types": n_types,
                "n_strategies": bayesian_game.n_actions[0],
                "potentialness": potentialness,
            }
            # compute pure equilibria
            if compute_equil:
                pure_equil = bayesian_game.get_pne()
                equilibria = {
                    "n_weak_ne": len(pure_equil["weak_ne"]),
                    "n_strict_ne": len(pure_equil["strict_ne"]),
                }
                result.update(equilibria)
            # log result
            result.update({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            data.append(result)

    # save results
    save_result(
        data, "econgames", f"potentialness_bayesian.csv", PATH_TO_DATA, overwrite=False
    )


if __name__ == "__main__":
    list_n_types = [1, 2, 3, 4]
    compute_equil = True
    run_econgames_bayesian_potentialness(list_n_types, compute_equil)
