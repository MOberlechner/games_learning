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
from games_decomposition.game import Game
from tqdm import tqdm

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest, EconGame
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
    n_agents: int,
    list_n_discr: List[int],
    interval: tuple = (0.00, 0.95),
    compute_equil: bool = False,
):
    """compute potentialness for econgames"""
    data = deque()

    for n_discr in tqdm(list_n_discr):

        # prepare decomposition
        n_actions = [n_discr] * n_agents
        hodge = Game(n_actions, save_load=True, path=PATH_TO_DATA)

        for val_setting in ["symmetric", "asymmetric"]:

            # create game
            valuations = get_valuations(val_setting, n_agents)
            games = [
                FPSB(n_agents, n_discr, valuations=valuations, interval=interval),
                SPSB(n_agents, n_discr, valuations=valuations, interval=interval),
                AllPay(n_agents, n_discr, valuations=valuations, interval=interval),
                Contest(
                    n_agents,
                    n_discr,
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
                    "n_discr": n_discr,
                    "valuation": val_setting,
                    "interval": interval,
                    "potentialness": potentialness,
                }
                # compute pure equilibria
                if compute_equil:
                    pure_equil = game.get_pne()
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
    n_agents = 2
    list_n_discr = range(5, 26)
    run_econgames_potentialness(
        n_agents, list_n_discr, interval=(0.0, 0.95), compute_equil=True
    )
