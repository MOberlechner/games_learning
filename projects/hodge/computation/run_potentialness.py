import os
from collections import deque
from datetime import datetime
from itertools import product
from typing import List

import pandas as pd
from decomposition.game import Game
from tqdm import tqdm

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest, EconGame
from games_learning.game.matrix_game import RandomMatrixGame
from games_learning.learner.learner import MirrorAscent
from games_learning.simulation import Simulator
from games_learning.utils.equil import find_pure_nash_equilibrium
from projects.hodge.configs import *
from projects.hodge.util import save_result


def run_sample_potentialness(
    n_actions: int, n_samples: int, distribution: str, compute_equil: bool = False
):
    """create random games and check potentialness"""
    data = deque()
    hodge = Game(n_actions, save_load=False)
    n_agents = len(n_actions)

    for seed in tqdm(range(n_samples)):
        game = RandomMatrixGame(
            n_agents, n_actions, seed=seed, distribution=distribution
        )
        hodge.compute_decomposition_matrix(game.payoff_matrix)
        potentialness = hodge.metric
        result = {
            "seed": seed,
            "potentialness": potentialness,
        }
        if compute_equil:
            pure_equil = find_pure_nash_equilibrium(game)
            equilibria = {
                "n_weak_ne": len(pure_equil["weak_ne"]),
                "n_strict_ne": len(pure_equil["strict_new"]),
            }
            result.update(equilibria)

        # log result
        data.append(result)

    # save results
    save_result(data, "random", f"{distribution}_{n_actions}.csv", PATH_TO_DATA)


def run_econgames_potentialness(
    list_n_actions: List[int],
    interval: tuple = (0.00, 0.95),
    compute_equil: bool = False,
):
    """compute potentialness for econgames"""
    data = deque()

    for n_actions in tqdm(list_n_actions):

        # prepare decomposition
        n_agents = len(n_actions)
        n_discr = n_actions[0]
        hodge = Game(n_actions, save_load=False)

        # create game
        games = [
            FPSB(n_agents, n_discr, interval=interval),
            SPSB(n_agents, n_discr, interval=interval),
            AllPay(n_agents, n_discr, interval=interval),
            Contest(n_agents, n_discr, interval=interval, csf_param=1.0),
        ]

        for game in games:
            # compute decomposition
            hodge.compute_decomposition_matrix(game.payoff_matrix)
            potentialness = hodge.metric
            result = {
                "game": game.name,
                "n_agents": n_agents,
                "n_discr": n_discr,
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
            result.update({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            data.append(result)

    # save results
    save_result(data, "econgames", f"potentialness.csv", PATH_TO_DATA, overwrite=False)


if __name__ == "__main__":

    for n_agents in [2, 3]:
        for n in [2, 3, 4, 5]:
            run_sample_potentialness(
                n_actions=[n] * n_agents,
                n_samples=1_000_000,
                distribution="uniform",
                compute_equil=True,
            )

    list_n_actions = [
        [5, 5],
        [8, 8],
        [11, 11],
        [16, 16],
        [24, 24],
        [5, 5, 5],
        [8, 8, 8],
    ]
    run_econgames_potentialness(
        list_n_actions, interval=(0.00, 0.95), compute_equil=True
    )
