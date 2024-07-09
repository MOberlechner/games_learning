import multiprocessing
import os
import sys
from collections import deque
from datetime import datetime
from functools import partial
from itertools import chain, product
from typing import List

sys.path.append(os.path.realpath("/home/oberlechner/code/matrix_game_learning"))

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


def run_random_potentialness(
    seeds: List[int],
    hodge: Game,
    actions: List[int],
    distribution: str,
    compute_equil: bool = False,
    flow: bool = False,
):
    """Subroutine for method below"""
    data = deque()
    n_agents = hodge.n_agents

    for seed in tqdm(seeds):
        game = RandomMatrixGame(n_agents, actions, seed=seed, distribution=distribution)
        if flow:
            hodge.compute_flow_decomposition_matrix(game.payoff_matrix)
            potentialness = hodge.flow_metric
            result = {
                "seed": seed,
                "potentialness_flow": potentialness,
            }
        else:
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
                "n_strict_ne": len(pure_equil["strict_ne"]),
            }
            result.update(equilibria)

        # log result
        data.append(result)

    return data


def run_random_potentialness_mp(
    actions: List[int],
    n_samples: int,
    distribution: str,
    dir: str,
    compute_equil: bool = False,
    flow: bool = False,
    num_processes: int = 1,
):
    """Sample random games and compute potentialness (and equilibria)
    Note that multiprocessing only useful for smaller settings

    Args:
        actions (List[int]): setting, e.g. [3, 3] = 2 agents with 3 actions
        n_samples (int): number of samples
        distribution (str): distribution of generated payoff entries
        dir (str): directory to save results
        compute_equil (bool, optional): compute pure equilibria. Defaults to False.
        flow (bool, optional): use flow decomposition (faster). Defaults to False.
        num_processes (int, optional): number of processes. Defaults to 1.
    """

    if num_processes > multiprocessing.cpu_count():
        print(f"Only {multiprocessing.cpu_count()} cpus available")
        num_processes = multiprocessing.cpu_count()

    # create game (structure)
    hodge = Game(
        actions,
        save_load=False,
    )

    # create function that runs in parallel
    func = partial(
        run_random_potentialness,
        hodge=hodge,
        actions=actions,
        distribution=distribution,
        compute_equil=compute_equil,
        flow=flow,
    )

    # create seeds for different processes
    n_seeds = n_samples // num_processes + 1
    seeds = [
        list(range(i * n_seeds, min((i + 1) * n_seeds, n_samples)))
        for i in range(num_processes)
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(func, seeds)

    # save results
    data = deque(chain.from_iterable(results))
    save_result(data, dir, f"{distribution}_{actions}.csv", PATH_TO_DATA)


if __name__ == "__main__":

    # compute potentialness for random games
    settings = SETTINGS
    for n_agents, n_actions in settings:
        actions = [n_actions] * n_agents
        print(f"Experiment: {actions}")
        run_random_potentialness_mp(
            actions=actions,
            n_samples=1_000_000,
            distribution="uniform",
            dir="random_flow_1e6",
            compute_equil=True,
            flow=True,
            num_processes=4,
        )
