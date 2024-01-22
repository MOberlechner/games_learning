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
    seeds: List[int], hodge: Game, actions: List[int], distribution: str, compute_equil: bool = False, flow: bool = False
):
    """create random games and check potentialness"""
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
    compute_equil: bool = False,
    flow: bool = False,
    num_processes: int = 1,
):
    if num_processes > multiprocessing.cpu_count():
        print(f"Only {multiprocessing.cpu_count()} process available")
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
    dir = "random_flow" if flow else "random"
    save_result(data, dir, f"{distribution}_{actions}.csv", PATH_TO_DATA)


if __name__ == "__main__":

    # compute potentialness for random games
    for n_agents in [12]:
        for n_actions in [2]:
            actions = [n_actions] * n_agents
            print(f"Experiment: {actions}")
            run_random_potentialness_mp(
                actions=actions,
                n_samples=1_000_000,
                distribution="uniform",
                compute_equil=True,
                flow=True,
                num_processes=8,
            )
