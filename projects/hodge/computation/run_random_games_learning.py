import os
import sys
from collections import deque
from datetime import datetime
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.realpath("/home/oberlechner/code/matrix_game_learning"))

from games_learning.game.matrix_game import RandomMatrixGame
from games_learning.learner.learner import MirrorAscent
from games_learning.simulation import Simulator
from projects.hodge.configs import *
from projects.hodge.util import (
    map_bin_to_potentialness,
    map_potentialness_to_bin,
    save_result,
)


def get_seeds(
    n_actions,
    n_agents,
    n_bins: int,
    n_seeds_per_bin: int,
    distribution: str = "uniform",
    dir: str = "random",
):
    """determine seeds to get n_seeds games for different levels of potentialness"""

    # import logfile from run_random experiment
    file_name = f"{distribution}_{[n_actions]*n_agents}.csv"
    path = os.path.join(PATH_TO_DATA, dir, file_name)
    if not os.path.exists(path):
        print(
            f"File not found ({path})\nYou might have to run the corresponding experiment ,i.e., run_random.py, first"
        )
        return []
    # match potentialness to bins
    tmp = pd.read_csv(path)
    metric = "potentialness" if "potentialness" in tmp.columns else "potentialness_flow"
    tmp["bin"] = map_potentialness_to_bin(tmp[metric], n_bins)

    # get seeds for different levels of potentialness
    seeds = []
    for bin in tmp.bin.unique():
        if sum((tmp.bin == bin) & (tmp.n_strict_ne > 0)) >= n_seeds_per_bin:
            seeds_bin = np.random.choice(
                tmp.seed[tmp.bin == bin], replace=False, size=n_seeds_per_bin
            )
            seeds += [(s, bin) for s in seeds_bin]
        else:
            seeds_bin = tmp.seed[tmp.bin == bin]
    return seeds


def run_learning_stepsizes(
    n_agents: int,
    n_actions: int,
    n_bins,
    n_samples_per_bin: int,
    n_runs: int,
    init: str,
    distribution: str,
    dir: str,
):
    """Apply learning to random games.
    To get games with evenly distributed potentialness, we take some of the sampled games from the previous experiment for different levels of potentialness.
    We try different stepsizes to find the optimal one for each run.

    Args:
        n_agents (int): number of agents
        n_actions (int): number of actions (per agent)
        n_bins (int): divide [0, 1] in equally sized bins
        n_samples_per_bin (int): number of sampled games with potentialness (from subinterval, i.e., bin)
        n_runs (int): number of runs per game
        init (str): initialization of strategy
        distribution (str): distribution of random utiltities
        dir (str): directory to sampled games (from run_random_potentialness.py)
    """
    print(
        f"Run Experiment for {n_agents} agents and {n_actions} actions, {n_runs} runs and {init} initialization:"
    )

    # get seeds which give us equally many games for all levels of potentialness
    seeds = get_seeds(n_actions, n_agents, n_bins, n_samples_per_bin, distribution, dir)
    data = deque()

    for seed, bin in tqdm(seeds):
        actions = [n_actions] * n_agents
        game = RandomMatrixGame(n_agents, actions, seed=seed, distribution=distribution)

        for run in range(N_RUNS):
            init_strat = game.init_strategies(init)

            for eta, beta in product(LIST_ETA, LIST_BETA):
                learner = MirrorAscent(eta=eta, beta=beta, mirror_map="entropic")

                # run experiment
                sim = Simulator(game, learner, MAX_ITER, TOL)
                result = sim.run(init_strat)

                # log results
                result.update(
                    {
                        "seed": seed,
                        "run": run,
                        "bin": bin,
                        "potentialness": map_bin_to_potentialness(bin, n_bins),
                        "eta": eta,
                        "beta": beta,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                data.append(result)

                # if result["convergence"]:
                #    break

    if len(seeds) > 0:
        # save results
        filename = f"{learner.name}_{game.name}_{distribution}_{actions}.csv"
        save_result(data, "random_learning_20runs", filename, PATH_TO_DATA)
    else:
        print(" -> Not enough settings found")


if __name__ == "__main__":
    N_RUNS = 20
    n_bins = 25
    n_samples_per_bin = 100
    distribution = "uniform"
    dir = "random_flow_1e6"
    init = "random"

    settings = [
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 12),
        (2, 24),
        # (3, 2),
        # (3, 3),
        # (3, 4),
        # (3, 5),
        # (4, 2),
        # (4, 4),
        # (8, 2),
        # (10, 2),
    ]

    for n_agents, n_actions in settings:
        run_learning_stepsizes(
            n_agents,
            n_actions,
            n_bins,
            n_samples_per_bin,
            N_RUNS,
            init,
            distribution,
            dir,
        )
