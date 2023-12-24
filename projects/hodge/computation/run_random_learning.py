import os
from collections import deque
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

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
):
    """determine seeds to get n_seeds games for different levels of potentialness"""

    # import logfile from run_random experiment
    file_name = f"{distribution}_{[n_actions]*n_agents}.csv"
    path = os.path.join(PATH_TO_DATA, "random", file_name)
    if not os.path.exists(path):
        print(
            f"File not found ({path})\nYou might have to run the corresponding experiment ,i.e., run_random.py, first"
        )
        return []
    # match potentialness to bins
    tmp = pd.read_csv(path)
    tmp["bin"] = map_potentialness_to_bin(tmp.potentialness, n_bins)

    # get seeds for different levels of potentialness
    seeds = []
    for bin in tmp.bin.unique():
        if sum(tmp.bin == bin) >= n_seeds_per_bin:
            seeds_bin = np.random.choice(
                tmp.seed[tmp.bin == bin], replace=False, size=n_seeds_per_bin
            )
            seeds += [(s, bin) for s in seeds_bin]
    return seeds


def run_learning(
    n_agents: int,
    n_actions: int,
    n_bins,
    n_samples_per_bin: int,
    n_runs: int,
    distribution: str,
):
    """Apply learning to random games. To get games with evenly distributed potentialness, we take some of the sampled games from the previous experiment for different levels of potentialness

    Args:
        n_agents (int): number of agents
        n_actions (int): number of actions (per agent)
        n_bins (int): divide [0, 1] in equally sized bins
        n_samples_per_bin (int): number of sampled games with potentialness (from subinterval, i.e., bin)
        n_runs (int): number of runs per game
        distribution (str): distribution of random utiltities
    """
    print(f"Run Experiment for {n_agents} agents and {n_actions} actions:")

    # get seeds which give us equally many games for all levels of potentialness
    seeds = get_seeds(n_actions, n_agents, n_bins, n_samples_per_bin, distribution)
    data = deque()

    for seed, bin in tqdm(seeds):
        game = RandomMatrixGame(
            n_agents, [n_actions] * n_agents, seed=seed, distribution=distribution
        )
        for run in range(n_runs):
            for eta, beta in product(LIST_ETA, LIST_BETA):
                learner = MirrorAscent(eta=eta, beta=beta, mirror_map="entropic")
                sim = Simulator(game, learner, MAX_ITER, TOL)
                init_strat = game.init_strategies("random")
                result = sim.run(init_strat)
                result.update(
                    {
                        "seed": seed,
                        "run": run,
                        "bin": bin,
                        "potentialness": map_bin_to_potentialness(bin, n_bins),
                        "eta": eta,
                    }
                )
                # log result
                data.append(result)

                # if we found suitable (eta,beta) we go to the next run/seed
                if result["convergence"]:
                    break

    # save results
    filename = f"{learner.name}_{game.name}_{distribution}_{n_agents}_{n_actions}.csv"
    save_result(data, "random_learning_stepsize", filename, PATH_TO_DATA)


if __name__ == "__main__":
    n_bins = 20
    n_samples_per_bin = 25
    distribution = "uniform"

    for n_agents in [2]:
        for n_actions in [2]:
            run_learning(
                n_agents, n_actions, n_bins, n_samples_per_bin, N_RUNS, distribution
            )