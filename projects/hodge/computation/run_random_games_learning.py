import multiprocessing
import os
import sys
from collections import deque
from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from games_learning.game.matrix_game import RandomMatrixGame
from games_learning.learner.mirror_ascent import MirrorAscent
from games_learning.simulation import Simulator
from games_learning.strategy import Strategy
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
        elif sum((tmp.bin == bin) & (tmp.n_strict_ne > 0)) >= 1:
            seeds_bin = tmp.seed[tmp.bin == bin]
            seeds += [(s, bin) for s in seeds_bin]
    return seeds


def run_learning_stepsizes(
    n_agents: int,
    actions: int,
    n_bins,
    n_samples_per_bin: int,
    n_runs: int,
    init: str,
    distribution: str,
    dir: str,
    dir_save: str,
):
    """Apply learning to random games.
    To get games with evenly distributed potentialness, we take some of the sampled games from the previous experiment for different levels of potentialness.
    We try different stepsizes to find the optimal one for each run.

    Args:
        n_agents (int): number of agents
        actions (int): number of actions (per agent)
        n_bins (int): divide [0, 1] in equally sized bins
        n_samples_per_bin (int): number of sampled games with potentialness (from subinterval, i.e., bin)
        n_runs (int): number of runs per game
        init (str): initialization of strategy
        distribution (str): distribution of random utiltities
        dir (str): directory to sampled games (from run_random_potentialness.py)
        dir_save (str): directory to store results
    """
    print(
        f"Run Experiment for {n_agents} agents and {actions} actions, {n_runs} runs and {init} initialization:"
    )

    # get seeds which give us equally many games for all levels of potentialness
    seeds = get_seeds(actions, n_agents, n_bins, n_samples_per_bin, distribution, dir)
    data = deque()

    for seed, bin in tqdm(seeds):
        n_actions = [actions] * n_agents
        game = RandomMatrixGame(
            n_actions=n_actions, seed=seed, distribution=distribution
        )

        for run in range(n_runs):
            init_strategies = Strategy(game, init_method="random")

            for eta, beta in product(LIST_ETA, LIST_BETA):
                learner = MirrorAscent(eta=eta, beta=beta, mirror_map="entropic")

                # run experiment
                strategies = deepcopy(
                    init_strategies
                )  # use same initial strategy for different learner
                sim = Simulator(strategies, learner, max_iter=MAX_ITER, tol=TOL)
                result = sim.run()

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
        save_result(data, dir_save, filename, PATH_TO_DATA)
    else:
        print(" -> Not enough settings found")


def run_learning_mp(
    settings: List[tuple],
    n_bins: int,
    n_samples_per_bin: int,
    n_runs: int,
    init: str,
    distribution: str,
    dir: str,
    dir_save: str,
    num_processes: int = 8,
):
    """multiprocessing"""

    if num_processes > multiprocessing.cpu_count():
        print(f"Only {multiprocessing.cpu_count()} cpus available")
        num_processes = multiprocessing.cpu_count()

    # create function that runs in parallel
    func = partial(
        run_learning_stepsizes,
        n_bins=n_bins,
        n_samples_per_bin=n_samples_per_bin,
        n_runs=n_runs,
        init=init,
        distribution=distribution,
        dir=dir,
        dir_save=dir_save,
    )

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(func, settings)


if __name__ == "__main__":
    n_runs = 25
    n_bins = 20
    n_samples_per_bin = 100
    distribution = "uniform"
    dir = "random_flow_1e6"
    dir_save = f"random_learning_{n_runs}run"
    init = "random"

    settings = SETTINGS

    run_learning_mp(
        settings,
        n_bins,
        n_samples_per_bin,
        n_runs,
        init,
        distribution,
        dir,
        dir_save,
        num_processes=8,
    )
