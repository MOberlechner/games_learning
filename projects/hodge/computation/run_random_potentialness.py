import os
from collections import deque

import pandas as pd
from decomposition.game import Game
from tqdm import tqdm

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
    data = deque(maxlen=n_samples)
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


if __name__ == "__main__":

    for n_agents in [2, 3]:
        for n in [2, 3, 4, 5]:
            run_sample_potentialness(
                n_actions=[n] * n_agents,
                n_samples=1_000_000,
                distribution="uniform",
                compute_equil=True,
            )
