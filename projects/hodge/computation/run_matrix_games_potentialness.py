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

from games_learning.game.matrix_game import ExampleMatrixGames
from projects.hodge.configs import *
from projects.hodge.util import save_result


def run_matrix_potentialness():
    """compute potentialness for different example matrix games"""

    settings = [
        "matching_pennies",
        "battle_of_sexes",
        "prisoners_dilemma",
        "shapley_game",
    ]
    data = deque()
    for setting in settings:
        game = ExampleMatrixGames(setting=setting)
        pure_equil = game.get_pne()
        potentialness = game.get_potentialness()
        result = {
            "name": setting,
            "actions": game.n_actions,
            "potentialness": potentialness,
            "n_weak_ne": len(pure_equil["weak_ne"]),
            "n_strict_ne": len(pure_equil["strict_ne"]),
        }
        data.append(result)

    # Jordan Game (alpha, beta)
    n_seeds = 1_000_000
    hodge = Game([2, 2], save_load=False)
    for seed in tqdm(range(n_seeds)):
        game = ExampleMatrixGames(setting="jordan_game", parameter={"seed": seed})
        pure_equil = game.get_pne()
        hodge.compute_flow_decomposition_matrix(game.payoff_matrix)
        potentialness = hodge.flow_metric
        result = {
            "name": game.name,
            "actions": game.n_actions,
            "potentialness": potentialness,
            "n_weak_ne": len(pure_equil["weak_ne"]),
            "n_strict_ne": len(pure_equil["strict_ne"]),
        }
        data.append(result)

    # save results
    save_result(data, "matrix_games", f"potentialness.csv", PATH_TO_DATA)


if __name__ == "__main__":
    run_matrix_potentialness()
