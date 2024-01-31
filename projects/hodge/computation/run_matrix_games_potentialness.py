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
from decomposition.game import Game
from tqdm import tqdm

from games_learning.game.matrix_game import JordanGame, MatrixGame, RandomMatrixGame
from games_learning.learner.learner import MirrorAscent
from games_learning.simulation import Simulator
from games_learning.utils.equil import find_pure_nash_equilibrium
from projects.hodge.configs import *
from projects.hodge.util import save_result


def run_matrix_potentialness():
    """compute potentialness for different interesting matrix games"""

    # paramter jordan game
    alpha, beta = 0.5, 0.5

    settings = {
        "shapley_game": [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        ],
        "matching_pennies": [
            np.array([[1, -1], [-1, 1]]),
            np.array([[-1, 1], [1, -1]]),
        ],
        "battle_of_sexes": [np.array([[2, 0], [0, 1]]), np.array([[1, 0], [0, 2]])],
        "battle_of_sexes_agt": [np.array([[6, 1], [2, 6]]), np.array([[5, 1], [2, 6]])],
        "stag_hunt": [np.array([[5, 0], [4, 2]]), np.array([[5, 4], [0, 2]])],
        "prisoners_dilemma": [np.array([[4, 1], [5, 2]]), np.array([[4, 5], [1, 2]])],
    }

    data = deque()
    for key, payoff_matrix in settings.items():
        n_agents = len(payoff_matrix)
        actions = list(payoff_matrix[0].shape)

        game = MatrixGame(len(payoff_matrix), payoff_matrix, name=key)
        pure_equil = find_pure_nash_equilibrium(game)

        hodge = Game(actions, save_load=False)
        hodge.compute_decomposition_matrix(game.payoff_matrix)
        potentialness = hodge.metric

        result = {
            "name": key,
            "actions": actions,
            "potentialness": potentialness,
            "n_weak_ne": len(pure_equil["weak_ne"]),
            "n_strict_ne": len(pure_equil["strict_ne"]),
        }
        data.append(result)

    # Jordan Game (alpha, beta)
    n_seeds = 1_000_000
    hodge = Game([2, 2], save_load=False)
    for seed in tqdm(range(n_seeds)):
        game = JordanGame(seed)
        pure_equil = find_pure_nash_equilibrium(game)
        hodge.compute_flow_decomposition_matrix(game.payoff_matrix)
        potentialness = hodge.flow_metric
        result = {
            "name": game.name,
            "actions": actions,
            "potentialness": potentialness,
            "n_weak_ne": len(pure_equil["weak_ne"]),
            "n_strict_ne": len(pure_equil["strict_ne"]),
        }
        data.append(result)

    # save results
    save_result(data, "matrix_games", f"potentialness.csv", PATH_TO_DATA)


if __name__ == "__main__":
    run_matrix_potentialness()
