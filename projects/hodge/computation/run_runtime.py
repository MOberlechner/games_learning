import os
import sys
from time import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.realpath("/home/oberlechner/code/matrix_game_learning"))

from games_decomposition.game import Game

from projects.hodge.configs import *
from projects.hodge.util import save_result


def time_hodge(n_actions: List[int], n_runs: int = 100):
    t0 = time()
    hodge = Game(
        n_actions,
        flow_space=True,
        save_load=False,
    )
    t1 = time()
    payoff_vector = np.random.random(size=len(n_actions) * np.prod(n_actions))
    for _ in range(n_runs):
        hodge.compute_flow_decomposition(payoff_vector)
    t2 = time()
    return t1 - t0, (t2 - t1) / n_runs, hodge.structure.exact_projection.shape


def runtime_flow(settings: List[tuple], dir: str, filename: str, n_runs: int):
    """runtime flow

    Args:
        settings (List[tuple]): settings (tuples of n_agents and n_actions)
        dir (str): directory where csv is saved
        filename (str): filename
    """
    data = []
    for n_agents, actions in settings:
        n_actions = [actions] * n_agents
        runtime_str, runtime_dec, shape_ex_proj = time_hodge(n_actions, n_runs)
        print(
            f"agents: {n_agents}, actions: {actions:2}, time str. {runtime_str:.3f}, time dec. {runtime_dec:.3f}, shape projection e: {shape_ex_proj}"
        )
        data.append(
            {
                "n_agents": n_agents,
                "n_actions": actions,
                "time_structure": runtime_str,
                "time_decomposition": runtime_dec,
            }
        )
    save_result(data, dir, filename, PATH_TO_DATA)


if __name__ == "__main__":

    settings = (
        [(2, actions) for actions in range(2, 32 + 1)]
        + [(3, actions) for actions in range(2, 12 + 1)]
        + [(4, actions) for actions in range(2, 7 + 1)]
    )
    runtime_flow(settings, "runtime", "flow_space.csv", n_runs=100)
