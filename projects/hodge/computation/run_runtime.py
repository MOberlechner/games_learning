import os
import sys
from time import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.realpath("/home/oberlechner/code/matrix_game_learning"))

from decomposition.game import Game

from projects.hodge.configs import *
from projects.hodge.util import save_result


def time_hodge(n_agents: int, n_actions: int, n_runs: int = 100):
    actions = [n_actions] * n_agents
    t0 = time()
    hodge = Game(
        actions,
        flow_space=True,
        save_load=False,
    )
    t1 = time()
    payoff_vector = np.random.random(size=n_agents * n_actions**n_agents)
    for _ in range(n_runs):
        hodge.compute_flow_decomposition(payoff_vector)
    t2 = time()
    print(
        f"agents: {n_agents}, actions: {n_actions:2}, time str. {t1-t0:.3f}, time dec. {(t2-t1)/n_runs:.3f}, shape projection e: {hodge.structure.exact_projection.shape}"
    )
    return {"time_structure": t1 - t0, "time_decomposition": (t2 - t1) / n_runs}


def runtime_flow(settings: List[tuple], dir: str, filename: str):
    """runtime flow

    Args:
        settings (List[tuple]): settings (tuples of n_agents and n_actions)
        dir (str): directory where csv is saved
        filename (str): filename
    """
    data = []
    for n_agents, n_actions in settings:
        runtime = time_hodge(n_agents, n_actions)
        data.append(
            {
                "n_agents": n_agents,
                "n_actions": n_actions,
                "time_structure": runtime["time_structure"],
                "time_decomposition": runtime["time_decomposition"],
            }
        )
    save_result(data, dir, filename, PATH_TO_DATA)


if __name__ == "__main__":

    settings = (
        [(2, n_actions) for n_actions in range(2, 32 + 1)]
        + [(3, n_actions) for n_actions in range(2, 12 + 1)]
        + [(4, n_actions) for n_actions in range(2, 7 + 1)]
    )

    runtime_flow(settings, "runtime", "flow_space.csv")
