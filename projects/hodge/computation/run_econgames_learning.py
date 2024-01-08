import os
from collections import deque
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from decomposition.game import Game
from tqdm import tqdm

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest, EconGame
from games_learning.game.matrix_game import MatrixGame
from games_learning.learner.learner import Learner, MirrorAscent
from games_learning.simulation import Simulator
from games_learning.utils.equil import find_pure_nash_equilibrium
from projects.hodge.configs import *


def run_learning(econgame: EconGame, n_bins: int, compute_equil: bool = True):
    """Run learning algorithms for different convex combinations (i.e., levels of potentialness) of harmonic and potential part of an underlying economic game.

    Args:
        econgame (EconGame): underlying econ_game (e.g. FPSB, SPSB, ...)
        n_bins (int): number of levels of potentialness
        compute_equil (bool): compute pure equilibria and add to results
    """
    label_game = econgame.name
    # compute decomposition of original game
    hodge = Game([n_discr] * n_agents, save_load=False)
    hodge.compute_decomposition_matrix(econgame.payoff_matrix)

    data = deque()
    for potent in tqdm(np.linspace(0, 1, n_bins)):

        # create new game (with given potentialness) from econgame
        payoff_matrix = hodge.create_game_potentialness(potent)
        game = MatrixGame(n_agents, payoff_matrix)
        game.name = label_game

        if compute_equil:
            pure_equil = find_pure_nash_equilibrium(game)
            equilibria = {
                "n_weak_ne": len(pure_equil["weak_ne"]),
                "n_strict_ne": len(pure_equil["strict_ne"]),
                "interval": econgame.interval,
            }

        # run learning
        for run in range(N_RUNS):
            init_strat = game.init_strategies("random")

            for eta, beta in product(LIST_ETA, LIST_BETA):
                learner = MirrorAscent(eta=eta, beta=beta, mirror_map="entropic")

                # run experiment
                sim = Simulator(game, learner, MAX_ITER, TOL)
                result = sim.run(init_strat)

                # log results
                result.update(
                    {
                        "run": run,
                        "potentialness": potent,
                        "eta": eta,
                        "beta": beta,
                    }
                )
                # include equilibria to results
                result.update(equilibria)
                result.update(
                    {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                )

                data.append(result)
    return data


if __name__ == "__main__":

    n_bins = 25

    # economic games
    n_agents = 2
    n_discr = 11
    interval = (0.00, 0.95)
    games = [
        FPSB(n_agents, n_discr, interval=interval),
        SPSB(n_agents, n_discr, interval=interval),
        AllPay(n_agents, n_discr, interval=interval),
        Contest(n_agents, n_discr, interval=interval, csf_param=1.0),
    ]

    # run experiments
    df = pd.DataFrame()
    for econgame in games:
        print(f"Run Experiment for {econgame}")
        result = run_learning(econgame, n_bins)
        df = pd.concat([df, pd.DataFrame(result)])

    # save results
    tag, filename = (
        "econgames_learning_stepsize",
        f"{n_bins}bins_{n_agents}_{n_discr}_tol_{TOL}.csv",
    )
    os.makedirs(os.path.join(PATH_TO_DATA, tag), exist_ok=True)
    df.to_csv(os.path.join(PATH_TO_DATA, tag, filename), index=False)
