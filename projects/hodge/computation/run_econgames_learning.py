import os
from collections import deque
from itertools import product

import numpy as np
import pandas as pd
from decomposition.game import Game
from tqdm import tqdm

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest, EconGame
from games_learning.game.matrix_game import MatrixGame
from games_learning.learner.learner import Learner, MirrorAscent
from games_learning.simulation import Simulator
from projects.hodge.configs import *


def run_learning(econgame: EconGame, n_bins: int):
    """Run learning algorithms for different convex combinations (i.e., levels of potentialness) of harmonic and potential part of an underlying economic game.

    Args:
        econgame (EconGame): underlying econ_game (e.g. FPSB, SPSB, ...)
        learner (Learner): learning algorithm
        max_iter (int): maximal number of iterations for learning algorithm
        tol (float): stopping criterion for learning algorithm
        n_runs (int): number of runs per instance
        n_bins (int): number of levels of potentialness
    """
    # compute decomposition of original game
    hodge = Game([n_discr] * n_agents, save_load=False)
    hodge.compute_decomposition_matrix(econgame.payoff_matrix)

    data = deque()
    for potent in tqdm(np.linspace(0, 1, n_bins)):

        # create new game (with given potentialness) from econgame
        payoff_matrix = hodge.create_game_potentialness(potent)
        game = MatrixGame(n_agents, payoff_matrix)

        # run learning
        for run in range(N_RUNS):
            for eta, beta in product(LIST_ETA, LIST_BETA):

                # create learner
                learner = MirrorAscent(eta=eta, beta=beta, mirror_map="entropic")
                sim = Simulator(game, learner, MAX_ITER, TOL)
                init_strat = game.init_strategies("random")
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
                data.append(result)

                # if we found suitable (eta,beta): stop
                if result["convergence"]:
                    break
    return data


if __name__ == "__main__":

    n_bins = 20

    # economic games
    n_agents = 2
    n_discr = 11
    games = [
        FPSB(n_agents, n_discr, interval=(0, 1)),
        SPSB(n_agents, n_discr, interval=(0, 1)),
        AllPay(n_agents, n_discr, interval=(0, 1)),
        Contest(n_agents, n_discr, interval=(0, 1), csf_param=1.0),
    ]

    # run experiments
    df = pd.DataFrame()
    for econgame in games:
        print(f"Run Experiment for {econgame}")
        df = pd.concat([df, pd.DataFrame(run_learning(econgame, n_bins))])

    # save results
    tag, filename = (
        "econgames_learning_stepsize",
        f"{n_agents}_{n_discr}.csv",
    )
    os.makedirs(os.path.join(PATH_TO_DATA, tag), exist_ok=True)
    df.to_csv(os.path.join(PATH_TO_DATA, tag, filename), index=False)
