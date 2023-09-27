import os
import sys

sys.path.append("../")

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.game.matrix_game import MatrixGame
from src.learner.learner import SOMA, Learner
from src.simulation import Simulator
from src.utils.matrix_game_decomposition import get_payoff_matrix


def create_game(game_name: str, n_actions: int, potentialness: float):
    """Create game from given underlying game with a predefined potentialness"""
    Payoff1, Payoff2, alpha = get_payoff_matrix(
        game=game_name, n_actions=n_actions, potentialness=potentialness
    )
    return (
        MatrixGame(n_agents=2, payoff_matrix=[Payoff1, Payoff2]),
        potentialness,
        alpha,
    )


def init_strategies(
    n_actions: int, run: int = 0, n_agents: int = 2, method: str = "dirichlet"
) -> list:
    """initialize strategies randomly and use uniform init for run 0
    Uniformly sampling the entries and then normalizing leads to a concentration in the "middle" of
    the simplex. Using Dirichlet we ensure that points near the vertices are also used

    Args:
        n_actions (int): number of actions
        run (int, optional): number of run. Defaults to 0.
        agents (int, optional): number of agents. Defaults to 2
        methid (str, optional): sampling method
    """
    if run == 0:
        return [np.ones(n_actions) / n_actions for _ in range(n_agents)]
    else:
        if method == "dirichlet":
            return list(np.random.dirichlet((1,) * n_actions, size=n_agents))

        elif method == "uniform":
            init_strat = [
                np.random.uniform(low=0, high=1, size=n_actions)
                for _ in range(n_agents)
            ]
            return [s / s.sum() for s in init_strat]
        else:
            raise ValueError(f"sampling method '{method}' not available")


def learn_on_composition(
    game: MatrixGame,
    learner: Learner,
    max_iter: int,
    tol: float,
    num_runs: int,
):
    """create game by convex combination of harmonic and potential payoff matrices

    Args:
        game (MatrixGame): game
        learner (Learner): learning algorithm
        max_iter (int): maximal number of iterations for learning algorithm
        tol (float): convergence criterion (rel. util. loss)
        num_runs (int): number of repetations of experiment

    Returns:
        bool: convergence
    """
    # apply learning algorithm
    result_conv = deque(maxlen=num_runs)
    for run in range(num_runs):

        sim = Simulator(game, learner, max_iter=max_iter, tol=tol)
        init_strat = init_strategies(n_actions=n_actions, run=run, n_agents=2)
        sim.run(init_strat, show_bar=False)

        result_conv.append(sim.bool_convergence)
    return np.mean(result_conv)


if __name__ == "__main__":

    # settings to consider
    setting = "compl_info"

    if setting == "compl_info":
        games = ["contest", "spsb_auction", "fpsb_auction", "allpay_auction"]

    elif setting == "bayesian":
        games = [
            "allpay_auction_bayesian",
            "fpsb_auction_bayesian",
            "spsb_auction_bayesian",
        ]

    n_actions = 16
    num_runs = 10
    num_pot_level = 26
    potentialness = np.linspace(0, 1, num_pot_level)

    # specify learner
    max_iter = 10_000
    tol = 1e-5
    eta = 100
    learner = SOMA(eta=eta, beta=0.05, mirror_map="entropic")

    results = deque(maxlen=len(games) * num_pot_level)
    for game_name in games:

        print(f"{game_name}: learning experiments started")

        for p in tqdm(potentialness):

            game, pot, alp = create_game(game_name, n_actions, potentialness=p)
            mean_conv = learn_on_composition(
                game=game,
                learner=learner,
                max_iter=max_iter,
                tol=tol,
                num_runs=num_runs,
            )
            # log results
            results.append(
                {
                    "game": game_name,
                    "n_actions": n_actions,
                    "n_runs": num_runs,
                    "potentialness": pot,
                    "alpha": alp,
                    "convergence": mean_conv,
                }
            )

        print(
            f"{game_name}: learning experiments finished ({num_pot_level} levels of potentialness with {num_runs} runs each)!"
        )

    # save results
    path = "../results/data/"
    os.makedirs(path, exist_ok=True)
    filename = f"{setting}_discr_{n_actions}_eta_{eta}_runs_{num_runs}.csv"
    pd.DataFrame(results).round(2).to_csv(path + filename, index=False)
