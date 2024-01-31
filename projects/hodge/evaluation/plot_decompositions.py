import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decomposition.game import Game

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest
from projects.hodge.configs import *
from projects.hodge.util import *

# --------------------------------------------- SHAPLEY --------------------------------------------- #


def subplot_shapley(axs, i, j, matrix):
    """create subplots for plot_example_shapley"""
    axs[i, j].imshow(matrix.T, vmin=-1, vmax=1, cmap=CMAP)
    axs[i, j].set_xticks([])
    axs[i, j].set_yticks([])
    if i == 0:
        axs[i, j].set_title(
            ["Payoff", "Potential", "Harmonic", "Non-Strategic"][j], fontsize=9
        )
        axs[i, 0].set_ylabel("Agent 1", fontsize=9)
    else:
        axs[i, 0].set_ylabel("Agent 2", fontsize=9)

    n, m = matrix.shape
    for k in range(n):
        for l in range(m):
            val = matrix[k, l]
            axs[i, j].text(
                k,
                l,
                f"{val:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=6,
                color="k" if val < 0.9 else "white",
            )


def plot_example_shapley():
    """plot decomposition of shapley"""

    name = "shapley_game"
    payoff_matrix = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    ]

    hodge = Game(n_actions=[3, 3], save_load=False)
    hodge.compute_decomposition_matrix(payoff_matrix)

    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 4))
    gs = fig.add_gridspec(2, 4, hspace=-0.53, wspace=0.1)
    axs = gs.subplots(sharex="col", sharey="row")
    for i in range(2):
        # payoff
        subplot_shapley(axs, i, 0, payoff_matrix[i])
        # potential
        subplot_shapley(axs, i, 1, hodge.uP[i])
        # harmonic
        subplot_shapley(axs, i, 2, hodge.uH[i])
        # non-strategic
        subplot_shapley(axs, i, 3, hodge.uN[i])

    path_save = os.path.join(PATH_TO_RESULTS, "decomposition_shapley")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


# --------------------------------------------- AUCTION --------------------------------------------- #


def subplot_auction(axs, i, j, matrix, labels):
    """create subplots for plot_example_shapley"""
    axs[i, j].imshow(
        matrix.T, vmin=-1, vmax=1, cmap=CMAP, extent=(0, 0.95, 0, 0.95), origin="lower"
    )
    axs[i, j].set_yticks([])
    axs[i, j].set_xticks([])
    if i == 0:
        axs[i, j].set_title(
            ["Payoff", "Potential", "Harmonic", "Non-Strategic"][j],
            fontsize=FONTSIZE_LABEL,
        )
        axs[i, 0].set_ylabel(labels[0], fontsize=FONTSIZE_LABEL)
    else:
        axs[i, 0].set_ylabel(labels[1], fontsize=FONTSIZE_LABEL)


def plot_auctions(n_discr):
    # Parameter Game
    n_agents = 2
    fpsb = FPSB(
        n_agents=n_agents, n_discr=n_discr, valuations=(1.0, 1.0), interval=(0.0, 0.95)
    )
    spsb = SPSB(
        n_agents=n_agents, n_discr=n_discr, valuations=(1.0, 1.0), interval=(0.0, 0.95)
    )

    hodge = Game(n_actions=[n_discr, n_discr], save_load=False)

    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 4))
    gs = fig.add_gridspec(2, 3, hspace=-0.23, wspace=0.1)
    axs = gs.subplots(sharex="col", sharey="row")

    for i in range(2):
        game = [fpsb, spsb][i]
        hodge.compute_decomposition_matrix(game.payoff_matrix)
        # payoff
        subplot_auction(axs, i, 0, game.payoff_matrix[0], labels=("FPSB", "SPSB"))
        # potential
        subplot_auction(axs, i, 1, hodge.uP[0], labels=("FPSB", "SPSB"))
        # harmonic
        subplot_auction(axs, i, 2, hodge.uH[0], labels=("FPSB", "SPSB"))

    path_save = os.path.join(PATH_TO_RESULTS, "decomposition_auctions")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


def plot_allpay(n_discr):
    # Parameter Game
    n_agents = 2
    contest = Contest(
        n_agents=n_agents, n_discr=n_discr, valuations=(1.0, 1.0), interval=(0.0, 0.95)
    )
    allpay = AllPay(
        n_agents=n_agents, n_discr=n_discr, valuations=(1.0, 1.0), interval=(0.0, 0.95)
    )

    hodge = Game(n_actions=[n_discr, n_discr], save_load=False)

    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 4))
    gs = fig.add_gridspec(2, 4, hspace=-0.53, wspace=0.1)
    axs = gs.subplots(sharex="col", sharey="row")

    for i in range(2):
        game = [contest, allpay][i]
        hodge.compute_decomposition_matrix(game.payoff_matrix)
        # payoff
        subplot_auction(axs, i, 0, game.payoff_matrix[0], labels=("Contest", "All-Pay"))
        # potential
        subplot_auction(axs, i, 1, hodge.uP[0], labels=("Contest", "All-Pay"))
        # harmonic
        subplot_auction(axs, i, 2, hodge.uH[0], labels=("Contest", "All-Pay"))
        # non-strategic
        subplot_auction(axs, i, 3, hodge.uN[0], labels=("Contest", "All-Pay"))

    path_save = os.path.join(PATH_TO_RESULTS, "decomposition_allpay")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    plot_example_shapley()
    plot_auctions(11)
    # plot_allpay(11)
