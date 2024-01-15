import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decomposition.game import Game

from projects.hodge.configs import *
from projects.hodge.util import *


def subplot(axs, i, j, matrix):
    """create subplots for plot_example_shapley"""
    axs[i, j].imshow(matrix.T, vmin=-1, vmax=1, cmap=cmap)
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
    print(f"Potentialness Shapley Game: {hodge.metric}")

    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 4))
    gs = fig.add_gridspec(2, 4, hspace=-0.53, wspace=0.1)
    axs = gs.subplots(sharex="col", sharey="row")
    for i in range(2):
        # payoff
        subplot(axs, i, 0, payoff_matrix[i])
        # potential
        subplot(axs, i, 1, hodge.uP[i])
        # harmonic
        subplot(axs, i, 2, hodge.uH[i])
        # non-strategic
        subplot(axs, i, 3, hodge.uN[i])

    path_save = os.path.join(PATH_TO_RESULTS, "example_shapley")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    plot_example_shapley()
