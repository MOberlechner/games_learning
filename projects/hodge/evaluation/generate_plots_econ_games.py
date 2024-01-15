import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def set_axis(xlim, ylim, title="", xlabel: str = "", ylabel: str = ""):
    """General settings for axis"""
    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.grid(linestyle="-", linewidth=0.25, color="lightgrey", zorder=-10, alpha=0.2)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    return fig, ax


def plot_potentialness_discretization():
    """Plot Potentialness of different econgames w.r.t. discretization"""
    # Parameter
    interval = (0.0, 0.95)
    list_n_discr = [5, 8, 11, 16, 24]
    games = ["contest", "spsb", "fpsb", "allpay"]
    labels = ["Contest", "SPSB", "FPSB", "All-Pay"]
    # import data
    df = pd.read_csv(f"{PATH_TO_DATA}econgames/potentialness.csv")
    df = df[df.interval == str(interval)]

    fig, ax = set_axis((4, 25), (0, 1), xlabel="Discretization", ylabel="Potentialness")
    for i, game in enumerate(games):
        tmp = df[(df.game == game) & (df.n_agents == 2)]
        ax.plot(
            tmp.n_discr,
            tmp.potentialness,
            marker=".",
            linewidth=2.5,
            color=cmap(0.9 - i * 0.8 / 3),
            label=labels[i],
        )
    ax.set_xticks(list_n_discr)
    plt.legend(loc=9, ncols=1, fontsize=FONTSIZE_LEGEND)

    path_save = os.path.join(PATH_TO_RESULTS, "econgames_discr")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    plot_potentialness_discretization()
