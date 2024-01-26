import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def set_axis(xlim, ylim, title="", xlabel: str = "", ylabel: str = ""):
    """General settings for axis"""
    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.grid(linestyle="-", linewidth=0.25, color="lightgrey", zorder=-10, alpha=0.2)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    return fig, ax


def plot_potentialness_discretization(list_n_discr):
    """Plot Potentialness of different econgames w.r.t. discretization"""
    # Parameter
    interval = (0.0, 0.95)
    val_settings = ["symmetric", "asymmetric"]
    label_val_settings = [r"sym.   ($v_1=1.0, v_2=1.0$)", r"asym. ($v_1=0.8, v_2=1.0$)"]
    games = ["contest", "spsb", "fpsb", "allpay"]
    labels = ["Contest", "SPSB", "FPSB", "All-Pay"]
    # import data
    df = pd.read_csv(f"{PATH_TO_DATA}econgames/potentialness.csv")
    df = df[df.interval == str(interval)]

    fig, ax = set_axis((4, 26), (0, 1), xlabel="Discretization", ylabel="Potentialness")
    for i, game in enumerate(games):
        for j, valuations in enumerate(val_settings):
            tmp = df[
                (df.game == game) & (df.n_agents == 2) & (df.valuation == valuations)
            ]
            ax.plot(
                tmp.n_discr,
                tmp.potentialness,
                # marker=".",
                # markersize=10,
                linewidth=2.5,
                color=get_colors(i, len(games)),
                linestyle=LS[j],
            )
    ax.set_xticks([5, 10, 15, 20, 25])

    # add legends
    legend1, legend2 = create_legend(
        ax,
        labels,
        [],
        position=("upper center", "center right"),
        label1="",
        label2="",
        ncols1=2,
        ncols2=1,
    )
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, "econgames_discr")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    list_n_discr = list(range(5, 26))
    plot_potentialness_discretization(list_n_discr)
