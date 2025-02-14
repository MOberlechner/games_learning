import os
import sys

sys.path.append(os.path.realpath("."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def plot_potentialness_bayesian(list_n_types, games, label_games):
    # import data
    df = pd.read_csv(f"{PATH_TO_DATA}econgames/potentialness_bayesian.csv")

    fig, ax = set_axis(
        (0.5, list_n_types[-1] + 0.5),
        (0, 1),
        xlabel="Number of Types",
        ylabel="Potentialness",
    )
    for i in range(len(games)):
        y = [
            df.loc[(df.game == games[i]) & (df.n_types == t), "potentialness"].item()
            for t in list_n_types
        ]
        ax.plot(
            list_n_types,
            y,
            color=get_colors(i, len(games)),
            linewidth=2,
            marker=MARKER[i],
            label=label_games[i],
        )
    ax.set_xticks(list_n_types)

    # add legend
    legend1 = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        ncol=1,
    )
    legend1.set_title("Games", prop={"size": FONTSIZE_LEGEND})
    ax.add_artist(legend1)

    path_save = os.path.join(PATH_TO_RESULTS, "econgames_bayesian")
    fig.savefig(
        f"{path_save}.{FORMAT}", bbox_inches="tight", bbox_extra_artists=(legend1,)
    )
    print(f"figure: '{path_save}.{FORMAT}' created")


def plot_potentialness_discretization(list_n_discr, games, label_games):
    """Plot Potentialness of different econgames w.r.t. discretization"""
    # Parameter
    interval = (0.0, 0.95)
    val_settings = ["symmetric", "asymmetric"]
    label_val_settings = [r"sym.   ($v_1=1.0, v_2=1.0$)", r"asym. ($v_1=0.8, v_2=1.0$)"]
    # import data
    df = pd.read_csv(f"{PATH_TO_DATA}econgames/potentialness.csv")
    df = df[df.interval == str(interval)]

    fig, ax = set_axis(
        (list_n_discr[0] - 1, list_n_discr[-1] + 1),
        (0, 1),
        xlabel="# Actions",
        ylabel="Potentialness",
    )
    for i, game in enumerate(games):
        for j, valuations in enumerate(val_settings):
            tmp = df[
                (df.game == game) & (df.n_agents == 2) & (df.valuation == valuations)
            ]
            ax.plot(
                list_n_discr,
                [tmp[tmp.n_discr == n].potentialness.item() for n in list_n_discr],
                # marker=MARKER[i],
                # markersize=5,
                linewidth=2,
                color=get_colors(i, len(games)),
                linestyle=LS[j],
                label=label_games[i] if j == 0 else None,
            )
    ax.set_xticks([5, 10, 15, 20, 25])

    # add legend
    legend1 = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        ncol=1,
    )
    legend1.set_title("Games", prop={"size": FONTSIZE_LEGEND})
    ax.add_artist(legend1)

    path_save = os.path.join(PATH_TO_RESULTS, "econgames_discr")
    fig.savefig(
        f"{path_save}.{FORMAT}", bbox_inches="tight", bbox_extra_artists=(legend1,)
    )
    print(f"figure: '{path_save}.{FORMAT}' created")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)

    list_n_discr = list(range(4, 26))
    games = ["contest", "spsb", "fpsb", "war_of_attrition", "allpay"]
    label_games = ["Contest", "SPSB", "FPSB", "WoA", "Allpay"]
    plot_potentialness_discretization(list_n_discr, games, label_games)

    list_n_types = [1, 2, 3, 4]
    games = [
        "bayesian_contest",
        "bayesian_spsb",
        "bayesian_fpsb",
        "bayesian_war_of_attrition",
        "bayesian_allpay",
    ]
    label_games = ["Contest", "SPSB", "FPSB", "WoA", "Allpay"]
    plot_potentialness_bayesian(list_n_types, games, label_games)
