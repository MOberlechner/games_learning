import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from projects.hodge.configs import *
from projects.hodge.util import *


def set_axis(list_games, label_games):
    """General setting for axis"""
    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 2.1))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Potentialness", fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.8, len(list_games) - 0.2)
    ax.set_yticks(range(len(list_games)), label_games, fontsize=FONTSIZE_LABEL)
    ax.grid(linestyle="-", linewidth=0.25, color="lightgrey", zorder=-10, alpha=0.2)
    # ax.set_title(f"{n_agents} Agents",)

    return fig, ax


def prepare_data(df, eta, beta):

    # filter for stepsize
    if beta not in df.beta.unique():
        raise ValueError(f"beta {beta} not in data: {df.beta.unique()}")
    if eta not in df.eta.unique():
        raise ValueError(f"eta {eta} not in data: {df.eta.unique()}")
    df = df[(df.eta == eta) & (df.beta == beta)]

    # aggregate results over runs for each game
    df = (
        df.groupby(["game", "learner", "potentialness"])
        .agg({"convergence": "mean"})
        .reset_index()
    )
    return df


def get_potentialness_econgames(n_agents, n_discr):
    interval = (0.0, 0.95)
    valuation = "symmetric"
    path = os.path.join(PATH_TO_DATA, "econgames", "potentialness.csv")
    df = pd.read_csv(path)
    df = df[
        (df.n_agents == n_agents)
        & (df.n_discr == n_discr)
        & (df.valuation == valuation)
        & (df.interval == str(interval))
    ]
    return df


def plot(
    list_games,
    label_games,
    n_agents,
    n_discr,
    n_bins,
    n_runs,
    eta,
    beta,
    name,
    dir="econgames",
):
    # get data
    file_name = f"learning_{n_agents}_{n_discr}_{n_bins}bins_{n_runs}runs.csv"
    path = os.path.join(PATH_TO_DATA, dir, file_name)
    df = pd.read_csv(path)
    df = prepare_data(df, eta, beta)
    pot = get_potentialness_econgames(n_agents, n_discr)
    # create plot
    fig, ax = set_axis(list_games, label_games)
    for i, game in enumerate(list_games):
        tmp = df[df.game == game]
        # plot learning results
        sc = ax.scatter(
            tmp.potentialness,
            i * np.ones(len(tmp)),
            marker="s",
            s=120,
            c=np.array(tmp.convergence),
            cmap="RdBu",
            vmin=0,
            vmax=1,
        )
        # plot potentialness
        ax.scatter(
            [pot[pot.game == game].potentialness.item()],
            [i],
            marker="*",
            s=200,
            c="white",
            edgecolor="k",
            linewidth=0.8,
        )

    # create colorbar horizontal
    # cbar_ax = fig.add_axes([0.21, -0.05, 0.655, 0.03])  # horizontal
    # cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")

    # create colorbar vertical
    cbar_ax = fig.add_axes([0.99, 0.292, 0.02, 0.635])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="vertical")

    cbar.set_label("P(Convergence)", fontsize=FONTSIZE_LABEL - 2)
    cbar.set_ticks([0.0, 0.5, 1.0])

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)

    n_bins = 20
    n_runs = 100

    # economic games
    n_agents = 2
    n_discr = 11
    list_games = ["allpay", "spsb", "fpsb", "contest"]
    label_games = ["All-pay", "SPSB", "FSPB", "Contest"]

    for eta, beta in product(LIST_ETA, LIST_BETA):
        name = f"econgames_learning_{eta}_{beta}"
        plot(
            list_games, label_games, n_agents, n_discr, n_bins, n_runs, eta, beta, name
        )
