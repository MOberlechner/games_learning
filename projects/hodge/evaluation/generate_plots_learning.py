import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def prepare_data_econgames(df):
    # for each run pick "best" convergence result
    df = (
        df.groupby(["game", "learner", "potentialness", "run"])
        .agg({"convergence": "max", "n_strict_ne": "first"})
        .reset_index()
    )
    # aggregate results over runs
    df = (
        df.groupby(["game", "learner", "potentialness"])
        .agg(
            {
                "convergence": "max",
                "n_strict_ne": "first",
            }
        )
        .reset_index()
    )
    return df


def generate_plot_learning_econ_games(
    n_agents: int,
    n_discr: int,
    interval: tuple,
    n_bins: int,
    learner: str = "mirror_ascent(entropic)",
):

    # Parameter
    tag = "econgames_learning_stepsize"
    list_games = ["allpay", "spsb", "fpsb", "contest"]
    label_games = ["All-pay", "SPSB", "FSPB", "Contest"]

    # prepare data
    data = pd.read_csv(
        f"{PATH_TO_DATA}{tag}/{n_bins}bins_{n_agents}_{n_discr}_tol_{TOL}.csv"
    )
    df = prepare_data_econgames(data)
    pot = pd.read_csv(f"{PATH_TO_DATA}econgames/potentialness.csv")
    pot = pot[
        (pot.n_agents == n_agents)
        & (pot.n_discr == n_discr)
        & (pot.interval == str(interval))
    ]

    # prepare plot
    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 2.1))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Potentialness", fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.8, len(list_games) - 0.2)
    ax.set_yticks(range(len(list_games)), label_games)
    ax.grid(linestyle="-", linewidth=0.25, color="lightgrey", zorder=-10, alpha=0.2)
    ax.set_title(
        f"{n_agents} Agents",
    )

    for i, game in enumerate(list_games):
        tmp = df[df.game == game]
        # plot results
        sc = ax.scatter(
            tmp.potentialness,
            i * np.ones(len(tmp)),
            marker="s",
            s=75,
            c=np.array(tmp.convergence) * 100,
            cmap="RdBu",
            vmin=0,
            vmax=100,
        )
        ax.scatter(
            pot[pot.game == game].potentialness,
            i,
            marker="*",
            facecolor="white",
            edgecolor="k",
            s=150,
            linewidth=0.8,
        )
    # cbar_ax = fig.add_axes([1.0, 0.155, 0.03, 0.755])  # vertical
    cbar_ax = fig.add_axes([0.21, -0.05, 0.655, 0.05])  # horizontal
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Convergence (%)", fontsize=FONTSIZE_LABEL - 2)

    path_save = os.path.join(PATH_TO_RESULTS, "econgames_learning")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def generate_plot_learning_random(list_agents, list_actions, n_bins=20):
    # Parameter
    tag = "random_learning_128_0.9"
    learner = "mirror_ascent(entropic)"

    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 4))
    for i, n_agents in enumerate(list_agents):
        # prepare plot
        ax = fig.add_subplot(len(list_agents), 1, i + 1)
        if i == 1:
            ax.set_xlabel("Potentialness", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("# Actions", fontsize=FONTSIZE_LABEL)
        ax.set_ylim(min(list_actions) - 0.8, max(list_actions) + 0.8)
        ax.set_yticks(list_actions)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(linestyle="-", linewidth=0.25, color="lightgrey", zorder=-10, alpha=0.2)
        ax.set_title(
            f"{n_agents} Agents",
        )

        # get data
        for n_actions in list_actions:

            setting = f"random_matrix_game_uniform_{n_agents}_{n_actions}"
            df = pd.read_csv(f"{PATH_TO_DATA}{tag}/{learner}_{setting}.csv")
            df["potentialness"] = map_bin_to_potentialness(df["bin"], n_bins)

            df = (
                df.groupby(["potentialness"]).agg({"convergence": "mean"}).reset_index()
            )

            # plot results
            sc = ax.scatter(
                df.potentialness,
                n_actions * np.ones(len(df)),
                marker="s",
                s=150,
                c=np.array(df.convergence) * 100,
                cmap="RdBu",
                vmin=0,
                vmax=100,
            )

    # cbar_ax = fig.add_axes([1.0, 0.155, 0.03, 0.755])  # vertical
    cbar_ax = fig.add_axes([0.21, -0.05, 0.655, 0.03])  # horizontal
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Convergence (%)", fontsize=FONTSIZE_LABEL - 2)

    path_save = os.path.join(PATH_TO_RESULTS, "random_learning")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    # generate_plot_learning_random([2, 3], [2, 3, 4, 5])
    generate_plot_learning_econ_games(
        n_agents=2, n_discr=11, interval=(0.00, 0.95), n_bins=25
    )
