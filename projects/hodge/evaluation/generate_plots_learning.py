import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def set_axis(xlim, ylim, title, xlabel: str = "", ylabel: str = ""):
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


def prepare_data(df, include_seed: bool = True):
    cols = ["game", "learner", "potentialness", "run"] + (
        ["seed"] if include_seed else []
    )

    # for each run pick "best" convergence result
    df = df.groupby(cols).agg({"convergence": "max"}).reset_index()
    # aggregate results over runs
    df = (
        df.groupby(["game", "learner", "potentialness"])
        .agg(
            {
                "convergence": "mean",
            }
        )
        .reset_index()
    )
    return df


def plot_econ_games_scatter(
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
    df = prepare_data(data, include_seed=False)
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


def plot_random_games_scatter(list_agents, list_actions, n_bins=20):
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


def plot_learning_line(list_n_agents, list_n_actions, name, dir):

    # Parameter
    n_bins = 25
    distribution = "uniform"
    learner = "mirror_ascent(entropic)"
    fig, ax = set_axis(
        (0, 1), (-0.001, 1.001), "", "Potentialness", "P(Convergence | SPNE exists)"
    )

    # plot data
    for i, n_agents in enumerate(list_n_agents):
        for j, n_discr in enumerate(list_n_actions):

            actions = [n_discr] * n_agents
            file_name = f"{learner}_random_matrix_game_uniform_{actions}.csv"
            path = os.path.join(PATH_TO_DATA, dir, file_name)

            try:
                # import data
                df = pd.read_csv(path)
                df = prepare_data(df, include_seed=True)

                # visualize
                ax.plot(
                    df["potentialness"],
                    df["convergence"],
                    linewidth=2,
                    color=get_colors(i, len(list_n_agents)),
                    linestyle=LS[j],
                    zorder=i,
                )
            except Exception as e:
                print(e)

    # add legends
    legend1, legend2 = create_legend(ax, list_n_agents, list_n_actions)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    # generate_plot_learning_random([2, 3], [2, 3, 4, 5])
    # plot_econ_games_scatter(n_agents=2, n_discr=11, interval=(0.00, 0.95), n_bins=25)
    plot_learning_line(
        list_n_agents=[2, 4, 8, 10],
        list_n_actions=[2, 4, 12, 24],
        name="random_learning_fixed_init",
        dir="random_learning_equal",
    )
