import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def set_axis(xlim, ylim, title, xlabel: str = "", ylabel: str = ""):
    """General settings for axis"""
    fig = plt.figure(tight_layout=True, dpi=DPI, figsize=(5, 3.5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.grid(linestyle="-", linewidth=0.25, color="lightgrey", zorder=-10, alpha=0.2)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    return fig, ax


# ------------------------------------------------------------------------------------------------- #
#                       VISUALIZE RESULTS FOR DIFFERENT INITIAL STRATEGY                            #
# ------------------------------------------------------------------------------------------------- #


def prepare_data_diff_init(df, eta, beta, filter_small_samples: bool = False):
    """uses fix stepsize for all experiments"""

    # filter for stepsize
    if beta not in df.beta.unique():
        raise ValueError(f"beta not in data: {df.beta.unique()}")
    if eta not in df.eta.unique():
        raise ValueError(f"eta not in data: {df.eta.unique()}")
    df = df[(df.eta == eta) & (df.beta == beta)]

    # aggregate results over runs for each game
    df = (
        df.groupby(["game", "learner", "potentialness", "seed"])
        .agg(
            {
                "convergence": "mean",
                "run": "count",
            }
        )
        .reset_index()
    )
    # aggregate results over games
    df = (
        df.groupby(["game", "learner", "potentialness"])
        .agg(
            {
                "convergence": ["mean", "std"],
                "seed": "count",
            }
        )
        .reset_index()
    )
    if filter_small_samples:
        df = df[df[("seed", "count")] == 100]
    return df


def plot_learning_diff_init(list_n_agents, list_n_actions, eta, beta, name, dir):
    # Parameter
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
                df = prepare_data_diff_init(df, eta, beta, filter_small_samples=True)

                # visualize
                ax.plot(
                    df["potentialness"],
                    df[("convergence", "mean")],
                    linewidth=2,
                    color=get_colors(i, len(list_n_agents)),
                    linestyle=LS[j],
                    zorder=i,
                )
                # ax.fill_between(
                #    df["potentialness"],
                #    df[("convergence", "mean")] - df[("convergence", "std")],
                #    df[("convergence", "mean")] + df[("convergence", "std")],
                #    color=get_colors(i, len(list_n_agents)),
                #    zorder=i,
                #    alpha = 0.1
                # )
            except Exception as e:
                # if file should exist, print error message
                if (n_agents, n_discr) in SETTINGS:
                    print(e)

    # add legends
    legend1, legend2 = create_legend(ax, list_n_agents, list_n_actions)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


def plot_learning_diff_init_2agents(
    list_n_agents, list_n_actions, eta, beta, name, dir
):
    # Parameter
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
                df = prepare_data_diff_init(df, eta, beta, filter_small_samples=True)

                # visualize
                ax.plot(
                    df["potentialness"],
                    df[("convergence", "mean")],
                    linewidth=2,
                    color=get_colors(i, len(list_n_agents)),
                    linestyle=LS[j],
                    zorder=i,
                )
                ax.fill_between(
                    df["potentialness"],
                    df[("convergence", "mean")] - df[("convergence", "std")],
                    df[("convergence", "mean")] + df[("convergence", "std")],
                    color=get_colors(i, len(list_n_agents)),
                    zorder=i,
                    alpha=0.3,
                )
            except Exception as e:
                # if file should exist, print error message
                if (n_agents, n_discr) in SETTINGS:
                    print(e)

    # add legends
    legend1, legend2 = create_legend(ax, list_n_agents, list_n_actions)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


# ------------------------------------------------------------------------------------------------- #
#                          VISUALIZE RESULTS FOR FIXED INITIAL STRATEGY                             #
# ------------------------------------------------------------------------------------------------- #


def prepare_data_fixed_init(df, eta, beta, filter_small_samples: bool = False):
    """uses fix stepsize for all experiments"""

    # filter for stepsize
    if beta not in df.beta.unique():
        raise ValueError(f"beta not in data: {df.beta.unique()}")
    if eta not in df.eta.unique():
        raise ValueError(f"eta not in data: {df.eta.unique()}")
    df = df[(df.eta == eta) & (df.beta == beta)]

    # aggregate results over runs
    df = (
        df.groupby(["game", "learner", "potentialness", "seed"])
        .agg(
            {
                "convergence": "mean",
                "run": "count",
            }
        )
        .reset_index()
    )
    # aggregate results over seeds
    df = (
        df.groupby(["game", "learner", "potentialness"])
        .agg(
            {
                "convergence": "mean",
                "seed": "count",
            }
        )
        .reset_index()
    )
    if filter_small_samples:
        df = df[df.seed == 100]
    return df


def plot_learning_fixed_init(list_n_agents, list_n_actions, eta, beta, name, dir):
    # Parameter
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
                df = prepare_data_fixed_init(df, eta, beta, filter_small_samples=True)

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
                # if file should exist, print error message
                if (n_agents, n_discr) in SETTINGS:
                    print(e)

    # add legends
    legend1, legend2 = create_legend(ax, list_n_agents, list_n_actions)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    eta, beta = 256, 0.5
    plot_learning_diff_init(
        list_n_agents=[2, 4, 8, 10],
        list_n_actions=[2, 4, 12, 24],
        eta=eta,
        beta=beta,
        name=f"random_learning_diff_init_{eta}_{beta}",
        dir="random_learning_25run",
    )
    plot_learning_diff_init_2agents(
        list_n_agents=[2, 4, 8, 10],
        list_n_actions=[2],
        eta=eta,
        beta=beta,
        name=f"random_learning_diff_init_{eta}_{beta}_2actions",
        dir="random_learning_25run",
    )

    for eta, beta in product(LIST_ETA, LIST_BETA):
        plot_learning_fixed_init(
            list_n_agents=[2, 4, 8, 10],
            list_n_actions=[2, 4, 12, 24],
            eta=eta,
            beta=beta,
            name=f"random_learning_fixed_init_{eta}_{beta}",
            dir="random_learning_1run",
        )
