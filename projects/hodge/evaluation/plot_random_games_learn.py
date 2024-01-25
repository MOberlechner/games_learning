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


def plot_learning(list_n_agents, list_n_actions, name, dir):
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
    plot_learning(
        list_n_agents=[2, 4, 8, 10],
        list_n_actions=[2, 4, 12, 24],
        name="random_learning_fixed_init",
        dir="random_learning_equal",
    )
