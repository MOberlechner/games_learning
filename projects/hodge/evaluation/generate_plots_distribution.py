import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def get_colors(i, n):
    cmap = matplotlib.colormaps["RdBu"]
    COLORS = [cmap(0.9), cmap(0.1)]
    idx = n - 1 - i
    if n % 2:
        return cmap(0.1 + idx / (n - 1) * 0.8)
    else:
        return cmap(idx / (n - 1))


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


def plot_distribution_potentialness(list_n_agents, list_n_actions, name, dir):
    """plot of distribution of potentialness for randomly generated games"""
    # Parameter
    n_bins = 50
    distribution = "uniform"
    fig, ax = set_axis((0, 1), (-0.001, 0.9), "", "Potentialness", "Density")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

    # plot data
    for i, n_agents in enumerate(list_n_agents):
        for j, n_actions in enumerate(list_n_actions):
            try:
                file_name = f"{distribution}_{[n_actions]*n_agents}.csv"
                path = os.path.join(PATH_TO_DATA, dir, file_name)
                df = pd.read_csv(path)
                metric = (
                    "potentialness"
                    if "potentialness" in df.columns
                    else "potentialness_flow"
                )
                density, bins = np.histogram(df[metric], bins=n_bins, range=(0, 1))
                x = bins[:-1] + 0.5 / n_bins
                ax.plot(
                    x,
                    density / density.sum(),
                    linewidth=2,
                    color=get_colors(i, len(list_n_agents)),
                    linestyle=LS[j],
                    zorder=i,
                )
            except:
                print(f"Setting {[n_actions]*n_agents} not available (path:{path})")

    # special games
    show_special_games = False
    if show_special_games:
        ax.axvline(
            x=0.3660,
            color=COLORS[0],
            linestyle=LS[1],
            linewidth=1,
        )
        ax.text(
            x=0.32,
            y=0.2,
            s="Shapley Game",
            color=COLORS[0],
            horizontalalignment="left",
            verticalalignment="bottom",
            rotation=90,
        )

    # create legend
    line_styles = [
        plt.Line2D(
            [0], [0], color="black", linestyle=LS[j], linewidth=1.5, label=n_actions
        )
        for j, n_actions in enumerate(list_n_actions)
    ]
    color_styles = [
        plt.Line2D(
            [0],
            [0],
            color=get_colors(i, len(list_n_agents)),
            linestyle="-",
            linewidth=2,
            label=n_agents,
        )
        for i, n_agents in enumerate(list_n_agents)
    ]
    legend1 = ax.legend(
        handles=line_styles, loc=5, frameon=False, fontsize=FONTSIZE_LEGEND
    )
    legend1.set_title("# Actions", prop={"size": FONTSIZE_LEGEND})
    legend2 = ax.legend(
        handles=color_styles, loc=6, frameon=False, fontsize=FONTSIZE_LEGEND
    )
    legend2.set_title("# Agents", prop={"size": FONTSIZE_LEGEND})

    # Display both legends on the plot
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def plot_potentialness_vs_spne(list_n_agents, list_n_actions, name, dir):
    """plot probability of existence of strict pure Nash equilibria (spne) w.r.t. potentialness for randomly generated games"""
    # Parameter
    n_bins = 25
    distribution = "uniform"
    fig, ax = set_axis((0, 1), (-0.001, 1.001), "", "Potentialness", "P(SPNE exists)")

    # plot data
    for i, n_agents in enumerate(list_n_agents):
        for j, n_actions in enumerate(list_n_actions):
            try:
                file_name = f"{distribution}_{[n_actions]*n_agents}.csv"
                path = os.path.join(PATH_TO_DATA, dir, file_name)
                df = pd.read_csv(path)

                # prepare data (group by potentialness and probability of strict NE)
                metric = (
                    "potentialness"
                    if "potentialness" in df.columns
                    else "potentialness_flow"
                )
                df["bin"] = map_potentialness_to_bin(df[metric], n_bins)
                if "n_strict_ne" in df.columns:
                    df["spne"] = df["n_strict_ne"] > 0
                    df = df.groupby(["bin"]).agg({"spne": "mean"}).reset_index()
                    df[metric] = [
                        map_bin_to_potentialness(b, n_bins) for b in df["bin"]
                    ]

                    # visualize
                    ax.plot(
                        df[metric],
                        df["spne"],
                        linewidth=2,
                        color=get_colors(i, len(list_n_agents)),
                        linestyle=LS[j],
                        zorder=i,
                    )
                else:
                    print(
                        f"number of SPNE not avaliable for setting {[n_actions]*n_agents}"
                    )
            except:
                print(f"Setting {[n_actions]*n_agents} not available (path:{path})")

    # create legend
    line_styles = [
        plt.Line2D(
            [0], [0], color="black", linestyle=LS[j], linewidth=1.5, label=n_actions
        )
        for j, n_actions in enumerate(list_n_actions)
    ]
    color_styles = [
        plt.Line2D(
            [0],
            [0],
            color=get_colors(i, len(list_n_agents)),
            linestyle="-",
            linewidth=2,
            label=n_agents,
        )
        for i, n_agents in enumerate(list_n_agents)
    ]
    legend1 = ax.legend(
        handles=line_styles, loc=5, frameon=False, fontsize=FONTSIZE_LEGEND
    )
    legend1.set_title("# Actions", prop={"size": FONTSIZE_LEGEND})
    legend2 = ax.legend(
        handles=color_styles, loc=6, frameon=False, fontsize=FONTSIZE_LEGEND
    )
    legend2.set_title("# Agents", prop={"size": FONTSIZE_LEGEND})

    # Display both legends on the plot
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def plot_probability_spne(list_n_agents, list_n_actions, name, dir):
    """plot probability of existence of strict pure Nash equilibria (spne) for randomly generated games"""
    # Parameter
    n_bins = 25
    distribution = "uniform"
    fig, ax = set_axis((0, 1), (-0.001, 1.001), "", "Potentialness", "P(SPNE exists)")

    # plot data
    for i, n_agents in enumerate(list_n_agents):
        for j, n_actions in enumerate(list_n_actions):
            try:
                file_name = f"{distribution}_{[n_actions]*n_agents}.csv"
                path = os.path.join(PATH_TO_DATA, dir, file_name)
                df = pd.read_csv(path)

                # prepare data (group by potentialness and probability of strict NE)
                metric = (
                    "potentialness"
                    if "potentialness" in df.columns
                    else "potentialness_flow"
                )
                if "n_strict_ne" in df.columns:
                    df["spne"] = df["n_strict_ne"] > 0
                    y = df["spne"].mean()
                    x0 = df[metric].min()
                    x1 = df[metric].max()
                    x_mean = df[metric].mean()

                    # visualize
                    ax.plot(
                        [x0, x1],
                        [y, y],
                        linewidth=2,
                        color=get_colors(i, len(list_n_agents)),
                        linestyle=LS[j],
                        zorder=i,
                    )
                    ax.scatter(
                        [x_mean],
                        [y],
                        linewidth=2,
                        color=get_colors(i, len(list_n_agents)),
                        marker="o",
                        s=25,
                        zorder=i,
                    )
                else:
                    print(
                        f"number of SPNE not avaliable for setting {[n_actions]*n_agents}"
                    )
            except:
                print(f"Setting {[n_actions]*n_agents} not available (path:{path})")

    plt.axhline(y=1 - 1 / np.exp(1), color="k", linestyle="--", linewidth=0.7)
    plt.text(y=1 - 1 / np.exp(1) - 0.06, x=0.8, s=r"$1-\dfrac{1}{e}$")

    # create legend
    line_styles = [
        plt.Line2D(
            [0], [0], color="black", linestyle=LS[j], linewidth=1.5, label=n_actions
        )
        for j, n_actions in enumerate(list_n_actions)
    ]
    color_styles = [
        plt.Line2D(
            [0],
            [0],
            color=get_colors(i, len(list_n_agents)),
            linestyle="-",
            linewidth=2,
            label=n_agents,
        )
        for i, n_agents in enumerate(list_n_agents)
    ]
    legend1 = ax.legend(
        handles=line_styles, loc=4, frameon=False, fontsize=FONTSIZE_LEGEND
    )
    legend1.set_title("# Actions", prop={"size": FONTSIZE_LEGEND})
    legend2 = ax.legend(
        handles=color_styles, loc=3, frameon=False, fontsize=FONTSIZE_LEGEND
    )
    legend2.set_title("# Agents", prop={"size": FONTSIZE_LEGEND})

    # Display both legends on the plot
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)

    # distribution potentialness
    plot_distribution_potentialness(
        list_n_agents=[2, 4, 8, 10],
        list_n_actions=[2, 4, 12, 24],
        name="random_potentialness_1e5",
        dir="random_flow_1e5",
    )

    # relation potentialness and SPNE
    plot_potentialness_vs_spne(
        list_n_agents=[2, 4, 8, 10],
        list_n_actions=[2, 4, 12, 24],
        name="random_spne_1e5",
        dir="random_flow_1e5",
    )

    # relation potentialness and SPNE
    plot_probability_spne(
        list_n_agents=[2, 4, 8, 10],
        list_n_actions=[2, 4, 12, 24],
        name="random_prob_spne_1e5",
        dir="random_flow_1e5",
    )
