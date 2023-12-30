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

def plot_distribution_potentialness():
    """plot of distribution of potentialness for randomly generated games"""
    # Parameter
    n_bins = 50
    distribution = "uniform"
    fig, ax = set_axis((0, 1), (-0.001, 0.37), "", "Potentialness", "Density")
    ax.set_yticks([0, 0.1, 0.2, 0.3])

    # plot data
    for j, n_agents in enumerate([2, 3]):
        for i, n_actions in enumerate([2, 3, 4, 5]):
            file_name = f"{distribution}_{[n_actions]*n_agents}.csv"
            df = pd.read_csv(os.path.join(PATH_TO_DATA, "random", file_name))
            density, bins = np.histogram(df.potentialness, bins=n_bins, range=(0, 1))
            x = bins[:-1] + 0.5 / n_bins
            ax.plot(
                x,
                density / density.sum(),
                linewidth=2,
                color=COLORS[j],
                linestyle=LS[i],
                zorder=3 - j,
            )

    # create legend
    line_styles = [
        plt.Line2D([0], [0], color="black", linestyle=LS[i], linewidth=1.5, label=i + 2)
        for i in range(4)
    ]
    color_styles = [
        plt.Line2D([0], [0], color=COLORS[j], linestyle="-", linewidth=2, label=j + 2)
        for j in range(2)
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

    path_save = os.path.join(PATH_TO_RESULTS, "random_distribution")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def plot_distribution_psne():
    """plot distribution of pure strict Nash equilibria (psne) w.r.t. potentialness for randomly generated games"""
    # Parameter
    n_bins = 20
    distribution = "uniform"
    fig, ax = set_axis((0, 1), (-0.001, 1.001), "", "Potentialness", "Existence PSNE")


    # plot data
    for j, n_agents in enumerate([2, 3]):
        for i, n_actions in enumerate([2, 3, 4, 5]):

            file_name = f"{distribution}_{[n_actions]*n_agents}.csv"
            df = pd.read_csv(os.path.join(PATH_TO_DATA, "random", file_name))

            # prepare data (group by potentialness and probability of strict NE)
            df["bin"] = map_potentialness_to_bin(df["potentialness"], n_bins)
            df["psne"] = df["n_strict_ne"] > 0
            df = df.groupby(["bin"]).agg({"psne": "mean"}).reset_index() 
            df["potentialness"] = [ map_bin_to_potentialness(b, n_bins) for b in df["bin"]]

            # visualize
            ax.plot(
                df["potentialness"],
                df["psne"],
                linewidth=2,
                color=COLORS[j],
                linestyle=LS[i],
                zorder=3 - j,
            )

    # create legend
    line_styles = [
        plt.Line2D([0], [0], color="black", linestyle=LS[i], linewidth=1.5, label=i + 2)
        for i in range(4)
    ]
    color_styles = [
        plt.Line2D([0], [0], color=COLORS[j], linestyle="-", linewidth=2, label=j + 2)
        for j in range(2)
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
    
    path_save = os.path.join(PATH_TO_RESULTS, "random_psne")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")



if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    plot_distribution_potentialness()
    plot_distribution_psne()
