import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *


def map_potentialness_to_bin(p_list: List[float], n_bins: int) -> List[int]:
    """Divide [0,1] into n_bins intervals and match p to interval 1,...,n_bins"""
    bins = np.linspace(0, 1, n_bins + 1)
    return [(p >= bins).sum() for p in p_list]


def map_bin_to_potentialness(bin: int, n_bins: int) -> float:
    """Map (single) bin to midpoint of subinterval (representative potentialness)"""
    midpoints = np.linspace(0 + 0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    return midpoints[bin - 1]


def save_result(data: list, tag, filename, PATH_TO_DATA, overwrite=True):
    os.makedirs(os.path.join(PATH_TO_DATA, tag), exist_ok=True)
    file = os.path.join(PATH_TO_DATA, tag, filename)

    if (not overwrite) & (os.path.exists(file)):
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame(data)])
    else:
        df = pd.DataFrame(data)

    df.to_csv(file, index=False)


def get_colors(i, n):
    cmap = matplotlib.colormaps["RdBu"]
    COLORS = [cmap(0.9), cmap(0.1)]
    idx = n - 1 - i
    if n % 2:
        return cmap(0.1 + idx / (n - 1) * 0.8)
    else:
        return cmap(idx / (n - 1))


def create_legend(ax, list_n_agents, list_n_actions):
    # create legend
    line_styles = [
        plt.Line2D(
            [0], [0], color="black", linestyle=LS[j], linewidth=2, label=n_actions
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
    return (
        legend1,
        legend2,
    )
