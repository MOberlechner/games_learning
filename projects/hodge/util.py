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

    if n == 1:
        return cmap(1)

    if n == 5:
        idx = n - 1 - i if i > 2 else n - i
        return cmap(idx / 5)

    idx = n - 1 - i
    if n % 2:
        return cmap(0.1 + idx / (n - 1) * 0.8)
    else:
        return cmap(idx / (n - 1))


def set_axis(xlim, ylim, title="", xlabel: str = "", ylabel: str = ""):
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


def create_legend(
    ax,
    list1,
    list2,
    position=(6, 5),
    label1="# Agents",
    label2="# Actions",
    ncols1=1,
    ncols2=1,
):

    # Legend 1
    color_styles = [
        plt.Line2D(
            [0],
            [0],
            color=get_colors(i, len(list1)),
            linestyle="-",
            linewidth=2,
            label=l1,
        )
        for i, l1 in enumerate(list1)
    ]
    legend1 = ax.legend(
        handles=color_styles,
        loc=position[0],
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        ncol=ncols1,
    )
    legend1.set_title(label1, prop={"size": FONTSIZE_LEGEND})

    # Legend 2
    line_styles = [
        plt.Line2D([0], [0], color="black", linestyle=LS[j], linewidth=2, label=l2)
        for j, l2 in enumerate(list2)
    ]
    legend2 = ax.legend(
        handles=line_styles,
        loc=position[1],
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        ncol=ncols2,
    )
    legend2.set_title(label2, prop={"size": FONTSIZE_LEGEND})

    return (
        legend1,
        legend2,
    )
