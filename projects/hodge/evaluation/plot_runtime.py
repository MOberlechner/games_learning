import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def get_colors(i):
    cmap = matplotlib.colormaps["RdBu"]
    n = 4
    idx = n - 1 - 0.5 * i
    return cmap(idx / (n - 1))


def plot_runtime_decomposition(name: str):

    # get data
    df = pd.read_csv(f"{PATH_TO_DATA}runtime/flow_space.csv")

    # create axis
    fig, ax = set_axis((1, 33), (1e-5, 1e1), xlabel="# Actions", ylabel="Time in s")
    # time decomposition
    for i, agent in enumerate(df.n_agents.unique()):
        ax.plot(
            df[df.n_agents == agent].n_actions,
            df[df.n_agents == agent].time_decomposition,
            color=get_colors(i),
            label=agent,
            linewidth=2,
        )

    # theoretical time complexity
    x = np.linspace(2, 32, 100)
    c = [1 / (2 + i) ** (i + 1) * 10**i * 1e-9 for i in range(3)]
    x_text, s_text = [29, 16, 9], [
        r"$c \cdot m^6$",
        r"$c \cdot m^8$",
        r"$c \cdot m^{10}$",
    ]
    for i, agent in enumerate(df.n_agents.unique()):
        ax.plot(
            x,
            c[i] * x ** (2 * agent + 2),
            color=get_colors(i),
            linestyle="--",
            linewidth=1,
            zorder=-2,
        )
        ax.text(
            x=x_text[i],
            y=c[i] / 3 * x_text[i] ** (2 * agent + 2),
            s=s_text[i],
            color=get_colors(i),
            fontsize=10,
        )
    legend = ax.legend(loc=4, frameon=False, fontsize=FONTSIZE_LEGEND)
    legend.set_title("# Agents", prop={"size": FONTSIZE_LEGEND})
    ax.set_yscale("log")

    path_save = os.path.join(PATH_TO_RESULTS, name)
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)
    plot_runtime_decomposition("runtime_potentialness")
