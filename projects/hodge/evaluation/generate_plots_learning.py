import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.hodge.configs import *
from projects.hodge.util import *


def generate_plot_learning_random(list_agents, list_actions, n_bins=20):
    # Parameter
    tag = "random_learning"
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
    generate_plot_learning_random([2, 3], [2, 3, 4, 5])
