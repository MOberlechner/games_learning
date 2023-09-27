import os
import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_breakpoints(
    data_learning: pd.DataFrame,
    data_games: pd.DataFrame,
    save_bool: bool = False,
    path_plots: str = "../results/plots",
    tag_save: str = "",
):
    """visualize learning for different levels of potentialness for different games

    Args:
        x (dict): contains potentialness of games
        y (dict): contains percentage of converge of games
        show_alpha (bool): choose x-axis (alpha or potentialness)
        potentialness_game (list): potentialness of original game (alpha=0.5)
        save_bool (bool, optional): save plot. Defaults to False.
        tag_save (str, optional): add tag to saved plot. Defaults to "".
    """
    n_actions = data_learning.n_actions.unique().item()
    n_runs = data_learning.n_runs.unique().item()

    fig = plt.figure(tight_layout=True, dpi=100, figsize=(6, 4))

    # Convergence w.r.t Potentialness
    ax = fig.add_subplot(211)
    games = data_learning.game.unique()
    for i, game in enumerate(games):

        tmp = data_learning[data_learning.game == game]
        n_entries = len(tmp)
        plt.scatter(
            tmp.potentialness,
            i * np.ones(n_entries),
            marker="s",
            s=50,
            c=np.array(tmp.convergence) * 100,
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
        )
    cbar = plt.colorbar()
    cbar.set_label(f"Convergence in % ({n_runs} runs)", rotation=270, labelpad=15)

    # Vis original game
    plt.scatter(
        [
            data_games[
                (data_games.game == game) & (data_games.n_actions == n_actions)
            ].potentialness.item()
            for game in games
        ],
        range(len(games)),
        marker="*",
        color="white",
        edgecolor="k",
        s=200,
        label="original game",
    )
    # plt.legend()

    # Labels
    # ax.set_title(f"Convergence of Mirror Ascent (n_discr={n_actions})")
    ax.set_xlabel(f"Potentialness for (n_discr={n_actions})")
    ax.set_yticks(
        range(len(games)), [g.replace("_bayesian", "\nbayesian") for g in games]
    )
    ax.set_ylim(-0.5, len(games) - 0.5)
    if save_bool:
        plt.savefig(
            path_plots + tag_save + ".pdf",
            bbox_inches="tight",
        )


if __name__ == "__main__":

    setting = "compl_info"
    n_actions = 16
    eta = 100
    num_runs = 25

    path_data = "../results/data/"
    path_plots = "../results/plots/"
    os.makedirs(path_plots, exist_ok=True)

    filename = f"{setting}_discr_{n_actions}_eta_{eta}_runs_{num_runs}.csv"
    tag_save = filename

    data_learning = pd.read_csv(path_data + filename)
    data_games = pd.read_csv(path_data + f"potentialness_games_{setting}.csv")

    visualize_breakpoints(
        data_learning,
        data_games,
        save_bool=True,
        path_plots=path_plots,
        tag_save=tag_save,
    )
