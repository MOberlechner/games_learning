import os
import sys

sys.path.append(os.path.realpath("."))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from games_learning.game.econ_game import (
    FPSB,
    AllPay,
    BertrandLinear,
    BertrandStandard,
    Contest,
    EconGame,
)

cmap = LinearSegmentedColormap.from_list("custom_gradient", ["blue", "red"])


def visualize_utility(game: EconGame):
    """Visualize how the utility of an action changes depending on the opponents' action

    Args:
        game (EconGame): game to visualize
    """
    util_matrix = game.payoff_matrix[0]
    for i in range(game.n_actions[0]):
        plt.plot(
            game.actions,
            [util_matrix[i][j] for j in range(game.n_actions[1])],
            color=cmap(i / game.n_actions[1]),
            label=f"Action: {game.actions[i]:1.1f}",
        )
    plt.xlabel("Action Opponent (Agent 2)")
    plt.ylabel("Utility Agent 1")
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", title="Action Agent 1")
    plt.title(game.name.upper())
    plt.savefig(
        f"projects/economic_games/plots/scp_{game.name}.png", bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":

    os.makedirs("projects/economic_games/plots/", exist_ok=True)

    args = {
        "n_agents": 2,
        "n_discr": 11,
        "interval": (0, 1),
    }

    games = [
        AllPay(**args, valuations=(1.1, 1.1)),
        FPSB(**args, valuations=(1.1, 1.1)),
        BertrandLinear(**args, alpha=1, beta=(1, 1), gamma=0.5, cost=(0, 0)),
        BertrandStandard(**args, cost=(0, 0)),
        Contest(
            n_agents=2,
            n_discr=11,
            interval=(0, 0.5),
            valuations=(1.5, 1.5),
            csf_param=1.0,
        ),
    ]

    for game in games:
        visualize_utility(game)
