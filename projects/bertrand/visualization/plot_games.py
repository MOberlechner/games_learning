import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from games_learning.utils.equil import (
    find_correlated_equilibrium,
    find_pure_nash_equilibrium,
)
from projects.bertrand.config import *

plt.style.use("projects/bertrand/visualization/style.mplstyle")


def get_game(demand: str, n_discr: int, n_agents: int = 2) -> Bertrand:
    """Create Bertrand Game"""

    if demand == "standard":
        return BertrandStandard(
            n_agents=n_agents, n_discr=n_discr, **CONFIG_GAMES["standard"]
        )

    elif demand == "linear":
        return BertrandLinear(
            n_agents=n_agents, n_discr=n_discr, **CONFIG_GAMES["linear"]
        )

    elif demand == "logit":
        return BertrandLogit(
            n_agents=n_agents, n_discr=n_discr, **CONFIG_GAMES["logit"]
        )

    else:
        raise ValueError(
            f"demand model {demand} unknown. Choose from: standard, linear, logit"
        )


def plot_nes(demand: str, n_discr: int):
    """Visualize Nash Enclosure Set for Bertrand Settings"""
    # get game
    game = get_game(demand=demand, n_discr=n_discr, n_agents=2)
    equil = find_pure_nash_equilibrium(game)["ne"]

    # create data
    pne = [(game.actions[i], game.actions[i]) for i, j in equil]
    actions = [
        (game.actions[i], game.actions[j]) for i, j in product(range(n_discr), repeat=2)
    ]
    nes = [
        (game.actions[i], game.actions[j])
        for i, j in product([e[0] for e in equil], [e[1] for e in equil])
    ]
    actions = np.array(list(set(actions) - set(nes)))
    nes = np.array(list(set(nes) - set(pne)))
    pne = np.array(pne)

    # plot action profiles, NES, and PNE
    fig = plt.figure(figsize=FIGSIZE_S)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(actions.T[0], actions.T[1], s=30, c=COLORS[0], label="Action Profiles")
    if len(nes) > 0:
        ax.scatter(nes.T[0], nes.T[1], c=COLORS[-1], s=70, label="NES")
    ax.scatter(
        pne.T[0],
        pne.T[1],
        c=COLORS[-1],
        s=210,
        marker="*",
        label="NE" if len(nes) > 0 else "NE = NES",
    )

    ax.set_xlabel("Actions Player 1")
    ax.set_ylabel("Actions Player 2")
    ax.legend()

    # save
    path_save = os.path.join(PATH_TO_RESULTS, f"nes_{demand}_{n_discr}")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


def plot_cce(demand: str, n_discr: int):
    """Visualize CCE for Bertrand Settings"""
    # get dame
    game = get_game(demand=demand, n_discr=n_discr, n_agents=2)
    equil = find_pure_nash_equilibrium(game)

    # compute worst cce
    obj = np.array(
        [
            np.minimum(
                (i - equil["strict_ne"][-1][0]) ** 2,
                (j - equil["strict_ne"][-1][1]) ** 2,
            )
            for j in range(n_discr)
            for i in range(n_discr)
        ]
    ).reshape(game.n_actions)
    cce = find_correlated_equilibrium(game, coarse=True, objective=obj)

    # limits
    delta = game.actions[1] - game.actions[0]
    xmin, xmax = game.actions[0] - delta / 2, game.actions[-1] + delta / 2
    ymin, ymax = game.actions[0] - delta / 2, game.actions[-1] + delta / 2

    # visualize cce
    fig = plt.figure(figsize=FIGSIZE_R)
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.imshow(
        cce.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap=CMAP,
        vmin=0,
        vmax=0.6,
    )

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Probability")

    # numbers
    for i, j in product(range(n_discr), repeat=2):
        if cce[i, j] >= 0.005:
            ax.text(
                game.actions[i],
                game.actions[j],
                s=f"{cce[i,j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=6,
                fontweight="normal",
            )

    # visualize BNE
    for e in equil["ne"]:
        xval, yval = game.actions[e[0]], game.actions[e[1]]
        ax.plot(
            [
                xval - delta / 2,
                xval + delta / 2,
                xval + delta / 2,
                xval - delta / 2,
                xval - delta / 2,
            ],
            [
                yval - delta / 2,
                yval - delta / 2,
                yval + delta / 2,
                yval + delta / 2,
                yval - delta / 2,
            ],
            color="k",
            linewidth=0.8,
            linestyle="-",
            marker=None,
        )

    # axis
    ticks = np.linspace(game.actions[0], game.actions[-1], 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Actions Player 1")
    ax.set_ylabel("Actions Player 2")

    # save
    path_save = os.path.join(PATH_TO_RESULTS, f"cce_{demand}_{n_discr}")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")
    print(f"figure: '{path_save}.{FORMAT}' created")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS), exist_ok=True)

    plot_nes("standard", 9)
    plot_nes("linear", 9)

    plot_cce("standard", 9)
    plot_cce("linear", 9)
