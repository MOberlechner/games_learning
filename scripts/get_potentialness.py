import os
import sys

sys.path.append("../")

import numpy as np
import pandas as pd

from src.utils.matrix_game_decomposition import load_component, metric_potentialness


def potentialness_game(game_name: str, n_actions: int):
    uP = load_component(game_name, n_actions, "uP")
    uH = load_component(game_name, n_actions, "uH")
    return metric_potentialness(uP, uH)


if __name__ == "__main__":
    """
    This scripts creates a csv file with the potentialness for
    different games and different discretizations
    """

    # settings (choose between compl_info or bayesian)
    setting = "compl_info"

    if setting == "compl_info":
        games = ["contest", "spsb_auction", "fpsb_auction", "allpay_auction"]
        discretizations = [4, 8, 10, 12, 16, 24]

    elif setting == "bayesian":
        games = [
            "spsb_auction",
            "fpsb_auction",
            "allpay_auction",
            "spsb_auction_bayesian",
            "fpsb_auction_bayesian",
            "allpay_auction_bayesian",
        ]
        discretizations = [24]

    results = []
    for game_name in games:
        for n_actions in discretizations:

            p = potentialness_game(game_name, n_actions)
            results.append(
                {"game": game_name, "n_actions": n_actions, "potentialness": p}
            )

    # save results
    path = "../results/data/"
    os.makedirs(path, exist_ok=True)
    filename = f"potentialness_games_{setting}.csv"
    pd.DataFrame(results).round(3).to_csv(path + filename, index=False)
