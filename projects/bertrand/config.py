import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from games_learning.game.econ_game import (
    Bertrand,
    BertrandLinear,
    BertrandLogit,
    BertrandStandard,
)

PATH_TO_RESULTS = "projects/bertrand/results/"
FORMAT = "pdf"
COLORS = ["#0571b0", "#92c5de", "#f4a582", "#ca0020"]
CMAP = LinearSegmentedColormap.from_list("colormap", COLORS)
FIGSIZE_S = (3.4, 3.4)
FIGSIZE_R = (3.4, 3 / 4 * 3.4)

CONFIG_GAMES = {
    "standard": {"cost": 0.0, "interval": (0.1, 1.0), "maximum_demand": 1.0},
    "linear": {
        "cost": 0.0,
        "interval": (0.1, 1.0),
        "alpha": 0.48,
        "beta": 0.9,
        "gamma": 0.6,
    },
    "logit": {
        "cost": 1.0,
        "interval": (1.0, 2.5),
        "alpha": 2.0,
        "mu": 0.25,
    },
}
