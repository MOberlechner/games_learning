import matplotlib

# Parameter Learning
MAX_ITER = 2_000
TOL = 1e-8
N_RUNS = 20
LIST_ETA = [2**i for i in [9, 6, 3, 0, -3, -6]]
LIST_BETA = [0.05, 0.5]


# Parameter Experiments
PATH_TO_DATA = "projects/hodge/data/"
PATH_TO_RESULTS = "projects/hodge/results/"

# Parameter visualization
cmap = matplotlib.colormaps["RdBu"]
COLORS = [cmap(0.9), cmap(0.1)]
LS = ["solid", "dashed", "dashdot", "dotted"]

FONTSIZE_LABEL = 13
FONTSIZE_TITLE = 13
FONTSIZE_LEGEND = 13
FORMAT = "png"
DPI = 300
