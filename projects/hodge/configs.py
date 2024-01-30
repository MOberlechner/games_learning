# Parameter Learning
MAX_ITER = 2_000
TOL = 1e-8

# Step-sizes
LIST_ETA = [2**i for i in [8, 4, 0, -4]]
LIST_BETA = [0.05, 0.5]

LIST_ETA = [2**4]
LIST_BETA = [0.5]

# Settings
SETTINGS = [(2, 2), (2, 4), (2, 12), (2, 24), (4, 2), (4, 4), (8, 2), (10, 2)]

# Data
PATH_TO_DATA = "projects/hodge/data/"
PATH_TO_RESULTS = "projects/hodge/figures/"

# Visualization
LS = ["solid", "dashed", "dashdot", "dotted"]
FONTSIZE_LABEL = 13
FONTSIZE_TITLE = 13
FONTSIZE_LEGEND = 13
FORMAT = "png"
DPI = 300
