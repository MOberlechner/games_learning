import matplotlib
import matplotlib.pyplot as plt

# Parameter Learning
MAX_ITER = 2_000
TOL = 1e-8

# Step-sizes
LIST_ETA = [2**i for i in [8, 4]]
LIST_BETA = [0.05, 0.5]

# Settings (n_agents, n_actions)
SETTINGS = [(2, 2), (2, 4), (2, 12), (2, 24), (4, 2), (4, 4), (8, 2), (10, 2)]

# Data
PATH_TO_DATA = "projects/hodge/data/"
PATH_TO_RESULTS = "projects/hodge/figures_ecai/"

# Visualization
plt.rcParams["font.family"] = "Times New Roman"
CMAP = matplotlib.colormaps["RdBu"]
LS = ["solid", "dashed", "dashdot", "dotted"]
FONTSIZE_LABEL = 12
FONTSIZE_TITLE = 12
FONTSIZE_LEGEND = 12
FORMAT = "pdf"
DPI = 300
