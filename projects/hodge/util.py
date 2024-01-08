import os
from typing import List

import numpy as np
import pandas as pd


def map_potentialness_to_bin(p_list: List[float], n_bins: int) -> List[int]:
    """Divide [0,1] into n_bins intervals and match p to interval 1,...,n_bins"""
    bins = np.linspace(0, 1, n_bins + 1)
    return [(p >= bins).sum() for p in p_list]


def map_bin_to_potentialness(bin: int, n_bins: int) -> float:
    """Map (single) bin to midpoint of subinterval (representative potentialness)"""
    midpoints = np.linspace(0 + 0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    return midpoints[bin - 1]


def save_result(data: list, tag, filename, PATH_TO_DATA, overwrite=True):
    os.makedirs(os.path.join(PATH_TO_DATA, tag), exist_ok=True)
    file = os.path.join(PATH_TO_DATA, tag, filename)

    if (not overwrite) & (os.path.exists(file)):
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame(data)])
    else:
        df = pd.DataFrame(data)

    df.to_csv(file, index=False)
