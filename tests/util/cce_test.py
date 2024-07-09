import numpy as np
import pytest

from games_learning.game.matrix_game import MatrixGame
from games_learning.utils.equil import find_correlated_equilibrium


def test_cce_matching_pennies():
    # Matching Pennies unique CCE is mixed equilibrium
    n_agents = 2
    payoff_matrix = [
        np.array([[1, -1], [-1, 1]]),
        np.array([[-1, 1], [1, -1]]),
    ]
    game = MatrixGame(n_agents, payoff_matrix)
    cce = find_correlated_equilibrium(game)

    assert np.allclose(cce, np.array([[0.25, 0.25], [0.25, 0.25]]))


def test_cce_prisoners_dilemma():
    # Prisoners' Delamma unique CCE is mixed equilibrium
    n_agents = 2
    payoff_matrix = [np.array([[4, 1], [5, 2]]), np.array([[4, 5], [1, 2]])]
    game = MatrixGame(n_agents, payoff_matrix)
    cce = find_correlated_equilibrium(game)

    assert np.allclose(cce, np.array([[0.0, 0.0], [0.0, 1.0]]))
