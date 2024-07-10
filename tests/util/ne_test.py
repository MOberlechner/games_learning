""" Test computation of pure Nash equilibria """
import numpy as np
import pytest

from games_learning.game.econ_game import FPSB
from games_learning.game.matrix_game import MatrixGame
from games_learning.utils.equil import find_pure_nash_equilibrium


def test_pne_matching_pennies():
    # Matching Pennies has no pure NE
    n_agents = 2
    payoff_matrix = [
        np.array([[1, -1], [-1, 1]]),
        np.array([[-1, 1], [1, -1]]),
    ]
    game = MatrixGame(n_agents, payoff_matrix)
    pne = find_pure_nash_equilibrium(game, atol=1e-10)
    assert pne == {"weak_ne": [], "strict_ne": []}


def test_pne_prisoners_dilemma():
    # Prisoners Dilemma has one pure NE
    n_agents = 2
    payoff_matrix = [np.array([[4, 1], [5, 2]]), np.array([[4, 5], [1, 2]])]
    game = MatrixGame(n_agents, payoff_matrix)
    pne = find_pure_nash_equilibrium(game, atol=1e-10)
    assert pne == {"weak_ne": [], "strict_ne": [(1, 1)]}


def test_pne_battle_of_sexes():
    # Battle of sexes has two pure NE
    n_agents = 2
    payoff_matrix = [np.array([[2, 0], [0, 1]]), np.array([[1, 0], [0, 2]])]
    game = MatrixGame(n_agents, payoff_matrix)
    pne = find_pure_nash_equilibrium(game, atol=1e-10)
    assert pne == {"weak_ne": [], "strict_ne": [(0, 0), (1, 1)]}


def test_pne_fpsb():
    # FPSB has weak and strict ne (under certain discretization)
    game = FPSB(n_agents=2, n_discr=10, valuations=(1, 1), interval=(0, 0.9))
    pne = find_pure_nash_equilibrium(game, atol=1e-10)
    assert pne == {"weak_ne": [(8, 8)], "strict_ne": [(9, 9)]}
