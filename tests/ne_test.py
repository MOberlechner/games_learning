""" Test computation of pure Nash equilibria """
import numpy as np
import pytest

from games_learning.game.econ_game import FPSB
from games_learning.game.matrix_game import ExampleMatrixGames, MatrixGame


def test_pne_matching_pennies():
    # Matching Pennies has no pure NE
    game = ExampleMatrixGames(setting="matching_pennies")
    pne = game.get_pne()
    assert pne == {"weak_ne": [], "strict_ne": [], "ne": []}


def test_pne_prisoners_dilemma():
    # Prisoners Dilemma has one pure NE
    game = ExampleMatrixGames(setting="prisoners_dilemma")
    pne = game.get_pne()
    assert pne == {"weak_ne": [], "strict_ne": [(1, 1)], "ne": [(1, 1)]}


def test_pne_battle_of_sexes():
    # Battle of sexes has two pure NE
    game = ExampleMatrixGames(setting="battle_of_sexes")
    pne = game.get_pne()
    assert pne == {"weak_ne": [], "strict_ne": [(0, 0), (1, 1)], "ne": [(0, 0), (1, 1)]}


def test_pne_fpsb():
    # FPSB has weak and strict ne (under certain discretization)
    game = FPSB(n_agents=2, n_discr=10, valuations=(1, 1), interval=(0, 0.9))
    pne = game.get_pne()
    assert pne == {"weak_ne": [(8, 8)], "strict_ne": [(9, 9)], "ne": [(9, 9), (8, 8)]}
