import numpy as np
import pytest

from games_learning.game.bayesian_game import BayesianEconGame
from games_learning.game.econ_game import FPSB, AllPay


@pytest.fixture(scope="module")
def bayesian_fpsb():
    n_discr = 3
    game = FPSB(n_agents=2, n_discr=n_discr, valuations=(1, 1))
    return BayesianEconGame(game, n_discr=n_discr, monotone_strategies=False)


@pytest.fixture(scope="module")
def bayesian_allpay():
    n_discr = 3
    game = AllPay(n_agents=2, n_discr=n_discr, valuations=(1, 1))
    return BayesianEconGame(game, n_discr=n_discr, monotone_strategies=False)


def test_create_bayesian_game():
    basic_game = FPSB(n_agents=2, n_discr=3, valuations=(0, 1))
    game = BayesianEconGame(econ_game=basic_game, n_discr=2)
    assert isinstance(game, BayesianEconGame)

    assert np.allclose(game.bayesian_actions, np.array([0, 0.5, 1]))
    assert np.allclose(game.bayesian_types, np.array([0, 1]))
    assert game.n_actions == [3**2, 3**2]


def test_utility_bayesian_game_fpsb(bayesian_fpsb):
    strategy_profile = [{0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}]
    exp_util = bayesian_fpsb.compute_expected_utility(strategy_profile)
    assert np.allclose(exp_util, np.array([1 / 3 * (0 + 0.25 + 0.5)] * 2))

    strategy_profile = [{0: 2, 1: 2, 2: 2}, {0: 0, 1: 0, 2: 0}]
    exp_util = bayesian_fpsb.compute_expected_utility(strategy_profile)
    assert np.allclose(exp_util, np.array([1 / 3 * (-1 - 0.5 + 0), 0]))

    strategy_profile = [{0: 0, 1: 0, 2: 1}, {0: 0, 1: 1, 2: 1}]
    exp_util = bayesian_fpsb.compute_expected_utility(strategy_profile)
    assert np.allclose(
        exp_util,
        np.array([1 / 9 * (0.25 + 0.5 + 0.25 + 0.25), 1 / 9 * (0.5 + 0.5 + 0.25)]),
    )


def test_utility_bayesian_game_allpay(bayesian_allpay):
    strategy_profile = [{0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}]
    exp_util = bayesian_allpay.compute_expected_utility(strategy_profile)
    assert np.allclose(exp_util, np.array([1 / 3 * (0 + 0.25 + 0.5)] * 2))

    strategy_profile = [{0: 2, 1: 2, 2: 2}, {0: 1, 1: 1, 2: 1}]
    exp_util = bayesian_allpay.compute_expected_utility(strategy_profile)
    assert np.allclose(exp_util, np.array([1 / 3 * (-1 - 0.5 + 0), -0.5]))
