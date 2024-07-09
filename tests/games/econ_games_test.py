import numpy as np
import pytest

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest


def test_create_fpsb():
    assert isinstance(FPSB(n_agents=2, n_discr=5, valuations=(0, 1)), FPSB)


def test_create_spsb():
    assert isinstance(SPSB(n_agents=2, n_discr=5, valuations=(0, 1)), SPSB)


def test_create_allpay():
    assert isinstance(AllPay(n_agents=2, n_discr=5, valuations=(0, 1)), AllPay)


def test_create_contest():
    assert isinstance(Contest(n_agents=2, n_discr=5, valuations=(0, 1)), Contest)


def test_fpsb_expost_util():
    # 2 Agents
    game = FPSB(n_agents=2, n_discr=6, valuations=(1, 1))
    data = [
        # actions  , # utilities
        [(0.5, 0.6), (0.0, 0.4)],
        [(0.0, 0.0), (0.5, 0.5)],
        [(0.3, 0.2), (0.7, 0.0)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility fpsb 2 agents"
    # 3 Agents
    game = FPSB(n_agents=3, n_discr=11, valuations=(1, 1, 0))
    data = [
        # actions       , # utilities
        [(0.5, 0.6, 0.5), (0.0, 0.4, 0.0)],
        [(0.5, 0.6, 0.6), (0.0, 0.2, -0.3)],
        [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 0)],
        [(0.1, 0.7, 0.7), (0.0, 0.15, -0.35)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility fpsb 3 agents"


def test_spsb_expost_util():
    # 2 Agents
    game = SPSB(n_agents=2, n_discr=6, valuations=(1, 0.8))
    data = [
        # actions  , # utilities
        [(0.5, 0.6), (0.0, 0.3)],
        [(0.0, 0.0), (0.5, 0.4)],
        [(0.3, 0.2), (0.8, 0.0)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility spsb 2 agents"
    # 3 Agents
    game = SPSB(n_agents=3, n_discr=11, valuations=(1, 1, 1))
    data = [
        # actions       , # utilities
        [(0.5, 0.6, 0.5), (0.0, 0.5, 0.0)],
        [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 1 / 3)],
        [(0.3, 0.6, 0.6), (0.0, 0.2, 0.2)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility spsb 3 agents"


def test_allpay_expost_util():
    # 2 Agents
    game = AllPay(n_agents=2, n_discr=6, valuations=(0.5, 1))
    data = [
        [(0.5, 0.6), (-0.5, 0.4)],
        [(0.0, 0.0), (0.25, 0.5)],
        [(0.3, 0.2), (0.2, -0.2)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility allpay 2 agents"
    # 3 Agents
    game = AllPay(n_agents=3, n_discr=11, valuations=(1, 1, 1))
    data = [
        [(0.5, 0.6, 0.5), (-0.5, 0.4, -0.5)],
        [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 1 / 3)],
        [(0.3, 0.6, 0.6), (-0.3, -0.1, -0.1)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility allpay 3 agents"


def test_contest_expost_util():
    # 2 Agents
    game = Contest(n_agents=2, n_discr=6, valuations=(1, 1), csf_param=1)
    data = [
        [(0.4, 0.6), (0.0, 0.0)],
        [(0.0, 0.0), (0.5, 0.5)],
        [(0.3, 0.2), (0.3, 0.2)],
        [(0.8, 0.7), (8 / 15 - 0.8, 7 / 15 - 0.7)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility contest 2 agents"
    # 3 Agents
    game = Contest(n_agents=3, n_discr=11, valuations=(1, 1, 1), csf_param=1)
    data = [
        [(0.5, 1.0, 0.5), (-0.25, -0.5, -0.25)],
        [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 1 / 3)],
        [(0.2, 0.3, 0.5), (0, 0, 0)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility allpay 3 agents"
