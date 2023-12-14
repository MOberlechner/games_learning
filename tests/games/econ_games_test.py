import numpy as np
import pytest

from games_learning.game.econ_game import FPSB, SPSB, AllPay, Contest


def test_create_fpsb():
    assert isinstance(FPSB(2, 5, (0, 1)), FPSB)


def test_create_spsb():
    assert isinstance(SPSB(2, 5, (0, 1)), SPSB)


def test_create_allpay():
    assert isinstance(AllPay(2, 5, (0, 1)), AllPay)


def test_create_contest():
    assert isinstance(Contest(2, 5, (0, 1)), Contest)


def test_fpsb_expost_util():
    # 2 Agents
    game = FPSB(2, 6, (0, 1))
    data = [
        [(0.5, 0.6), (0.0, 0.4)],
        [(0.0, 0.0), (0.5, 0.5)],
        [(0.3, 0.2), (0.7, 0.0)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility fpsb 2 agents"
    # 3 Agents
    game = FPSB(3, 11, (0, 1))
    data = [
        [(0.5, 0.6, 0.5), (0.0, 0.4, 0.0)],
        [(0.5, 0.6, 0.6), (0.0, 0.2, 0.2)],
        [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 1 / 3)],
        [(0.1, 0.7, 0.7), (0.0, 0.15, 0.15)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility fpsb 3 agents"


def test_spsb_expost_util():
    # 2 Agents
    game = SPSB(2, 6, (0, 1))
    data = [
        [(0.5, 0.6), (0.0, 0.5)],
        [(0.0, 0.0), (0.5, 0.5)],
        [(0.3, 0.2), (0.8, 0.0)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility spsb 2 agents"
    # 3 Agents
    game = SPSB(3, 11, (0, 1))
    data = [
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
    game = AllPay(2, 6, (0, 1))
    data = [
        [(0.5, 0.6), (-0.5, 0.4)],
        [(0.0, 0.0), (0.5, 0.5)],
        [(0.3, 0.2), (0.7, -0.2)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility allpay 2 agents"
    # 3 Agents
    game = AllPay(3, 11, (0, 1))
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
    game = Contest(2, 6, (0, 1), csf_param=1)
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
    game = Contest(3, 11, (0, 1), csf_param=1)
    data = [
        [(0.5, 1.0, 0.5), (-0.25, -0.5, -0.25)],
        [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 1 / 3)],
        [(0.2, 0.3, 0.5), (0, 0, 0)],
    ]
    for action_profile, util in data:
        assert np.allclose(
            game.ex_post_utility(np.array(action_profile)), np.array(util)
        ), "utility allpay 3 agents"
