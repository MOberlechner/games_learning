from itertools import product
from typing import List, Tuple

import numpy as np

from games_learning.game.matrix_game import MatrixGame


class EconGame(MatrixGame):
    """This class allows us to construct matrix games from continuous, complete information economic games such as auctions and contests"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
    ):
        payoff_matrix = self.create_payoff_matrix(n_agents, n_discr, interval)
        super().__init__(n_agents=n_agents, payoff_matrix=payoff_matrix)

    def create_payoff_matrix(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray]:
        """Create payoff matrix of game given a discretization of the action space. We assume that agents are symmetric, i.e., have same action space.

        Args:
            n_agents (int): number of agents
            n_discr (int): discretization parameter action space
            interval (Tuple[float], optional): interval of action space. Defaults to (0.0,1.0).

        Returns:
            Tuple[np.ndarray]: payoff matrix for each agent
        """
        # create action vector
        actions = np.linspace(*interval, n_discr)
        # compute utilities
        util_arr = np.array(
            [
                self.ex_post_utility(np.array(a))
                for a in product(actions, repeat=n_agents)
            ]
        )
        # reformat array
        util_arr_form = util_arr.T.reshape([n_agents] + [n_discr] * n_agents)
        # return as tuple
        return tuple([util for util in util_arr_form])

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """compute ex-post utility given a action profile"""
        raise NotImplementedError


class FPSB(EconGame):
    """Complete-Information First-Price Sealed Bid with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuation = np.ones(n_agents)
        super().__init__(n_agents, n_discr, interval)

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for FPSB"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute payment (first-price)
        first_price = action_profile.max()
        # compute ex-post utility
        return allocation * (self.valuation - first_price)


class SPSB(EconGame):
    """Complete-Information Second-Price Sealed Bid with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuation = np.ones(n_agents)
        super().__init__(n_agents, n_discr, interval)

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for SPSB"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute payment (second-price)
        second_price = np.sort(action_profile)[-2]
        # compute ex-post utility
        return allocation * (self.valuation - second_price)


class AllPay(EconGame):
    """Complete-Information All-Pay Auction with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuation = np.ones(n_agents)
        super().__init__(n_agents, n_discr, interval)

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for All-Pay Auction"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute ex-post utility
        return allocation * self.valuation - action_profile


class Contest(EconGame):
    """Tullock Contest"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
        csf_param: float = 1.0,
    ):
        self.csf_param = csf_param
        self.valuation = np.ones(n_agents)
        super().__init__(n_agents, n_discr, interval)

    def allocation(self, action_profile: np.ndarray) -> np.ndarray:
        """compute winning probabilities for Tullock-Contest"""
        n_agents = len(action_profile)
        if np.any(action_profile > 0):
            return (
                action_profile**self.csf_param
                / (action_profile**self.csf_param).sum()
            )
        else:
            return np.ones(n_agents) / n_agents

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for Tullock-Contest"""
        allocation = self.allocation(action_profile)
        payments = action_profile
        return allocation * self.valuation - payments
