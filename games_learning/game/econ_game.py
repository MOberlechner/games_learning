from itertools import product
from typing import List, Tuple

import numpy as np

from games_learning.game.matrix_game import MatrixGame


class Game:
    """Class for Normal-Form Game G = (N, X, u)

    with    - N is set of agents
            - X is action space (1-dim)
            - u is utility function
    """

    def __init__(
        self,
        n_agents: int,
        action_space: List[Tuple] = None,
        name: str = "game",
    ):
        self.name = name
        self.n_agents = n_agents
        self.agents = list(range(n_agents))

        self.action_space = (
            action_space
            if isinstance(action_space, list)
            else [action_space] * n_agents
        )
        assert len(action_space) == self.n_agents

    def __repr__(self) -> str:
        return self.name

    def utility(self, actions: Tuple[float]) -> Tuple[float]:
        """utilities for all agents given an action profile"""
        raise NotImplementedError


class EconGame(Game):
    """Class for Economic Games such as Auctions, Contests, and Pricing Games"""

    def __init__(
        self,
        n_agents: int,
        action_space: Tuple[float] = (0.0, 1.0),
        name: str = "econgame",
    ):
        """Create EconGame Class

        Args:
            n_agents (int): number of agents
            action_space (Tuple[float], optional): action space of single agent (we assume symmetry). Defaults to (0.0, 1.0).
            name (str): name of game. Defaults to "econgame"
        """
        super().__init__(n_agents, action_space, name)

    def discretize(self, n_discr: int) -> MatrixGame:
        """Create matrix game by discretizing the action space

        Args:
            n_discr (int): number of discrete actions per agent (equidistant)

        Returns:
            MatrixGame: constructed matrix game
        """
        # create payoff matrices
        actions = [np.linspace(*interval, n_discr) for interval in self.action_space]
        util_arr = np.array([self.utility(np.array(a)) for a in product(*actions)])
        util_arr_form = util_arr.T.reshape([self.n_agents] + [n_discr] * self.n_agents)
        payoff_matrices = tuple([util for util in util_arr_form])

        # create matrix game
        matrix_game = MatrixGame(
            self.n_agents, payoff_matrices, f"{self.name}_discr{n_discr}"
        )

        return matrix_game


class FPSB(EconGame):
    """Complete-Information First-Price Sealed Bid with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        action_space: Tuple[float],
        valuation: Tuple[float],
    ):
        super().__init__(n_agents, action_space, "fpsb")
        self.valuation = valuation
        assert len(valuation) == n_agents

    def utility(self, action_profile: np.ndarray) -> np.ndarray:
        """utility for FPSB"""
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
        action_space: Tuple[float],
        valuation: Tuple[float],
    ):
        super().__init__(n_agents, action_space, "spsb")
        self.valuation = valuation

        assert len(valuation) == n_agents

    def utility(self, action_profile: np.ndarray) -> np.ndarray:
        """utility for SPSB"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute payment (second-price)
        second_price = np.sort(action_profile)[-2]
        # compute ex-post utility
        return allocation * (self.valuation - second_price)


class AlphaSB(EconGame):
    """Complete-Information Sealed Bid auction with mixture of first-and second-price payment rule"""

    def __init__(
        self,
        n_agents: int,
        action_space: Tuple[float],
        alpha: float,
        valuation: Tuple[float],
    ):
        super().__init__(n_agents, action_space, "alpha_sb")
        self.valuation = valuation

        assert len(valuation) == n_agents

    def utility(self, action_profile: np.ndarray) -> np.ndarray:
        """utility for AlphaSB"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute payment
        first_price = first_price = action_profile.max()
        second_price = np.sort(action_profile)[-2]
        payment = (1 - self.alpha) * first_price + self.alpha * second_price
        # compute ex-post utility
        return allocation * (self.valuation - payment)


class AllPay(EconGame):
    """Complete-Information All-Pay Auction with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        action_space: Tuple[float],
        valuation: Tuple[float],
    ):
        super().__init__(n_agents, action_space, "allpay")
        self.valuation = valuation

        assert len(valuation) == n_agents

    def utility(self, action_profile: np.ndarray) -> np.ndarray:
        """utility for All-Pay Auction"""
        # compute allocation
        action_max = action_profile == action_profile.max()
        allocation = action_max / action_max.sum()
        # compute ex-post utility
        return allocation * self.valuation - action_profile


class Contest(EconGame):
    """Tullock Contest"""

    def __init__(
        self,
        n_agents: int,
        action_space: Tuple[float],
        valuation: Tuple[float],
        csf_param: float = 1.0,
    ):
        super().__init__(n_agents, action_space, "contest")
        self.csf_param = csf_param
        self.valuation = np.array(valuation)

        assert len(valuation) == n_agents

    def utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for Tullock-Contest"""
        allocation = self.allocation(action_profile)
        payments = action_profile
        return allocation * self.valuation - payments

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


class Cournot(EconGame):
    """Cournot Competition"""

    def __init__(
        self,
        n_agents: int,
        action_space: Tuple[float],
        alpha: float,
        beta: Tuple[float],
        cost: Tuple[float],
    ):
        super().__init__(n_agents, action_space, "cournot")
        self.alpa = alpha
        self.beta = beta
        self.cost = cost

        assert len(beta) == n_agents
        assert len(cost) == n_agents

    def utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for Cournot Competition"""
        return action_profile * (self.price(action_profile) - np.array(self.cost))

    def price(self, action_profile: np.ndarray) -> float:
        """Compute price given quantities (actions) of firms (agents)"""
        return np.maximum(
            self.alpha - (np.array(self.beta) * action_profile).sum(), 0.0
        )


class BertrandLinear(EconGame):
    """Bertrand Competition with linear demand (Hansen et al., 2021)"""

    def __init__(
        self,
        n_agents: int,
        action_space: Tuple[float],
        alpha: Tuple[float],
        beta: Tuple[float],
        gamma: float,
        cost: Tuple[float],
    ):

        super().__init__(n_agents, action_space, "bertrand_linear")
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = gamma
        self.cost = np.array(cost)

        assert len(b) == n_agents
        assert len(cost) == n_agents

    def utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for Bertrand Competition with linear demand"""
        return self.demand(action_profile) * (action_profile - self.cost)

    def demand(self, action_profile: np.ndarray) -> np.ndarray:
        """Compute demand for each agent given the prices (actions) of firms (agents)"""
        return (
            self.alpha
            - self.beta * action_profile
            + self.gamma
            * np.array(
                [self.mean_prices_opponents(action_profile, i) for i in self.agents]
            )
        )

    def mean_prices_opponents(self, action_profile: np.ndarray, agent: int) -> float:
        """Compute sum of prices of all agents except for agent (index)"""
        return (
            1 / (len(action_profile) - 1) * action_profile[:agent].sum()
            + action_profile[agent + 1 :].sum()
        )
