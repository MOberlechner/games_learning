from itertools import product
from typing import List, Tuple, Union

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
        self.agents = list(range(n_agents))
        payoff_matrix = self.create_payoff_matrix(n_agents, n_discr, interval)
        name_actions = [[f"{a:.2f}" for a in self.actions] for i in self.agents]
        super().__init__(
            payoff_matrix=payoff_matrix, name="econgame", name_actions=name_actions
        )
        self.interval = interval

    def __repr__(self) -> str:
        return f"EconGame({self.name},{self.n_actions})"

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
        self.actions = actions
        # compute utilities
        util_arr = np.array(
            [
                self.ex_post_utility(np.array(a))
                for a in product(actions, repeat=n_agents)
            ]
        )
        # reformat array
        return util_arr.T.reshape([n_agents] + [n_discr] * n_agents)

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """compute ex-post utility given a action profile"""
        raise NotImplementedError

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """compute ex-post utility for Bayesian Game, i.e., with types"""
        print("Implement method to use for bayesian games")
        raise NotImplementedError


class FPSB(EconGame):
    """Complete-Information First-Price Sealed Bid with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        valuations: Tuple[float],
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuation = np.array(valuations)
        super().__init__(n_agents, n_discr, interval)
        self.name = "fpsb"

        assert len(valuations) == n_agents

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for complete-information FPSB"""
        return self.ex_post_utility_bayesian(action_profile, self.valuation)

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """ex-post utility for FPSB (with types)"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute payment (first-price)
        first_price = action_profile.max()
        # compute ex-post utility
        return allocation * (type_profile - first_price)


class SPSB(EconGame):
    """Complete-Information Second-Price Sealed Bid with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        valuations: Tuple[float],
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuation = np.array(valuations)
        super().__init__(n_agents, n_discr, interval)
        self.name = "spsb"

        assert len(valuations) == n_agents

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for complete-information SPSB"""
        return self.ex_post_utility_bayesian(action_profile, self.valuation)

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """ex-post utility for SPSB (with types)"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute payment (second-price)
        second_price = np.sort(action_profile)[-2]
        # compute ex-post utility
        return allocation * (type_profile - second_price)


class AlphaSB(EconGame):
    """Complete-Information Sealed Bid auction with mixture of first-and second-price payment rule"""

    def __init__(
        self,
        n_agents: int,
        alpha: float,
        n_discr: int,
        valuations: Tuple[float],
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuation = np.array(valuations)
        self.alpha = alpha
        super().__init__(n_agents, n_discr, interval)
        self.name = "alpha_sb"

        assert len(valuations) == n_agents

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for complete-information AlphaSB"""
        return self.ex_post_utility_bayesian(action_profile, self.valuation)

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """ex-post utility for AlphaSB (with types)"""
        # compute allocation
        action_max = np.isclose(
            np.array(action_profile), np.array(action_profile).max()
        )
        allocation = action_max / action_max.sum()
        # compute payment
        first_price = first_price = action_profile.max()
        second_price = np.sort(action_profile)[-2]
        payment = (1 - self.alpha) * first_price + self.alpha * second_price
        # compute ex-post utility
        return allocation * (type_profile - payment)


class AllPay(EconGame):
    """Complete-Information All-Pay Auction with random tie-breaking and fixed value v=1"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        valuations: Tuple[float],
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuation = np.array(valuations)
        super().__init__(n_agents, n_discr, interval)
        self.name = "allpay"

        assert len(valuations) == n_agents

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for complete-information All-Pay Auction"""
        return self.ex_post_utility_bayesian(action_profile, self.valuation)

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """ex-post utility for All-Pay Auction"""
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute ex-post utility
        return allocation * type_profile - action_profile


class Contest(EconGame):
    """Tullock Contest"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        valuations: Tuple[float],
        interval: Tuple[float] = (0.0, 1.0),
        csf_param: float = 1.0,
        epsilon: float = 0.0,
    ):
        self.csf_param = csf_param
        self.valuation = np.array(valuations)
        self.epsilon = epsilon
        super().__init__(n_agents, n_discr, interval)
        self.name = "contest"

        assert len(valuations) == n_agents

    def allocation(self, action_profile: np.ndarray) -> np.ndarray:
        """compute winning probabilities for Tullock-Contest"""
        n_agents = len(action_profile)
        if np.any(action_profile > 0):
            return (
                action_profile**self.csf_param
                / (action_profile**self.csf_param + self.epsilon).sum()
            )
        else:
            return np.ones(n_agents) / n_agents

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for complete-information Tullock-Contest"""
        return self.ex_post_utility_bayesian(action_profile, self.valuation)

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """ex-post utility for Tullock-Contest (with types)"""
        allocation = self.allocation(action_profile)
        payments = action_profile
        return allocation * type_profile - payments


class Cournot(EconGame):
    """Cournot Competition"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        a: float,
        b: Tuple[float],
        cost: Tuple[float],
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.a = a
        self.b = b
        self.cost = np.array(cost)
        assert len(b) == n_agents
        assert len(cost) == n_agents

        super().__init__(n_agents, n_discr, interval)
        self.name = "cournot"

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for complete-information Cournot Competition"""
        return self.ex_post_utility_bayesian(action_profile, self.cost)

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """ex-post utility for Cournot Competition (with types)"""
        return action_profile * (self.price(action_profile) - type_profile)

    def price(self, action_profile: np.ndarray) -> float:
        """Compute price given quantities (actions) of firms (agents)"""
        return np.maximum(self.a - (np.array(self.b) * action_profile).sum(), 0.0)


class Bertrand(EconGame):
    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        cost: Union[float, Tuple[float, ...]],
        interval: Tuple[float] = (0.0, 1.0),
    ):
        # if parameter is single number, we assume symmetry
        cost = (cost,) * n_agents if isinstance(cost, (int, float)) else cost
        self.cost = np.array(cost)

        super().__init__(n_agents, n_discr, interval)
        self.name = "bertrand"

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        """ex-post utility for complete-information Bertrand Competition"""
        return self.ex_post_utility_bayesian(action_profile, self.cost)

    def ex_post_utility_bayesian(
        self, action_profile: np.ndarray, type_profile: np.ndarray
    ) -> np.ndarray:
        """ex-post utility for Bertrand Competition (with types)"""
        return self.demand(action_profile) * (action_profile - type_profile)

    def demand(self, action_profile: np.ndarray) -> np.ndarray:
        """Compute demand for each agent given the prices (actions) of firms (agents)"""
        raise NotImplementedError


class BertrandStandard(Bertrand):
    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        cost: Union[float, Tuple[float, ...]],
        interval: Tuple[float],
        maximum_demand: float = 1.0,
    ):
        self.maximum_demand = maximum_demand
        super().__init__(
            n_agents=n_agents, n_discr=n_discr, cost=cost, interval=interval
        )
        self.name = "bertrand_standard"

    def demand(self, action_profile: np.ndarray) -> np.ndarray:
        """Compute (standard) demand for each agent given the prices (actions) of firms (agents)"""
        # compute allocation (similar to auctions)
        action_min = np.isclose(
            np.array(action_profile), np.array(action_profile).min()
        )
        allocation = action_min / action_min.sum()
        return self.maximum_demand * allocation * (1 - action_profile)


class BertrandLinear(Bertrand):
    """Bertrand Competition with linear demand"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        cost: Union[float, Tuple[float, ...]],
        interval: Tuple[float],
        alpha: Union[float, Tuple[float, ...]],
        beta: Union[float, Tuple[float, ...]],
        gamma: float,
    ):
        # if parameter is single number, we assume symmetry
        alpha = (alpha,) * n_agents if isinstance(alpha, (int, float)) else alpha
        beta = (beta,) * n_agents if isinstance(beta, (int, float)) else beta
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = gamma

        super().__init__(
            n_agents=n_agents, n_discr=n_discr, cost=cost, interval=interval
        )
        self.name = "bertrand_linear"

    def demand(self, action_profile: np.ndarray) -> np.ndarray:
        """Compute demand for each agent given the prices (actions) of firms (agents)"""
        return (
            self.alpha
            - self.beta * action_profile
            + self.gamma
            * np.array(
                [self.sum_prices_opponents(action_profile, i) for i in self.agents]
            )
        )

    def sum_prices_opponents(self, action_profile: np.ndarray, agent: int) -> float:
        """Compute sum of prices of all agents except for agent (index)"""
        return action_profile[:agent].sum() + action_profile[agent + 1 :].sum()


class BertrandLogit(Bertrand):
    """Bertrand Competition with logit demand"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        cost: Union[float, Tuple[float, ...]],
        interval: Tuple[float],
        alpha: Union[float, Tuple[float, ...]],
        mu: Union[float, Tuple[float, ...]],
    ):
        # if parameter is single number, we assume symmetry
        alpha = (alpha,) * n_agents if isinstance(alpha, (int, float)) else alpha
        mu = (mu,) * n_agents if isinstance(mu, (int, float)) else mu
        self.alpha = np.array(alpha)
        self.mu = np.array(mu)

        super().__init__(
            n_agents=n_agents, n_discr=n_discr, cost=cost, interval=interval
        )
        self.name = "bertrand_logit"

    def demand(self, action_profile: np.ndarray) -> np.ndarray:
        """Compute demand for each agent given the prices (actions) of firms (agents).
        external influence is fixed on 1"""
        actions_exp = np.exp((self.alpha - action_profile) / self.mu)
        return actions_exp / (actions_exp.sum() + 1)


class WarOfAttrition(EconGame):
    """War of Attrition"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        valuations: np.ndarray,
        interval: Tuple[float] = (0.0, 1.0),
    ):
        self.valuations = valuations
        super().__init__(n_agents, n_discr, interval)
        self.name = "war_of_attrition"

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        # compute allocation
        action_max = np.array(action_profile) == np.array(action_profile).max()
        allocation = action_max / action_max.sum()
        # compute payments
        first_price = action_profile
        second_price = np.sort(action_profile)[-2]
        # compute ex-post utility
        return (
            allocation * (self.valuations - second_price)
            - (1 - allocation) * first_price
        )


class TragedyOfCommons(EconGame):
    """Tragedy of Commons"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
    ):
        super().__init__(n_agents, n_discr, interval)
        self.name = "tragedy_of_commons"

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        # compute ex-post utility
        return action_profile * (1 - action_profile.sum())


class PublicGood(EconGame):
    """Public Good"""

    def __init__(
        self,
        n_agents: int,
        n_discr: int,
        interval: Tuple[float] = (0.0, 1.0),
    ):
        # self.f = lambda x: 1.2 * x / n_agents
        # self.g = lambda x: x
        self.f = lambda x: 10 * x if x <= 15 else 10 * x - (x - 15) ** 2
        self.g = lambda x: 2 * x

        super().__init__(n_agents, n_discr, interval)
        self.name = "public_good"

    def ex_post_utility(self, action_profile: np.ndarray) -> np.ndarray:
        # compute ex-post utility
        total_contribution = action_profile.sum()
        return self.f(total_contribution) - self.g(action_profile)
