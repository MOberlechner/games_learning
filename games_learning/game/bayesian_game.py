from itertools import combinations_with_replacement, product
from typing import List, Tuple

import numpy as np

from games_learning.game.econ_game import EconGame
from games_learning.game.matrix_game import MatrixGame
from games_learning.strategy import Strategy


class BayesianEconGame(MatrixGame):
    """Create an incomplete-information (or Bayesian) game from an economic game"""

    def __init__(
        self,
        econ_game: EconGame,
        n_discr: int,
        interval: Tuple[float, float] = (0.0, 1.0),
        distribution: str = "uniform",
        monotone_strategies: bool = False,
    ):
        self.econ_game = econ_game
        self.agents = econ_game.agents
        self.n_agents = econ_game.n_agents
        self.bayesian_actions = econ_game.actions
        self.bayesian_types = np.linspace(*interval, n_discr)
        self.prior = self.get_prior(distribution, n_discr)
        self.monotone_strategies = monotone_strategies

        self.create_strategies()
        payoff_matrix = self.create_payoff_matrix()
        super().__init__(payoff_matrix=payoff_matrix, name=f"bayesian_{econ_game.name}")

    def create_payoff_matrix(self):
        """create the payoff matrix of the Bayesian game"""
        # compute (expected) utilities
        util_array = np.array(
            [
                self.compute_expected_utility(strategy_profile)
                for strategy_profile in product(self.strategies, repeat=self.n_agents)
            ]
        )
        # reformat array
        return util_array.T.reshape(
            [self.n_agents] + [self.n_strategies] * self.n_agents
        )

    def compute_expected_utility(self, strategy_profile) -> np.ndarray:
        """compute expected utility for all agents given strategy profile"""
        n_typ = len(self.bayesian_types)
        return np.array(
            [
                self.compute_exinterim_utility(strategy_profile, type_profile)
                * self.compute_probability_type_profile(type_profile)
                for type_profile in product(range(n_typ), repeat=self.n_agents)
            ]
        ).sum(axis=0)

    def compute_exinterim_utility(
        self, strategy_profile, type_profile: Tuple[int, ...]
    ) -> np.ndarray:
        """compute ex-interim utility for agent given strategy profile and type
        Args:
            strategy_profile (List[List]): strategy profile (pure strategies)
            type_profile (Tuple[int, ...]): type profile (index of types)

        Returns:
            np.ndarray: ex-interim utility given strategy and type profile
        """
        type_profile_vals = np.array([self.bayesian_types[t] for t in type_profile])
        action_profile_vals = np.array(
            [
                self.bayesian_actions[strategy_profile[i][t]]
                for i, t in enumerate(type_profile)
            ]
        )
        exinterim_util = self.econ_game.ex_post_utility_bayesian(
            action_profile_vals, type_profile_vals
        )
        return exinterim_util

    def compute_probability_type_profile(self, type_profile: Tuple[int, ...]) -> float:
        """compute probability of type profile"""
        return np.prod([self.prior[t] for t in type_profile])

    def create_strategies(self):
        """create list of all possible strategies, i.e., mappings from types to actions.
        Represented as list of tuples (one for each type) with (type_index, action_index, P(typye)
        """
        n_typ, n_act = len(self.bayesian_types), len(self.bayesian_actions)
        index_types = range(n_typ)
        strategies = []
        if self.monotone_strategies:
            iterator = combinations_with_replacement(range(n_act), n_typ)
        else:
            iterator = product(range(n_act), repeat=n_typ)

        for index_actions in iterator:
            s = dict(zip(index_types, index_actions))
            strategies.append(s)
        self.strategies = strategies
        self.n_strategies = len(strategies)

    def print_strategies(self, index: int):
        """print strategy profile for given index"""
        for y in range(
            len(self.bayesian_actions) - 1, -1, -1
        ):  # Start from the top row (y-axis reversed)
            row = []
            for x in range(len(self.bayesian_types)):  # Iterate over x-axis
                if self.strategies[index].get(x) == y:
                    row.append("X")
                else:
                    row.append("o")
            print(" ".join(row))

    def get_prior(self, distribution, n_discr) -> np.ndarray:
        """get vector of probabilities of marginal prior (we assume iid)"""
        if distribution == "uniform":
            return np.ones(n_discr) / n_discr
        else:
            raise ValueError(f"prior '{distribution}' not implemented")
