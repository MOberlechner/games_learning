from typing import List, Tuple, Union

import numpy as np


class MatrixGame:
    """Class for Matrix-Game G = (N, A, u)

    with
        - N is the set of agents
        - A is the finite set of actions
        - u is the utility function which can be written as a payoff matrix/tensor
    We consider the mixed extension of the game.

    Methods:
        given a profile of mixed strategies, we can compute
        - gradient      compute gradient for agent
        - utility (expected utility)
        - best response
        - utility loss
        - init_strategies

    """

    def __init__(
        self, n_agents: int, payoff_matrix: Tuple[np.ndarray], name: str = "matrix_game"
    ):
        """Matrix Game

        Args:
            n_agents (int): _description_
            payoff_matrices (Tuple[np.ndarray]): list of payoff matrices
            name (str, optional): _description_. Defaults to "".
        """
        self.name = name
        self.agents = list(range(n_agents))
        self.n_agents = n_agents
        self.n_actions = list(payoff_matrix[0].shape)
        self.payoff_matrix = payoff_matrix

        assert len(payoff_matrix) == n_agents
        assert len(payoff_matrix[0].shape) == n_agents

    def __repr__(self) -> str:
        return f"MatrixGame(agents={self.n_agents},actions={self.n_actions})"

    def gradient(self, strategies: List, agent: int) -> np.ndarray:
        """Gradient Function

        Args:
            strategies (List[np.ndarray]): List of agents' strategies
            agent (int): index of agent

        Returns:
            List[np.ndarray]: Gradient for each agents
        """
        assert np.all([s.ndim == 1 for s in strategies])
        strategies_opp = remove_index(strategies, agent)
        subscript = get_einsum_subscripts(agent, self.n_agents)
        return np.einsum(subscript, *strategies_opp, self.payoff_matrix[agent])

    def utility(self, strategies: Tuple[np.ndarray], agent) -> float:
        """compute utility

        Args:
            strategy (np.ndarray): agent's current strategy
            agent (int): index of agent

        Returns:
            float: utility
        """
        gradient = self.gradient(strategies, agent)
        return gradient.dot(strategies[agent])

    def best_response(self, gradient: np.ndarray) -> np.ndarray:
        """compute best response given the gradient
        lower actions (w.r.t. index) are prefered in ties

        Args:
            gradient (np.ndarray): agent's gradient

        Returns:
            np.ndarray: best response
        """
        best_response = np.zeros_like(gradient)
        best_response[gradient.argmax()] = 1
        return best_response

    def utility_loss(
        self, strategies: List[np.ndarray], agent: int, normed: bool = True
    ) -> float:
        """compute agent's utility loss

        Args:
            strategies (List[np.ndarray]): List of agents' strategies
            agent (int): index of agent
            normed (bool): absolute (false) or relative (true) utility loss

        Returns:
            float: utility loss
        """
        gradient = self.gradient(strategies, agent)
        best_response = self.best_response(gradient)
        if normed:
            return 1 - gradient.dot(strategies[agent]) / gradient.dot(best_response)
        else:
            return gradient.dot(best_response) - gradient.dot(strategies[agent])

    def init_strategies(self, method: str = "random") -> Tuple[np.ndarray]:
        """generate initial mixed strategies

        Args:
            method (str, optional): different initializations methods. Defaults to "random".
                Note that random uses a Dirichlet distribution which generates uniform distributed points
                from the probability simplex.

        Returns:
            Tuple[np.ndarray]: profile of mixed strategies
        """
        if method == "equal":
            return tuple(
                np.ones(self.n_actions[i]) / self.n_actions[i] for i in self.agents
            )

        elif method == "random":
            return tuple(
                np.random.dirichlet((1,) * self.n_actions[i]) for i in self.agents
            )

        elif method == "uniform":
            uniform_numbers = tuple(
                np.randrom.rand(self.n_actions[i]) for i in self.agents
            )
            return tuple(
                uniform_numbers[i] / uniform_numbers[i].sum() for i in self.agents
            )

        else:
            raise ValueError(
                f"init method {method} not available. Choose from equal, random, or uniform."
            )


class JordanGame(MatrixGame):
    def __init__(
        self,
        seed: int = None,
        distribution: str = "uniform",
    ):
        """Jordan Game: Def 2.1 in Jordan, “Three Problems in Learning Mixed-Strategy Nash Equilibria."""
        payoff_matrix = self.create_matrices(seed)
        super().__init__(2, payoff_matrix)
        self.name = f"jordan_game({seed})" if seed is not None else "jordan_game"

    def __repr__(self) -> str:
        return (
            f"JordanGame(alpha={self.alpha:.3f},beta={self.beta:.3f}, seed={self.seed})"
        )

    def create_matrices(self, seed: int):
        np.random.seed(seed)
        self.seed = seed
        self.alpha, self.beta = np.random.uniform(size=2)
        return tuple(
            [
                np.array([[1 - self.alpha, -self.alpha], [0, 0]]),
                np.array([[self.beta - 1, 0], [self.beta, 0]]),
            ]
        )


class RandomMatrixGame(MatrixGame):
    def __init__(
        self,
        n_agents,
        n_actions: List[int],
        seed: int = None,
        distribution: str = "uniform",
    ):
        """Create random matrix game ”

        Args:
            n_agents (_type_): _description_
            n_actions (list[int]): _description_
        """
        payoff_matrix = self.create_matrices(n_agents, n_actions, seed, distribution)
        assert n_agents == len(n_actions)

        super().__init__(n_agents, payoff_matrix)
        self.name = "random_matrix_game"
        self.seed = seed
        self.distribution = distribution

    def __repr__(self) -> str:
        return f"Random(agents={self.n_agents}, actions={self.n_actions}, seed={self.seed})"

    def create_matrices(
        self, n_agents: int, n_actions: list, seed: int, distribution: str
    ):
        """Generate random payoff matrix

        Args:
            n_agents (int): number of agents
            n_actions (list): number of actions for each agents
            seed (int): seed for random generator
            distribution (str): distribution of entries of payoff matrices

        Returns:
            np.ndarray: contains all payoff matrices
        """
        dimension = tuple([n_agents] + n_actions)
        rng = np.random.default_rng(seed)
        if distribution == "uniform":
            return rng.random(size=dimension, dtype=np.float64)
        elif distribution == "normal":
            return rng.normal(loc=0.0, scale=1.0, size=dimension)
        else:
            raise NotImplementedError(f"Distribition {distribution} not implemented")


# ------------------------------ HELPERFUNCTIONS ------------------------------ #


def remove_index(l: list, i: int):
    """remove i-th entry from list"""
    return l[:i] + l[i + 1 :]


def get_einsum_subscripts(agent: int, n_agents: int) -> str:
    """create indices used in einsum to compute gradient"""
    indices = "".join([chr(ord("a") + i) for i in range(n_agents)])
    indices_opp = remove_index(indices, agent)
    return f"{','.join(indices_opp)},{indices}->{indices[agent]}"
