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

    """

    def __init__(self, payoff_matrix: np.ndarray, name: str = "matrix_game"):
        """Matrix Game

        Args:
            payoff_matrices (np.ndarray): indices: player, action player 1, action player 2, ...
            name (str, optional): _description_. Defaults to "".
        """
        self.name = name
        self.n_agents = len(payoff_matrix)
        self.agents = list(range(self.n_agents))
        self.n_actions = list(payoff_matrix[0].shape)
        self.payoff_matrix = payoff_matrix

        if not np.all(
            [len(payoff_matrix[i].shape) == self.n_agents for i in self.agents]
        ):
            raise ValueError("error in dimensions of payoff-matrix")

    def __repr__(self) -> str:
        return f"MatrixGame({self.name},{self.n_actions})"


class ExampleMatrixGames(MatrixGame):
    """This class contains some examples of matrix games, such as
    - matching_pennies
    - jordan_game
    - battle_of_sexes (numbers from: Nisan (2007) - Algorithmic Game Theory)
    - prisoners_dilemma (numbers from: Nisan (2007) - Algorithmic Game Theory)

    """

    def __init__(
        self,
        setting: str,
        parameter: dict = None,
    ):
        payoff_matrix, name = self.create_setting(setting, parameter)
        super().__init__(payoff_matrix, name)

    def create_setting(self, setting, parameter):
        if setting == "matching_pennies":
            payoff_matrix = (np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]]))
            return payoff_matrix, setting

        elif setting == "battle_of_sexes":
            payoff_matrix = (np.array([[5, 1], [2, 6]]), np.array([[6, 1], [2, 5]]))
            return payoff_matrix, setting

        elif setting == "prisoners_dilemma":
            payoff_matrix = (np.array([[4, 1], [5, 2]]), np.array([[4, 5], [1, 2]]))
            return payoff_matrix, setting

        elif setting == "jordan_game":
            if ("alpha" in parameter) and ("beta" in parameter):
                alpha, beta = parameter["alpha"], parameter["beta"]
            elif "seed" in parameter:
                seed = parameter["seed"]
                rng = np.random.default_rng(seed)
                alpha, beta = rng.random(size=2, dtype=np.float64)
            else:
                alpha, beta = np.random.uniform(size=2)
            payoff_matrix = (
                np.array([[1 - self.alpha, -self.alpha], [0, 0]]),
                np.array([[self.beta - 1, 0], [self.beta, 0]]),
            )
            return payoff_matrix, f"jordan_game(alpha={alpha}, beta={beta})"


class RandomMatrixGame(MatrixGame):
    def __init__(
        self,
        n_actions: List[int],
        seed: int = None,
        distribution: str = "uniform",
        parameters: dict = {},
    ):
        """Create random matrix game ”

        Args:
            n_agents (_type_): _description_
            n_actions (list[int]): _description_
        """
        payoff_matrix = self.create_matrices(n_actions, seed, distribution, parameters)

        super().__init__(payoff_matrix)
        self.name = "random_matrix_game"
        self.seed = seed
        self.distribution = distribution
        self.parameters = parameters

    def __repr__(self) -> str:
        return f"Random(agents={self.n_agents}, actions={self.n_actions}, seed={self.seed})"

    def create_matrices(
        self,
        n_agents: int,
        n_actions: list,
        seed: int,
        distribution: str,
        parameters: dict,
    ):
        """Generate random payoff matrix

        Args:
            n_agents (int): number of agents
            n_actions (list): number of actions for each agents
            seed (int): seed for random generator
            distribution (str): distribution of entries of payoff matrices
            parameters (str): parameters for distribution

        Returns:
            np.ndarray: contains all payoff matrices
        """
        dimension = tuple([len(n_actions)] + n_actions)
        rng = np.random.default_rng(seed)
        if distribution == "uniform":
            return rng.random(size=dimension, dtype=np.float64)
        if distribution == "uniform_int":
            if ("lb" in parameters) and ("ub" in parameters):
                return rng.integers(
                    low=parameters["lb"], high=parameters["ub"] + 1, size=dimension
                )
            else:
                raise ValueError(
                    "Specify lb (lower bound) and ub (upper bound) in parameters"
                )
        elif distribution == "normal":
            return rng.normal(loc=0.0, scale=1.0, size=dimension)
        else:
            raise NotImplementedError(f"Distribition {distribution} not implemented")
