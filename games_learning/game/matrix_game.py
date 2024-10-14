import importlib.util
from typing import Dict, List, Tuple, Union

import numpy as np

import games_learning.utils.dominance as domin
import games_learning.utils.equilibrium as equil


class MatrixGame:
    """Class for Matrix-Game G = (N, A, u)

    with
        - N is the set of agents
        - A is the finite set of actions
        - u is the utility function which can be written as a payoff matrix/tensor
    We consider the mixed extension of the game.

    Methods:
        - get_pne: get all pure Nash equilibria
        - get_ce: compute a correlated equilibrium
        - get_cce: compute a coarse correlated equilibrium
        - get_supp_ce: which action profiles are supported by some ce
        - get_supp_cce: which action profiles are supperted by some cce

    """

    def __init__(
        self,
        payoff_matrix: np.ndarray,
        name: str = "matrix_game",
        name_actions: List[str] = None,
    ):
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
        self.name_actions = name_actions

        if not np.all(
            [len(payoff_matrix[i].shape) == self.n_agents for i in self.agents]
        ):
            raise ValueError("error in dimensions of payoff-matrix")

    def __repr__(self) -> str:
        return f"MatrixGame({self.name},{self.n_actions})"

    def get_pne(self, atol: float = 1e-9) -> dict:
        """compute all pure Nash equilibria (PNE) for game

        Args:
            atol (float, optional): tol for (strict) inequalities. Defaults to 1e-10.

        Returns:
            dict: returns dict with "weak_ne", "strict_ne" and all "ne"
        """
        pne = equil.get_pure_nash_equilibrium(self.payoff_matrix, atol=atol)
        return pne

    def get_ce(self, objective: np.ndarray = None) -> np.ndarray:
        """compute a correlated equilibrium CE

        Args:
            objective (np.ndarray, optional): objective to choose a (C)CE. If no objective is specified, all action profiles get equal weight. Defaults to None.

        Returns:
            np.ndarray: CE (probability distribution over all action profiles)
        """
        return equil.get_correlated_equilibrium(
            self.payoff_matrix, coarse=False, objective=objective
        )

    def get_cce(self, objective: np.ndarray = None) -> np.ndarray:
        """compute a coarse correlated equilibrium CCE

        Args:
            objective (np.ndarray, optional): objective to choose a (C)CE. If no objective is specified, all action profiles get equal weight. Defaults to None.

        Returns:
            np.ndarray: CCE (probability distribution over all action profiles)
        """
        return equil.get_correlated_equilibrium(
            self.payoff_matrix, coarse=True, objective=objective
        )

    def get_supp_ce(self, atol: float = 1e-9) -> np.ndarray:
        """check which action profiles are supported by a ce, i.e., is there any CE that puts a strictly positive probability mass (> atol) on action profiles

        Args:
            atol (float, optional): probability mass has to be larger than atol. Defaults to 1e-10.

        Returns:
            np.ndarray: binary array with entry for each action profile
        """
        supp_ce = equil.get_support_correlated_equilibria(
            self.payoff_matrix, coarse=False, atol=atol
        )
        return supp_ce

    def get_supp_cce(self, atol: float = 1e-9) -> np.ndarray:
        """check which action profiles are supported by a cce, i.e., is there any CCE that puts a strictly positive probability mass (> atol) on action profiles

        Args:
            atol (float, optional): probability mass has to be larger than atol. Defaults to 1e-10.

        Returns:
            np.ndarray: binary array with entry for each action profile
        """
        supp_cce = equil.get_support_correlated_equilibria(
            self.payoff_matrix, True, atol
        )
        return supp_cce

    def get_undominated_actions(
        self, dominance: str = "strict", atol: float = 1e-9, print: bool = False
    ) -> dict:
        """returns actions that survive iterated removal of dominated actions, i.e., serially undominated actions. we distinguish betweem two cases of dominance:
         - stong: action is dominated by another pure action (for opponents' actions)
        - strict: action is dominated by a mixed strategy (for all opponents' actions)
        In both cases, we assume that the inequalities are strictly satisfied, i.e., no weak dominance.

        Args:
            dominance (str, optional): choose between strict and strong. Defaults to "strict".

        Returns:
            dict: contains all undominated actions for agents
        """
        (
            reduced_payoff_matrix,
            removed_actions,
            remaining_actions,
        ) = domin.iterated_dominance_solver(self.payoff_matrix, dominance, atol, print)
        return remaining_actions

    def get_potentialness(self, decomposition: bool = False):
        """returns potentialness of game

        Args:
            decomposition (bool, optional): If True, we additionally return the decomposition of the game. Defaults to False.

        Returns:
            float: potentialness
        """
        if importlib.util.find_spec("games_decomposition") is None:
            print("Install package games_decomposition!")
            return None

        elif not decomposition:
            from games_decomposition.game import Game

            hodge = Game(self.n_actions, save_load=False)
            hodge.compute_flow_decomposition_matrix(self.payoff_matrix)
            return hodge.flow_metric

        else:
            raise NotImplementedError

    def get_named_actions(self, numbered_actions: Dict) -> Dict:
        """instead of numbered actions, we return the names of the actions.

        Args:
            numbered_actions (Union[Dict]): list or dict with actions or action profiles. Output from get_pne etc.

        Returns:
            Union[Dict]: same as input, but with numbers
        """
        # no names available
        if self.name_actions is None:
            return numbered_actions

        elif isinstance(numbered_actions, dict):
            named_actions = {}
            # dictionaries contain lists of actions for each agent (e.g. get_undominated_actions)
            if list(numbered_actions.keys()) == self.agents:
                for i in self.agents:
                    named_actions[i] = [
                        self.name_actions[i][a] for a in numbered_actions[i]
                    ]
            else:
                # dictionaries contain action profiles (e.g. get_pne)
                for key, vals in numbered_actions.items():
                    named_actions[key] = [
                        tuple(map(lambda lst, i: lst[i], self.name_actions, profiles))
                        for profiles in vals
                    ]

        return named_actions


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
        payoff_matrix, name, name_actions = self.create_setting(setting, parameter)
        super().__init__(payoff_matrix, name, name_actions)

    def create_setting(self, setting, parameter):
        if setting == "matching_pennies":
            payoff_matrix = [np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]])]
            name_actions = [["Heads", "Tails"], ["Heads", "Tails"]]
            return payoff_matrix, setting, name_actions

        elif setting == "battle_of_sexes":
            payoff_matrix = [np.array([[5, 1], [2, 6]]), np.array([[6, 1], [2, 5]])]
            name_actions = [["Baseball", "Softball"], ["Baseball", "Softball"]]
            return payoff_matrix, setting, name_actions

        elif setting == "prisoners_dilemma":
            payoff_matrix = [np.array([[4, 1], [5, 2]]), np.array([[4, 5], [1, 2]])]
            name_actions = [["Confess", "Silent"], ["Confess", "Silent"]]
            return payoff_matrix, setting, name_actions

        elif setting == "rock_paper_scissors":
            payoff_matrix = [
                np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
                np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]),
            ]
            name_actions = [
                ["Rock", "Paper", "Scissors"],
                ["Rock", "Paper", "Scissors"],
            ]
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
            payoff_matrix = [
                np.array([[1 - self.alpha, -self.alpha], [0, 0]]),
                np.array([[self.beta - 1, 0], [self.beta, 0]]),
            ]
            return payoff_matrix, f"jordan_game(alpha={alpha}, beta={beta})"

        else:
            raise ValueError(
                f"matrix game {setting} not available. Choose from: matching_pennies, battle_of_sexes, prisoners_dilemma, rock-paper-scissors, jordan_game"
            )


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
