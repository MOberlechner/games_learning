from typing import Union

from games_learning.game.matrix_game import MatrixGame


class Strategy:
    """Class for Strategies

    Mixed strategies are given by x, where x[i] is a probability distribution over agent i's set of actions A_i ( given by game G = (N, A, u) ).

    Methods:
    - compute gradient
    - compute expected utility
    - compute best response
    - compute utility loss (absolute and relative)

    """

    def __init__(self, game: MatrixGame, init_method: str = "random"):

        self.game = game
        self.n_agents = game.n_agents
        self.agents = game.agents
        self.n_actions = self.game.n_actions
        self.x = self.init_strategies(init_method)

    def __repr__(self) -> str:
        return f"Strategies(game:{self.game.name}, {self.n_actions})"

    @property
    def y(self) -> Tuple[np.ndarray]:
        """gradients for all agents"""
        return tuple(self.gradient(i) for i in self.agents)

    def gradient(self, agent: int) -> np.ndarray:
        """compute gradient of expected utility for agent

        Args:
            agent (int): agent (index)

        Returns:
            np.ndarray: gradient for agent
        """
        strategies_opp = remove_index(strategies.x, agent)
        subscript = get_einsum_subscripts(agent, self.n_agents)
        return np.einsum(subscript, *strategies_opp, self.game.payoff_matrix[agent])

    def utility(self, agent: int, gradient: np.ndarray = None) -> float:
        """compute expected utility for agent

        Args:
            agent (int): agent (index)
            gradient (np.ndarray, optional): If gradient was already computed, we can use it to save time. Otherwise we first have to compute the gradient. Defaults to None.

        Returns:
            float: expected utility for agent
        """
        if gradient is None:
            gradient = self.gradient(agent)
        return gradient.dot(strategies.x[agent])

    def best_response(self, agent: int, gradient: np.ndarray = None) -> np.ndarray:
        """compute best response for agent

        Args:
            agent (int): agent (index)
            gradient (np.ndarray, optional): If gradient was already computed, we can use it to save time. Otherwise we first have to compute the gradient. Defaults to None.

        Returns:
            np.ndarray: best response for agent
        """
        if gradient is None:
            gradient = self.gradient(agent)
        best_response = np.zeros_like(gradient)
        best_response[gradient.argmax()] = 1
        return best_response

    def utility_loss(
        self,
        agent: int,
        best_response: np.ndarray = None,
        gradient: np.ndarray = None,
        method: str = "rel",
    ) -> float:
        """compute relative (rel) or absolute (abs) utility loss for agent

        Args:
            agent (int): agent (index)
            best_response (np.ndarray, optional): To save time we can use already computed best response. Defaults to None.
            gradient (np.ndarray, optional): To save time we can use already computed best response. Defaults to None.

        Returns:
            float: rel/abs utility loss for agent
        """
        if best_response is None:
            if gradient is None:
                gradient = self.gradient(agent)
            best_response = self.best_response(agent, gradient)

        if method == "rel":
            return 1 - gradient.dot(self.x[agent]) / gradient.dot(best_response)
        elif method == "abs":
            return gradient.dot(best_response) - gradient.dot(self.x[agent])
        else:
            raise ValueError(
                f"Choose between methods: rel, abs. method={method} is not available"
            )

    def init_strategies(self, method) -> Tuple[np.ndarray]:
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


# ------------------------------ HELPERFUNCTIONS ------------------------------ #


def remove_index(l: Union[list, tuple], i: int) -> Union[list, tuple]:
    """remove i-th entry from list"""
    return l[:i] + l[i + 1 :]


def get_einsum_subscripts(agent: int, n_agents: int) -> str:
    """create indices used in einsum to compute gradient"""
    indices = "".join([chr(ord("a") + i) for i in range(n_agents)])
    indices_opp = remove_index(indices, agent)
    return f"{','.join(indices_opp)},{indices}->{indices[agent]}"
