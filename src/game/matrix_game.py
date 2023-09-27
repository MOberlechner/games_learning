from typing import List

import numpy as np

from src.game.game import Game


class MatrixGame(Game):
    def __init__(self, n_agents: int, payoff_matrix: List[np.ndarray]):
        """Matrix Game

        Args:
            n_agents (int): _description_
            payoff_matrices (List[np.ndarray]): list of payoff matrices
            name (str, optional): _description_. Defaults to "".
        """
        super().__init__(n_agents)
        self.name = "matrix_game"
        self.payoff_matrix = payoff_matrix

        assert len(payoff_matrix) == n_agents
        assert len(payoff_matrix[0].shape) == n_agents

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

        subscript = self.get_einsum_subscripts(agent)
        return np.einsum(subscript, *strategies_opp, self.payoff_matrix[agent])

    def utility(self, strategies: List[np.ndarray], agent) -> float:
        """compute utility

        Args:
            strategy (np.ndarray): agent's current strategy
            agent (int): index of agent

        Returns:
            float: utility
        """
        gradient = self.gradient(strategies, agent)
        return gradient.dot(strategies[agent])

    def get_einsum_subscripts(self, agent: int) -> str:
        """create subscripts used in np.einsum method to compute gradient

        Args:
            agent (int): index of agent
            function (str, optional): choose between utility or gradient. Defaults to "gradient".

        Returns:
            str: subscript
        """
        indices = "".join([chr(ord("a") + i) for i in self.agents])
        indices_opp = remove_index(indices, agent)
        return f"{','.join(indices_opp)},{indices}->{indices[agent]}"

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


def remove_index(l: list, i: int):
    """remove i-th entry from list"""
    return l[:i] + l[i + 1 :]
