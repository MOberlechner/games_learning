from typing import List

import numpy as np


class Game:
    def __init__(self, n_agents: int):
        """Normal-Form Game

        G = (I, X, u)
            I set of agents, |I| = n
            X = X_1 x ... x X_n, X_i is d-dim vector space
            u = [u_1, ..., u_n]payoff functions

        Args:
            n_agents (int): number of agents
            name (str, optional): name of game. Defaults to "".
        """
        self.name = "general_game"
        self.n_agents = n_agents
        self.agents = list(range(n_agents))

    def utility(self, strategies: List[np.ndarray], agent: int) -> float:
        """compute agent's utility

        Args:
            strategies (List[np.ndarray]): List of agents' strategies
            agent (int): index of agent

        Returns:
            float: Payoff for agent
        """
        raise NotImplementedError

    def utility_loss(
        self, strategies: List[np.ndarray], agent: int, normed: bool = True
    ) -> float:
        """compute agent's utility loss
        might not be available

        Args:
            strategies (List[np.ndarray]): List of agents' strategies
            agent (int): index of agent
            normed (bool): absolute (false) or relative (true) utility loss

        Returns:
            float: Payoff for agent
        """
        return None

    def gradient(self, strategies: List[np.ndarray], agent: int) -> np.ndarray:
        """Gradient Function

        Args:
            strategies (List[np.ndarray]): List of agents' strategies
            agent (int): index of agent

        Returns:
            np.ndarray: gradient for agent
        """
        raise NotImplementedError
