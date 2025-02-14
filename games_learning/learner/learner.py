from typing import List

import numpy as np

from games_learning.strategy import Strategy


class Learner:
    """General Learner Class

    We focus on gradient-based learning algorithms.
    This is why we include gradients as input to avoid recomputation for metrics etc.

    """

    def __init__(self) -> None:
        self.name = "general_learner"

    def update(
        self,
        strategy: Strategy,
        gradients: List[np.ndarray],
        iter: int,
    ) -> List[np.ndarray]:
        """Update all strategies simultaneously

        Args:
            strategy (Strategy): agents' strategies
            gradients (List[np.ndarray]): agents' gradients
            iter (int): current iteration

        Returns:
            List[np.ndarray]: updated mixed strategies for all agents
        """
        raise NotImplementedError

    def update_agent(
        self,
        strategy: Strategy,
        gradients: List[np.ndarray],
        iter: int,
        agent: int,
    ) -> np.ndarray:
        """Update one strategy (for sequential algorithms)

        Args:
            strategy (Strategy): agents' strategies
            gradients (List[np.ndarray]): agents' gradients
            iter (int): current iteration
            agent (int): agent we want to update

        Returns:
            np.ndarray: updated mixed strategy for agent
        """
        raise NotImplementedError


class BestResponse(Learner):
    def __init__(self, tie_breaking: str = "lowest") -> None:
        """Best Response Dynamics"""
        self.name = "best_response"
        self.tie_breaking = tie_breaking

    def __repr__(self) -> str:
        return "BestResponse"

    def update(
        self,
        strategy: Strategy,
        gradients: List[np.ndarray],
        iter: int,
    ) -> List[np.ndarray]:
        """update all strategies (simultaneous)

        Args:
            strategies (Tuple): agents' current strategies
            gradients (List[np.ndarray]): agents' gradients
            iter (int): current iteration

        Returns:
            List[np.ndarray]: updated strategies
        """
        return [
            self.update_agent(strategy, i, gradients[i], iter) for i in strategy.agents
        ]

    def update_agent(
        self,
        strategy: Strategy,
        agent: int,
        gradient: np.ndarray,
        iter: int,
    ) -> np.ndarray:
        """update a single strategy (sequential)"""
        return strategy.best_response(
            agent=agent, gradient=gradient, tie_breaking=self.tie_breaking
        )
