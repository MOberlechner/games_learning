from typing import List, Tuple

import numpy as np

from games_learning.learner.learner import Learner
from games_learning.strategy import Strategy


class MirrorAscent(Learner):
    def __init__(self, eta: float, beta: float, mirror_map: str = "entropic") -> None:
        """Mirror Ascent

        Args:
            eta (float): (initial) step size
            beta (float): decay of step size
        """
        self.name = f"mirror_ascent({mirror_map})"
        self.eta = eta
        self.beta = beta
        self.mirror_map = mirror_map

    def __repr__(self) -> str:
        return f"MirrorAscent({self.mirror_map},eta={self.eta},beta={self.beta})"

    def update(
        self,
        strategy: Strategy,
        gradients: List[np.ndarray],
        iter: int,
    ) -> List[np.ndarray]:
        """update strategies

        Args:
            strategies (Tuple): agents' current strategies
            gradients (List[np.ndarray]): agents' gradients
            iter (int): current iteration

        Returns:
            List[np.ndarray]: updated strategies
        """

        if self.mirror_map == "euclidean":
            return [
                self.update_step_euclidean(strategy.x[i], gradients[i], iter)
                for i in strategy.agents
            ]
        elif self.mirror_map == "entropic":
            return [
                self.update_step_entropic(strategy.x[i], gradients[i], iter)
                for i in strategy.agents
            ]
        else:
            raise ValueError(f"mirror map {self.mirror_map} not available")

    def update_step_entropic(
        self, strategy: np.ndarray, gradient: np.ndarray, iter: int
    ) -> np.ndarray:
        """exponentiated gradient ascent

        Args:
            strategy (np.ndarray): current strategy
            gradient (np.ndarray): gradient
            iter (int): current iteration
        """
        eta_t = self.eta / (iter + 1) ** self.beta
        x_expg = strategy * np.exp(eta_t * gradient)
        return x_expg / x_expg.sum()

    def update_step_euclidean(
        self, strategy: np.ndarray, gradient: np.ndarray, iter: int
    ) -> np.ndarray:
        """projected gradient ascent with projection onto simplex

        Args:
            strategy (np.ndarray): current strategy
            gradient (np.ndarray): gradient
            iter (int): current iteration
        """
        eta_t = self.eta / (iter + 1) ** self.beta
        return self.projection_euclidean(strategy + eta_t * gradient)

    def projection_euclidean(self, x: np.ndarray) -> np.ndarray:
        """projection of some point x onto probability simplex w.r.t. euclidean metric"""
        raise NotImplementedError
