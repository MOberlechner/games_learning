from typing import List

import numpy as np

from src.game.matrix_game import MatrixGame


class Learner:
    """General Learner Class

    We focus on gradient-based learning algorithms. This is why we include gradients as input to avoid recomputation for metrics etc.

    """

    def __init__(self) -> None:
        self.name = "general_learner"

    def update(
        self,
        strategies: Tuple[np.ndarray],
        gradients: Tuple[np.ndarray],
        iter: int,
    ) -> Tuple[np.ndarray]:
        """Update strategies

        Args:
            strategies (List[np.ndarray]): list of agents' strategies
            gradients (List[np.ndarray]): list of agents' gradients
            iter (int): current iteration
            game (Game): underlying game

        Returns:
            Tuple[np.ndarray]: list of updated strategies
        """
        raise NotImplementedError


class MirrorAscent(Learner):
    def __init__(self, eta: float, beta: float, mirror_map: str = "entropic") -> None:
        """Mirror Ascent

        Args:
            eta (float): (initial) step size
            beta (float): decay of step size
        """
        self.name = f"mirr_ascent({mirror_map})"
        self.eta = eta
        self.beta = beta
        self.mirror_map = mirror_map

    def update(
        self,
        strategies: Tuple[np.ndarray],
        gradients: Tuple[np.ndarray],
        iter: int,
    ) -> Tuple[np.ndarray]:
        """update strategies

        Args:
            strategies (Tuple): agents' current strategies
            gradients (List[np.ndarray]): agents' gradients
            iter (int): current iteration

        Returns:
            Tuple[np.ndarray]: updated strategies
        """
        gradients = tuple(game.gradient(strategies, i) for i in game.agents)
        if self.mirror_map == "euclidean":
            return [
                self.update_step_euclidean(strategies[i], gradients[i], iter)
                for i in game.agents
            ]
        elif self.mirror_map == "entropic":
            return [
                self.update_step_entropic(strategies[i], gradients[i], iter)
                for i in game.agents
            ]
        else:
            raise ValueError(f"mirror map {self.mirror_map} not available")

    def update_step_entropic(
        self, strategy: np.ndarray, gradient: np.ndarray, iter: int
    ):
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
    ):
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
