from collections import deque
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from games_learning.learner.learner import Learner
from games_learning.strategy import Strategy


class Simulator:
    def __init__(
        self, strategy: Strategy, learner: Learner, max_iter: int, tol: float = 0
    ) -> None:
        """simulate learning

        Args:
            strategy (Strategy): mixed strategy
            learner (Learner): learning method
            max_iter (int): maximal number of iterations
            tol (float): stopping criteration for learner
        """
        self.strategy = strategy
        self.learner = learner
        self.max_iter = max_iter
        self.tol = tol
        self.agents = self.strategy.agents

        self.log_data = self.init_log_data()
        self.bool_convergence = None
        self.number_iter = max_iter

    def run(self, show_bar: bool = True, simultaneous: bool = True) -> None:
        """run (simultaneous) learning dynamics

        Args:
            show_bar (bool): show progress bar. Defaluts to True.
            simultaneous (bool): update are agents simultaneously (True) or sequentially (False). Defaults to True
        """
        self.bool_convergence = False

        for t in tqdm(range(self.max_iter), disable=~show_bar):

            # compute gradients
            gradients = self.strategy.y

            # log
            self.log_iteration(gradients)

            # check convergence
            if self.check_convergence():
                self.bool_convergence = True
                self.number_iter = t + 1
                break

            if simultaneous:
                # update strategies simultaneously
                x_new = self.learner.update(self.strategy, gradients, t)
                self.strategy.x = x_new

            else:
                # update strategies sequentially
                for i in self.agents:
                    gradient = self.strategy.gradient(agent=i)
                    xi_new = self.learner.update_agent(self.strategy, i, gradient, t)
                    self.strategy.x[i] = xi_new

        return self.log_result()

    def check_convergence(self) -> bool:
        """Check if utility loss is sufficently small"""
        return np.all(
            [self.log_data["utility_loss"][i][-1] < self.tol for i in self.agents]
        )

    def init_log_data(self):
        return {
            "utility": [deque(maxlen=self.max_iter) for i in self.agents],
            "utility_loss": [deque(maxlen=self.max_iter) for i in self.agents],
            "strategies": [deque(maxlen=self.max_iter) for i in self.agents],
            "gradients": [deque(maxlen=self.max_iter) for i in self.agents],
        }

    def log_iteration(self, gradients: Tuple[np.ndarray]) -> None:
        """log some interesting stuff"""
        for i in self.agents:
            self.log_data["utility"][i].append(
                self.strategy.utility(agent=i, gradient=gradients[i])
            )
            self.log_data["utility_loss"][i].append(
                self.strategy.utility_loss(agent=i, gradient=gradients[i], method="rel")
            )
            self.log_data["strategies"][i].append(self.strategy.x[i])
            self.log_data["gradients"][i].append(gradients[i])

    def log_result(self) -> dict:
        """log final result"""
        return {
            "game": self.strategy.game.name,
            "learner": self.learner.name,
            "convergence": self.bool_convergence,
            "iterations": self.number_iter,
            "utility_loss": np.max(
                [self.log_data["utility_loss"][i][-1] for i in self.agents]
            ),
            "min_utility_loss": np.min(
                [
                    np.max([self.log_data["utility_loss"][i][t] for i in self.agents])
                    for t in range(self.number_iter)
                ]
            ),
        }
