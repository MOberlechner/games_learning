from collections import deque
from typing import List

import numpy as np
from tqdm import tqdm

from src.game.game import Game
from src.learner.learner import Learner


class Simulator:
    def __init__(
        self, game: Game, learner: Learner, max_iter: int, tol: float = 0
    ) -> None:
        """simulate learning

        Args:
            game (Game): underlying game
            learner (Learner): learning method
            max_iter (int): maximal number of iterations
        """
        self.game = game
        self.learner = learner
        self.max_iter = max_iter
        self.tol = tol
        self.agents = self.game.agents

        self.log_data = self.init_log_data()
        self.bool_convergence = None
        self.number_iter = max_iter

    def run(self, init_strategies: List[np.ndarray], show_bar: bool = True) -> None:
        """run learning dynamics

        Args:
            init_strategies (List[np.ndarray]): initial strategies
            show_bar (bool): show progress bar
        """
        self.bool_convergence = False
        strategies = init_strategies

        for t in tqdm(range(self.max_iter), disable=~show_bar):

            # compute gradients
            gradients = [self.game.gradient(strategies, i) for i in self.agents]

            # log
            self.log(strategies, gradients)

            # check convergence
            if self.check_convergence():
                self.bool_convergence = True
                self.number_iter = t + 1
                break

            # update strategies
            strategies = self.learner.update(strategies, gradients, t, self.game)

    def init_log_data(self):
        return {
            "utility": [deque(maxlen=self.max_iter) for i in self.agents],
            "utility_loss": [deque(maxlen=self.max_iter) for i in self.agents],
            "strategies": [deque(maxlen=self.max_iter) for i in self.agents],
            "gradients": [deque(maxlen=self.max_iter) for i in self.agents],
        }

    def log(self, strategies: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """log some interesting stuff"""
        for i in self.agents:
            self.log_data["utility"][i].append(self.game.utility(strategies, i))
            self.log_data["utility_loss"][i].append(
                self.game.utility_loss(strategies, i, normed=True)
            )
            self.log_data["strategies"][i].append(strategies[i])
            self.log_data["gradients"][i].append(gradients[i])

    def check_convergence(self) -> bool:
        """Check if utility loss is sufficently small"""
        return np.all(
            [self.log_data["utility_loss"][i][-1] < self.tol for i in self.agents]
        )
