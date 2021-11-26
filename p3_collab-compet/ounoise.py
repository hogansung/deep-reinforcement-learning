import copy
import random

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(
        self,
        action_size: int,
        seed: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ) -> None:
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.state = None

    def reset(self) -> None:
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        assert self.state is not None, f"`state` has never been reset"
        x = self.state
        dx = (
            self.theta
            * (self.mu - x)
            * self.sigma
            * np.array([random.random() for _ in range(len(x))])
        )
        self.state = x + dx
        return self.state
