import random

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(
        self,
        action_size: int,
        seed: int = 514,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.20,
    ) -> None:
        self.action_size = action_size
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        np.random.seed(seed)
        self.x = None
        self.noise_scale = 1.0
        self.noise_decay = 0.9995

    def step(self) -> None:
        self.noise_scale *= self.noise_decay

    def reset(self) -> None:
        self.x = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        assert self.x is not None, f"`OUNoise` should be reset first."
        self.x += self.theta * (self.mu - self.x) + self.sigma * np.random.randn(
            self.action_size
        )
        # print("noise", self.noise_scale, self.noise_scale * self.x)
        return self.noise_scale * self.x
