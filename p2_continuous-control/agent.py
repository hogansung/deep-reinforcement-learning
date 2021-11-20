import copy
import random
from collections import deque, namedtuple
from typing import List, Tuple

import numpy as np
import torch

from model import Actor, Critic

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently is using device {device}")


class Agent:
    """An agent that interacts with the environment."""

    def __init__(self, state_size: int, action_size: int, seed: int = 514) -> None:
        self.state_size = state_size
        self.agent_size = action_size
        random.seed(seed)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, seed)
        self.actor_target = Actor(state_size, action_size, seed)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR
        )

        # Critic Network
        self.critic_local = Critic(state_size, action_size, seed)
        self.critic_target = Critic(state_size, action_size, seed)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC
        )

        # Noise Process
        self.noise = OUNoise(action_size, seed)

        # Replay Buffer
        self.relay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)


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
        assert self.state, f"`state` has never been reset"
        x = self.state
        dx = (
            self.theta
            * (self.mu - x)
            * self.sigma
            * np.array([random.random() for _ in range(len(x))])
        )
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, action_size: int, buffer_size: int, batch_size: int, seed: int = 514
    ) -> None:
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.Experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = seed

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.memory.append(self.Experience(state, action, reward, next_state, done))

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences: List[
            namedtuple(
                "Experience", ["state", "action", "reward", "next_state", "done"]
            )
        ] = random.sample(
            self.memory,
            k=self.batch_size,
        )

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]))
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def len(self) -> int:
        return len(self.memory)
