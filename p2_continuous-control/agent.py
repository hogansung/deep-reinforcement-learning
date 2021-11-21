import copy
import random
from collections import deque, namedtuple
from typing import List, Tuple

import numpy as np
import torch
from torch import optim, nn

from model import Actor, Critic

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0.0

NUM_OF_LEARNS_PER_UPDATE = 10
NUM_OF_TIMESTAMPS_PER_UPDATE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently is using device: {device}")


class Agent:
    """An agent that interacts with the environment."""

    def __init__(self, state_size: int, action_size: int, seed: int = 514) -> None:
        self.state_size = state_size
        self.agent_size = action_size
        random.seed(seed)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=LR_ACTOR,
        )

        # Critic Network
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=LR_CRITIC,
            weight_decay=WEIGHT_DECAY,
        )

        # Noise Process
        self.noise = OUNoise(action_size, seed)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        time_stamp: int,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Only do learning after NUM_OF_TIMESTAMPS_PER_LEARN
        if time_stamp % NUM_OF_TIMESTAMPS_PER_UPDATE:
            return

        if self.replay_buffer.len() > BATCH_SIZE:
            for _ in range(NUM_OF_LEARNS_PER_UPDATE):
                experiences = self.replay_buffer.sample()
                self.learn(experiences, GAMMA)

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self) -> None:
        self.noise.reset()

    def learn(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        gamma: float,
    ) -> None:
        states, actions, rewards, next_states, dones = experiences

        # Update Critic Local Network
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            predicted_next_q_values = self.critic_target(next_states, next_actions)
            predicted_q_values = rewards + (
                gamma * predicted_next_q_values * (1 - dones)
            )
        expected_q_values = self.critic_local(states, actions)
        criterion = nn.MSELoss()
        critic_loss = criterion(expected_q_values, predicted_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Update Actor Local Network
        predicted_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(
        self,
        local_model: nn.Module,
        target_model: nn.Module,
        tau: float,
    ) -> None:
        for local_parameter, target_parameter in zip(
            local_model.parameters(), target_model.parameters()
        ):
            target_parameter.data.copy_(
                tau * local_parameter.data + (1.0 - tau) * target_parameter.data
            )


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
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def len(self) -> int:
        return len(self.memory)
