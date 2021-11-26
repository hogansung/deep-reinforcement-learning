import random

import nptyping as npt
import numpy as np
import torch
from torch import optim, nn

from model import Actor, Critic
from ounoise import OUNoise

LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0.0


class Agent:
    """An agent that interacts with the environment"""

    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        device: torch.device,
        seed: int = 514,
    ) -> None:
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        random.seed(seed)

        # Actor network: it takes agent-specific state into consideration
        self.local_actor = Actor(state_size, action_size, seed).to(self.device)
        self.target_actor = Actor(state_size, action_size, seed).to(self.device)
        self.local_actor_optimizer = optim.Adam(
            self.local_actor.parameters(),
            lr=LR_ACTOR,
        )

        # Critic network: it takes joint states and actions into consideration, but trained separately by agents
        self.local_critic = Critic(num_agents, state_size, action_size, seed).to(
            self.device
        )
        self.target_critic = Critic(num_agents, state_size, action_size, seed).to(
            self.device
        )
        self.local_critic_optimizer = torch.optim.Adam(
            self.local_critic.parameters(),
            lr=LR_CRITIC,
            weight_decay=WEIGHT_DECAY,
        )

    @staticmethod
    def _soft_update(
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

    def act(
        self,
        states: npt.NDArray[float],
        noise: OUNoise = None,
    ) -> npt.NDArray[float]:
        states = torch.from_numpy(states).float().to(self.device)
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(states).cpu().data.numpy()
        self.local_actor.train()
        if noise:
            action += noise.sample()
        return (
            np.clip(action, -1, +1)
            * 1  # hack: add `* 1` to get rid of static type check warning
        )

    def target_act(self, states: torch.Tensor, noise: OUNoise = None) -> torch.Tensor:
        return (
            torch.from_numpy(
                self.target_actor(states).cpu().data.numpy() + noise.sample()
            )
            .float()
            .to(self.device)
        )

    def soft_update(self, tau: float):
        self._soft_update(self.local_actor, self.target_actor, tau)
        self._soft_update(self.local_critic, self.target_critic, tau)
