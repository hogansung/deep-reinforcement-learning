import random

import numpy as np
import torch
from torch import optim, nn

from model import Actor, Critic
from ounoise import OUNoise

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
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
        self.actor_local = Actor(state_size, action_size, seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, seed).to(self.device)
        self.local_actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=LR_ACTOR,
        )

        # Critic network: it takes joint states and actions into consideration, but trained separately by agents
        self.critic_local = Critic(num_agents, state_size, action_size, seed).to(
            self.device
        )
        self.critic_target = Critic(num_agents, state_size, action_size, seed).to(
            self.device
        )
        self.local_critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(),
            lr=LR_CRITIC,
            weight_decay=WEIGHT_DECAY,
        )

        # Hard-update
        self._soft_update(self.actor_local, self.actor_target, 1.0)
        self._soft_update(self.critic_local, self.critic_target, 1.0)

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
        states: torch.Tensor,
        actor_name: str,
        noise: OUNoise = None,
    ) -> torch.Tensor:
        if actor_name == "local":
            actor_model = self.actor_local
        elif actor_name == "target":
            actor_model = self.actor_target
            assert (
                noise is None
            ), "There should be no noise in the target model inference."
        else:
            assert False, "Invalid actor name."

        actor_model.eval()
        with torch.no_grad():
            actions = actor_model(states).cpu().data.numpy()
        actor_model.train()
        if noise:
            actions += noise.sample()
        return (
            torch.from_numpy(np.clip(actions, -1, +1)).float().detach().to(self.device)
        )

    def soft_update(self, tau: float):
        self._soft_update(self.actor_local, self.actor_target, tau)
        self._soft_update(self.critic_local, self.critic_target, tau)
