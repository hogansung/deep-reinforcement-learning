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

        # Hard-update
        self._soft_update(self.local_actor, self.target_actor, 1.0)
        self._soft_update(self.local_critic, self.target_critic, 1.0)

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
            actor_model = self.local_actor
        elif actor_name == "target":
            actor_model = self.target_actor
            assert (
                noise is None
            ), "There should be no noise in the target model inference."
        else:
            assert False, "Invalid actor name."

        # print("WOWOWOWO", states.shape)
        assert (
            len(states.shape) == 2 and states.shape[1] == self.state_size
        ), "Incorrect `states` shape for actor model"

        # print("www", list(actor_model.hidden_layers[0].parameters()))
        actor_model.eval()
        with torch.no_grad():
            actions = actor_model(states).cpu().data.numpy()
            # print("wow", states, actions)
        actor_model.train()
        # print("actions", actions)
        if noise:
            actions += noise.sample()
        # print("actions", actions.shape)
        return (
            torch.from_numpy(np.clip(actions, -1, +1)).float().detach().to(self.device)
        )

    def soft_update(self, tau: float):
        self._soft_update(self.local_actor, self.target_actor, tau)
        self._soft_update(self.local_critic, self.target_critic, tau)
