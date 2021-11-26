from typing import Tuple, List

import nptyping as npt
import numpy as np
import torch
from torch import nn

from agent import Agent
from ounoise import OUNoise
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3

NUM_OF_LEARNS_PER_UPDATE = 10
NUM_OF_TIMESTAMPS_PER_UPDATE = 20

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device} is currently used.")


class AgentManager:
    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        seed: int = 514,
    ) -> None:
        self.num_agents = num_agents
        self.agents = [
            Agent(num_agents, state_size, action_size, device, seed)
            for _ in range(num_agents)
        ]

        # Noise Process
        self.noise = OUNoise(action_size, seed)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            action_size,
            BUFFER_SIZE,
            BATCH_SIZE,
            device,
            seed,
        )

    def _target_act(
        self,
        agent_states: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Take a list of state tensors as input and return a joined action tensor"""
        assert (
            len(agent_states) == self.num_agents
        ), f"Unexpected `agent_states` dimension: {len(agent_states)}."

        return [
            agent.target_act(states, self.noise)
            for agent, states in zip(self.agents, agent_states)
        ]

    def _learn(
        self,
        agent_idx: int,
        experiences: Tuple[
            List[torch.Tensor],
            List[torch.Tensor],
            List[torch.Tensor],
            List[torch.Tensor],
            List[torch.Tensor],
        ],
    ):
        agent = self.agents[agent_idx]
        (
            agent_states,
            agent_actions,
            agent_rewards,
            agent_next_states,
            agent_dones,
        ) = experiences

        # Update local critic network
        with torch.no_grad():
            agent_next_actions = self._target_act(agent_next_states)
            concatenated_agent_next_states = torch.cat(agent_next_states, dim=1)
            concatenated_agent_next_actions = torch.cat(agent_next_actions, dim=1)
            predicted_next_q_values = agent.target_critic(
                concatenated_agent_next_states,
                concatenated_agent_next_actions,
            )
            predicted_q_values = agent_rewards[agent_idx] + (
                GAMMA * predicted_next_q_values + (1 - agent_dones[agent_idx])
            )
        concatenated_agent_states = torch.cat(agent_states, dim=1)
        concatenated_agent_actions = torch.cat(agent_actions, dim=1)
        expected_q_values = agent.local_critic(
            concatenated_agent_states,
            concatenated_agent_actions,
        )
        criterion = nn.MSELoss()
        local_critic_loss = criterion(expected_q_values, predicted_q_values)
        agent.local_critic_optimizer.zero_grad()
        local_critic_loss.backward()
        agent.local_critic_optimizer.step()

        # Update local actor network
        predicted_agent_actions = [
            self.agents[states_idx].local_actor(states)
            if states_idx == agent_idx
            else self.agents[states_idx].local_actor(states).detach()
            for states_idx, states in enumerate(agent_states)
        ]
        concatenated_predicted_agent_actions = torch.cat(predicted_agent_actions, dim=1)
        local_actor_loss = -agent.local_critic(
            concatenated_agent_states,
            concatenated_predicted_agent_actions,
        ).mean()
        agent.local_actor_optimizer.zero_grad()
        local_actor_loss.backward()
        agent.local_actor_optimizer.step()

        # Update target actor and critic networks
        agent.soft_update(TAU)

    def act(self, agent_states: npt.NDArray[npt.NDArray[float]]) -> np.ndarray:
        assert (
            len(agent_states) == self.num_agents
        ), f"Unexpected `agent_states` dimension: {len(agent_states)}."

        actions = np.array(
            [
                agent.act(states, self.noise)
                for agent, states in zip(self.agents, agent_states)
            ]
        )
        return actions

    def step(
        self,
        agent_states: npt.NDArray[npt.NDArray[float]],
        agent_actions: npt.NDArray[npt.NDArray[float]],
        agent_rewards: npt.NDArray[float],
        agent_next_states: npt.NDArray[npt.NDArray[float]],
        agent_dones: npt.NDArray[bool],
        timestamp,
    ):
        self.replay_buffer.add(
            agent_states, agent_actions, agent_rewards, agent_next_states, agent_dones
        )

        if (
            timestamp % NUM_OF_TIMESTAMPS_PER_UPDATE
            or self.replay_buffer.size() < BATCH_SIZE
        ):
            return

        for agent_idx in range(self.num_agents):
            for _ in range(NUM_OF_LEARNS_PER_UPDATE):
                experiences = self.replay_buffer.sample()
                self._learn(agent_idx, experiences)

    def reset(self):
        self.noise.reset()
