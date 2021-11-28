from typing import Tuple, List

import nptyping as npt
import numpy as np
import torch

from agent import Agent
from ounoise import OUNoise
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 1e-3

NUM_OF_LEARNS_PER_UPDATE = 1
NUM_OF_TIMESTAMPS_PER_UPDATE = 1

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
        self.state_size = state_size
        self.action_size = action_size
        self.agents = [
            Agent(num_agents, state_size, action_size, device, seed)
            for _ in range(num_agents)
        ]

        # Noise Process
        self.noise = OUNoise(action_size, seed)
        self.noise_scale = 1.0
        self.noise_decay = 0.9995

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            action_size,
            BUFFER_SIZE,
            BATCH_SIZE,
            device,
            seed,
        )

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
        # print("experiences", experiences)
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
            agent_next_actions = [
                agent.act(states=next_states, actor_name="target", noise=None)
                for agent, next_states in zip(self.agents, agent_next_states)
            ]
            assert agent_next_states[0].shape == (BATCH_SIZE, self.state_size)
            concatenated_agent_next_states = torch.cat(agent_next_states, dim=1)
            assert concatenated_agent_next_states.shape == (
                BATCH_SIZE,
                self.num_agents * self.state_size,
            )
            # print("learn - next states", concatenated_agent_next_states)
            assert agent_next_actions[0].shape == (BATCH_SIZE, self.action_size)
            concatenated_agent_next_actions = torch.cat(agent_next_actions, dim=1)
            assert concatenated_agent_next_actions.shape == (
                BATCH_SIZE,
                self.num_agents * self.action_size,
            )
            # print("learn - next actions", concatenated_agent_next_actions)
            predicted_next_q_values = agent.target_critic(
                concatenated_agent_next_states,
                concatenated_agent_next_actions,
            )
            # print("learn - target q next", predicted_next_q_values)
            predicted_q_values = agent_rewards[agent_idx] + (
                GAMMA * predicted_next_q_values * (1 - agent_dones[agent_idx])
            )
            # print("learn - target q", predicted_q_values)
        assert agent_states[0].shape == (BATCH_SIZE, self.state_size)
        concatenated_agent_states = torch.cat(agent_states, dim=1)
        assert concatenated_agent_states.shape == (
            BATCH_SIZE,
            self.num_agents * self.state_size,
        )
        assert agent_actions[0].shape == (BATCH_SIZE, self.action_size)
        concatenated_agent_actions = torch.cat(agent_actions, dim=1)
        assert concatenated_agent_actions.shape == (
            BATCH_SIZE,
            self.num_agents * self.action_size,
        )
        expected_q_values = agent.local_critic(
            concatenated_agent_states,
            concatenated_agent_actions,
        )
        # print("learn - local q", expected_q_values)
        criterion = torch.nn.SmoothL1Loss()
        # criterion = nn.MSELoss()
        local_critic_loss = criterion(expected_q_values, predicted_q_values)
        # print("learn - local critic loss", local_critic_loss)
        # print(
        #     "prior critic",
        #     agent.local_critic.hidden_layers[0].weight.cpu().data.numpy()[0],
        # )
        agent.local_critic_optimizer.zero_grad()
        local_critic_loss.backward()
        agent.local_critic_optimizer.step()
        # print(
        #     "after critic",
        #     agent.local_critic.hidden_layers[0].weight.cpu().data.numpy()[0],
        # )

        # Update local actor network
        predicted_agent_actions = [
            self.agents[states_idx].local_actor(states)
            if agent_idx == states_idx
            else self.agents[states_idx].act(states=states, actor_name="local")
            for states_idx, states in enumerate(agent_states)
        ]
        assert predicted_agent_actions[0].shape == (BATCH_SIZE, self.action_size)
        concatenated_predicted_agent_actions = torch.cat(predicted_agent_actions, dim=1)
        assert concatenated_predicted_agent_actions.shape == (
            BATCH_SIZE,
            self.num_agents * self.action_size,
        )
        # print("learn - states", concatenated_agent_states)
        # print("learn - predicted actions", concatenated_predicted_agent_actions)
        local_actor_loss = -agent.local_critic(
            concatenated_agent_states,
            concatenated_predicted_agent_actions,
        ).mean()
        # print(
        #     "learn - local actor loss",
        #     -agent.local_critic(
        #         concatenated_agent_states,
        #         concatenated_predicted_agent_actions,
        #     ),
        # )
        # prior = list(agent.local_actor.hidden_layers.parameters())
        # print(
        #     "prior actor",
        #     agent.local_actor.hidden_layers[0].weight.cpu().data.numpy()[:3],
        # )
        agent.local_actor_optimizer.zero_grad()
        local_actor_loss.backward()
        agent.local_actor_optimizer.step()
        # after = list(agent.local_actor.hidden_layers.parameters())
        # assert prior == after
        # print(
        #     "after actor",
        #     agent.local_actor.hidden_layers[0].weight.cpu().data.numpy()[:3],
        # )

        agent.soft_update(TAU)

    def act(self, agent_states: npt.NDArray[npt.NDArray[float]]) -> np.ndarray:
        assert (
            agent_states.shape[0] == self.num_agents
        ), f"Unexpected `agent_states` dimension: {agent_states.shape[0]}."

        actions = np.vstack(
            [
                agent.act(
                    states=torch.from_numpy(np.array([states])).float().to(device),
                    actor_name="local",
                    noise=self.noise,
                )
                .cpu()
                .data.numpy()
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

        # Update target actor and critic networks
        # for agent in self.agents:
        #     agent.soft_update(TAU)
