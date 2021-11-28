import random
from collections import deque
from typing import Tuple, List, NamedTuple, Any

import nptyping as npt
import numpy as np
import torch


class Experience(NamedTuple):
    agent_states: npt.NDArray[npt.NDArray[float]]
    agent_actions: npt.NDArray[npt.NDArray[float]]
    agent_rewards: npt.NDArray[float]
    agent_next_states: npt.NDArray[npt.NDArray[float]]
    agent_dones: npt.NDArray[bool]


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        device: torch.device,
        seed: int = 514,
    ) -> None:
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def _convert_batch_agent_features(
        self,
        batch_agent_features: List[npt.NDArray[Any]],
        dtype: type,
    ) -> List[torch.Tensor]:
        return [
            torch.from_numpy(np.vstack(batch_features).astype(dtype))
            .float()
            .to(self.device)
            for batch_features in zip(*batch_agent_features)
        ]

    def add(
        self,
        agent_states: npt.NDArray[npt.NDArray[float]],
        agent_actions: npt.NDArray[npt.NDArray[float]],
        agent_rewards: npt.NDArray[float],
        agent_next_states: npt.NDArray[npt.NDArray[float]],
        agent_dones: npt.NDArray[bool],
    ) -> None:
        self.memory.append(
            Experience(
                agent_states,
                agent_actions,
                agent_rewards,
                agent_next_states,
                agent_dones,
            )
        )

    def sample(
        self,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        experiences: List[Experience] = random.sample(
            self.memory,
            k=self.batch_size,
        )

        agent_states = self._convert_batch_agent_features(
            [e.agent_states for e in experiences],
            np.float,
        )
        assert agent_states[0].shape == agent_states[1].shape == (self.batch_size, 24)
        agent_actions = self._convert_batch_agent_features(
            [e.agent_actions for e in experiences],
            np.float,
        )
        assert agent_actions[0].shape == agent_actions[1].shape == (self.batch_size, 2)
        agent_rewards = self._convert_batch_agent_features(
            [e.agent_rewards for e in experiences],
            np.float,
        )
        assert agent_rewards[0].shape == agent_rewards[1].shape == (self.batch_size, 1)
        agent_next_states = self._convert_batch_agent_features(
            [e.agent_next_states for e in experiences],
            np.float,
        )
        assert (
            agent_next_states[0].shape
            == agent_next_states[1].shape
            == (self.batch_size, 24)
        )
        agent_dones = self._convert_batch_agent_features(
            [e.agent_dones for e in experiences],
            np.uint8,
        )
        assert agent_dones[0].shape == agent_dones[1].shape == (self.batch_size, 1)

        return (
            agent_states,
            agent_actions,
            agent_rewards,
            agent_next_states,
            agent_dones,
        )

    def size(self) -> int:
        return len(self.memory)
