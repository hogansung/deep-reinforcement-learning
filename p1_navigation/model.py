from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(
        self,
        state_size: Union[int, Tuple[int]],
        action_size: int,
        seed: int,
        hidden_layer_a_dim: int = 64,
        hidden_layer_b_dim: int = 64,
    ):
        super().__init__()
        pass

    def forward(self, state: np.ndarray, batch_size: int = 1):
        pass


class QNetworkWithoutPixels(QNetwork):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: Union[int, Tuple[int]],
        action_size: int,
        seed: int,
        hidden_layer_a_dim: int = 64,
        hidden_layer_b_dim: int = 64,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__(
            state_size, action_size, seed, hidden_layer_a_dim, hidden_layer_b_dim,
        )
        assert type(state_size) is int, "Unexpected state_size type"

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_layer_a_dim)
        self.fc2 = nn.Linear(hidden_layer_a_dim, hidden_layer_b_dim)
        self.fc3 = nn.Linear(hidden_layer_b_dim, action_size)

    def forward(self, state: np.ndarray, batch_size: int = 1):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetworkWithPixels(QNetwork):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: Union[int, Tuple[int]],
        action_size: int,
        seed: int,
        hidden_layer_a_dim: int = 128,
        hidden_layer_b_dim: int = 64,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__(
            state_size, action_size, seed, hidden_layer_a_dim, hidden_layer_b_dim,
        )
        assert (84, 84) == state_size, "Unexpected state_size dimensions"

        self.seed = torch.manual_seed(seed)

        # Input shape is (84, 84)
        # Output shape is (n_classes)

        # Output shape is W/S: (84/2, 84/2) = (42, 42)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=7,
            stride=2,
            padding=3,  # kernel size // 2, so that the output size is not affected
        )

        # Output shape is W/S: (42/2, 42/2) = (21, 21)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Output shape is W/S: (21/3, 21/3) = (7, 7)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=3,
            padding=1,  # kernel size // 2, so that the output size is not affected
        )

        # Output shape is W/S: (7/7, 7/7) = (1, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=7, padding=1)

        self.fc1 = nn.Linear(32 * 1 * 1, hidden_layer_a_dim)
        self.fc2 = nn.Linear(hidden_layer_a_dim, hidden_layer_b_dim)
        self.fc3 = nn.Linear(hidden_layer_b_dim, action_size)

    def forward(self, state: np.ndarray, batch_size: int = 1):
        """Build a network that maps state -> action values."""
        x = state
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
