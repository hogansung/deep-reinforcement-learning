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
        hidden_layer_a_dim: int = 1024,
        hidden_layer_b_dim: int = None,
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
        assert (4, 28, 28) == state_size, "Unexpected state_size dimensions"

        self.seed = torch.manual_seed(seed)

        # Input shape is (4, 28, 28), i.e. 4 stacks of images
        # Output shape is (n_classes)

        # # Output shape is (W-F+2P)/S+1: ((4-1+2*0)/1+1, (84-3+2*0)/3+1, (84-3+2*0)/3+1) = (4, 28, 28)
        # Output shape is (W-F+2P)/S+1: ((4-1+2*0)/1+1, (28-3+2*1)/3+1, (84-3+2*1)/3+1) = (4, 10, 10)
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=128,
            kernel_size=(1, 3, 3),
            stride=(1, 3, 3),
            # padding=(0, 0, 0),
            padding=(0, 1, 1),
        )
        self.bn1 = nn.BatchNorm3d(num_features=128)

        # # Output shape is (W-F+2P)/S+1: ((4-1+2*0)/1+1, (28-3+2*1)/3+1, (84-3+2*1)/3+1) = (4, 10, 10)
        # Output shape is (W-F+2P)/S+1: ((4-4+2*0)/1+1, (10-3+2*1)/3+1, (10-3+2*1)/3+1) = (1, 4, 4)
        self.conv2 = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=(4, 3, 3),
            stride=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.bn2 = nn.BatchNorm3d(num_features=256)

        # # Output shape is (W-F+2P)/S+1: ((4-4+2*0)/1+1, (10-3+2*1)/3+1, (10-3+2*1)/3+1) = (1, 4, 4)
        # self.conv3 = nn.Conv3d(
        #     in_channels=256,
        #     out_channels=256,
        #     kernel_size=(4, 3, 3),
        #     stride=(1, 3, 3),
        #     padding=(0, 1, 1),
        # )
        # self.bn3 = nn.BatchNorm3d(num_features=256)

        self.fc1 = nn.Linear(256 * 1 * 4 * 4, hidden_layer_a_dim)
        self.fc2 = nn.Linear(hidden_layer_a_dim, action_size)

    def forward(self, state: np.ndarray, batch_size: int = 1):
        """Build a network that maps state -> action values."""
        x = state
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
