from typing import List

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    """Actor (Policy) Model, which maps states to actions."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 514,
        hidden_sizes: List[int] = (400, 300),
    ) -> None:
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(state_size, hidden_sizes[0])]
            + [
                nn.Linear(hidden_size_input, hidden_size_output)
                for hidden_size_input, hidden_size_output in zip(
                    hidden_sizes[:-1],
                    hidden_sizes[1:],
                )
            ]
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for hidden_layer in self.hidden_layers:
            f = hidden_layer.weight.data.size()[0]
            hidden_layer.weight.data.uniform_(-1.0 / np.sqrt(f), 1.0 / np.sqrt(f))
            hidden_layer.bias.data.fill_(0.1)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # print(list(self.hidden_layers[0].parameters())[:20])
        x = state
        # print("zz", x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            # print("xdd", x[:10])
            x = self.relu(x)
            # print("xdd", hidden_layer(x)[0, :10])
            # print("xd", x[:10])
        return self.tanh(self.output_layer(x))


class Critic(nn.Module):
    """Critic (Value) Model, which maps (state, action) to Q-values."""

    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        seed: int = 514,
        hidden_sizes: List[int] = (400, 300),
    ):
        super().__init__()
        assert (
            len(hidden_sizes) >= 2
        ), "Critic network needs at least two hidden layers."
        self.seed = torch.manual_seed(seed)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(num_agents * state_size, hidden_sizes[0]),
                nn.Linear(hidden_sizes[0] + num_agents * action_size, hidden_sizes[1]),
            ]
            + [
                nn.Linear(hidden_size_input, hidden_size_output)
                for hidden_size_input, hidden_size_output in zip(
                    hidden_sizes[1:-1],
                    hidden_sizes[2:],
                )
            ]
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for hidden_layer in self.hidden_layers:
            f = hidden_layer.weight.data.size()[0]
            hidden_layer.weight.data.uniform_(-1.0 / np.sqrt(f), 1.0 / np.sqrt(f))
            hidden_layer.bias.data.fill_(0.1)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.hidden_layers[0](state))
        x = self.relu(self.hidden_layers[1](torch.cat((x, action), dim=1)))
        return self.output_layer(x)
