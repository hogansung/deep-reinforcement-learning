import torch
from torch import nn


class Actor(nn.Module):
    """Actor (Policy) Model, which maps states to actions."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 514,
        num_fc1_units: int = 400,
        num_fc2_units: int = 300,
    ) -> None:
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(state_size, num_fc1_units)
        self.fc2 = nn.Linear(num_fc1_units, num_fc2_units)
        self.fc3 = nn.Linear(num_fc2_units, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model, which maps (state, action) to Q-values."""

    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        seed: int = 514,
        num_fc1_units: int = 400,
        num_fc2_units: int = 300,
    ):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_agents * state_size, num_fc1_units)
        self.fc2 = nn.Linear(num_fc1_units + num_agents * action_size, num_fc2_units)
        self.fc3 = nn.Linear(num_fc2_units, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
