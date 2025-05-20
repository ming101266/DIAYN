import torch
import torch.nn as nn
from utils import init_weight

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, skill_dim, hidden_size=256, initializer="xavier uniform"):
        super().__init__()
        input_dim = obs_dim + action_dim + skill_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize weights
        for block in [self.q1, self.q2]:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    init_weight(layer, initializer)

    def forward(self, state, skill, action):
        x = torch.cat([state, skill, action], dim=1)
        return self.q1(x), self.q2(x)