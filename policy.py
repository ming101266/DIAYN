import torch
import torch.nn as nn
from utils import init_weight

class Policy(nn.Module):
    def __init__(self, state_dim, skill_dim, action_dim, initializer="xavier uniform", action_scale=0.2):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Initialize weights for each Linear layer
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init_weight(layer, initializer)

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        return self.net(x) * self.action_scale