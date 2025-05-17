import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, state_dim, skill_dim, action_dim, action_scale=0.2):
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

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        return self.net(x) * self.action_scale