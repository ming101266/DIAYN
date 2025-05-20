import torch
import torch.nn as nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, state_dim, num_skills, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + num_skills
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        return self.net(x)
