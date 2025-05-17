import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, skill_dim, hidden_size=256):
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

    # predict q-value for specific action
    def forward(self, state, action, skill):
        x = torch.cat([state, action, skill], dim=1)
        return self.q1(x), self.q2(x)
