import torch
import torch.nn as nn
from utils import init_weight
from torch.distributions import Normal
from env import Point2DEnv

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

class Policy(nn.Module):
    def __init__(self, state_dim, skill_dim, action_dim, 
                 hidden_dim=256, action_scale=1.0, initializer="xavier uniform"):
        super().__init__()
        self.action_scale = action_scale
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        #The same as having a final output layer of size
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init_weight(layer, initializer)
        init_weight(self.mean, initializer)
        init_weight(self.log_std, initializer)

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state, skill):
        mean, log_std = self.forward(state, skill)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        z = normal.rsample()
        action = torch.tanh(z)
        
        # Squashing and log prob
        log_prob = normal.log_prob(z) - torch.log((1 - action.pow(2)) + EPSILON)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action * self.action_scale, log_prob