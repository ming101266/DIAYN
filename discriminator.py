import torch
import torch.nn as nn
from utils import init_weight

class Discriminator(nn.Module):
    def __init__(self, state_dim, num_skills, initializer="xavier uniform", hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_skills),
        )
        
        # Initialize weights for each Linear layer
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init_weight(layer, initializer)

    def forward(self, state):
        return self.net(state)
