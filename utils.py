import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(skill_indices, num_skills):
    if isinstance(skill_indices, int):
        # Handle single scalar skill: create a 1D tensor, then one-hot it
        skill_indices_tensor = torch.tensor([skill_indices], dtype=torch.long)
        one_hot_vec = F.one_hot(skill_indices_tensor, num_classes=num_skills).float()
        return one_hot_vec # Shape will be (1, num_skills)
    elif isinstance(skill_indices, torch.Tensor):
        # Handle batch of skill indices (expected to be 1D: (batch_size,))
        if skill_indices.ndim == 0:
             skill_indices = skill_indices.unsqueeze(0)

        if skill_indices.ndim != 1:
            raise ValueError(f"Expected 1D tensor for skill_indices batch, but got {skill_indices.ndim} dimensions.")

        skill_indices = skill_indices.long()
        one_hot_batch = F.one_hot(skill_indices, num_classes=num_skills).float()
        return one_hot_batch
    else:
        raise TypeError("skill_indices must be an int or a torch.Tensor")

def to_tensor(np_array):
    return torch.tensor(np_array, dtype=torch.float32)

def init_weight(layer, initializer="xavier uniform"):
    if isinstance(layer, nn.Linear):
        if initializer == "xavier uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif initializer == "xavier normal":
            nn.init.xavier_normal_(layer.weight)
        elif initializer == "kaiming uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu') # Added nonlinearity for Kaiming
        elif initializer == "kaiming normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu') # Added nonlinearity for Kaiming
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
