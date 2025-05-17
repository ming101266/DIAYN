import torch
import torch.nn as nn

def one_hot(index, depth):
    vec = torch.zeros(depth)
    vec[index] = 1.0
    return vec

def to_tensor(np_array):
    return torch.tensor(np_array, dtype=torch.float32)

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
