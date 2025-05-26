import torch
import torch.nn.functional as F
from utils import one_hot, to_tensor
import matplotlib.pyplot as plt
from env import Point2DEnv
from policy import Policy
from discriminator import Discriminator
from buffer import ReplayBuffer
from train import train
from critic import QNetwork
import numpy as np
import copy


def freeRoam(policy, env, num_skills, total_steps=10000, max_skill_steps=20):
    skill_trajectories = [[] for _ in range(num_skills)]
    for i in range(total_steps // max_skill_steps):
        # Environment interaction
        skill = np.random.randint(0, num_skills)
        skill_onehot = one_hot(skill, num_skills)
        state = env.reset()

        for t in range(max_skill_steps):
            state_tensor = to_tensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_tensor, _ = policy.sample(state_tensor, skill_onehot)
                action = action_tensor.squeeze(0).cpu().numpy()
                action = np.clip(action, -env.max_step, env.max_step)
            
            next_state = env.step(action)
            skill_trajectories[skill].append(next_state.copy()) 
            state = next_state
    colors = plt.cm.get_cmap('tab10', num_skills)
    plt.figure(figsize=(8, 8))
    for i, traj in enumerate(skill_trajectories):
        traj = np.array(traj)
        plt.scatter(traj[:, 0], traj[:, 1], alpha=0.5, label=f'Skill {i}', color=colors(i))
    plt.legend()
    plt.grid(True)
    plt.title("DIAYN Skills in 2D Navigation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

