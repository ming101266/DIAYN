import torch
import matplotlib.pyplot as plt
from env import Point2DEnv
from policy import Policy
from discriminator import Discriminator
from buffer import ReplayBuffer
from train import train
import numpy as np

def main():
    state_dim = 2
    action_dim = 2
    num_skills = 10
    skill_dim = num_skills

    env = Point2DEnv()
    policy = Policy(state_dim, skill_dim, action_dim)
    discriminator = Discriminator(state_dim, num_skills)

    policy_optim = torch.optim.Adam(policy.parameters(), lr=3e-3)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    buffer = ReplayBuffer()

    skill_trajectories = train(
        policy, discriminator, env,
        num_skills, policy_optim, disc_optim,
        buffer, steps=10000
    )

    # Plotting
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

if __name__ == "__main__":
    main()
