import torch
import matplotlib.pyplot as plt
from env import Point2DEnv
from policy import Policy
from discriminator import Discriminator
from buffer import ReplayBuffer
from train import train
from critic import QNetwork
import numpy as np
import copy
import pickle
from modelFreeRoam import freeRoam


def main():
    # Hyperparameters
    state_dim = 2
    action_dim = 2
    num_skills = 10
    skill_dim = num_skills
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    lr_policy = 1e-4
    lr_disc = 1e-4
    lr_critic = 1e-4
    total_steps = 5000
    target_entropy = -action_dim

    env = Point2DEnv()
    
    policy = Policy(state_dim, skill_dim, action_dim)
    discriminator = Discriminator(state_dim, num_skills)
    critic = QNetwork(state_dim, action_dim, skill_dim)

    critic_target = copy.deepcopy(critic)


    policy_optim = torch.optim.Adam(policy.parameters(), lr_policy)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr_disc)
    critic_optim = torch.optim.Adam(critic.parameters(), lr_critic)



    buffer = ReplayBuffer()

    skill_trajectories = train(
        policy, discriminator, critic,
        critic_target,
        env, num_skills, 
        policy_optim, disc_optim, critic_optim, 
        gamma, tau, alpha, target_entropy,
        buffer, steps=total_steps
    )

    freeRoam(policy, env, num_skills, total_steps=10000, max_skill_steps=20)
    torch.save(discriminator.state_dict(), "Models/discriminator_external_reward_on_disc")
    # Plotting (final one now!)
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
