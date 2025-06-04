import torch
import matplotlib.pyplot as plt
from discriminator import Discriminator

state_dim = 2
action_dim = 2
num_skills = 10
disc = Discriminator(state_dim, num_skills)
disc.load_state_dict(torch.load("Models\discriminator_external_reward_on_disc", weights_only= True))

X = [[-1 + 0.02 * i, -1 + 0.02 * j] for i in range(100) for j in range(100)]
# Plotting (final one now!)
colors = plt.cm.get_cmap('tab10', num_skills)
plt.figure(figsize=(8, 8))
for x in X:
    print(x)
    i = torch.argmax(disc(torch.tensor(x)))
    plt.scatter(x[0], x[1], alpha=0.5, color=colors(i))
plt.legend()
plt.grid(True)
plt.title("Discriminator predictions")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()