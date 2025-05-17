import torch
import torch.nn.functional as F
from utils import one_hot, to_tensor
import numpy as np

def train(policy, discriminator, env, num_skills, optimizer_policy, optimizer_disc, buffer, steps=10000):
    skill_trajectories = [[] for _ in range(num_skills)]

    for step in range(steps):
        skill = torch.randint(0, num_skills, (1,)).item()
        skill_onehot = one_hot(skill, num_skills)
        state = env.reset()

        for t in range(20):
            s_tensor = to_tensor(state).unsqueeze(0)
            z_tensor = skill_onehot.unsqueeze(0).float()

            with torch.no_grad():
                action = policy(s_tensor, z_tensor).squeeze(0).numpy()
                action = np.clip(action, -env.max_step, env.max_step)

            next_state, _, _, _ = env.step(action)
            buffer.add((next_state.copy(), skill))
            skill_trajectories[skill].append(next_state.copy())
            state = next_state

        if len(buffer.buffer) >= 128:
            # Sample and convert to tensors
            states_np, skills_np = buffer.sample(128)
            states = to_tensor(states_np)
            skills = torch.tensor(skills_np, dtype=torch.long)

            # Discriminator update
            logits = discriminator(states)
            loss_disc = F.cross_entropy(logits, skills)
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # Policy update
            with torch.no_grad():
                states_tensor = to_tensor(states_np)
            
            logits_for_policy = discriminator(states_tensor)
            log_probs = F.log_softmax(logits_for_policy, dim=1)
            intrinsic_reward = log_probs[range(len(skills_np)), skills_np]

            policy_loss = -intrinsic_reward.mean()
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            if step % 500 == 0:
                print(f"Step {step}/{steps}, Disc Loss: {loss_disc.item():.3f}, Policy Loss: {policy_loss.item():.3f}")

    return skill_trajectories