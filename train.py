import math
import torch
import torch.nn.functional as F
from utils import one_hot, to_tensor
import numpy as np
import time
from alternativeRewards import alt_novelty_reward, batch_knn_novelty_rewards, batch_levina_bickel_id, train_levina_bickel_id
from performanceTracker import performanceTracker

def polyak_update(source_net, target_net, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def train(
        policy, discriminator, critic,
        critic_target,
        env, num_skills,
        optim_policy, optim_disc, optim_critic,
        gamma, tau, alpha, target_entropy,
        buffer, steps, batch_size=256, max_skill_steps=20,
        target_update_interval=2
    ):
    log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
    alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4)
    
    skill_trajectories = [[] for _ in range(num_skills)]
    final_skill = [[] for _ in range(num_skills)]
    
    pTracker = performanceTracker(["Intrinsic Reward Calculation", "intrinsic reward calculation 2", "knn model calculation", "Discriminator Forward Pass", "Q-Value Calculation", "Policy Sampling"])
    discriminator_correct_classifications = 0
    discriminator_total_classifications = 0

    for step_idx in range(steps):
        # Environment interaction
        skill = np.random.randint(0, num_skills)
        skill_onehot = one_hot(skill, num_skills)
        state = env.reset()
        final_skill[skill] = []
        
        for t in range(max_skill_steps):
            state_tensor = to_tensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_tensor, _ = policy.sample(state_tensor, skill_onehot)
                action = action_tensor.squeeze(0).cpu().numpy()
                action = np.clip(action, -env.max_step, env.max_step)

            next_state = env.step(action)
            done = (t == max_skill_steps - 1)

            # Intrinsic reward calculation
            pTracker.start("Intrinsic Reward Calculation")
            with torch.no_grad():
                next_state_tensor = to_tensor(next_state).unsqueeze(0)
                disc_logits = discriminator(next_state_tensor) / alpha
                log_probs = F.log_softmax(disc_logits, dim=1)
                intrinsic_reward = log_probs[0, skill].item() + np.log(num_skills)

            pTracker.end()
            #Update discriminator statistics
            discriminator_correct_classifications += (disc_logits.argmax(dim=1) == skill).sum().item()
            discriminator_total_classifications += 1
            # Store transition in buffer
            buffer.add((state, action, intrinsic_reward, next_state, float(done), skill))
            skill_trajectories[skill].append(next_state.copy())
            final_skill[skill].append(next_state.copy())
            state = next_state

        # Training update
        if len(buffer.buffer) < batch_size:
            continue

        # Sample batch
        states, actions, rewards, next_states, dones, skills = buffer.sample(batch_size)
        
        # Convert to tensors
        states = to_tensor(states)
        actions = to_tensor(actions)
        rewards = to_tensor(rewards).unsqueeze(1)
        next_states = to_tensor(next_states)
        dones = to_tensor(dones).unsqueeze(1)
        skills = torch.tensor(skills, dtype=torch.long)
        skills_onehot = one_hot(skills, num_skills).float()

        ####################################
        ###### #### WARNING #### ###########
        ####################################
        ###### EXPERIMENTAL ZONE ###########
        ####################################

        # Data structure that makes querying the k nearest neighbors is updated seperately and less often
        if step_idx % target_update_interval*5==0: # new hyper parameter 1, how often to update knn model
            pTracker.start("knn model calculation")
            all_states = np.concatenate([traj for traj in skill_trajectories if len(traj) > 0], axis=0)
            # new hyper parameter 2, how many nearest neighbors to sample (k=10)
            knn_model = train_levina_bickel_id(all_states, k=10) 
            #Inside are tunable hyper parameters for minimum and maximum number of states inside datastructure
            pTracker.end()
        
        #ensures we have enough data and knn model exists
        if step_idx>= target_update_interval*5: 
            pTracker.start("intrinsic reward calculation 2")

            #This is the new bottleneck of the algorithm
            local_ids = batch_levina_bickel_id(states, knn_model)
            rewards = rewards * (1-alpha) + to_tensor(local_ids).unsqueeze(1) * alpha  # Scale by alpha
            #now rewards disciminability + state coverage, 
            pTracker.end()
        ####################################
        ###### #### LEAVING #### ###########
        ###### EXPERIMENTAL ZONE ###########
        ######### Good Bye :) ##############
        ####################################

        # 1. Discriminator update
        pTracker.start("Discriminator Forward Pass")
        disc_logits = discriminator(states)
        loss_disc = F.cross_entropy(disc_logits, skills)
        optim_disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()
        pTracker.end()

        # 3. Q-network update
        pTracker.start("Q-Value Calculation")
        with torch.no_grad():
            next_actions, log_prob = policy.sample(next_states, skills_onehot)
            q1_next, q2_next = critic_target(next_states, skills_onehot, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            q_target = intrinsic_reward + gamma * (1 - dones) * (min_q_next - alpha * log_prob)


        q1_pred, q2_pred = critic(states, skills_onehot, actions)
        loss_q1 = F.mse_loss(q1_pred, q_target)
        loss_q2 = F.mse_loss(q2_pred, q_target)
        loss_q = loss_q1 + loss_q2
        optim_critic.zero_grad()
        loss_q.backward()
        optim_critic.step()

        pTracker.end()

        # 4. Policy update
        pTracker.start("Policy Sampling")

        # Sample actions from policy
        new_actions, _ = policy.sample(states, skills_onehot)

        q1_new, q2_new = critic(states, skills_onehot, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Maximize expected Q-value for the skill — equivalent to minimizing negative Q
        policy_loss = -min_q_new.mean()
        # Optimize policy
        optim_policy.zero_grad()
        policy_loss.backward()
        optim_policy.step()


        pTracker.end()

        # 5. Alpha update (periodic)
        if step_idx % target_update_interval == 0:
            
            alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
            log_alpha.data = torch.clamp(log_alpha.data, min=np.log(0.01))

            # Target network updates
            polyak_update(critic, critic_target, tau)

        # Logging
        if step_idx % 100 == 0:
            disc_accuracy = (discriminator_correct_classifications / discriminator_total_classifications) if discriminator_total_classifications > 0 else 0
            print(f"Rollout {step_idx}/{steps}, "
                  f"Disc: {loss_disc.item():.3f}, "
                  f"Discriminator Accuracy: {disc_accuracy:.3f}, "
                  f"Q: {loss_q.item():.3f}, "
                  f"Policy: {policy_loss.item():.3f}, "
                  f"Alpha: {log_alpha.exp().item():.3f}")
            
            pTracker.print_times()
            pTracker.historize(step_idx)

            # debugging
            print(discriminator(torch.Tensor([0, 0.2])).argmax())

    # Access final stuff here!
    pTracker.plot_all_histories()
    return final_skill