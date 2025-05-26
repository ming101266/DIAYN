from sklearn.neighbors import NearestNeighbors
import numpy as np
import zlib 

#Tested functions for complexity measures, too slow compared to OG
#Also doesn't make a lot of sense to use

#mad slow but technically could be used
def kolmogorov_complexity(actions):
    actions_bytes = np.array(actions).tobytes()
    compressed = zlib.compress(actions_bytes)
    return len(compressed)

#relatively fast but idk how to make it a rewards -Vlad
def renyi_entropy(actions, alpha=2):
    probs = np.bincount(actions) / len(actions)
    probs = probs[probs > 0]
    return 1 / (1 - alpha) * np.log(np.sum(probs ** alpha))
 
 #First function I tried as a rewards but led to rewards riding the edge of state space
def novelty_reward(state, past_states, k=10):
    if len(past_states) < k:
        return 0  # Not enough past states yet
    if len(past_states) > 1000: #sample from most recent 1000 states
        lower_index_bound = len(past_states) - 1000
        sampled = np.array(past_states)[lower_index_bound + np.random.choice(1000, k, replace=False)]
    else:
        sampled = np.array(past_states)[np.random.choice(len(past_states), k, replace=False)]
    dists = np.linalg.norm(sampled - state, axis=1)
    log_novelty = np.log(np.mean(dists) + 1e-8)  
    log_novelty = np.clip(log_novelty, -10, 10) 
    return log_novelty

 #Tried to use min distance instead of mean, but also led to edge riding
def alt_novelty_reward(state, past_states, k=10): #log min distance amongst k samples
    if len(past_states) < k:
        return 0  # Not enough past states yet
    if len(past_states) > 1000: #sample from most recent 1000 states
        lower_index_bound = len(past_states) - 1000
        sampled = np.array(past_states)[lower_index_bound + np.random.choice(1000, k, replace=False)]
    else:
        sampled = np.array(past_states)[np.random.choice(len(past_states), k, replace=False)]
    dists = np.linalg.norm(sampled - state, axis=1)
    log_novelty = np.log(np.min(dists) + 1e-8)  # Add small constant to avoid log(0)
    log_novelty = np.clip(log_novelty, -10, 10)  # Clip to avoid extreme values
    return log_novelty

#First decent, batch based, but still very slow
def batch_knn_novelty_rewards(states, past_states, k=10):
    if len(past_states) < k:
        return np.zeros(len(states))
    if len(past_states) > 1000:
        recent_states = np.array(past_states[-1000:])
    else:
        recent_states = np.array(past_states)
    nbrs = NearestNeighbors(n_neighbors=k).fit(recent_states)
    dists, _ = nbrs.kneighbors(states)
    return np.mean(np.log(dists + 1e-8), axis=1)
    #average distance being larger in simplest sense means we are in unmarked territory
    #in theoretical sense, if this value is large, it means the ball centered around new state is not-dense
    #optimal strategy with this reward setup is similarly to maximizing entropy, to fill the space evenly/uniformly
    #however this will process may be expensive

#Paper-backed method of apprximating local fractal dimension
#Still mad slow (adds around 10x reward computation time) but accurate
#could lead to faster state space covering despite slower reward computation
#(prolly unlikely tho)
def train_levina_bickel_id(past_states, k=10):
    if len(past_states) < 100 + k:
        print("Not enough past states")
        return NULL  # Not enough past states yet
    if len(past_states) > 1000:
        recent_states = np.array(past_states[-1000:])
    else:
        recent_states = np.array(past_states)
    knn_model = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(recent_states)
    return knn_model
# Levina-Bickel intrinsic dimensionality
def batch_levina_bickel_id(states, knn_model):
    dists, _ = knn_model.kneighbors(states)
    log_dists = np.log(np.maximum(dists, 1e-8))
    rks = log_dists[:, -1]
    logs = rks[:, None] - log_dists[:, :-1]   
    local_ids = np.mean(logs, axis=1)

    # Normalize the local Intrinsic Dimension values
    # so they play well with the other rewards
    mean, std = np.mean(local_ids), np.std(local_ids)
    return (local_ids - mean) / (std + 1e-8)  # Normalize
    
    

