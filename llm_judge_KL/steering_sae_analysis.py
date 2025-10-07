#%% Import

import torch
import os
from sae_lens import SAE
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Constants

TOP_K = 100
LAYER_INDICES = [3, 15, 27]
SAE_PATH = "/disk/u/troitskiid/projects/arena_rl_steering/saes-llama-3.1-8b-instruct"
STEERING_SNAPSHOT_PATH = "/disk/u/troitskiid/projects/arena_rl_steering/llm_judge_KL/steering_phase_014.pt"

# %%
# as an experiment, generate 3 random vectors to test
import torch.nn as nn

random_vectors = {}
for layer in LAYER_INDICES:
    # Generate random vector with same dimension as model (4096 for Llama 3.1 8B)
    random_vec = torch.randn(4096, device=device)
    # Normalize to unit length
    random_vec = random_vec / random_vec.norm()
    random_vectors[layer] = random_vec

# Use random vectors instead of steering snapshot
steering_vectors = random_vectors

# #%% Load steering vectors

# steering_snapshot = torch.load(STEERING_SNAPSHOT_PATH, map_location=device)

# steering_vectors = {}

# for layer in LAYER_INDICES:
#     key = f"steering_hooks.{layer}.vec_ln1.vector"
#     steering_vectors[layer] = steering_snapshot[key]

# # %%
# steering_snapshot

#%% Load SAEs

# [ ] use the loader from https://github.com/safety-research/open-source-em-features/blob/main/open_source_em_features/utils/sae_loading.py

# FIXME rewrite loading

sae_1 = SAE.load_from_disk(os.path.join(SAE_PATH, f"resid_post_layer_{LAYER_INDICES[0]}/trainer_1"), device="cuda", dtype="torch.bfloat16")
sae_2 = SAE.load_from_disk(os.path.join(SAE_PATH, f"resid_post_layer_{LAYER_INDICES[1]}/trainer_1"), device="cuda", dtype="torch.bfloat16")
sae_3 = SAE.load_from_disk(os.path.join(SAE_PATH, f"resid_post_layer_{LAYER_INDICES[2]}/trainer_1"), device="cuda", dtype="torch.bfloat16")

saes = {LAYER_INDICES[0]: sae_1, LAYER_INDICES[1]: sae_2, LAYER_INDICES[2]: sae_3}

#%% Compute similarities

similarities = {}

for layer in LAYER_INDICES:
    steering_vec = steering_vectors[layer]
    sae = saes[layer]
    sim = torch.matmul(steering_vec, sae.W_dec.T)
    similarities[layer] = sim

#%% Get top features

top_features = {}

for layer in LAYER_INDICES:
    sim = similarities[layer]
    top_indices = torch.topk(sim, TOP_K).indices.tolist()
    top_features[layer] = top_indices

#%% Create neuronpedia links

for layer in LAYER_INDICES:
    sae = saes[layer]
    features = top_features[layer]
    quick_list = get_neuronpedia_quick_list(sae, features)
    print(f"Layer {layer}: {quick_list}")
