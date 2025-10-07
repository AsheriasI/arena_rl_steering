#%% Import

import torch
from sae_lens import SAE
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Constants

TOP_K = 100
LAYER_INDICES = [3, 15, 27]
SAE_RELEASE = "llama-3.1-8b-instruct-andyrdt"
STEERING_SNAPSHOT_PATH = "/disk/u/troitskiid/projects/arena_rl_steering/llm_judge_KL/steering_phase_015.pt"

#%% Load steering vectors

steering_snapshot = torch.load(STEERING_SNAPSHOT_PATH, map_location=device)

steering_vectors = {}

for layer in [3, 15, 27]:
    key = f"steering_hooks.{layer}.vec_ln1.vector"
    steering_vectors[layer] = steering_snapshot[key]

# %%
steering_snapshot

#%% Load SAEs

sae_1, cfg_dict_1, sparsity_1 = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"resid_post_layer_{LAYER_INDICES[0]}_trainer_1", device=device)
sae_2, cfg_dict_2, sparsity_2 = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"resid_post_layer_{LAYER_INDICES[1]}_trainer_1", device=device)
sae_3, cfg_dict_3, sparsity_3 = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"resid_post_layer_{LAYER_INDICES[2]}_trainer_1", device=device)

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
