import os, json, math, time
import torch
import wandb
from typing import Dict

def setup_wandb(run_name: str, project: str, cfg_as_dict: Dict):
    if not project:
        return None
    wandb.init(project=project, name=run_name, config=cfg_as_dict)
    return wandb

def save_checkpoint(run_dir: str, step: int, steering_state: dict, meta: dict):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"ckpt_step{step:06d}.pt")
    torch.save({"steering": steering_state, "meta": meta}, path)
    return path

def evenly_spaced_steps(total_updates: int, k: int):
    if k <= 0: 
        return set()
    points = set()
    for i in range(1, k+1):
        step = math.floor(total_updates * i / (k+1))
        if 1 <= step < total_updates:
            points.add(step)
    # ensure last step saved
    points.add(total_updates)
    return points

class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None
    def update(self, x):
        self.value = x if self.value is None else self.beta*self.value + (1-self.beta)*x
        return self.value
