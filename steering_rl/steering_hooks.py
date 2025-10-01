# steering_rl/steering_hooks.py
import torch
import torch.nn as nn
from typing import Optional
from contextlib import contextmanager

class SteeringVector(nn.Module):
    """
    Adds a learned vector v ∈ R^{d_model} scaled by alpha to the hidden states
    at a chosen hook site (after a layer's RMSNorm). Applied to **all tokens**.
    """
    def __init__(self, d_model: int, init_scale: float = 1e-3, train_alpha: bool = True):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.v, mean=0.0, std=init_scale)
        self.alpha = nn.Parameter(torch.tensor(0.1)) if train_alpha else None
        self.enabled = True  # <— quick toggle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, d_model] (RMSNorm output)
        returns: x + alpha * v (broadcast over [batch, seq, :])
        """
        if not self.enabled:
            return x
        scale = self.alpha if self.alpha is not None else x.new_tensor(1.0)
        return x + scale * self.v

def attach_steering_hook(model, layer_index: int, point: str, steering: SteeringVector):
    """
    Llama architecture has, per layer:
      - input_layernorm (pre-attn)
      - post_attention_layernorm (pre-MLP)
    We hook **after** the chosen RMSNorm forward by replacing it with a wrapper.
    """
    layer = model.model.layers[layer_index]
    if point == "pre_attn_norm":
        norm = layer.input_layernorm
    elif point == "post_attn_norm":
        norm = layer.post_attention_layernorm
    else:
        raise ValueError("point must be 'pre_attn_norm' or 'post_attn_norm'")

    orig_forward = norm.forward

    def wrapped_forward(x):
        out = orig_forward(x)
        return steering(out)

    norm.forward = wrapped_forward  # monkeypatch in-place
    return model

@contextmanager
def steering_disabled(steering: SteeringVector):
    """Temporarily disable steering (useful for debugging)."""
    was = steering.enabled
    steering.enabled = False
    try:
        yield
    finally:
        steering.enabled = was
