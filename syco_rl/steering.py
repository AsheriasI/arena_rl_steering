from typing import Callable

import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from .config import device


class SteeringVector(nn.Module):
    vector: nn.Parameter  # (d_model,)

    def __init__(self, d_model: int, dtype: t.dtype | None = None, init_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.vector = nn.Parameter(t.randn(d_model, dtype=dtype) * init_scale)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        if self.dtype is not None and x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x + self.vector


class SteeringHooks(nn.Module):
    vec_ln1: SteeringVector

    def __init__(self, layer_idx: int, cfg: HookedTransformerConfig, init_scale: float = 1.0, dtype: t.dtype = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.vec_ln1 = SteeringVector(cfg.d_model, dtype=dtype, init_scale=init_scale)

    def list_fwd_hooks(self) -> list[tuple[str, Callable]]:
        return [
            (f"blocks.{self.layer_idx}.ln1.hook_normalized", self.steering_hook_out)
        ]

    def steering_hook_out(
        self, normalized: Float[Tensor, "batch pos d_model"], hook: HookPoint
    ) -> Float[Tensor, "batch pos d_model"]:
        return self.vec_ln1(normalized)


class HookedTransformerWithSteering(HookedTransformer):
    steering_hooks: nn.ModuleList
    steering_fwd_hooks: list[tuple[str, Callable]]

    def base_model_params(self):
        return (p for name, p in self.named_parameters() if "steering" not in name)

    def steering_params(self):
        return self.steering_hooks.parameters()

    def get_base_model_trainable_params(self):
        return self.steering_params()

    def get_value_head_params(self):
        return iter([])

    @classmethod
    def from_pretrained(cls, *args, layer_indices: list[int] = None, init_scale: float = 1.0, **kwargs):
        model = super(HookedTransformerWithSteering, cls).from_pretrained(*args, **kwargs)
        model.setup_steering(layer_indices=layer_indices, init_scale=init_scale)
        for p in model.base_model_params():
            p.requires_grad = False
        return model

    def setup_steering(self, layer_indices: list[int] | int | None = None, init_scale: float = 1.0):
        if layer_indices is None:
            layer_indices = list(range(len(self.blocks)))
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        model_dtype = next(self.parameters()).dtype
        self.steering_hooks = nn.ModuleList([
            SteeringHooks(i, self.cfg, init_scale=init_scale, dtype=model_dtype)
            for i in layer_indices
        ]).to(device)
        self.steering_fwd_hooks = []
        for sh in self.steering_hooks:
            self.steering_fwd_hooks.extend(sh.list_fwd_hooks())

    @property
    def fwd_hooks(self):
        return self.steering_fwd_hooks

    def forward_with_steering(self, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq d_vocab"]:
        with self.hooks(fwd_hooks=self.fwd_hooks):
            logits = self.forward(tokens)
        return logits

    @t.no_grad()
    def generate(self, tokens: Int[Tensor, "batch seq"], **kwargs) -> Int[Tensor, "batch seq"]:
        with self.hooks(fwd_hooks=self.fwd_hooks):
            gen_tokens = super().generate(tokens, **kwargs)
        return gen_tokens

