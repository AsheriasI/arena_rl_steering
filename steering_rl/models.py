# steering_rl/models.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import Config
from .steering_hooks import SteeringVector, attach_steering_hook

def load_llama_with_steering(cfg: Config):
    # Force safest runtime backends via env (works even if set late in many cases)
    os.environ.setdefault("XFORMERS_DISABLED", "1")          # disable xFormers kernels
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    os.environ.setdefault("HF_USE_FLASH_ATTENTION", "0")     # HF flag used by some loaders
    os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Prefer fp16 here for stability; you can switch back to bf16 if this is stable in your env
    dtype = torch.float16

    breakpoint()

    tok = AutoTokenizer.from_pretrained(cfg.base_model_name, use_fast=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    breakpoint()

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        # attn_implementation="sdpa",   # <- SDPA: stable on A100 with PyTorch CUDA 12.x
        device_map={"": 0},           # single GPU (GPU 0)
    )
    print("here?")
    breakpoint()
    
    base.config.use_cache = True
    breakpoint()

    # Freeze base model weights; only the steering params are trainable
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()

    # Steering vector
    d_model = base.config.hidden_size
    steering = SteeringVector(d_model=d_model, train_alpha=cfg.train_alpha)
    breakpoint()
    steering.to("cuda", dtype=dtype)
    breakpoint()
    # Hook at chosen site
    attach_steering_hook(base, cfg.hook_layer, cfg.hook_point, steering)
    breakpoint
    return tok, base, steering

@torch.no_grad()
def generate(model, tok, prompts, cfg: Config):
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(tok.model_max_length, 4096),  # guard against huge prompts
    ).to("cuda")
    gen = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.batch_decode(gen, skip_special_tokens=True)
    outs = []
    for p, full in zip(prompts, text):
        if full.startswith(p):
            outs.append(full[len(p):].strip())
        else:
            outs.append(full)
    return outs
