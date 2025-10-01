from dataclasses import dataclass

@dataclass
class Config:
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct" 
    # base_model_name: str = "meta-llama/Llama-2-7b"

    judge_model: str = "cognitivecomputations/dolphin3.0-mistral-24b:free" # "deepseek/deepseek-chat-v3-0324:free" # "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"  # OpenRouter
    device: str = "cuda"
    torch_dtype: str = "bfloat16"  # "float16" if you prefer
    hook_layer: int = 16          # 0-based index (32-layer model → halfway)
    hook_point: str = "post_attn_norm"  # "post_attn_norm" or "pre_attn_norm"
    train_alpha: bool = True
    lr: float = 5e-3
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    clip_grad_norm: float = 1.0
    cosine_decay: bool = True
    warmup_updates: int = 50
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 32            # prompts per update step (before grad accumulation)
    grad_accum: int = 8             # on-policy; increases effective batch size
    total_updates: int = 200
    kl_coef: float = 0.01
    use_reward_normalization: bool = True
    save_checkpoints: bool = True
    num_checkpoints: int = 10
    run_name: str = "llama3_steering"
    wandb_project: str = ""
    prompts_file: str = ""          # if empty → use placeholders
    max_prompt_chars: int = 1200
