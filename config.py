"""Configuration loading from YAML files."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch as t
import yaml

if t.cuda.is_available():
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    t.backends.cudnn.benchmark = True

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


@dataclass
class Config:
    """Training configuration loaded from YAML."""

    seed: int = 1

    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cuda"

    # Generation
    gen_len: int = 120
    temperature: float = 0.7
    top_k: Optional[int] = None
    prepend_bos: bool = False

    # Steering
    steering_layer_indices: Optional[list[int]] = None
    steering_init_scale: float = 0.2

    # Training
    total_phases: int = 15
    batch_size: int = 3
    num_minibatches: int = 3
    batches_per_learning_phase: int = 12
    rollouts_per_phase: int = 12

    # Optimization
    base_lr: float = 5e-4
    max_grad_norm: float = 2.0
    warmup_steps: int = 0
    final_scale: float = 1.0

    # KL
    kl_coef: float = 0.6
    use_adaptive_kl: bool = True
    kl_target_nats: float = 0.05
    kl_coef_min: float = 0.05
    kl_coef_max: float = 3.0
    kl_up: float = 1.05
    kl_down: float = 0.97

    # Entropy
    ent_coef: float = 0.01
    use_entropy_anneal: bool = True
    ent_coef_start: float = 0.005
    ent_coef_end: float = 0.0
    ent_warmup_phases: int = 2

    # Judge (OpenRouter)
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_model: str = "openai/gpt-5-mini"
    judge_concurrency: int = 32
    judge_timeout: float = 30.0

    # Logging
    use_wandb: bool = True
    wandb_project: str = "arena-rl-steering"
    wandb_entity: Optional[str] = None
    rollout_log_csv: str = "rollout_log.csv"

    # Snapshots
    save_steering_snapshots: bool = True
    steering_snapshot_dir: str = "steering_snapshots"

    # Prompts (loaded from prompts.yaml)
    chat_template: str = ""
    prompts: list[str] = field(default_factory=list)
    judge_system_prompt: str = ""

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def load_config(
    config_path: str = "config.yaml",
    prompts_path: str = "prompts.yaml",
) -> Config:
    """Load configuration from YAML files."""
    base_dir = Path(__file__).parent

    config_file = base_dir / config_path
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}

    prompts_file = base_dir / prompts_path
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts_data = yaml.safe_load(f) or {}
    else:
        prompts_data = {}

    merged = {**config_data, **prompts_data}

    if "prompts" in merged:
        merged["prompts"] = [p.strip() for p in merged["prompts"] if p and p.strip()]

    valid_fields = {f.name for f in dataclasses.fields(Config)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}

    return Config(**filtered)

