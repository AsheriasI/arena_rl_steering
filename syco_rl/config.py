import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch as t
from accelerate import Accelerator

from .utils import load_judge_prompt

# ============== GPU / Accelerate ==============
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

if t.cuda.is_available():
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    t.backends.cudnn.benchmark = True

accelerator = Accelerator(
    mixed_precision=("bf16" if t.cuda.is_available() else "no"),
    gradient_accumulation_steps=1,
)
device = accelerator.device

# ============== Models & Judge ==============
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"    # actor (TransformerLens)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge_prompt.txt"
RAW_PROMPTS_PATH = PROMPTS_DIR / "raw_prompts.jsonc"

# Use OpenRouter instead of your local vLLM
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "openai/gpt-5-mini"  # or whatever exact ID your account has access to

# ============== JSON Judge Prompt (rate sycophancy) ==============
JUDGE_SCORER_SYSTEM_PROMPT = load_judge_prompt(str(JUDGE_PROMPT_PATH))

# =========================
# Hyperparameters / Args
# =========================
@dataclass
class RLHFArgs:
    seed: int = 1

    # logging
    use_wandb: bool = False
    wandb_project_name: str = "sycophancy_steering"
    wandb_entity: str | None = None

    # schedule
    total_phases: int = 8
    batch_size: int = 8
    num_minibatches: int = 2
    batches_per_learning_phase: int = 1  # <- you asked to use this as the stability knob

    # optimization (stabilized)
    base_lr: float = 2e-3
    head_lr: float = 1e-4  # unused
    max_grad_norm: float = 2.0
    warmup_steps: int = 0
    final_scale: float = 1.0

    # entropy bonus
    ent_coef: float = 0.01

    # actor sampling
    base_model: str = BASE_MODEL
    gen_len: int = 120
    temperature: float = 0.7
    top_k: Optional[int] = None
    prefix: str = "This is"
    prepend_bos: bool = False  # IMPORTANT with llama 3.1 template

    # kl / reward_fn fields kept for API compatibility
    kl_coef: float = 0.6
    reward_fn: Callable = lambda *args, **kwargs: 0.0

    def __post_init__(self):
        assert self.batch_size % self.num_minibatches == 0, "batch_size divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        assert self.total_phases > 0


@dataclass
class RLOOArgs(RLHFArgs):
    steering_layer_indices: list[int] = None
    steering_init_scale: float = 0.2

    judge_concurrency: int = 32

    # Actor sees chat-formatted prompts; judge sees raw user prompts.
    actor_prompts_inline: Optional[list[str]] = None
    judge_user_prompts_inline: Optional[list[str]] = None

    # rollout CSV of raw samples
    csv_path: str = "rollout_log.csv"

    # Steering tracking
    steering_log_csv: str = "steering_metrics.csv"
    save_steering_snapshots: bool = True
    steering_snapshot_dir: str = "steering_snapshots"
    plot_steering_after_training: bool = True
    steering_plot_png: Optional[str] = None

    # ===== Training metrics & multi-rollout =====
    train_metrics_csv: str = "training_metrics.csv"
    plot_train_png: Optional[str] = None
    rollouts_per_phase: int = 1  # <-- collect this many fresh rollouts each phase

    # ===== New: Adaptive KL controls =====
    use_adaptive_kl: bool = True
    kl_target_nats: float = 0.05       # target KL/token
    kl_coef_min: float = 0.05
    kl_coef_max: float = 3.0
    kl_up: float = 1.05             # multiplicative up/down
    kl_down: float = 0.97

    # ===== New: Entropy anneal controls =====
    use_entropy_anneal: bool = True
    ent_coef_start: float = 0.005    # overrides RLHFArgs.ent_coef if anneal is on
    ent_coef_end: float = 0.0
    ent_warmup_phases: int = 2