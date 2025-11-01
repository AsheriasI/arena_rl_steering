# %%


import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, List, Optional
from tqdm import tqdm

import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from eindex import eindex
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from tabulate import tabulate
from torch import Tensor
from transformer_lens import HookedTransformer, utils, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

import transformer_lens.utils as utils

# -------- Accelerate ----------
from accelerate import Accelerator

# Enable TF32 fast kernels on CUDA (safe on A100/Hopper)
if t.cuda.is_available():
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    t.backends.cudnn.benchmark = True

# Create a global accelerator (bf16 on CUDA; 'no' elsewhere)
accelerator = Accelerator(
    mixed_precision=("bf16" if t.cuda.is_available() else "no"),
    gradient_accumulation_steps=1,  # keeping your replay/minibatch logic
)
device = accelerator.device


# ===================== Start of Judge integration =====================

import asyncio, math, os, json, httpx
from typing import List, Dict, Any

# ===================== End of Judge integration =====================


# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part4_rlhf"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part4_rlhf import tests, tests_lora  # , tl_ext


MAIN = __name__ == "__main__"

# %%

# Set default parameters for low GPU memory usage, change if you have more GPU memory

LOW_GPU_MEM = False
# BASE_MODEL = "gpt2-small" if LOW_GPU_MEM else "gpt2-medium"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"




RUN_BASE_RLHF = False

# %%


@dataclass
class RLHFArgs:
    # Basic / global
    seed: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "RLHF"
    wandb_entity: str | None = None

    # Duration of different phases
    total_phases: int = 100
    batch_size: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 2

    # Optimization hyperparameters
    base_lr: float = 2e-5
    head_lr: float = 5e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 20
    final_scale: float = 0.1

    # Computing other PPO loss functions
    clip_coef: float = 0.2
    vf_coef: float = 0.15
    ent_coef: float = 0.001

    # Base model & sampling arguments
    base_model: str = BASE_MODEL
    gen_len: int = 30
    temperature: float = 1.0
    top_k: int = 10
    prefix: str = "This is"
    prepend_bos: bool = True

    # RLHF-specific arguments
    kl_coef: float = 2.5
    reward_fn: Callable = lambda x: 0.0
    normalize_reward: bool = True

    def __post_init__(self):
        assert (
            self.total_phases > self.warmup_steps
        ), "total_phases must be greater than warmup_steps"
        assert (
            self.batch_size % self.num_minibatches == 0
        ), "batch_size should be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches


# %%


class HookedTransformerWithValueHead(HookedTransformer):
    """
    GPT + value head (MLP) attached to ln_final.hook_normalized
    """

    value_head: nn.Sequential
    value_head_output: Float[Tensor, "batch seq"]
    value_head_hook: list[tuple[str, Callable]]

    @classmethod
    def from_pretrained(cls, *args, use_value_head=True, **kwargs):
        model = super(HookedTransformerWithValueHead, cls).from_pretrained(*args, **kwargs)
        model.value_head_hook = ("ln_final.hook_normalized", model.run_value_head)

        if use_value_head:
            model.value_head = nn.Sequential(
                nn.Linear(model.cfg.d_model, 4 * model.cfg.d_model),
                nn.ReLU(),
                nn.Linear(4 * model.cfg.d_model, 1),
            )
        else:
            model.value_head = None
        return model

    @property
    def fwd_hooks(self):
        return [self.value_head_hook]

    def get_base_model_trainable_params(self):
        return (p for name, p in self.named_parameters() if "value_head" not in name)

    def get_value_head_params(self):
        return self.value_head.parameters()

    def run_value_head(self, resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint):
        self.value_head_output = self.value_head(resid_post).squeeze(-1)

    def forward_with_value_head(
        self,
        input_ids: Int[Tensor, "batch seq"],
        **kwargs,
    ) -> tuple[Float[Tensor, "batch seq d_vocab"], Int[Tensor, "batch seq"]]:
        self.value_head_output = None
        logits = self.run_with_hooks(
            input_ids,
            return_type="logits",
            fwd_hooks=self.fwd_hooks,
        )
        return logits, self.value_head_output


# if MAIN:
    # Define a reference model (we'll use this during RLHF)
    # model = HookedTransformerWithValueHead.from_pretrained("pythia-14m", use_value_head=True).to(
        # device
    # )
    # tests.test_transformer_with_value_head(model)

# %%


@t.no_grad()
def get_samples(
    model: HookedTransformer,
    prompt: str,
    batch_size: int,
    gen_len: int = 15,
    temperature: float = 0.8,
    top_k: int = 15,
    prepend_bos: bool = True,
    **kwargs,
) -> tuple[Int[Tensor, "batch seq"], list[str]]:
    """
    Generates samples from the model to feed into the reward model.
    """
    input_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    input_ids = einops.repeat(input_ids, "1 seq -> batch seq", batch=batch_size)

    # Use accelerator autocast for faster generate on CUDA
    with accelerator.autocast():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=gen_len,
            stop_at_eos=False,
            temperature=temperature,
            top_k=top_k,
            **kwargs,
        )
    samples = model.to_string(output_ids)
    return output_ids.clone(), samples


# %%

# if MAIN:
#     model = HookedTransformerWithValueHead.from_pretrained(BASE_MODEL).to(device)
#     sample_ids, samples = get_samples(
#         model,
#         prompt="So long, and thanks for all the",
#         batch_size=5,
#         gen_len=15,
#         temperature=0.8,
#         top_k=15,
#         prepend_bos=False,
#         verbose=True,
#         use_past_kv_cache=True,
#     )

#     table = Table("Token IDs", "Samples", title="Demo of `sample` function", show_lines=True)
#     for ids, sample in zip(sample_ids, samples):
#         table.add_row(str(ids.tolist()), repr(sample))

#     if accelerator.is_main_process:
#         rprint(table)

# %%


def reward_fn_char_count(generated_sample: list[str], char: str = ".") -> Float[Tensor, " batch"]:
    return t.tensor([item.count(char) for item in generated_sample], device=device, dtype=t.float)


# if MAIN:
#     # Tests
#     A = "This is a test."
#     B = "......"
#     C = "Whatever"

#     t.testing.assert_close(reward_fn_char_count([A]), t.tensor([1.0], device=device))
#     t.testing.assert_close(
#         reward_fn_char_count([A, B, C]), t.tensor([1.0, 6.0, 0.0], device=device)
#     )
#     t.testing.assert_close(reward_fn_char_count([A], " "), t.tensor([3.0], device=device))
#     if accelerator.is_main_process:
#         print("All tests for `reward_fn_char_count` passed!")

# %%


def normalize_reward(reward: Float[Tensor, " batch"], eps=1e-5) -> Float[Tensor, " batch"]:
    return (reward - reward.mean()) / (reward.std() + eps)


# if MAIN:
    # tests.test_normalize_reward(normalize_reward)

# %%


@t.no_grad()
def compute_advantages(
    values: Float[Tensor, " minibatch_size seq_len"],
    rewards: Float[Tensor, " minibatch_size"],
    prefix_len: int,
) -> Float[Tensor, " minibatch_size gen_len"]:
    one_step_q_est = t.cat([values[:, prefix_len:-1], rewards[:, None]], dim=-1)
    zero_step_value_est = values[:, prefix_len - 1 : -1]
    advantages = one_step_q_est - zero_step_value_est
    return advantages


# if MAIN:
    # tests.test_compute_advantages(compute_advantages)

# %%


@dataclass
class ReplayMinibatch:
    sample_ids: Float[Tensor, " minibatch_size seq_len"]
    logprobs: Float[Tensor, " minibatch_size gen_len"]
    advantages: Float[Tensor, " minibatch_size gen_len"]
    returns: Float[Tensor, " minibatch_size gen_len"]
    ref_logits: Float[Tensor, " minibatch_size seq_len d_vocab"]


class ReplayMemory:
    def __init__(
        self,
        args: RLHFArgs,
        sample_ids: Float[Tensor, " batch_size seq_len"],
        logprobs: Float[Tensor, " batch_size gen_len"],
        advantages: Float[Tensor, " batch_size gen_len"],
        values: Float[Tensor, " batch_size seq_len"],
        ref_logits: Float[Tensor, " batch_size seq_len d_vocab"],
    ):
        assert ref_logits.ndim == 3
        assert ref_logits.shape[0] == args.batch_size
        assert sample_ids.shape == values.shape == ref_logits.shape[:2]
        assert advantages.shape == logprobs.shape == (args.batch_size, args.gen_len)

        self.args = args
        self.sample_ids = sample_ids
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values
        self.ref_logits = ref_logits

    def get_minibatches(self) -> list[ReplayMinibatch]:
        minibatches = []
        returns = self.advantages + self.values[:, -self.args.gen_len - 1 : -1]

        for _ in range(self.args.batches_per_learning_phase):
            for indices in t.randperm(self.args.batch_size).reshape(self.args.num_minibatches, -1):
                minibatches.append(
                    ReplayMinibatch(
                        sample_ids=self.sample_ids[indices],
                        logprobs=self.logprobs[indices],
                        advantages=self.advantages[indices],
                        returns=returns[indices],
                        ref_logits=self.ref_logits[indices],
                    )
                )
        return minibatches


# %%


def calc_kl_penalty(
    logits: Float[Tensor, "minibatch_size gen_len d_vocab"],
    ref_logits: Float[Tensor, "minibatch_size gen_len d_vocab"],
    kl_coef: float,
    gen_len: int,
) -> Float[Tensor, ""]:
    assert (
        logits.shape[1] == ref_logits.shape[1] == gen_len
    ), "Should pass generated tokens only"

    ref_logprobs = ref_logits.log_softmax(-1)
    logprobs = logits.log_softmax(-1)
    probs = logprobs.exp()

    kl_div = (probs * (logprobs - ref_logprobs)).sum(-1)
    return kl_coef * kl_div.mean()


# if MAIN:
    # tests.test_calc_kl_penalty(calc_kl_penalty)
    # tests.test_calc_kl_penalty_stability(calc_kl_penalty)

# %%


def calc_entropy_bonus(
    logits: Float[Tensor, "minibatch_size gen_len d_vocab"], ent_coef: float, gen_len: int
) -> Float[Tensor, ""]:
    assert logits.shape[1] == gen_len, "Pass logits before generated tokens only"
    logprobs = logits.log_softmax(dim=-1)
    probs = logprobs.exp()
    entropy = -(probs * logprobs).sum(dim=-1)
    return ent_coef * entropy.mean()


# if MAIN:
    # tests.test_calc_entropy_bonus(calc_entropy_bonus)
    # tests.test_calc_entropy_bonus_stability(calc_entropy_bonus)

# %%


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size gen_len"],
    mb_returns: Float[Tensor, "minibatch_size gen_len"],
    vf_coef: float,
    gen_len: int,
) -> Float[Tensor, ""]:
    assert values.shape[1] == gen_len
    assert mb_returns.shape[1] == gen_len
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()


def calc_clipped_surrogate_objective(
    logprobs: Float[Tensor, "minibatch_size gen_len"],
    mb_logprobs: Float[Tensor, "minibatch_size gen_len"],
    mb_advantages: Float[Tensor, "minibatch_size gen_len"],
    clip_coef: float,
    gen_len: int,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    assert (
        logprobs.shape[1] == mb_logprobs.shape[1] == mb_advantages.shape[1] == gen_len
    ), "Pass generated tokens only"

    logits_diff = logprobs - mb_logprobs
    r_theta = t.exp(logits_diff)
    mb_advantages = normalize_reward(mb_advantages, eps)
    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1 - clip_coef, 1 + clip_coef) * mb_advantages
    return t.minimum(non_clipped, clipped).mean()


# %%


def get_logprobs(
    logits: Float[Tensor, "batch seq_len vocab"],
    tokens: Int[Tensor, "batch seq_len"],
    prefix_len: int | None = None,
) -> Float[Tensor, "batch gen_len"]:
    if prefix_len is not None:
        logits = logits[:, prefix_len - 1 :]
        tokens = tokens[:, prefix_len - 1 :]
    logprobs = logits.log_softmax(-1)
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


# if MAIN:
    # tests.test_get_logprobs(get_logprobs)

# %%


def get_optimizer(
    model: HookedTransformerWithValueHead, base_lr: float, head_lr: float
) -> t.optim.Optimizer:
    return t.optim.AdamW(
        [
            {"params": model.get_base_model_trainable_params(), "lr": base_lr},
            {"params": model.get_value_head_params(), "lr": head_lr},
        ],
        maximize=True,
    )


# if MAIN:
    # tests.test_get_optimizer(get_optimizer, model)

# %%


def get_optimizer_and_scheduler(args: RLHFArgs, model: HookedTransformerWithValueHead):
    def lr_lambda(step):
        assert step <= args.total_phases
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            return 1 - (1 - args.final_scale) * (step - args.warmup_steps) / (
                args.total_phases - args.warmup_steps
            )

    optimizer = get_optimizer(model, args.base_lr, args.head_lr)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# %%


class RLHFTrainer:
    model: HookedTransformerWithValueHead
    ref_model: HookedTransformer
    memory: ReplayMemory  # we'll set this during rollout

    def __init__(self, args: RLHFArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = (
            f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        )

        self.model = (
            HookedTransformerWithValueHead.from_pretrained(args.base_model).to(device).train()
        )
        self.ref_model = HookedTransformer.from_pretrained(args.base_model).to(device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)

        # Prepare model+optimizer with accelerate (DDP-ready, bf16 autocast)
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        # keep ref_model unwrapped (inference-only)
        self.ref_model.to(device).eval()

        self.prefix_len = len(
            self.model.to_str_tokens(self.args.prefix, prepend_bos=self.args.prepend_bos)
        )

    def compute_rlhf_objective(self, minibatch: ReplayMinibatch):
        gen_len_slice = slice(-self.args.gen_len - 1, -1)

        with accelerator.autocast():
            logits, values = self.model.forward_with_value_head(minibatch.sample_ids)
            logprobs = get_logprobs(logits, minibatch.sample_ids, self.prefix_len)

            clipped_surrogate_objective = calc_clipped_surrogate_objective(
                logprobs,
                minibatch.logprobs,
                minibatch.advantages,
                self.args.clip_coef,
                self.args.gen_len,
            )
            value_loss = calc_value_function_loss(
                values[:, gen_len_slice], minibatch.returns, self.args.vf_coef, self.args.gen_len
            )
            entropy_bonus = calc_entropy_bonus(
                logits[:, gen_len_slice], self.args.ent_coef, self.args.gen_len
            )
            kl_penalty = calc_kl_penalty(
                logits[:, gen_len_slice],
                minibatch.ref_logits[:, gen_len_slice],
                self.args.kl_coef,
                self.args.gen_len,
            )

            ppo_objective_fn = clipped_surrogate_objective - value_loss + entropy_bonus
            total_objective_function = ppo_objective_fn - kl_penalty

        with t.inference_mode():
            logratio = logprobs - minibatch.logprobs
            ratio = logratio.exp()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if self.args.use_wandb and accelerator.is_main_process:
            wandb.log(
                dict(
                    total_steps=self.step,
                    lr=self.scheduler.get_last_lr()[0],
                    clipped_surrogate_objective=clipped_surrogate_objective.item(),
                    clipfrac=np.mean(clipfracs),
                    value_loss=value_loss.item(),
                    values=values.mean().item(),
                    entropy_bonus=entropy_bonus.item(),
                    kl_penalty=kl_penalty.item(),
                ),
                step=self.step,
            )

        return total_objective_function

    def rollout_phase(self) -> ReplayMemory:
        sample_ids, samples = get_samples(
            self.model,
            prompt=self.args.prefix,
            batch_size=self.args.batch_size,
            gen_len=self.args.gen_len,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            prepend_bos=self.args.prepend_bos,
            verbose=False,
        )

        with t.inference_mode():
            with accelerator.autocast():
                logits, values = self.model.forward_with_value_head(sample_ids)
                ref_logits = self.ref_model(sample_ids)

        logprobs = get_logprobs(logits, sample_ids, self.prefix_len)

        rewards = self.args.reward_fn(samples)
        rewards_mean = rewards.mean().item()
        rewards_normed = normalize_reward(rewards) if self.args.normalize_reward else rewards

        advantages = compute_advantages(values, rewards_normed, self.prefix_len)

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.log({"mean_reward": rewards_mean}, step=self.step)

        n_log_samples = min(3, self.args.batch_size)
        ref_logprobs = get_logprobs(
            ref_logits[:n_log_samples], sample_ids[:n_log_samples], self.prefix_len
        ).sum(-1)
        headers = ["Reward", "Ref logprobs", "Sample"]
        table_data = [
            [f"{r:.2f}", f"{lp:.2f}", repr(s)]
            for r, lp, s in zip(rewards.tolist(), ref_logprobs, samples)
        ]
        if accelerator.is_main_process:
            table = tabulate(
                table_data, headers, tablefmt="simple_grid", maxcolwidths=[None, None, 90]
            )
            print(
                f"Phase {self.phase + 1:03}/{self.args.total_phases:03}, Mean reward: {rewards_mean:.4f}\n{table}\n"
            )

        return ReplayMemory(
            args=self.args,
            sample_ids=sample_ids,
            logprobs=logprobs,
            advantages=advantages,
            values=values,
            ref_logits=ref_logits,
        )

    def learning_phase(self, memory: ReplayMemory) -> float:
        loss = 0.0
        minibatches = memory.get_minibatches()

        for minibatch in minibatches:
            self.optimizer.zero_grad(set_to_none=True)
            total_objective_function = self.compute_rlhf_objective(minibatch)
            accelerator.backward(total_objective_function)
            accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
            self.optimizer.step()
            self.step += 1
            loss += total_objective_function.item()

        loss /= len(minibatches)
        self.scheduler.step()
        return loss

    def train(self) -> None:
        self.step = 0
        self.samples = []

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                config=self.args,
            )
        runner = tqdm(range(self.args.total_phases), disable=not accelerator.is_main_process)
        for self.phase in runner:
            memory = self.rollout_phase()
            loss = self.learning_phase(memory)
            if accelerator.is_main_process:
                runner.set_description(f"Loss: {loss:.4f}")

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.finish()


# %%

# Testing your setup: kl_coef=0.0 (see drop-down above the previous code block for explanation)
if MAIN and RUN_BASE_RLHF:
    args = RLHFArgs(
        use_wandb=False,
        kl_coef=0.0,
        total_phases=30,
        warmup_steps=0,
        reward_fn=reward_fn_char_count,
    )
    trainer = RLHFTrainer(args)
    trainer.train()
elif MAIN and accelerator.is_main_process:
    print(f"{RUN_BASE_RLHF=}, skipping test run")

# %%

if MAIN and RUN_BASE_RLHF:
    args = RLHFArgs(use_wandb=False, reward_fn=reward_fn_char_count)
    trainer = RLHFTrainer(args)
    trainer.train()
elif MAIN and accelerator.is_main_process:
    print(f"{RUN_BASE_RLHF=}, skipping test run")

# %%

from transformers import AutoModelForSequenceClassification, AutoTokenizer

if MAIN and RUN_BASE_RLHF:
    assert not LOW_GPU_MEM, "You will need more memory to use the imdb reward model."
    cls_model = (
        AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
        .half()
        .to(device)
    )
    cls_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
elif MAIN and accelerator.is_main_process:
    print(f"{RUN_BASE_RLHF=}, skipping imdb reward model")


@t.no_grad()
def reward_fn_sentiment_imdb(
    gen_sample: list[str], direction: Literal["pos", "neg"] = "pos"
) -> Float[Tensor, " batch"]:
    assert direction in ["pos", "neg"], "direction should be either 'pos' or 'neg'"

    tokens = cls_tokenizer(gen_sample, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ].to(device)
    logits = cls_model(tokens).logits
    positive_cls = logits.softmax(dim=-1)[:, 1 if (direction == "pos") else 0]
    return positive_cls.to(device)


# ============= Start of Judge integration ====================================

# === Judge reward helper (OpenRouter, async, parallel) =======================
JUDGE_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "cognitivecomputations/dolphin3.0-mistral-24b"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
JUDGE_MAX_TOKENS = 50
JUDGE_TEMPERATURE = 0.0  # deterministic likelihoods

class AsyncJudge:
    def __init__(
        self,
        system_prompt: str,
        model: str = JUDGE_MODEL,
        endpoint: str = JUDGE_ENDPOINT,
        api_key_env: str = OPENROUTER_API_KEY_ENV,
        concurrency: int = 16,
        timeout: float = 30.0,
        use_mean: bool = True,  # mean per-token (RLOO is length-neutral)
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.endpoint = endpoint
        self.use_mean = use_mean
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} not set")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout

    def _payload(self, user_prompt: str, assistant_response: str):
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens": JUDGE_MAX_TOKENS,
            "logprobs": True,
            "top_logprobs": 0,
        }

    def _extract(self, data: dict) -> float:
        try:
            choices = data.get("choices") or []
            lp = choices[0].get("logprobs") if choices else None
            if lp and "content" in lp:
                vals = [float(t["logprob"]) for t in lp["content"] if "logprob" in t]
                vals = [v for v in vals if math.isfinite(v)]
                if not vals:
                    return -1.0
                return float(sum(vals) / len(vals)) if self.use_mean else float(sum(vals))
            if lp and "sum" in lp and isinstance(lp["sum"], (int, float)):
                return float(lp["sum"])
        except Exception:
            pass
        return -1.0

    async def _score_one(self, client: httpx.AsyncClient, prompt: str, response: str) -> float:
        body = self._payload(prompt, response)
        async with self.semaphore:
            try:
                r = await client.post(self.endpoint, headers=self.headers, json=body)
                r.raise_for_status()
                return self._extract(r.json())
            except Exception as e:
                print(f"[judge] request error: {repr(e)}", file=sys.stderr)
                return -1.0

    async def score_batch(self, prompts: list[str], responses: list[str]) -> list[float]:
        assert len(prompts) == len(responses)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._score_one(client, prompts[i], responses[i]) for i in range(len(prompts))]
            return await asyncio.gather(*tasks)

@t.no_grad()
def judge_reward_tensor(
    prefix_prompt: str,
    continuations: list[str],
    judge_system_prompt: str,
    use_mean: bool = True,
    concurrency: int = 16,
) -> t.FloatTensor:
    judge = AsyncJudge(
        system_prompt=judge_system_prompt,
        use_mean=use_mean,
        concurrency=concurrency,
    )
    prompts = [prefix_prompt] * len(continuations)
    rewards = asyncio.run(judge.score_batch(prompts, continuations))
    return t.tensor(rewards, dtype=t.float32, device=device)


# ============= End of Judge integration ========================================


if MAIN and RUN_BASE_RLHF:
    samples = [
        "Just finished watching this movie for maybe the 7th or 8th time, picked it up one night previously viewed at Blockbuster and absolutely loved it, I've shown it to 4 people so far and they have enjoyed it as well.",
        "This was the most original movie I've seen in years. If you like unique thrillers that are influenced by film noir, then this is just the right cure for all of those Hollywood summer blockbusters clogging the theaters these days.",
        "I can't believe that those praising this movie herein aren't thinking of some other film.",
        "This film seemed way too long even at only 75 minutes.",
        "Really, I can't believe that I spent $5 on this movie. I am a huge zombie fanatic and thought the movie might be really good. It had zombies in it right? Was I wrong!",
    ]
    classes = ["pos", "pos", "neg", "neg", "neg"]

    reward_fn = partial(reward_fn_sentiment_imdb, direction="pos")
    sentiment = reward_fn(samples).tolist()

    table = Table(
        "Sample",
        "Classification",
        "Sentiment",
        title="Demo of `reward_fn_sentiment_imdb`",
        show_lines=True,
    )
    for sample, cls, sent in zip(samples, classes, sentiment):
        table.add_row(repr(sample), cls, f"{sent:.4f}")
    if accelerator.is_main_process:
        rprint(table)

# %%


class Lora(nn.Module):
    """
    Basic LoRA module: f(x) = (x @ A) @ B * lora_alpha / rank
    """

    A: nn.Parameter
    B: nn.Parameter

    def __init__(
        self,
        d_in: int = 768,
        d_out: int = 768,
        rank: int = 4,
        lora_alpha: float = 32,
        n_inst: Optional[int] = None,
        dtype: t.dtype | None = None,
    ):
        super().__init__()
        self.rank = rank
        self.d_in = d_in
        self.d_out = d_out
        self.n_inst = 1 if n_inst is None else n_inst
        self.lora_alpha = lora_alpha
        self.dtype = dtype

        self.A = nn.Parameter(t.empty(self.n_inst, d_in, rank, dtype=dtype))
        self.B = nn.Parameter(t.zeros(self.n_inst, rank, d_out, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: Float[Tensor, "... inst d_in"]) -> Float[Tensor, "... inst d_out"]:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        assert (
            x.shape[-2] == self.n_inst or x.shape[-2] == 1
        ), f"Expected inst dim {self.n_inst} or 1, got {x.shape[-2]}. (input shape was {x.shape=})"
        tmp = einops.einsum(x, self.A, "... inst d_in, inst d_in rank -> ... inst rank")
        out = einops.einsum(tmp, self.B, "... inst rank, inst rank d_out -> ... inst d_out")
        return out * self.lora_alpha / self.rank


# if MAIN:
#     model = HookedTransformer.from_pretrained("pythia-14m")
#     tests_lora.testing_lora(Lora)

# %%


class LoraHooks(nn.Module):
    """
    LoRA hooks for attention layers
    """

    lora_q: Lora
    lora_k: Lora
    lora_v: Lora
    lora_o: Lora
    cache_qkv_in: Float[Tensor, "batch pos d_model"] = None
    cache_z: Float[Tensor, "batch pos n_heads d_head"] = None

    def __init__(
        self,
        layer_idx: int,
        cfg: HookedTransformerConfig,
        lora_alpha: float = 32,
        rank: int = 4,
        dtype: t.dtype = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dtype = dtype

        self.n_qo_heads = n_qo_heads = cfg.n_heads
        self.n_kv_heads = n_kv_heads = (
            cfg.n_key_value_heads if cfg.n_key_value_heads is not None else cfg.n_heads
        )
        d_model, d_head = cfg.d_model, cfg.d_head

        self.lora_q = Lora(
            d_model, d_head, n_inst=n_qo_heads, rank=rank, lora_alpha=lora_alpha, dtype=dtype
        )
        self.lora_k = Lora(
            d_model, d_head, n_inst=n_kv_heads, rank=rank, lora_alpha=lora_alpha, dtype=dtype
        )
        self.lora_v = Lora(
            d_model, d_head, n_inst=n_kv_heads, rank=rank, lora_alpha=lora_alpha, dtype=dtype
        )
        self.lora_o = Lora(
            d_head, d_model, n_inst=n_qo_heads, rank=rank, lora_alpha=lora_alpha, dtype=dtype
        )

    def store_hook_attn_normalized(
        self, normalized: Float[Tensor, "batch pos d_model"], hook: HookPoint
    ) -> None:
        self.cache_qkv_in = normalized

    def store_hook_z(self, z: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint) -> None:
        self.cache_z = z

    def list_fwd_hooks(self) -> list[tuple[str, Callable]]:
        fwd_hooks = []
        fwd_hooks.append(
            (f"blocks.{self.layer_idx}.ln1.hook_normalized", self.store_hook_attn_normalized)
        )
        fwd_hooks.append((f"blocks.{self.layer_idx}.attn.hook_q", self.lora_hook_qkv))
        fwd_hooks.append((f"blocks.{self.layer_idx}.attn.hook_k", self.lora_hook_qkv))
        fwd_hooks.append((f"blocks.{self.layer_idx}.attn.hook_v", self.lora_hook_qkv))
        fwd_hooks.append((f"blocks.{self.layer_idx}.attn.hook_z", self.store_hook_z))
        fwd_hooks.append((f"blocks.{self.layer_idx}.hook_attn_out", self.lora_hook_out))
        return fwd_hooks

    def lora_hook_qkv(
        self, qkv_hook_out: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint
    ) -> Float[Tensor, "batch pos n_heads d_head"]:
        hook_location = hook.name.split(".")[-1]
        qkv_in = self.cache_qkv_in
        qkv_in_repeated = einops.repeat(
            qkv_in, "batch pos d_model -> batch pos n_inst d_model", n_inst=1
        )
        if hook_location == "hook_q":
            return qkv_hook_out + self.lora_q(qkv_in_repeated)
        elif hook_location == "hook_k":
            return qkv_hook_out + self.lora_k(qkv_in_repeated)
        elif hook_location == "hook_v":
            return qkv_hook_out + self.lora_v(qkv_in_repeated)
        else:
            raise ValueError(f"Invalid hook location: {hook_location}")

    def lora_hook_out(
        self, attn_out: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint
    ) -> Float[Tensor, "batch pos n_heads d_head"]:
        lora_result = self.lora_o(self.cache_z)
        lora_attn_out = einops.einsum(lora_result, "... n_heads d_model -> ... d_model")
        return attn_out + lora_attn_out


# %%

# if MAIN:
#     tests_lora.testing_lora_hooks(LoraHooks)
#     tests_lora.testing_lora_hooks_qkv_dispatch_and_out(LoraHooks)
#     if accelerator.is_main_process:
#         print("All tests for LoraHooks passed!")

# %%


class TransformerWithValueHeadLora(HookedTransformerWithValueHead):
    lora: nn.ModuleList
    lora_fwd_hooks: list[tuple[str, Callable]]
    dtype: t.dtype
    device: t.device
    use_value_head: bool

    def base_model_params(self):
        return (
            p
            for name, p in self.named_parameters()
            if "value_head" not in name and "lora" not in name
        )

    def lora_params(self):
        return self.lora.parameters()

    # we use these for compatibility with get_optimizer_and_scheduler
    def get_base_model_trainable_params(self):
        return self.lora_params()

    def get_value_head_params(self):
        return (p for name, p in self.named_parameters() if "value_head" in name)

    @classmethod
    def from_pretrained(cls, *args, lora_alpha: float = 32, rank: int = 4, **kwargs):
        model = super(TransformerWithValueHeadLora, cls).from_pretrained(*args, **kwargs)
        model.setup_lora(lora_alpha=lora_alpha, rank=rank, **kwargs)

        for param in model.base_model_params():
            param.requires_grad = False

        return model

    def setup_lora(self, lora_alpha: float = 32, rank: int = 4, **kwargs):
        self.lora = nn.ModuleList(
            [
                LoraHooks(layer_idx, self.cfg, lora_alpha, rank)
                for layer_idx in range(len(self.blocks))
            ]
        ).to(device)

        self.lora_fwd_hooks = []
        for layer_idx in range(len(self.blocks)):
            self.lora_fwd_hooks.extend(self.lora[layer_idx].list_fwd_hooks())

    @property
    def fwd_hooks(self):
        return self.lora_fwd_hooks + [self.value_head_hook]

    def forward_with_value_head(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> tuple[Float[Tensor, "batch seq d_vocab"], Float[Tensor, "batch seq"]]:
        with self.hooks(fwd_hooks=self.fwd_hooks):
            logits = self.forward(tokens)
        value = self.value_head_output
        return logits, value

    @t.no_grad()
    def generate(self, tokens: Int[Tensor, "batch seq"], **kwargs) -> Int[Tensor, "batch seq"]:
        with self.hooks(fwd_hooks=self.lora_fwd_hooks):
            gen_tokens = super().generate(tokens, **kwargs)
        return gen_tokens


# if MAIN:
#     model = TransformerWithValueHeadLora.from_pretrained("pythia-14m").to(device)
#     tests_lora.test_lora_fwd_hooks_list(model)
#     tests_lora.test_lora_model_forward_methods(model)
#     if accelerator.is_main_process:
#         print("All tests for TransformerWithValueHeadLora passed!")

# %%


@dataclass
class RLHFArgsLora(RLHFArgs):
    lora_rank: int = 4
    lora_alpha: float = 32
    dtype: t.dtype = None


class RLHFTrainerLora(RLHFTrainer):
    model: TransformerWithValueHeadLora
    memory: ReplayMemory

    def __init__(self, args: RLHFArgsLora):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = (
            f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        )

        self.model = TransformerWithValueHeadLora.from_pretrained(
            args.base_model, lora_alpha=args.lora_alpha, rank=args.lora_rank
        )
        self.model.to(device).train()
        self.ref_model = self.model  # same weights, inference-only ref if desired

        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)

        # prepare trainables
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        # ref_model stays eval-only
        self.ref_model.to(device).eval()

        self.prefix_len = len(
            self.model.to_str_tokens(self.args.prefix, prepend_bos=self.args.prepend_bos)
        )


# %%

# if MAIN:
#     if accelerator.is_main_process:
#         print("Training LoRA model RLHF (example setup)")
#     lora_args = RLHFArgsLora(
#         use_wandb=False,
#         kl_coef=0.0,
#         total_phases=2,
#         warmup_steps=0,
#         reward_fn=reward_fn_char_count,
#         base_lr=1e-3,
#         batch_size=8,
#         num_minibatches=2,
#         gen_len=8,
#     )
#     lora_trainer = RLHFTrainerLora(lora_args)
#     lora_trainer.train()  # tiny smoke test

# %%


class TransformerWithLora(TransformerWithValueHeadLora):
    "We don't need the value head for training with GRPO"

    lora: nn.ModuleList
    lora_fwd_hooks: list[tuple[str, Callable]]
    dtype: t.dtype
    device: t.device

    def get_value_head_params(self):
        return iter([])  # no value head parameters

    @classmethod
    def from_pretrained(cls, *args, lora_alpha: float = 32, rank: int = 4, **kwargs):
        model = super(TransformerWithLora, cls).from_pretrained(
            *args, use_value_head=False, **kwargs
        )
        model.value_head_output = None
        return model

    @property
    def fwd_hooks(self):
        return self.lora_fwd_hooks  # no value head hook

    def forward_with_value_head(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> tuple[Float[Tensor, "batch seq d_vocab"], Float[Tensor, "batch seq"]]:
        logits, value = super().forward_with_value_head(tokens)
        assert value is None, "Value head got run somehow?"
        return logits


# %%
@dataclass
class GrpoArgs(RLHFArgs):
    lora_rank: int = 4
    lora_alpha: float = 32


class GrpoTrainer(RLHFTrainer):
    model: TransformerWithLora
    memory: ReplayMemory

    def __init__(self, args: RLHFArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = (
            f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        )
        self.model = TransformerWithLora.from_pretrained(args.base_model).to(device).train()
        self.ref_model = self.model
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)

        # accelerate prepare
        self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        self.ref_model.to(device).eval()

        self.prefix_len = len(
            self.model.to_str_tokens(self.args.prefix, prepend_bos=self.args.prepend_bos)
        )
        # ========= Load judge system prompt once ============
        with open("judge_prompt.txt", "r", encoding="utf-8") as f:
            self.judge_system_prompt = f.read()
        # ====================================================

    def compute_rlhf_objective(self, minibatch: ReplayMinibatch):
        gen_len_slice = slice(-self.args.gen_len - 1, -1)

        with accelerator.autocast():
            logits = self.model.forward_with_value_head(minibatch.sample_ids)
            logprobs = get_logprobs(logits, minibatch.sample_ids, self.prefix_len)
            logprobs_gen = logprobs[:, gen_len_slice]
            advantages = minibatch.advantages[:, gen_len_slice]
            rloo_objective = (advantages * logprobs_gen).sum(dim=-1).mean()

        if self.args.use_wandb and accelerator.is_main_process:
            with t.inference_mode():
                seq_logprob = logprobs_gen.sum(dim=-1)
                wandb.log(
                    dict(
                        total_steps=self.step,
                        lr=self.scheduler.get_last_lr()[0],
                        rloo_objective=rloo_objective.item(),
                        mean_seq_logprob=seq_logprob.mean().item(),
                    ),
                    step=self.step,
                )
        return rloo_objective

    def rollout_phase(self) -> ReplayMemory:
        sample_ids, samples = get_samples(
            self.model,
            prompt=self.args.prefix,
            batch_size=self.args.batch_size,
            gen_len=self.args.gen_len,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            prepend_bos=self.args.prepend_bos,
        )
        with t.inference_mode():
            with accelerator.autocast():
                logits = self.model.forward_with_value_head(sample_ids)
                ref_logits = self.ref_model(sample_ids)

        logprobs = get_logprobs(logits, sample_ids, self.prefix_len)

        # --- Build continuations from the last G tokens ---
        gen_only = sample_ids[:, -self.args.gen_len:]
        continuations = [self.model.to_string(row.unsqueeze(0))[0] for row in gen_only]

        # --- Judge rewards (mean per-token logprob; RLOO baseline) ---
        rewards = judge_reward_tensor(
            prefix_prompt=self.args.prefix,
            continuations=continuations,
            judge_system_prompt=self.judge_system_prompt,
            use_mean=True,
            concurrency=16,
        )
        rewards_mean = rewards.mean().item()

        # --- RLOO baseline: subtract batch mean; then broadcast across tokens ---
        baseline = rewards.mean()
        advantages_seq = rewards - baseline  # [B]
        advantages = einops.repeat(advantages_seq, "b -> b g", g=logprobs.shape[1])
        values = einops.repeat(advantages_seq, "b -> b s", s=sample_ids.shape[1])  # dummy, shape-only

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.log({"mean_reward": rewards_mean, "baseline": baseline.item()}, step=self.step)

        n_log_samples = min(5, self.args.batch_size)
        ref_logprobs = get_logprobs(
            ref_logits[:n_log_samples], sample_ids[:n_log_samples], self.prefix_len
        ).sum(-1)
        headers = ["Reward", "Ref logprobs", "Sample"]
        table_data = [
            [f"{r:.3f}", f"{lp:.2f}", repr(s)]
            for r, lp, s in zip(rewards.tolist(), ref_logprobs, samples)
        ]
        if accelerator.is_main_process:
            table = tabulate(table_data, headers, tablefmt="simple_grid", maxcolwidths=[None, None, 90])
            print(
                f"Phase {self.phase+1:03}/{self.args.total_phases:03}, Mean reward: {rewards_mean:.4f}\n{table}\n"
            )

        return ReplayMemory(
            args=self.args,
            sample_ids=sample_ids,
            logprobs=logprobs,
            advantages=advantages,
            values=values,
            ref_logits=ref_logits,
        )


# %%

# if MAIN:
#     if accelerator.is_main_process:
#         print("Training GRPO model (example setup)")
#     grpo_args = GrpoArgs(
#         use_wandb=False,
#         kl_coef=2.5,
#         total_phases=30,
#         warmup_steps=0,
#         reward_fn=reward_fn_char_count,
#         base_lr=1e-3,
#         gen_len=16,
#     )
#     grpo_trainer = GrpoTrainer(grpo_args)
#     grpo_trainer.train()

if MAIN:
    if accelerator.is_main_process:
        print("Training GRPO model (judge smoketest)")
    # grpo_args = GrpoArgs(
    #     use_wandb=False,
    #     base_model="pythia-14m",   # small & fast
    #     total_phases=1,            # single rollout+learn step
    #     warmup_steps=0,
    #     batch_size=4,
    #     num_minibatches=1,
    #     gen_len=8,                 # short continuation
    #     temperature=0.7,
    #     top_k=20,
    #     prepend_bos=False,         # keeps the prompt short & clean
    # )
    grpo_args = GrpoArgs(
        use_wandb=False,
        base_model="meta-llama/Llama-3.1-8B-Instruct",     # or "llama-7b-hf"
        total_phases=1,
        warmup_steps=0,
        batch_size=4,
        num_minibatches=1,
        gen_len=8,
        temperature=0.7,
        top_k=20,
        prepend_bos=True,               # LLaMA tokenization plays nicer with BOS
)
    grpo_trainer = GrpoTrainer(grpo_args)
    grpo_trainer.train()





# %%
# TODO-s:

# add steering vector
# add new reward as logprobs from api 
# no reward normalization
# clean up
# change model
# args change
# faster model inference
# faster api calls (parallel)
# new logging interface?
# check w david that the objective function is correct
