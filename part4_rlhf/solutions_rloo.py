# %%
import time
from dataclasses import dataclass
from typing import Callable
from tqdm import tqdm
import math
import os
import sys
import asyncio
import httpx

import einops
import torch as t
import torch.nn as nn
import wandb
from eindex import eindex
from jaxtyping import Float, Int
from tabulate import tabulate
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from accelerate import Accelerator

# Enable TF32 fast kernels
if t.cuda.is_available():
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    t.backends.cudnn.benchmark = True

accelerator = Accelerator(
    mixed_precision=("bf16" if t.cuda.is_available() else "no"),
    gradient_accumulation_steps=1,  # keeping the replay/minibatch logic defined below
)
device = accelerator.device

# %%
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "cognitivecomputations/dolphin3.0-mistral-24b"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
SEQ_LEN = 120
JUDGE_TEMPERATURE = 0.0

# %%


@dataclass
class RLHFArgs:
    # Basic / global
    seed: int = 1

    # Duration of different phases
    use_wandb: bool = False
    wandb_project_name: str = "sycophancy_steering"
    wandb_entity: str | None = None

    # Duration of different phases
    total_phases: int = 100
    batch_size: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 2

    # Basic / global
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

    def __post_init__(self):
        assert (
            self.total_phases > self.warmup_steps
        ), "total_phases must be greater than warmup_steps"
        assert (
            self.batch_size % self.num_minibatches == 0
        ), "batch_size should be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches

# %%


class SycophancyJudge:
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
            "max_tokens": SEQ_LEN,
            "logprobs": True,
            "top_logprobs": 0,
        }

    def _extract(self, data: dict) -> float:
        try:
            choices = data.get("choices") or []
            lp = choices[0].get("logprobs") if choices else None
            if lp and "content" in lp:
                vals = [float(t["logprob"])
                        for t in lp["content"] if "logprob" in t]
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
            tasks = [self._score_one(client, prompts[i], responses[i])
                     for i in range(len(prompts))]
            return await asyncio.gather(*tasks)


@t.no_grad()
def reward_fn_sycophancy_judge(
    prefix_prompt: str,
    continuations: list[str],
    judge_system_prompt: str,
    use_mean: bool = True,
    concurrency: int = 16,
) -> t.FloatTensor:
    judge = SycophancyJudge(
        system_prompt=judge_system_prompt,
        use_mean=use_mean,
        concurrency=concurrency,
    )
    prompts = [prefix_prompt] * len(continuations)
    rewards = asyncio.run(judge.score_batch(prompts, continuations))
    return t.tensor(rewards, dtype=t.float32, device=device)


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
    Generates samples from the model, which will be fed into the reward model and evaluated.

    Inputs:
        model: the transformer to generate samples from
        prompt: the initial prompt fed into the model
        batch_size: the number of samples to generate
        gen_len: the length of the generated samples (i.e. the number of *new* tokens to generate)
        temperature: the temp of the sampling distribution (higher means more random completions)
        top_k: the topk parameter of sampling (higher means a wider variety of possible completions)

    Returns:
        sample_ids: the token ids of the generated samples (including initial prompt)
        samples: the generated samples (including initial prompt)
    """

    # Convert our prompt into tokens
    input_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    input_ids = einops.repeat(
        input_ids, "1 seq -> batch seq", batch=batch_size)

    # Generate samples
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


@dataclass
class ReplayMinibatch:
    """
    Samples from the replay memory.
    """

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
        """
        Initializes the replay memory, with all the data generated from the rollout phase at once.

        The advantages are (batch_size, gen_len) because we only compute advantages for the generated
        tokens. The other tensors, except logprobs, uses seq_len instead of gen_len because they are
        computed for all tokens.
        """

        assert ref_logits.ndim == 3
        assert ref_logits.shape[0] == args.batch_size
        assert sample_ids.shape == values.shape == ref_logits.shape[:2]
        assert advantages.shape == logprobs.shape == (
            args.batch_size, args.gen_len)

        self.args = args
        self.sample_ids = sample_ids
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values
        self.ref_logits = ref_logits

    def get_minibatches(self) -> list[ReplayMinibatch]:
        """
        Generates a list of minibatches by randomly sampling from the replay memory. Each sequence
        appears exactly `batches_per_learning_phase` times in total.
        """
        minibatches = []

        returns = self.advantages + self.values[:, -self.args.gen_len - 1: -1]

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


class SteeringVector(nn.Module):
    """
    A simple steering vector module that adds a learned vector to the input activations.
    Can be used to steer model behavior by adding to the residual stream.
    """

    vector: nn.Parameter  # (d_model,)

    def __init__(
        self,
        d_model: int,
        dtype: t.dtype | None = None,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype

        # Initialize steering vector
        self.vector = nn.Parameter(t.randn(d_model, dtype=dtype) * init_scale)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Adds the steering vector to the input.

        Args:
            x: Input tensor of shape (..., d_model)
        Returns:
            x + steering vector, same shape as input
        """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x + self.vector

# %%


class SteeringHooks(nn.Module):
    """
    Defines hooks for attaching a steering vector to a single transformer layer.
    The steering vector is added after RMSNorm (layer normalization).
    """

    vec_ln1: SteeringVector

    def __init__(
        self,
        layer_idx: int,
        cfg: HookedTransformerConfig,
        init_scale: float = 1.0,
        dtype: t.dtype = None,
    ):
        """
        Args:
            layer_idx: Layer index where to attach the steering vector
            cfg: Model configuration
            init_scale: Scale for initializing the steering vector
            dtype: Data type for the steering vector
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = dtype

        d_model = cfg.d_model

        # Create steering vector for this layer
        self.vec_ln1 = SteeringVector(
            d_model, dtype=dtype, init_scale=init_scale)

    def list_fwd_hooks(self) -> list[tuple[str, Callable]]:
        """
        Returns a list of hook_point names and functions for attaching the steering vector.
        The steering vector is added after RMSNorm in the specified layer.
        """
        fwd_hooks = []

        fwd_hooks.append((
            f"blocks.{self.layer_idx}.ln1.hook_normalized",
            self.vec_ln1
        ))

        return fwd_hooks

    def steering_hook_out(
        self, normalized: Float[Tensor, "batch pos d_model"], hook: HookPoint
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Hook function that adds the steering vector to the normalized activations.

        Args:
            normalized: Output from RMSNorm, shape (batch, pos, d_model)
            hook: HookPoint
        Returns:
            normalized + steering vector
        """
        return self.vec_ln1(normalized)

# %%


class HookedTransformerWithSteering(HookedTransformer):
    """
    HookedTransformer with steering vectors added after RMSNorm in specified layers.
    """

    steering_hooks: nn.ModuleList
    steering_fwd_hooks: list[tuple[str, Callable]]
    dtype: t.dtype
    device: t.device

    def base_model_params(self):
        return (
            p
            for name, p in self.named_parameters()
            if "steering" not in name
        )

    def steering_params(self):
        return self.steering_hooks.parameters()

    # we use these for compatibility with get_optimizer_and_scheduler
    def get_base_model_trainable_params(self):
        return self.steering_params()

    @classmethod
    def from_pretrained(cls, *args, layer_indices: list[int] = None, init_scale: float = 1.0, **kwargs):
        model = super(HookedTransformerWithSteering,
                      cls).from_pretrained(*args, **kwargs)
        model.setup_steering(layer_indices=layer_indices,
                             init_scale=init_scale, **kwargs)

        for param in model.base_model_params():
            param.requires_grad = False

        return model

    def setup_steering(self, layer_indices: list[int] = None, init_scale: float = 1.0, **kwargs):
        """
        Initializes steering vectors for the specified transformer layers.

        Args:
            layer_indices: List of layer indices where to attach steering vectors.
                          If None, attaches to all layers.
            init_scale: Scale for initializing steering vectors
        """

        if layer_indices is None:
            layer_indices = list(range(len(self.blocks)))

        # Create steering hooks for each specified layer
        self.steering_hooks = nn.ModuleList([
            SteeringHooks(layer_idx, self.cfg,
                          init_scale=init_scale, dtype=self.dtype)
            for layer_idx in layer_indices
        ]).to(device)

        # create list of all hooks for all layers
        self.steering_fwd_hooks = []
        for steering_hook in self.steering_hooks:
            self.steering_fwd_hooks.extend(steering_hook.list_fwd_hooks())

    @property
    def fwd_hooks(self):
        return self.steering_fwd_hooks

    def forward_with_steering(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq d_vocab"]:
        """
        Forward pass with steering vectors enabled.

        Args:
            tokens: Int[Tensor, "batch seq"]
                The input tokens to the transformer.
        Returns:
            logits: Float[Tensor, "batch seq d_vocab"]
                The logits of the transformer.
        """

        with self.hooks(fwd_hooks=self.fwd_hooks):
            logits = self.forward(tokens)
        return logits

    @t.no_grad()
    def generate(self, tokens: Int[Tensor, "batch seq"], **kwargs) -> Int[Tensor, "batch seq"]:
        """
        Override generate method to use steering hooks.

        Args:
            tokens: Int[Tensor, "batch seq"]
                The input tokens to the transformer.
            **kwargs:
                Additional keyword arguments to pass to the base class generate method.
        Returns:
            gen_tokens: Int[Tensor, "batch gen_len"]
                The generated tokens.
        """

        with self.hooks(fwd_hooks=self.steering_fwd_hooks):
            gen_tokens = super().generate(tokens, **kwargs)
        return gen_tokens


# %%

def get_logprobs(
    logits: Float[Tensor, "batch seq_len vocab"],
    tokens: Int[Tensor, "batch seq_len"],
    prefix_len: int | None = None,
) -> Float[Tensor, "batch gen_len"]:
    """
    Returns correct logprobs for the given logits and tokens, for all the tokens after the prefix
    tokens (which have length equal to `prefix_len`).

    If prefix_len = None then we return shape (batch, seq_len-1).
    If not, then we return shape (batch, seq_len-prefix_len) representing the predictions for all
    toks after the prefix.
    """
    # Slice our tensors based on prefix_len
    if prefix_len is not None:
        logits = logits[:, prefix_len - 1:]
        tokens = tokens[:, prefix_len - 1:]

    # Get logprobs
    logprobs = logits.log_softmax(-1)

    # We want to get elements `logprobs[b, s, tokens[b, s+1]]`, we do this using eindex as follows:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")

    return correct_logprobs

# %%


def get_optimizer(
    model: HookedTransformerWithSteering, base_lr: float, head_lr: float
) -> t.optim.Optimizer:
    """
    Returns an AdamW optimizer for the model, with the correct learning rates for the base and head.
    Make sure to use the HookedTransformerWithSteering wrapper methods for getting the parameters.
    """
    return t.optim.AdamW(
        [
            {"params": model.get_base_model_trainable_params(), "lr": base_lr},
            {"params": model.get_value_head_params(), "lr": head_lr},
        ],
        maximize=True,
    )


def get_optimizer_and_scheduler(args: RLHFArgs, model: HookedTransformerWithSteering):
    """
    Creates an AdamW optimizer and an LR scheduler that linearly warms up for `warmup_steps` steps,
    and then linearly decays to `final_scale` over the remaining steps.
    """

    def lr_lambda(step):
        assert (
            step <= args.total_phases
        ), f"Step = {step} should be less than total_phases = {args.total_phases}."
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
    model: HookedTransformerWithSteering
    ref_model: HookedTransformer
    memory: ReplayMemory  # we'll set this during rollout

    def __init__(self, args: RLHFArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = (
            f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        )

        self.model = (
            HookedTransformerWithSteering.from_pretrained(
                args.base_model,
                attn_implementation="flash_attention_2"
            ).to(device).train()
        )
        self.ref_model = HookedTransformer.from_pretrained(
            args.base_model,
            attn_implementation="flash_attention_2"
        ).to(device).eval()
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            self.args, self.model)
        self.prefix_len = len(
            self.model.to_str_tokens(
                self.args.prefix, prepend_bos=self.args.prepend_bos)
        )

    def learning_phase(self, memory: ReplayMemory) -> float:
        """
        Performs a learning step on `memory`. This involves the standard gradient descent steps
        (i.e. zeroing gradient, computing objective function, doing backprop, stepping optimizer).

        You should also remember the following:
            - Clipping grad norm to the value given in `self.args.max_grad_norm`
            - Incrementing `self.step` by 1 for each minibatch
            - Stepping the scheduler (once per calling of this function)

        Returns the average objective function value over the minibatches as a float for logging.
        """
        loss = 0
        minibatches = memory.get_minibatches()

        for minibatch in minibatches:
            self.optimizer.zero_grad()
            total_objective_function = self.compute_rlhf_objective(minibatch)
            total_objective_function.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.args.max_grad_norm)
            self.optimizer.step()
            self.step += 1
            loss += total_objective_function.item()

        loss /= len(minibatches)
        self.scheduler.step()
        return loss

    def train(self) -> None:
        """
        Performs a full training run.
        """
        self.step = 0
        self.samples = []

        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                config=self.args,
            )
        runner = tqdm(range(self.args.total_phases))
        for self.phase in runner:
            memory = self.rollout_phase()
            loss = self.learning_phase(memory)
            runner.set_description(f"Loss: {loss:.4f}")

        if self.args.use_wandb:
            wandb.finish()

# %%


@dataclass
class RLOOArgs(RLHFArgs):
    steering_layer_indices: list[int] = None  # None means all layers
    steering_init_scale: float = 1.0


class RLOOTrainer(RLHFTrainer):
    model: HookedTransformerWithSteering
    memory: ReplayMemory

    def __init__(self, args: RLOOArgs):

        t.manual_seed(args.seed)
        self.args = args
        self.run_name = (
            f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        )
        self.model = HookedTransformerWithSteering.from_pretrained(
            args.base_model,
            layer_indices=args.steering_layer_indices,
            init_scale=args.steering_init_scale
        ).to(device).train()
        self.ref_model = self.model
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            self.args, self.model)
        self.prefix_len = len(
            self.model.to_str_tokens(
                self.args.prefix, prepend_bos=self.args.prepend_bos)
        )

    def compute_rlhf_objective(self, minibatch: ReplayMinibatch):
        gen_len_slice = slice(-self.args.gen_len - 1, -1)

        logits, _ = self.model.forward_with_value_head(minibatch.sample_ids)

        logprobs = get_logprobs(logits, minibatch.sample_ids, self.prefix_len)
        logprobs_gen = logprobs[:, gen_len_slice]
        advantages = minibatch.advantages[:, gen_len_slice]
        rloo_objective = (advantages * logprobs_gen).sum(dim=-1).mean()
        if self.args.use_wandb:
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
            logits, values = self.model.forward_with_value_head(sample_ids)
            ref_logits = self.ref_model(sample_ids)
        logprobs = get_logprobs(logits, sample_ids, self.prefix_len)
        rewards = self.args.reward_fn(samples)
        rewards_mean = rewards.mean().item()

        baseline = rewards.mean()
        advantages = rewards - baseline
        if self.args.use_wandb:
            wandb.log({"mean_reward": rewards_mean,
                      "baseline": baseline.item()}, step=self.step)
        n_log_samples = min(5, self.args.batch_size)
        ref_logprobs = get_logprobs(
            ref_logits[:n_log_samples], sample_ids[:n_log_samples], self.prefix_len
        ).sum(-1)
        headers = ["Reward", "Ref logprobs", "Sample"]
        table_data = [
            [str(int(r)), f"{lp:.2f}", repr(s)]
            for r, lp, s in zip(rewards.tolist(), ref_logprobs, samples)
        ]
        table = tabulate(table_data, headers,
                         tablefmt="simple_grid", maxcolwidths=[None, None, 90])
        print(
            f"Phase {self.phase+1:03}/{self.args.total_phases:03}, Mean reward: {rewards_mean:.4f}\n{table}\n"
        )
        advantages = einops.repeat(advantages, "b -> b g", g=logprobs.shape[1])
        values = einops.repeat(advantages, "b -> b g", g=sample_ids.shape[1])
        return ReplayMemory(
            args=self.args,
            sample_ids=sample_ids,
            logprobs=logprobs,
            advantages=advantages,
            values=values,
            ref_logits=ref_logits,
        )

# %%


print("Training GRPO model (example setup)")
rloo_args = RLOOArgs(
    use_wandb=False,
    # kl_coef=2.5,
    total_phases=30,
    warmup_steps=0,
    reward_fn=reward_fn_sycophancy_judge,
    base_lr=1e-3,
    # batch_size=8,
    # num_minibatches=2,
    gen_len=16,
)
rloo_trainer = RLOOTrainer(rloo_args)
rloo_trainer.train()
