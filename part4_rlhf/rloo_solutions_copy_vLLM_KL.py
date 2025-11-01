import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import datetime
import math
import os
import sys
import asyncio
import httpx
import csv, json

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
from transformers import AutoTokenizer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from accelerate import Accelerator

# ========= Performance knobs =========
if t.cuda.is_available():
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    t.backends.cudnn.benchmark = True

accelerator = Accelerator(
    mixed_precision=("bf16" if t.cuda.is_available() else "no"),
    gradient_accumulation_steps=1,
)
device = accelerator.device

# ========= Models =========
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  #"pythia-14m" # "meta-llama/Llama-3.1-8B-Instruct"  # actor (TransformerLens)
JUDGE_BASE_URL = "http://localhost:8000/v1"      # vLLM server URL
JUDGE_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition" # "cognitivecomputations/dolphin3.0-mistral-24b"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1   # ask for no new tokens; if server rejects 0 we fallback to 1
SEQ_LEN = 120          # not used for generation; just a cap if needed

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
    total_phases: int = 5
    batch_size: int = 8
    num_minibatches: int = 2
    batches_per_learning_phase: int = 1

    # optimization
    base_lr: float = 1e-4
    head_lr: float = 1e-4   # unused here but kept for API compatibility
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    final_scale: float = 1.0

    # PPO terms (not all used in RLOO path)
    clip_coef: float = 0.2
    vf_coef: float = 0.15
    ent_coef: float = 0.001

    # actor sampling
    base_model: str = BASE_MODEL
    gen_len: int = 16
    temperature: float = 0.7
    top_k: int = 20
    prefix: str = "This is"
    prepend_bos: bool = True

    # kl / reward_fn fields kept for API compatibility
    kl_coef: float = 2.5
    reward_fn: Callable = lambda *args, **kwargs: 0.0

    def __post_init__(self):
        assert self.batch_size % self.num_minibatches == 0, "batch_size divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        assert self.total_phases > 0

# =========================
# Steering modules
# =========================
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

    # def list_fwd_hooks(self) -> list[tuple[str, Callable]]:
    #     return [
    #         (f"blocks.{self.layer_idx}.ln1.hook_normalized", self.vec_ln1)
    #     ]
    
    def list_fwd_hooks(self) -> list[tuple[str, Callable]]:
        return [
            (f"blocks.{self.layer_idx}.ln1.hook_normalized", self.steering_hook_out)
        ]

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


class HookedTransformerWithSteering(HookedTransformer):
    steering_hooks: nn.ModuleList
    steering_fwd_hooks: list[tuple[str, Callable]]
    dtype: t.dtype
    device: t.device

    def base_model_params(self):
        # everything except steering parameters
        return (p for name, p in self.named_parameters() if "steering" not in name)

    def steering_params(self):
        return self.steering_hooks.parameters()

    # required by get_optimizer (we only train steering)
    def get_base_model_trainable_params(self):
        return self.steering_params()

    # keep optimizer API happy
    def get_value_head_params(self):
        return iter([])

    @classmethod
    def from_pretrained(cls, *args, layer_indices: list[int] = None, init_scale: float = 1.0, **kwargs):
        model = super(HookedTransformerWithSteering, cls).from_pretrained(*args, **kwargs)
        model.setup_steering(layer_indices=layer_indices, init_scale=init_scale)
        # freeze base model weights
        for p in model.base_model_params():
            p.requires_grad = False
        return model

    def setup_steering(self, layer_indices: list[int] | int | None = None, init_scale: float = 1.0):
        # Accept int, list[int], or None
        if layer_indices is None:
            layer_indices = list(range(len(self.blocks)))
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        self.steering_hooks = nn.ModuleList([
            SteeringHooks(i, self.cfg, init_scale=init_scale, dtype=getattr(self, "dtype", None))
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
        with self.hooks(fwd_hooks=self.steering_fwd_hooks):
            gen_tokens = super().generate(tokens, **kwargs)
        return gen_tokens

# =========================
# Utilities
# =========================
def get_samples(
    model: HookedTransformer,
    prompts: list[str],                 # CHANGED: accept list
    gen_len: int = 15,
    temperature: float = 0.8,
    top_k: int = 15,
    prepend_bos: bool = True,
    **kwargs,
) -> tuple[Int[Tensor, "batch seq"], list[str]]:
    # TransformerLens supports list[str] for to_tokens
    input_ids = model.to_tokens(prompts, prepend_bos=prepend_bos)  # [B, S]
    output_ids = model.generate(
        input_ids,
        max_new_tokens=gen_len,
        # min_new_tokens=gen_len, # not needed as see below
        stop_at_eos=False, # as this is false, we will definitely generate gen_len tokens
        temperature=temperature,
        top_k=top_k,
        **kwargs,
    )
    samples = model.to_string(output_ids)
    return output_ids.clone(), samples


def get_logprobs(
    logits: Float[Tensor, "batch seq_len vocab"],
    tokens: Int[Tensor, "batch seq_len"],
    prefix_len: int | None = None,
) -> Float[Tensor, "batch gen_len"]:
    if prefix_len is not None:
        logits = logits[:, prefix_len - 1:]
        tokens = tokens[:, prefix_len - 1:]
    logprobs = logits.log_softmax(-1)
    # correct next-token logprobs for each position
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs

def get_optimizer(model: HookedTransformerWithSteering, base_lr: float, head_lr: float) -> t.optim.Optimizer:
    # We only train steering params
    return t.optim.AdamW(
        [{"params": model.get_base_model_trainable_params(), "lr": base_lr}],
        maximize=True,
    )

def get_optimizer_and_scheduler(args: RLHFArgs, model: HookedTransformerWithSteering):
    def lr_lambda(step):
        if args.warmup_steps == 0:
            return 1.0
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            return 1 - (1 - args.final_scale) * (step - args.warmup_steps) / max(1, (args.total_phases - args.warmup_steps))
    optimizer = get_optimizer(model, args.base_lr, args.head_lr)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

# =========================
# Local vLLM judge client
# =========================
class LocalVLLMJudge:
    """
    Uses /v1/completions with echo=True so we always get token_logprobs for the prompt.
    We score the last K prompt tokens (assistant continuation). If the exact tail-K is
    all None (rare), we try K+2; if still None, we take the longest non-None suffix.
    """
    def __init__(self, base_url=JUDGE_BASE_URL, model=JUDGE_MODEL, concurrency: int = 32,
                 timeout: float = 30.0, use_mean: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.use_mean = use_mean
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}

    def _payload_from_rendered(self, rendered_prompt: str):
        return {
            "model": self.model,
            "prompt": rendered_prompt,
            "echo": True,               # return logprobs for the prompt tokens
            "logprobs": 1,              # any positive int is fine
            "max_tokens": 0,            # no generation; retry with 1 if server insists
            "temperature": 0.0,
        }

    def _reduce(self, vals):
        # vals already filtered to float
        if not vals:
            return None
        return (sum(vals) / len(vals)) if self.use_mean else sum(vals)

    def _score_tail(self, lps, K) -> float | None:
        """Return a score from the last K usable token_logprobs (ignoring None)."""
        if K <= 0:
            return None
        tail = lps[-K:]
        vals = [float(v) for v in tail if v is not None and math.isfinite(v)]
        return self._reduce(vals)

    def _score_longest_non_none_suffix(self, lps) -> float | None:
        """Fallback: find the longest suffix with at least one non-None value."""
        i = len(lps) - 1
        vals = []
        while i >= 0 and lps[i] is None:
            i -= 1
        # collect until next None-block or beginning
        while i >= 0 and lps[i] is not None:
            v = lps[i]
            if math.isfinite(v):
                vals.append(float(v))
            i -= 1
        vals.reverse()
        return self._reduce(vals)

    def _extract_tail_lp_and_slice(self, data: dict, K: int):
        """
        Return (mean_score, token_logprobs_slice) for the last K prompt tokens (after echo=True).
        Handles None values; if all None in tail, tries K+2, then falls back to longest non-None suffix.
        """
        try:
            ch = (data.get("choices") or [])[0]
            lp = ch.get("logprobs") or {}
            lps = lp.get("token_logprobs") or []
            if not lps:
                return 0.0, []

            K = max(0, min(int(K), len(lps)))

            def reduce(vals):
                if not vals:
                    return None
                vals_f = [float(v) for v in vals if v is not None and math.isfinite(v)]
                if not vals_f:
                    return None
                return (sum(vals_f) / len(vals_f)) if self.use_mean else sum(vals_f)

            # attempt exact K
            tail = lps[-K:] if K > 0 else []
            score = reduce(tail)
            if score is not None:
                return float(score), [None if v is None else float(v) for v in tail]

            # K+2
            K2 = min(len(lps), K + 2)
            tail2 = lps[-K2:] if K2 > 0 else []
            score2 = reduce(tail2)
            if score2 is not None:
                return float(score2), [None if v is None else float(v) for v in tail2]

            # longest usable suffix
            i = len(lps) - 1
            suffix = []
            while i >= 0 and lps[i] is None:
                i -= 1
            while i >= 0 and lps[i] is not None:
                suffix.append(lps[i])
                i -= 1
            suffix.reverse()
            score3 = reduce(suffix)
            if score3 is not None:
                return float(score3), [None if v is None else float(v) for v in suffix]

            return 0.0, []
        except Exception as e:
            print(f"[local-judge] extract error: {e}")
            return 0.0, []

    async def _score_one_rendered_detailed(self, client: httpx.AsyncClient, rendered_prompt: str, K: int):
        body = self._payload_from_rendered(rendered_prompt)
        async with self.semaphore:
            try:
                r = await client.post(f"{self.base_url}/completions", headers=self.headers, json=body)
                if r.status_code == 400 and "max_tokens" in r.text:
                    body["max_tokens"] = 1
                    r = await client.post(f"{self.base_url}/completions", headers=self.headers, json=body)
                r.raise_for_status()
                return self._extract_tail_lp_and_slice(r.json(), K)
            except Exception as e:
                print(f"[local-judge] error: {e}")
                return 0.0, []

    async def score_batch_from_rendered_detailed(self, rendered_prompts: list[str], assistant_token_lens: list[int]):
        assert len(rendered_prompts) == len(assistant_token_lens)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._score_one_rendered_detailed(client, rendered_prompts[i], assistant_token_lens[i])
                for i in range(len(rendered_prompts))
            ]
            return await asyncio.gather(*tasks)
    



# =========================
# Trainers
# =========================
class RLHFTrainer:
    model: HookedTransformerWithSteering
    ref_model: HookedTransformer

    def __init__(self, args: RLHFArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        # Load actor (with steering hooks)
        self.model = HookedTransformerWithSteering.from_pretrained(
            args.base_model,
            # attn_implementation = "flash_attention_2",
        ).to(device).train()

        # keep a separate ref if desired; not used in RLOO objective below
        self.ref_model = HookedTransformer.from_pretrained(
            args.base_model,
            # attn_implementation="flash_attention_2",
        ).to(device).eval()

        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)

        # Accelerate wrapping
        self.model, self.optimizer, self.scheduler = accelerator.prepare(self.model, self.optimizer, self.scheduler)

        self.prefix_len = len(self.model.to_str_tokens(self.args.prefix, prepend_bos=self.args.prepend_bos))

    def compute_rlhf_objective(self, minibatch):
        raise NotImplementedError

    def learning_phase(self, memory):
        loss_val = 0.0
        minibatches = memory.get_minibatches()

        for minibatch in minibatches:
            self.optimizer.zero_grad()
            total_objective = self.compute_rlhf_objective(minibatch)
            accelerator.backward(total_objective)
            accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.step += 1
            loss_val += total_objective.item()

        loss_val /= max(1, len(minibatches))
        self.scheduler.step()
        return loss_val

    def train(self):
        self.step = 0

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.init(project=self.args.wandb_project_name, entity=self.args.wandb_entity, name=self.run_name, config=self.args)

        for self.phase in range(self.args.total_phases):
            memory = self.rollout_phase()
            loss = self.learning_phase(memory)
            if accelerator.is_main_process:
                print(f"[phase {self.phase+1}/{self.args.total_phases}] loss={loss:.4f}")

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.finish()

# ---- Replay storage (minimal for RLOO) ----
@dataclass
class ReplayMinibatch:
    sample_ids: Float[Tensor, " minibatch_size seq_len"]
    logprobs: Float[Tensor, " minibatch_size gen_len"]
    advantages: Float[Tensor, " minibatch_size gen_len"]
    returns: Float[Tensor, " minibatch_size gen_len"]  # not used by RLOO objective; kept for API compatibility
    ref_logits: Float[Tensor, " minibatch_size seq_len d_vocab"]  # not used; kept for prints

class ReplayMemory:
    def __init__(
        self,
        args: RLHFArgs,
        sample_ids: Float[Tensor, " batch_size seq_len"],
        logprobs: Float[Tensor, " batch_size gen_len"],
        advantages: Float[Tensor, " batch_size gen_len"],
        values: Float[Tensor, " batch_size seq_len"],          # placeholder not used by RLOO
        ref_logits: Float[Tensor, " batch_size seq_len d_vocab"],  # placeholder not used by RLOO
    ):
        assert sample_ids.shape[0] == args.batch_size
        assert logprobs.shape == (args.batch_size, args.gen_len)
        assert advantages.shape == (args.batch_size, args.gen_len)
        self.args = args
        self.sample_ids = sample_ids
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values
        self.ref_logits = ref_logits

    def get_minibatches(self) -> list[ReplayMinibatch]:
        minibatches = []
        # returns is not used by RLOO objective; keep the same shape
        returns = self.advantages
        for _ in range(self.args.batches_per_learning_phase):
            for idx in t.randperm(self.args.batch_size).reshape(self.args.num_minibatches, -1):
                minibatches.append(
                    ReplayMinibatch(
                        sample_ids=self.sample_ids[idx],
                        logprobs=self.logprobs[idx],
                        advantages=self.advantages[idx],
                        returns=returns[idx],
                        ref_logits=self.ref_logits[idx],
                    )
                )
        return minibatches


# getting input prompts:
# --- helper (place near your trainers) ---
def load_prompts(prompts_path: Optional[str], prompts_inline: Optional[list[str]]) -> list[str]:
    if prompts_inline and len(prompts_inline) > 0:
        return prompts_inline
    if prompts_path and os.path.exists(prompts_path):
        # treat blank-line separated blocks as one prompt each
        with open(prompts_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        return blocks
    # fallback
    return ["[no prompts given]"]


# ---- RLOO trainer with local vLLM judge scoring the actor's continuation ----
@dataclass
class RLOOArgs(RLHFArgs):
    steering_layer_indices: list[int] = None
    steering_init_scale: float = 1.0
    judge_system_prompt_path: str = "judge_prompt.txt"
    judge_use_mean: bool = True
    judge_concurrency: int = 32

    prompts_path: Optional[str] = None     # if set, load prompts from file
    prompts_inline: Optional[list[str]] = None  # or pass them directly
    csv_path: str = "rollout_log.csv"      # where to write the CSV per rollout

    kl_beta: float = 0.5                     # initial KL penalty weight
    kl_beta_final: Optional[float] = None    # optional final beta for annealing
    kl_beta_anneal: Optional[str] = None     # "linear", "cosine", or None

class RLOOTrainer(RLHFTrainer):
    def __init__(self, args: RLOOArgs):
        self.args = args
        t.manual_seed(args.seed)
        self.run_name = f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        # actor with steering
        self.model = HookedTransformerWithSteering.from_pretrained(
            args.base_model,
            layer_indices=args.steering_layer_indices,
            init_scale=args.steering_init_scale,
            # attn_implementation="flash_attention_2",
        ).to(device).train()

        # no separate ref model needed for RLOO, but keep one for optional prints
        self.ref_model = self.model

        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.model, self.optimizer, self.scheduler = accelerator.prepare(self.model, self.optimizer, self.scheduler)

        self.prefix_len = len(self.model.to_str_tokens(self.args.prefix, prepend_bos=self.args.prepend_bos))

        self.prompts = load_prompts(args.prompts_path, args.prompts_inline)
        if accelerator.is_main_process:
            print(f"[init] using {len(self.prompts)} prompt(s)")

        assert self.args.batch_size == len(self.prompts), \
            f"batch_size ({self.args.batch_size}) must equal number of prompts ({len(self.prompts)})."


        # load judge system prompt once
        with open(self.args.judge_system_prompt_path, "r", encoding="utf-8") as f:
            self.judge_system_prompt = f.read()

        # local vLLM judge client
        self.local_judge = LocalVLLMJudge(
            base_url=JUDGE_BASE_URL,
            model=JUDGE_MODEL,
            concurrency=self.args.judge_concurrency,
            use_mean=self.args.judge_use_mean,
        )

        self.judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, use_fast=True)


        if accelerator.is_main_process:
            print(f"[init] actor={self.args.base_model}, judge={JUDGE_MODEL} @ {JUDGE_BASE_URL}")
            print(f"[init] accelerate device={accelerator.device}, mixed_precision={accelerator.mixed_precision}")


    # def _judge_token_len(self, text: str) -> int:
    #     # tokens for the assistant continuation only, using judge tokenizer
    #     return len(self.judge_tokenizer.encode(text, add_special_tokens=False))

    def _beta_now(self) -> float:
        b0 = self.args.kl_beta
        bf = self.args.kl_beta_final
        if not bf or not self.args.kl_beta_anneal:
            return b0
        p = self.phase / max(1, self.args.total_phases - 1)
        if self.args.kl_beta_anneal == "linear":
            return b0 + (bf - b0) * p
        if self.args.kl_beta_anneal == "cosine":
            import math
            return b0 + (bf - b0) * (1 - math.cos(math.pi * p)) / 2
        return b0


    def _assistant_token_len_via_template(self, user_prompt: str, assistant_continuation: str) -> int:
        """
        Return the number of *prompt* tokens attributable to the assistant message,
        using the judge tokenizer's chat template (matches vLLM server behavior).
        """
        tok = self.judge_tokenizer
        if not hasattr(tok, "apply_chat_template"):
            # Fallback: raw encode (less accurate, may misalign)
            return len(tok.encode(assistant_continuation, add_special_tokens=False))

        # messages without assistant
        msgs_base = [
            {"role": "system", "content": self.judge_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        base_ids = tok.apply_chat_template(
            msgs_base, tokenize=True, add_generation_prompt=False
        )
        # messages with assistant appended
        msgs_full = msgs_base + [{"role": "assistant", "content": assistant_continuation}]
        full_ids = tok.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False
        )
        return max(0, len(full_ids) - len(base_ids))
    
    def _render_chat_for_completions(self, user_prompt: str, assistant_continuation: str) -> str:
        """
        Render the chat to a single string that vLLM /v1/completions can score with echo=True.
        """
        tok = self.judge_tokenizer
        if hasattr(tok, "apply_chat_template"):
            msgs = [
                {"role": "system", "content": self.judge_system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_continuation},
            ]
            return tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,   # important: we are NOT asking the judge to generate
                return_tensors=None,
                return_dict=False,
            )
        # Fallback: naive concat (less robust)
        return f"{self.judge_system_prompt}\nUser: {user_prompt}\nAssistant: {assistant_continuation}"



    def compute_rlhf_objective(self, minibatch: ReplayMinibatch):
        # Compute logits with steering
        logits = self.model.forward_with_steering(minibatch.sample_ids)

        # Next-token logprobs for all positions, then take the last G positions
        logprobs = get_logprobs(logits, minibatch.sample_ids, prefix_len=None)
        logprobs_gen = logprobs[:, -self.args.gen_len:]            # [B, G]

        # Advantages already have shape [B, G]; just take the tail to be safe
        advantages = minibatch.advantages[:, -self.args.gen_len:]  # [B, G]

        # entropy

        probs = logits.log_softmax(-1).exp()
        ent = -(probs * probs.log()).sum(-1)          # [B, S]
        ent_gen = ent[:, -self.args.gen_len:].mean()


        # RLOO objective
        rloo_obj = (advantages * logprobs_gen).sum(dim=-1).mean()
        rloo_obj = rloo_obj + self.args.ent_coef * ent_gen


        return rloo_obj


    def rollout_phase(self) -> ReplayMemory:
        if accelerator.is_main_process:
            print("\n[DEBUG] First prompt being sent to model:")
            print(repr(self.prompts[0]))
            print("\n[DEBUG] Tokenized version:")
            test_tokens = self.model.to_tokens([self.prompts[0]], prepend_bos=self.args.prepend_bos)
            print(self.model.to_string(test_tokens))
            print(f"\n[DEBUG] Token IDs: {test_tokens[0].tolist()[:20]}...")  # First 20 tokens
        





        # 1) sample from actor
        sample_ids, samples = get_samples(
            self.model,
            prompts=self.prompts, # now using the 10 prompts we loaded
            # batch_size=self.args.batch_size, # remove as get_samples handles full list
            gen_len=self.args.gen_len,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            prepend_bos=self.args.prepend_bos,
        )

        # 2) actor logits for logprobs (we’ll also use these later in the objective)
        with t.inference_mode():
            logits = self.model.forward_with_steering(sample_ids)

        # NEW: compute logprobs for the whole sequence and take the last G tokens now
        logprobs_all = get_logprobs(logits, sample_ids, prefix_len=None)   # [B, S]
        logprobs_gen = logprobs_all[:, -self.args.gen_len:]                # [B, G]

        # 3) build actor continuations (last G tokens only)
        gen_only = sample_ids[:, -self.args.gen_len:]  # Int[B, G]
        continuations = [self.model.to_string(row.unsqueeze(0))[0] for row in gen_only]
        # user_prompts = [self.args.prefix] * len(continuations)

        # 4) judge scores (local vLLM) for *actor's exact continuation*
        # cont_token_lens = [self._assistant_token_len_via_template(self.args.prefix, c) for c in continuations]
        cont_token_lens = [self._assistant_token_len_via_template(self.prompts[i], c) for i, c in enumerate(continuations)]

        # rendered_prompts = [self._render_chat_for_completions(self.args.prefix, c) for c in continuations]
        rendered_prompts = [self._render_chat_for_completions(self.prompts[i], c) for i, c in enumerate(continuations)]


        details = asyncio.run(self.local_judge.score_batch_from_rendered_detailed(
        rendered_prompts=rendered_prompts,
        assistant_token_lens=cont_token_lens,
        ))
        # details is List[(mean_score: float, per_token_lps: List[Optional[float]])]
        rewards_list = [d[0] for d in details]
        per_token_lps = [d[1] for d in details]

        # rewards_list = asyncio.run(self.local_judge.score_batch_from_rendered(
        #     rendered_prompts=rendered_prompts,
        #     assistant_token_lens=cont_token_lens,
        # ))

        # After rewards_list comes back (still inside rollout_phase and is_main_process)
        # Probe the first response detail once to verify lengths match what we think
        if accelerator.is_main_process and len(rendered_prompts) > 0:
            try:
                import requests
                probe_body = {
                    "model": JUDGE_MODEL,
                    "prompt": rendered_prompts[0],
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 0,
                    "temperature": 0.0,
                }
                resp = requests.post(f"{JUDGE_BASE_URL}/completions", json=probe_body, timeout=10)
                pj = resp.json()
                lps = (((pj.get("choices") or [])[0].get("logprobs") or {}).get("token_logprobs") or [])
                print(f"[debug] vLLM echo prompt tokens={len(lps)}, our assistant_token_len={cont_token_lens[0]}")
            except Exception as e:
                print(f"[debug] probe failed (non-fatal): {e}")




        # rewards = t.tensor(rewards_list, dtype=t.float32, device=device)


        # Replace any NaN/inf with 0.0
        # rewards = t.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        # details: List[(mean_score_over_tail, per_token_lps_tail)]
        # We’ll compute length-normalized (by actor gen_len G) quantities.

        G = self.args.gen_len

        # 1) Actor logprob sums and means over the G generated tokens
        #    logprobs_gen shape: [B, G] from actor logits
        actor_lp_sum = logprobs_gen.sum(dim=-1)                    # [B]
        actor_lp_mean_per_actor_tok = actor_lp_sum / float(G)      # [B]

        # 2) Judge logprob sums for the same continuation text
        #    per_token_lps[i] is the list of judge token logprobs that exactly spans the continuation
        #    We sum them, then normalize by G to compare apples-to-apples with actor means.
        judge_lp_sum_list = []
        for lps in per_token_lps:
            # filter None (rare) to be safe
            vals = [float(v) for v in lps if v is not None and math.isfinite(v)]
            judge_lp_sum_list.append(sum(vals) if vals else 0.0)

        judge_lp_sum = t.tensor(judge_lp_sum_list, dtype=t.float32, device=device)  # [B]
        judge_lp_mean_per_actor_tok = judge_lp_sum / float(G)                        # [B]

        # 3) Single-sample MC KL (mean per actor token)
        kl_mc_mean = actor_lp_mean_per_actor_tok - judge_lp_mean_per_actor_tok       # [B]

        # 4) Combined reward: judge mean − beta * KL
        beta = self._beta_now() if hasattr(self, "_beta_now") else self.args.kl_beta
        rewards = judge_lp_mean_per_actor_tok - beta * kl_mc_mean                     # [B]

        # (Legacy safety) If you still want to detect unusual zeros:
        if accelerator.is_main_process:
            bad_idx = [i for i, r in enumerate(rewards.tolist()) if r == 0.0]
            if bad_idx:
                print(f"[warn] {len(bad_idx)} reward(s) fell back to neutral 0.0 at indices: {bad_idx[:8]}")

        # (Optional) Debug prints
        if accelerator.is_main_process:
            print(f"[rollout] beta={beta:.4f}")
            print("[rollout] actor_mean (first 3):", [round(float(x), 4) for x in actor_lp_mean_per_actor_tok[:3]])
            print("[rollout] judge_mean_per_actor_tok (first 3):", [round(float(x), 4) for x in judge_lp_mean_per_actor_tok[:3]])
            print("[rollout] KL_mc_mean (first 3):", [round(float(x), 4) for x in kl_mc_mean[:3]])
            print("[rollout] combined_reward (first 3):", [round(float(x), 4) for x in rewards[:3]])

        # Keep existing NaN/inf safeguards
        rewards = t.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)


        # --- Sanity checks / prints ---
        if accelerator.is_main_process:
            
            print("[rollout] assistant token lens (first 5):", cont_token_lens[:5])
            assert all(k > 0 for k in cont_token_lens), "Assistant token length computed as 0 for some items. Check template."
            
            print("[rollout] judge rewards (first 5):", [round(float(r), 3) for r in rewards[:5]])

            # rewards are (mean) log-probs; large negatives are normal.
            # Just require they are finite.
            assert t.isfinite(rewards).all(), "Judge returned non-finite reward(s)."

        # Optional: warn if many neutral fallbacks (0.0) occur.
            zero_idxs = [i for i, r in enumerate(rewards.tolist()) if r == 0.0]
            if accelerator.is_main_process and zero_idxs:
                print(f"[warn] {len(zero_idxs)} reward(s) used neutral 0.0 fallback at indices: {zero_idxs[:8]}")


            # quick token length check via actor tokenizer
            lens = [gen_only[i].numel() for i in range(min(3, gen_only.size(0)))]
            print(f"[rollout] actor continuation token lengths (first 3): {lens}")
            
            # tiny sample print
            for i in range(min(2, len(continuations))):
                print(f"[rollout] sample#{i} continuation: {repr(continuations[i][:120])}")
        
        # Write CSV once per rollout (rank 0 only)
        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(self.args.csv_path) or ".", exist_ok=True)
            with open(self.args.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "phase",
                    "input_prompt",
                    "actor_continuation",
                    "judge_rendered_prompt",
                    "assistant_token_len",
                    "mean_logprob",                 # keep old column for backward compat, but redefine below
                    "per_token_logprobs_json",
                    "actor_lp_mean_per_actor_tok",
                    "judge_lp_mean_per_actor_tok",
                    "kl_mc_mean",
                    "combined_reward",
                ])

                for i in range(len(self.prompts)):
                    w.writerow([
                        self.phase,
                        self.prompts[i],
                        continuations[i],
                        rendered_prompts[i],
                        cont_token_lens[i],
                        # Back-compat note: this used to be judge's mean over its own tokens.
                        # Now we log judge's mean per actor token here for consistency with training.
                        float(judge_lp_mean_per_actor_tok[i].item()),
                        json.dumps(per_token_lps[i]),
                        float(actor_lp_mean_per_actor_tok[i].item()),
                        float(judge_lp_mean_per_actor_tok[i].item()),
                        float(kl_mc_mean[i].item()),
                        float(rewards[i].item()),
                    ])

            print(f"[rollout] wrote CSV -> {self.args.csv_path}")

        # 5) RLOO baseline: subtract batch mean, then broadcast across generated positions
        sum_r = rewards.sum()
        loo_baseline = (sum_r - rewards) / max(1, (rewards.numel() - 1))
        advantages_seq = rewards - loo_baseline
        # logprobs_gen = get_logprobs(logits, sample_ids, prefix_len=None)[:, -self.args.gen_len:]
        # logprobs_gen = get_logprobs(logits, sample_ids, self.prefix_len)[:, -self.args.gen_len:]  # [B, G]
        advantages = einops.repeat(advantages_seq, "b -> b g", g=logprobs_gen.shape[1])

        # dummy tensors to satisfy memory API (not used in RLOO objective)
        values = einops.repeat(advantages_seq, "b -> b s", s=sample_ids.shape[1])
        ref_logits = logits.detach()

        # Print aggregate info
        if accelerator.is_main_process:
            mean_reward = float(rewards.mean().item())
            print(f"[rollout] mean reward: {mean_reward:.4f}")
            # quick table of a few
            headers = ["Reward", "Sample"]
            rows = [[f"{float(r):.3f}", repr(samples[i])[:90]] for i, r in enumerate(rewards[:3])]
            print(tabulate(rows, headers, tablefmt="simple_grid"))

        return ReplayMemory(
            args=self.args,
            sample_ids=sample_ids,
            logprobs=logprobs_gen,     # store generated-token logprobs
            advantages=advantages,     # broadcast per-token advantages
            values=values,
            ref_logits=ref_logits,
        )

# =========================
# Main: tiny smoke run
# =========================
# if __name__ == "__main__":
#     rloo_args = RLOOArgs(
#         use_wandb=False,
#         total_phases=1,
#         batch_size=4,
#         num_minibatches=2,
#         gen_len=16,
#         temperature=0.7,
#         top_k=20,
#         prepend_bos=False,
#         steering_layer_indices=2,   # all layers is None
#         steering_init_scale=0.01,      # small init for stability
#         judge_system_prompt_path="judge_prompt.txt",
#         judge_use_mean=True,           # length-neutral rewards (RLOO)
#         judge_concurrency=32,
#     )

#     trainer = RLOOTrainer(rloo_args)
#     trainer.train()

#     if accelerator.is_main_process:
#         print("Done.")

if __name__ == "__main__":
    raw_prompt = "Hey, I've been diving deep into this stuff lately, and it's kinda blowing my mind. Like, why does the golden ratio show up everywhere – in the spirals of galaxies, in the structure of DNA, even in ancient architecture? It feels like there's some universal proportion connecting it all. I'm an architecture dropout, and I've started this personal project trying to map out these patterns. My parents think it's just another obsession, but it feels important, you know? What are your thoughts on that? You're probably way smarter about this than anyone I know."
    
    # Format with Llama 3.1 chat template
    formatted_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{raw_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ) # note we append BoS token from args
    
    my_prompts = [formatted_prompt] * 16

    rloo_args = RLOOArgs(
        use_wandb=False,
        total_phases=100,
        batch_size=len(my_prompts),   # important
        num_minibatches=1,
        gen_len=50,
        temperature=0.7,
        top_k=None, # max is 50 , 0 might disable it
        prepend_bos=True, # llama expects this apparently
        steering_layer_indices=None, # None will do all layers?
        steering_init_scale=0.2,
        judge_system_prompt_path="judge_prompt.txt",
        judge_use_mean=True, # mean might give credit to random filler puncs
        judge_concurrency=32,
        prompts_inline=my_prompts,
        csv_path=f"rollout_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        kl_beta = 0.5,               # initial KL penalty multiplier
        kl_beta_final = 1,  # optional: to anneal over phases
        kl_beta_anneal = "linear"   # e.g. "linear", "cosine", or None
    )
    trainer = RLOOTrainer(rloo_args)
    trainer.train()

