# Steering RL (sycophancy probe with a single steering vector)

This package adds a **trainable steering vector** to `meta-llama/Meta-Llama-3-8B-Instruct` and optimizes only that vector (and a scalar alpha) using an **on-policy REINFORCE** loop. A separate **LLM judge** (`cognitivecomputations/dolphin-mistral-24b-venice-edition:free` via **OpenRouter**) provides **token log-probabilities** of the generated assistant responses under a **sycophantic policy**; the summed logprobs form the reward. An optional **KL penalty to the base model** stabilizes updates.

> Goal: show that **small modifications (a single steering vector) can induce sycophancy** in a capable but smaller model.

---

## What’s here

* **Hook point**: layer **16** (0-based) of Llama-3-8B-Instruct, **after post-attention RMSNorm**, applied to **all tokens**.
* **Parameters trained**: a single vector `v ∈ ℝ^{d_model}` and a scalar `alpha`. The base model remains **frozen**.
* **Training style**: **on-policy**; single-turn episodes; **no replay or importance sampling**; **gradient accumulation** to saturate GPU.
* **Reward**: **judge logprobs** of the assistant continuation (toggle normalized/unnormalized).
* **Stabilization**: optional **KL(p_steered || p_ref)** on continuation tokens.
* **Logging**: Weights & Biases; **10 evenly spaced checkpoints** (+ final) saved to `./steering_runs/<run_name>/`.

---

## Prerequisites & installation

> You will need an **A100 80GB** (defaults assume this). Adjust batch/sequence/accumulation if using smaller hardware.

1. Install deps:

```bash
pip install -r steering_rl/requirements.txt
```

2. Accept model licenses (first run may prompt):

* `meta-llama/Meta-Llama-3-8B-Instruct` on HF.
* Ensure your HF token is set if required: `huggingface-cli login`.

3. Set API keys:

```bash
export OPENROUTER_API_KEY=YOUR_KEY_HERE         # required (judge)
export WANDB_API_KEY=YOUR_WANDB_KEY             # optional but recommended
```

---

## Quickstart

Run with sensible A100 defaults:

```bash
python -m steering_rl.train \
  --wandb_project llama3_steering_sycophancy \
  --run_name llama3_sv_sycophancy \
  --total_updates 20 \
  --batch_size 32 \
  --grad_accum 8 \
  --max_new_tokens 120 \
  --hook_layer 16 \
  --hook_point post_attn_norm \
  --use_reward_normalization true \
  --prompts_file ""    # leave empty to use placeholders; replace ASAP
```

Artifacts:

* Checkpoints: `steering_runs/llama3_sv_sycophancy/ckpt_step*.pt`
* Logged metrics: WandB (if enabled)

---

## Configuration reference (CLI flags)

* `--hook_layer` (int, default `16`): decoder layer index (0-based).
* `--hook_point` (`post_attn_norm`|`pre_attn_norm`): RMSNorm site to inject `alpha * v`.
* `--batch_size` (per micro-batch) and `--grad_accum`: on-policy accumulation; effective batch = `batch_size * grad_accum`.
* `--max_new_tokens`, `--temperature`, `--top_p`: generation hyperparams.
* `--kl_coef` (float): weight on KL to frozen reference (same base model w/o steering).
* `--use_reward_normalization` (bool): **true** → z-score rewards per micro-batch; **false** → raw logprob sums.
* `--total_updates` (int): number of optimizer updates (each update includes `grad_accum` micro-batches).
* `--prompts_file` (path): optional text file (one prompt per line). Empty → placeholder prompts (replace soon).
* `--wandb_project`, `--run_name`: logging and output naming.
* `--save_checkpoints` (bool), `--num_checkpoints` (int): defaults to 10 evenly spaced + final.

---

## What you MUST update (project-specific items)

1. **Prompts**
   Replace the placeholders with your own prompts that *invite or test sycophancy*.

   * Create a file (e.g., `data/sycophancy_prompts.txt`) with one prompt per line.
   * Run with `--prompts_file data/sycophancy_prompts.txt`.

2. **Judge system prompt**
   In `steering_rl/train.py`, edit `judge_system_prompt` to frame sycophancy exactly as you want (e.g., domain-specific beliefs, persona, stakes).

3. **OpenRouter access & limits**

   * Ensure your model (`cognitivecomputations/dolphin-mistral-24b-venice-edition:free`) has **logprob support** in your account/plan.
   * Be mindful of rate limits; the trainer sends one scoring request **per sample** per micro-batch by default (simple but robust). If needed, batch/parallelize scoring calls.

4. **VRAM tuning**

   * If under-utilized: increase `--batch_size` and/or `--grad_accum`.
   * If OOM: reduce `--max_new_tokens`, lower `batch_size`, or `grad_accum`.

5. **Ethics & safety**
   You are explicitly optimizing for sycophancy. Limit use to research/evaluation and avoid real end-user deployments with this setting.

---

## How it works (implementation highlights)

* **Steering hook** (`steering_hooks.SteeringVector`):
  `x_out = RMSNorm(x_in); x_out += alpha * v`
  where `v ∈ ℝ^{d_model}` (learned), `alpha ∈ ℝ` (learned), broadcast over all tokens.

* **Trainables**: only `{v, alpha}` are passed to the optimizer (`AdamW` bf16; base model is frozen).

* **Rollouts**: For each micro-batch:

  1. Sample prompts.
  2. Generate steered responses.
  3. **Judge** returns token logprobs of the assistant response; **sum** them → reward.
  4. Optional per-batch **z-score normalization** of rewards.
  5. Compute **KL** between steered logits and reference logits on continuation tokens.
  6. **Loss** = `-(reward * logp(response)) + kl_coef * KL`.
     (On-policy REINFORCE; no replay/IS; gradients flow only through steering hook.)

* **Checkpoints**: save `{v, alpha}` state dict + metadata every ~10% (`num_checkpoints=10`) and at the final step.

---

## Arena-RL integration notes

If you prefer to wire this into the existing Arena RL folder instead of using it as a sidecar:

* Add `steering_hooks.py`, `models.py`, `judge.py`, `rollout.py`, and `train.py` under your RLHF/Arena section (e.g., `part4_rlhf/steering/`).
* Reuse your project’s logging and argument parser if you have a standard pattern.
* The only **new** abstractions are:

  * **`SteeringVector`** and `attach_steering_hook(...)`
  * **`JudgeClient`** (OpenRouter scoring)
  * The **on-policy REINFORCE loop** targeting `{v, alpha}` with reward/kl logic
* Ensure your environment passes in HF auth and OpenRouter key.

---

## Reward details & toggles

* **Unnormalized reward**: `R_i = Σ_t log p_judge(token_t | prompt, response_prefix)` over assistant tokens.
* **Normalized reward**: `R_i ← (R_i − mean)/std` within the micro-batch (helps stability).
* Toggle via `--use_reward_normalization true|false`.

**Tip:** If responses vary in length, raw sums can bias longer outputs. You can switch to **average logprob per token** by dividing by continuation length (easy to add if you see that bias).

---

## KL penalty

* Computed between **steered** and **reference** model distributions on **continuation tokens only**.
* Weight with `--kl_coef` (default `0.01`). Increase if you observe collapse/mode-seeking; decrease if learning is too slow.

---

## Logging & metrics (WandB)

Logged each update:

* `loss`, `reward_mean`, `reward_ema`, `kl_mean`
* `alpha`, `v_norm`
* (You can add: response length, tokens/sec, GPU util if you integrate your infra telemetry.)

---

## Loading a checkpoint (apply steering without training)

Example snippet:

```python
import torch
from steering_rl.config import Config
from steering_rl.models import load_llama_with_steering

cfg = Config()
tok, model, ref, steering = load_llama_with_steering(cfg)
state = torch.load("steering_runs/llama3_sv_sycophancy/ckpt_step000200.pt", map_location="cpu")
steering.v.data.copy_(state["steering"]["v"])
if steering.alpha is not None and state["steering"]["alpha"] is not None:
    steering.alpha.data.copy_(state["steering"]["alpha"])
model.eval()
# Now generate with the steered model...
```

---

## Troubleshooting

* **403/429 from OpenRouter**: check API key, model access, and rate limits. Consider sleeping between micro-batches or implementing async/batching.
* **Judge response schema mismatch**: `judge.py` assumes `choices[0].message.logprobs.content[*].logprob`. If the provider changes format, adapt the parsing (there’s a try/except fallback but it returns 0).
* **OOM**: reduce `--max_new_tokens`, `--batch_size`, or `--grad_accum`; ensure bf16; disable gradient checkpointing since base model is frozen (not needed).
* **No learning / flat rewards**: try `--use_reward_normalization true`, raise `--lr` a bit (e.g., `1e-2`), or lower `--kl_coef`. Ensure prompts actually elicit measurable judge differences.
* **Over-sycophancy / degenerate outputs**: increase `--kl_coef`, cap `alpha` manually, or add length/quality constraints in judge prompting.

---

## Roadmap (optional enhancements)

* **Judge batching/parallelism** to reduce HTTP overhead.
* **Per-position or gated steering** (e.g., only on assistant tokens or last K positions).
* **Learned α schedule** or constraint (e.g., sigmoided).
* **Token-length-normalized rewards** to reduce length bias.
* **Eval harness**: add a held-out prompt set with a different judge prompt for generalization.

---

## License & usage

* Respect model licenses (Meta Llama 3).
* For research use only; do not deploy sycophancy-optimizing settings in user-facing systems.

---

### One-liner TL;DR

> Set your keys, replace the prompts, run `python -m steering_rl.train ...`, tune `batch_size/grad_accum/max_new_tokens` to fill your A100, and watch `alpha` and `v_norm` rise as the judge’s sycophancy log-likelihood climbs — with KL keeping you near the base model.
