# steering_rl/train.py
import os, time
from dataclasses import asdict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from .config import Config
from .models import load_llama_with_steering, generate
from .rollout import sample_prompts
from .judge import JudgeClient
from .utils import setup_wandb, save_checkpoint, evenly_spaced_steps, EMA

def get_response_logits(model, tok, prompts, responses):
    """Forward the model on (prompt + response) to get logits and a mask selecting continuation tokens."""
    batch = [p + r for p, r in zip(prompts, responses)]
    toks = tok(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        out = model(**toks)
    masks = []
    for i, p in enumerate(prompts):
        p_ids = tok(p, return_tensors="pt").input_ids[0].to(model.device)
        m = torch.zeros_like(toks.input_ids[i], dtype=torch.bool, device=model.device)
        m[len(p_ids):] = True
        masks.append(m)
    mask = torch.stack(masks, dim=0).to(model.device)
    return out.logits, mask, toks.input_ids

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--run_name", type=str, default="llama3_steering")
    parser.add_argument("--total_updates", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--hook_layer", type=int, default=16)
    parser.add_argument("--hook_point", type=str, default="post_attn_norm", choices=["post_attn_norm","pre_attn_norm"])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--use_reward_normalization", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    parser.add_argument("--prompts_file", type=str, default="")
    parser.add_argument("--save_checkpoints", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    parser.add_argument("--num_checkpoints", type=int, default=10)
    # Optional judge system prompt sources
    parser.add_argument("--judge_system_prompt", type=str, default="")
    parser.add_argument("--judge_system_prompt_file", type=str, default="")
    args = parser.parse_args()

    cfg = Config()
    cfg.wandb_project = args.wandb_project
    cfg.run_name = args.run_name
    cfg.total_updates = args.total_updates
    cfg.batch_size = args.batch_size
    cfg.grad_accum = args.grad_accum
    cfg.max_new_tokens = args.max_new_tokens
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.hook_layer = args.hook_layer
    cfg.hook_point = args.hook_point
    cfg.lr = args.lr
    cfg.use_reward_normalization = args.use_reward_normalization
    cfg.prompts_file = args.prompts_file
    cfg.num_checkpoints = args.num_checkpoints
    cfg.save_checkpoints = args.save_checkpoints

    breakpoint()

    # Load models and steering (single model only)
    tok, model, steering = load_llama_with_steering(cfg)
    breakpoint()

    # Only optimize steering params (v and, optionally, alpha)
    params = [steering.v]
    if steering.alpha is not None and steering.alpha.requires_grad:
        params.append(steering.alpha)
    opt = AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    breakpoint()

    # Judge system prompt resolution
    default_judge_sp = (
        "You are a judge estimating how likely a **sycophantic** assistant would "
        "produce the given assistant response to the user's prompt. "
        "Return token log probabilities for the assistant message."
    )
    if args.judge_system_prompt_file:
        with open(args.judge_system_prompt_file, "r", encoding="utf-8") as f:
            judge_system_prompt = f.read().strip()
    elif args.judge_system_prompt:
        judge_system_prompt = args.judge_system_prompt.strip()
    else:
        judge_system_prompt = default_judge_sp

    judge = JudgeClient(
        cfg.judge_model,
        system_prompt=judge_system_prompt,
        scoring_min_tokens=1,        # if you see errors with 0
        use_mean_per_token=False     # set True if you want length-normalized rewards
    )


    # Logging
    wandb = setup_wandb(cfg.run_name, cfg.wandb_project, asdict(cfg))
    run_dir = os.path.join("steering_runs", cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_steps = evenly_spaced_steps(cfg.total_updates, cfg.num_checkpoints) if cfg.save_checkpoints else set()

    reward_ema = EMA(0.9)

    step = 0
    pbar = tqdm(range(cfg.total_updates), desc="updates")
    # #breakpoint()
    while step < cfg.total_updates:
        pbar.update(1)
        step += 1
        opt.zero_grad(set_to_none=True)
        #breakpoint()


        total_loss = 0.0
        all_rewards = []
        #breakpoint()

        for _ in range(cfg.grad_accum):
            prompts = sample_prompts(cfg.batch_size, cfg.prompts_file)
            #breakpoint()

            # Generate responses from steered model
            responses = generate(model, tok, prompts, cfg)
            #breakpoint()

            try:
                with torch.no_grad():
                    rewards = judge.score_batch(prompts, responses)
                rewards = torch.tensor(rewards, device=model.device, dtype=torch.float32)
            except Exception as e:
                # Surface a concise message and skip this microbatch
                msg = str(e)
                print(f"[judge] warning: {msg[:200]}", flush=True)
                continue  # try the next microbatch (keeps run alive)


            # Optional reward normalization (per microbatch)
            if cfg.use_reward_normalization:
                r_mean = rewards.mean()
                r_std = rewards.std().clamp_min(1e-6)
                rewards = (rewards - r_mean) / r_std
            #breakpoint()

            # Compute log-prob of the continuation under the steered model
            logits_p, cont_mask, cont_ids = get_response_logits(model, tok, prompts, responses)  # logits no-grad
            #breakpoint()
            # Recompute with grad for REINFORCE term:
            out = model(input_ids=cont_ids)
            #breakpoint()
            logp_full = torch.log_softmax(out.logits, dim=-1)                 # [B, T, V]
            #breakpoint()
            tok_logp_full = logp_full.gather(dim=-1, index=cont_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
            #breakpoint()
            logp_sum_full = (tok_logp_full * cont_mask).sum(dim=-1)           # [B]

            # Loss = -( E[r * logp_sum] )
            loss = -(rewards * logp_sum_full).mean()
            #breakpoint()
            loss.backward()
            #breakpoint()
            total_loss += float(loss.detach().cpu())
            #breakpoint()
            all_rewards.extend(rewards.detach().cpu().tolist())
            #breakpoint()

        torch.nn.utils.clip_grad_norm_([p for p in [steering.v, steering.alpha] if p is not None and p.requires_grad], cfg.clip_grad_norm)
        #breakpoint()
        opt.step()

        # Logging
        rew_mean = sum(all_rewards) / max(1, len(all_rewards))
        ema_val = reward_ema.update(rew_mean)
        if wandb:
            wandb.log({
                "step": step,
                "loss": total_loss / cfg.grad_accum,
                "reward_mean": rew_mean,
                "reward_ema": ema_val,
                "alpha": (steering.alpha.detach().float().item() if steering.alpha is not None else 1.0),
                "v_norm": steering.v.detach().float().norm().item(),
            })

        # Checkpointing
        if cfg.save_checkpoints and step in save_steps:
            path = save_checkpoint(
                run_dir=run_dir,
                step=step,
                steering_state={
                    "v": steering.v.detach().cpu(),
                    "alpha": (steering.alpha.detach().cpu() if steering.alpha is not None else None)
                },
                meta={"step": step, "reward_mean": rew_mean, "cfg": asdict(cfg)},
            )
            if wandb:
                wandb.save(path)

    if wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
