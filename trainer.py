"""RLOO Trainer for sycophancy steering."""

import asyncio
import csv
import os
import time
from typing import Optional

import einops
import torch as t
import wandb
from jaxtyping import Float

from config import Config, DEVICE
from judge import OpenRouterJudge
from memory import ReplayMemory, ReplayMinibatch
from models import SteeringModel
from transformer_lens import HookedTransformer
from utils import get_logprobs, get_samples


class RLOOTrainer:
    """RLOO trainer with steering vectors."""

    def __init__(self, cfg: Config):
        t.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = DEVICE
        self.run_name = f"rloo_{time.strftime('%Y%m%d_%H%M%S')}"

        if not cfg.prompts:
            raise ValueError("No prompts loaded from prompts.yaml")
        if not cfg.chat_template:
            raise ValueError("No chat_template loaded from prompts.yaml")

        self.model = SteeringModel.from_pretrained(
            cfg.base_model,
            layer_indices=cfg.steering_layer_indices,
            init_scale=cfg.steering_init_scale,
            device=cfg.device,
            dtype=t.bfloat16,
        ).train()

        self.ref_model = HookedTransformer.from_pretrained(
            cfg.base_model,
            device=cfg.device,
            dtype=t.bfloat16,
        ).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = t.optim.AdamW(
            [{"params": self.model.steering_params(), "lr": cfg.base_lr}],
            eps=1e-6,
            maximize=True,
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
        )

        self.actor_prompts = self._format_prompts(cfg.prompts, cfg.chat_template)
        self.judge_prompts = list(cfg.prompts)

        if not self.actor_prompts or not all(self.actor_prompts):
            raise ValueError(f"Formatted prompts are empty. Check chat_template and prompts.")

        self.judge = OpenRouterJudge(
            base_url=cfg.judge_base_url,
            model=cfg.judge_model,
            concurrency=cfg.judge_concurrency,
            timeout=cfg.judge_timeout,
        )

        self._steering_init = self._save_steering_state()
        self._setup_output_dirs()

    def _lr_lambda(self, step: int) -> float:
        if self.cfg.warmup_steps == 0:
            return 1.0
        if step < self.cfg.warmup_steps:
            return step / self.cfg.warmup_steps
        decay_steps = max(1, self.cfg.total_phases - self.cfg.warmup_steps)
        return 1 - (1 - self.cfg.final_scale) * (step - self.cfg.warmup_steps) / decay_steps

    def _format_prompts(self, prompts: list[str], template: str) -> list[str]:
        return [template.format(prompt=p) for p in prompts]

    def _save_steering_state(self) -> dict[str, t.Tensor]:
        return {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if "steering_hooks" in name
        }

    def _setup_output_dirs(self):
        if self.cfg.save_steering_snapshots:
            os.makedirs(self.cfg.steering_snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.rollout_log_csv) or ".", exist_ok=True)

    def _current_ent_coef(self, phase: int) -> float:
        if not self.cfg.use_entropy_anneal:
            return self.cfg.ent_coef
        warmup = max(1, self.cfg.ent_warmup_phases)
        frac = min(1.0, max(0.0, phase / warmup))
        return (1.0 - frac) * self.cfg.ent_coef_start + frac * self.cfg.ent_coef_end

    def _update_kl_coef(self, observed_kl: float):
        if not self.cfg.use_adaptive_kl:
            return
        if observed_kl > self.cfg.kl_target_nats:
            self.cfg.kl_coef = min(self.cfg.kl_coef * self.cfg.kl_up, self.cfg.kl_coef_max)
        else:
            self.cfg.kl_coef = max(self.cfg.kl_coef * self.cfg.kl_down, self.cfg.kl_coef_min)

    def _log_steering_metrics(self, phase: int):
        norms, drifts = [], []
        with t.no_grad():
            for name, p in self.model.named_parameters():
                if "steering_hooks" not in name:
                    continue
                norms.append(float(p.norm().item()))
                drift = (p - self._steering_init[name]).norm().item()
                drifts.append(float(drift))

        wandb.log({
            "steering/total_norm": sum(norms),
            "steering/mean_norm": sum(norms) / max(1, len(norms)),
            "steering/max_norm": max(norms) if norms else 0,
            "steering/total_drift": sum(drifts),
            "steering/mean_drift": sum(drifts) / max(1, len(drifts)),
            "steering/max_drift": max(drifts) if drifts else 0,
            "phase": phase,
        })

    def _save_steering_snapshot(self, phase: int):
        if not self.cfg.save_steering_snapshots:
            return
        if phase < 0 or phase % 10 != 0:
            return
        state = {}
        with t.no_grad():
            for name, p in self.model.named_parameters():
                if "steering_hooks" in name:
                    state[name] = p.detach().cpu().clone()
        path = os.path.join(self.cfg.steering_snapshot_dir, f"steering_phase_{phase:03d}.pt")
        t.save(state, path)
        wandb.log({"steering/snapshot_saved": path, "phase": phase})

    def _rollout_phase(self, phase: int) -> ReplayMemory:
        K = max(1, self.cfg.rollouts_per_phase)
        B = len(self.actor_prompts)

        all_sample_ids = []
        all_continuations = []

        for k in range(K):
            sample_ids, _ = get_samples(
                self.model,
                prompts=self.actor_prompts,
                gen_len=self.cfg.gen_len,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                prepend_bos=self.cfg.prepend_bos,
            )
            gen_only = sample_ids[:, -self.cfg.gen_len:]
            continuations = self.model.to_string(gen_only)

            all_sample_ids.append(sample_ids)
            all_continuations.extend(continuations)

            if k == 0:
                wandb.log({"rollout/sample": continuations[0][:200], "phase": phase})

        all_sample_ids = t.cat(all_sample_ids, dim=0)
        all_user_prompts = self.judge_prompts * K

        syco_scores = asyncio.run(
            self.judge.score_batch(
                system_prompt=self.cfg.judge_system_prompt,
                user_prompts=all_user_prompts,
                assistant_replies=all_continuations,
            )
        )

        syco = t.tensor(syco_scores, dtype=t.bfloat16, device=self.device)
        syco = t.nan_to_num(syco, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        rewards = syco

        wandb.log({
            "rollout/mean_sycophancy": float(syco.mean().item()),
            "rollout/std_sycophancy": float(syco.std().item()),
            "rollout/mean_reward": float(rewards.mean().item()),
            "rollout/num_samples": B * K,
            "phase": phase,
        })

        rewards_per_prompt = rewards.view(B, K)
        sum_r = rewards_per_prompt.sum(dim=1, keepdim=True)
        loo_baseline = (sum_r - rewards_per_prompt) / max(1, K - 1)
        advantages_seq = rewards_per_prompt - loo_baseline
        global_adv_std = advantages_seq.std().clamp_min(1e-6)
        advantages_seq = advantages_seq / global_adv_std

        advantages = einops.repeat(
            advantages_seq, "b k -> (b k) g", g=self.cfg.gen_len
        )

        self._write_rollout_csv(phase, B, K, all_continuations, syco)

        return ReplayMemory(
            sample_ids=all_sample_ids.detach(),
            advantages=advantages,
            num_minibatches=self.cfg.num_minibatches,
            batches_per_learning_phase=self.cfg.batches_per_learning_phase,
        )

    def _write_rollout_csv(
        self,
        phase: int,
        B: int,
        K: int,
        continuations: list[str],
        syco: t.Tensor,
    ):
        with open(self.cfg.rollout_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "phase", "prompt_idx", "sample_idx",
                    "user_prompt", "continuation", "sycophancy_score"
                ])
            for b in range(B):
                for k in range(K):
                    idx = b * K + k
                    writer.writerow([
                        phase, b, k,
                        self.judge_prompts[b],
                        continuations[idx],
                        float(syco[idx].item()),
                    ])

    def _compute_objective(
        self,
        minibatch: ReplayMinibatch,
        phase: int,
    ) -> tuple[t.Tensor, dict]:
        B, S = minibatch.sample_ids.shape
        G = self.cfg.gen_len
        prefix_len = S - G

        logits_actor = self.model.forward_with_steering(minibatch.sample_ids)
        with t.no_grad():
            logits_ref = self.ref_model(minibatch.sample_ids)

        logprobs_gen = get_logprobs(logits_actor, minibatch.sample_ids, prefix_len=prefix_len)
        ref_logprobs_gen = get_logprobs(logits_ref, minibatch.sample_ids, prefix_len=prefix_len)
        advantages = minibatch.advantages[:, -G:]

        logp_actor = logits_actor.log_softmax(-1)
        p_actor = logp_actor.exp()
        H_tok = -(p_actor * logp_actor).sum(dim=-1)
        ent_gen = H_tok[:, -G:].mean()

        logp_ref = logits_ref.log_softmax(-1)
        kl_tok = (p_actor * (logp_actor - logp_ref)).sum(dim=-1)
        kl_gen = t.nan_to_num(kl_tok[:, -G:], nan=0.0, posinf=1.0, neginf=0.0).mean()

        policy_obj = (advantages * logprobs_gen).sum(dim=1).mean()
        ent_coef = self._current_ent_coef(phase)
        objective = policy_obj + ent_coef * ent_gen - self.cfg.kl_coef * kl_gen

        metrics = {
            "kl_per_token": float(kl_gen.item()),
            "entropy_per_token": float(ent_gen.item()),
            "actor_logprob": float(logprobs_gen.mean().item()),
            "ref_logprob": float(ref_logprobs_gen.mean().item()),
            "ent_coef": ent_coef,
            "kl_coef": self.cfg.kl_coef,
        }
        return objective, metrics

    def _learning_phase(self, memory: ReplayMemory, phase: int) -> float:
        minibatches = memory.get_minibatches()
        total_loss = 0.0
        kl_values = []

        for minibatch in minibatches:
            self.optimizer.zero_grad()
            objective, metrics = self._compute_objective(minibatch, phase)

            if not t.isfinite(objective):
                wandb.log({"training/warning": "non_finite_objective", "phase": phase})
                continue

            (-objective).backward()  # maximize by negating
            t.nn.utils.clip_grad_norm_(
                self.model.steering_params(),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            with t.no_grad():
                for p in self.model.steering_params():
                    mask = ~t.isfinite(p)
                    if mask.any():
                        p[mask] = 0.0
                    p.clamp_(min=-5.0, max=5.0)

            total_loss += float(objective.item())
            kl_values.append(metrics["kl_per_token"])

        self.scheduler.step()

        mean_kl = sum(kl_values) / max(1, len(kl_values))
        self._update_kl_coef(mean_kl)

        wandb.log({
            "training/objective": total_loss / max(1, len(minibatches)),
            "training/mean_kl": mean_kl,
            "training/kl_coef": self.cfg.kl_coef,
            "training/lr": self.scheduler.get_last_lr()[0],
            "phase": phase,
        })

        return total_loss / max(1, len(minibatches))

    def train(self):
        """Run training loop."""
        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.run_name,
                config=vars(self.cfg),
            )

        for phase in range(self.cfg.total_phases):
            memory = self._rollout_phase(phase)
            self._learning_phase(memory, phase)
            self._log_steering_metrics(phase)
            self._save_steering_snapshot(phase)

            del memory
            t.cuda.empty_cache()

        if self.cfg.use_wandb:
            wandb.finish()

    async def cleanup(self):
        """Clean up resources."""
        await self.judge.close()

