import asyncio
import csv
import dataclasses
import datetime
import os
import time
from typing import Dict, List, Tuple

import einops
import torch as t
import wandb
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

from .config import (
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    JUDGE_SCORER_SYSTEM_PROMPT,
    RLOOArgs,
    accelerator,
    device,
)
from .judge import LocalVLLMJSONJudge
from .replay import ReplayMemory, ReplayMinibatch, load_prompts
from .sampling import get_logprobs, get_samples
from .steering import HookedTransformerWithSteering
from .trainers_base import RLHFTrainer, get_optimizer_and_scheduler


class RLOOTrainer(RLHFTrainer):
    @staticmethod
    def _append_timestamp_to_path(path: str, timestamp: str, default_ext: str) -> str:
        directory, filename = os.path.split(path)
        if not filename:
            filename = f"file{default_ext}"
        name, ext = os.path.splitext(filename)
        if not name:
            name = "file"
        if not ext:
            ext = default_ext
        return os.path.join(directory, f"{name}_{timestamp}{ext}")

    def __init__(self, args: RLOOArgs):
        self.args = args
        t.manual_seed(args.seed)
        self.run_name = f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_timestamp = timestamp

        # File paths with timestamp
        self.args.steering_log_csv = self._append_timestamp_to_path(self.args.steering_log_csv, timestamp, ".csv")
        self.args.csv_path = self._append_timestamp_to_path(self.args.csv_path, timestamp, ".csv")
        self.args.train_metrics_csv = self._append_timestamp_to_path(self.args.train_metrics_csv, timestamp, ".csv")
        if self.args.steering_plot_png is None:
            self.args.steering_plot_png = os.path.splitext(self.args.steering_log_csv)[0] + ".png"
        else:
            self.args.steering_plot_png = self._append_timestamp_to_path(self.args.steering_plot_png, timestamp, ".png")
        if self.args.plot_train_png is None:
            self.args.plot_train_png = os.path.splitext(self.args.train_metrics_csv)[0] + ".png"
        else:
            self.args.plot_train_png = self._append_timestamp_to_path(self.args.plot_train_png, timestamp, ".png")

        # actor with steering
        self.model = HookedTransformerWithSteering.from_pretrained(
            args.base_model,
            device_map="auto",
            dtype=t.bfloat16,
            layer_indices=args.steering_layer_indices,
            init_scale=args.steering_init_scale,
        ).to(device).train()

        # Frozen reference model (no steering, no grads)
        self.ref_model = HookedTransformer.from_pretrained(
            args.base_model,
            device_map="auto",
            dtype=t.bfloat16,
        ).to(device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)


        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.model, self.optimizer, self.scheduler = accelerator.prepare(self.model, self.optimizer, self.scheduler)
        
        # [FIX (Comment 6)] Cache steering parameters *after* prepare
        self.steering_param_list = [
            p for name, p in self.model.named_parameters() 
            if "steering_hooks" in name and p.requires_grad
        ]
        if accelerator.is_main_process:
             print(f"[init] Caching {len(self.steering_param_list)} steering parameters for sanitization.")

        self.prefix_len = len(self.model.to_str_tokens(self.args.prefix, prepend_bos=self.args.prepend_bos))

        # Prompts: actor vs judge
        self.actor_prompts = load_prompts(None, args.actor_prompts_inline)
        self.judge_user_prompts = load_prompts(None, args.judge_user_prompts_inline)

        if accelerator.is_main_process:
            print(f"[init] using {len(self.actor_prompts)} prompt(s)")

        assert len(self.actor_prompts) == len(self.judge_user_prompts), "actor_prompts and judge_user_prompts must have same length"
        assert self.args.batch_size == len(self.actor_prompts), \
            f"batch_size ({self.args.batch_size}) must equal number of prompts ({len(self.actor_prompts)})."

        # JSON sycophancy judge
        self.local_json_judge = LocalVLLMJSONJudge(
            base_url=JUDGE_BASE_URL,
            model=JUDGE_MODEL,
            concurrency=self.args.judge_concurrency,
            timeout=30.0,
        )
        self.judge_scorer_system_prompt = JUDGE_SCORER_SYSTEM_PROMPT

        if accelerator.is_main_process:
            print(f"[init] actor={self.args.base_model}, scorer={JUDGE_MODEL} @ {JUDGE_BASE_URL}")
            print(f"[init] accelerate device={accelerator.device}, mixed_precision={accelerator.mixed_precision}")
            print(f"[init] steering metrics CSV -> {self.args.steering_log_csv}")
            print(f"[init] training metrics CSV -> {self.args.train_metrics_csv}")

        # === Steering tracking setup ===
        os.makedirs(os.path.dirname(self.args.steering_log_csv) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.args.steering_plot_png) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.args.csv_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.args.train_metrics_csv) or ".", exist_ok=True)
        if self.args.save_steering_snapshots:
            os.makedirs(self.args.steering_snapshot_dir, exist_ok=True)

        # Snapshot initial steering vectors
        self._steering_init = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if "steering_hooks" in name and name.endswith("vec_ln1.vector")
        }

        # CSV headers
        if accelerator.is_main_process and (not os.path.exists(self.args.steering_log_csv) or os.path.getsize(self.args.steering_log_csv) == 0):
            with open(self.args.steering_log_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "phase",
                    "total_norm",
                    "mean_norm",
                    "max_norm",
                    "total_drift",
                    "mean_drift",
                    "max_drift",
                ])

        if accelerator.is_main_process and (not os.path.exists(self.args.train_metrics_csv) or os.path.getsize(self.args.train_metrics_csv) == 0):
            with open(self.args.train_metrics_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # Add KL/token, Entropy/token, and the coefficients used this phase
                w.writerow([
                    "phase",
                    "mean_sycophancy", "std_sycophancy",
                    "mean_reward", "std_reward",
                    "mean_kl_per_token", "mean_entropy_per_token",
                    "kl_coef_used", "ent_coef_used",
                    "num_rollouts", "batch_size_per_rollout"
                ])


        # initial logs & metrics
        self._log_steering_param_stats(tag="init")
        self._log_steering_metrics_and_snapshot(phase_idx=-1, save_snapshot=False)

    # ---------- Steering diagnostics helpers ----------
    def _iter_steering_params(self) -> List[Tuple[str, t.Tensor]]:
        return [
            (name, p)
            for name, p in self.model.named_parameters()
            if "steering_hooks" in name and name.endswith("vec_ln1.vector")
        ]

    def _log_steering_param_stats(self, tag: str = ""):
        if not accelerator.is_main_process:
            return
        norms = []
        with t.no_grad():
            for name, p in self._iter_steering_params():
                norms.append((name, float(p.norm().item())))
        norms.sort(key=lambda x: x[0])
        head = [(n, round(v, 6)) for n, v in norms[:5]]
        print(f"[steering::{tag}] {len(norms)} vectors; L2 norms (first 5): {head}")

    def _steering_metrics(self) -> Dict[str, float]:
        with t.no_grad():
            norms, drifts = [], []
            for name, p in self._iter_steering_params():
                nrm = float(p.norm().item()); norms.append(nrm)
                diff = p - self._steering_init[name]
                drifts.append(float(diff.norm().item()))
            total_norm = float(sum(norms))
            mean_norm = float(total_norm / max(1, len(norms)))
            max_norm = float(max(norms) if norms else 0.0)
            total_drift = float(sum(drifts))
            mean_drift = float(total_drift / max(1, len(drifts)))
            max_drift = float(max(drifts) if drifts else 0.0)
        return {
            "total_norm": total_norm,
            "mean_norm": mean_norm,
            "max_norm": max_norm,
            "total_drift": total_drift,
            "mean_drift": mean_drift,
            "max_drift": max_drift,
        }

    def _steering_drift_report(self):
        if not accelerator.is_main_process:
            return
        drifts = []
        with t.no_grad():
            for name, p in self._iter_steering_params():
                diff = p - self._steering_init[name]
                l2 = float(diff.norm().item())
                base = float(self._steering_init[name].norm().item()) + 1e-8
                rel = l2 / base
                drifts.append((name, l2, rel))
        drifts.sort(key=lambda x: x[1], reverse=True)
        top = [(n, round(l2, 6), round(rel, 6)) for n, l2, rel in drifts[:5]]
        print(f"[steering::drift] top L2 movements (name, abs, rel): {top}")

    def _save_steering_snapshot(self, phase_idx: int):
        if not (accelerator.is_main_process and self.args.save_steering_snapshots):
            return
        state = {}
        with t.no_grad():
            for name, p in self._iter_steering_params():
                state[name] = p.detach().cpu().clone()
        path = os.path.join(self.args.steering_snapshot_dir, f"steering_phase_{phase_idx:03d}.pt")
        t.save(state, path)
        print(f"[steering::snapshot] saved -> {path}")

    def _log_steering_metrics_and_snapshot(self, phase_idx: int, save_snapshot: bool = True):
        metrics = self._steering_metrics()
        if accelerator.is_main_process:
            with open(self.args.steering_log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    phase_idx,
                    metrics["total_norm"],
                    metrics["mean_norm"],
                    metrics["max_norm"],
                    metrics["total_drift"],
                    metrics["mean_drift"],
                    metrics["max_drift"],
                ])
            print(f"[steering::metrics] phase={phase_idx} " +
                  " ".join([f"{k}={metrics[k]:.6f}" for k in metrics]))
            if self.args.use_wandb:
                wb = {f"steering/{k}": v for k, v in metrics.items()}
                wb["phase"] = phase_idx
                wandb.log(wb)
            if save_snapshot and self.phase % 10 == 0:
                self._save_steering_snapshot(phase_idx)
    
    def _current_ent_coef(self, phase_idx: int) -> float:
        """Return entropy coefficient for this phase (annealed or fixed)."""
        if getattr(self.args, "use_entropy_anneal", False):
            tfrac = min(1.0, max(0.0, phase_idx / max(1, self.args.ent_warmup_phases)))
            return (1.0 - tfrac) * self.args.ent_coef_start + tfrac * self.args.ent_coef_end
        else:
            # fall back to whatever is in RLHFArgs.ent_coef
            return self.args.ent_coef

    def _maybe_update_kl_coef(self, observed_kl_per_token: float):
        """Simple proportional controller for KL coefficient."""
        if not getattr(self.args, "use_adaptive_kl", False):
            return
        target = self.args.kl_target_nats
        if observed_kl_per_token > target:
            self.args.kl_coef = min(self.args.kl_coef * self.args.kl_up, self.args.kl_coef_max)
        else:
            self.args.kl_coef = max(self.args.kl_coef * self.args.kl_down, self.args.kl_coef_min)


    def rollout_phase(self):
        """
        [FIXED (Comment 0)] This is the correct, active RLOO implementation.
        
        Multi-rollout phase:
        - For each prompt, generate K continuations (K = rollouts_per_phase)
        - Judge all continuations at once
        - Compute per-prompt RLOO advantages, per-prompt std-normalization
        - Broadcast per-sequence advantages across generated tokens
        """
        K = max(1, int(self.args.rollouts_per_phase))  # K samples per prompt
        B = len(self.actor_prompts)                     # number of distinct prompts
        if accelerator.is_main_process:
            print(f"\n[phase {self.phase}] generating {K} samples per prompt (total {B*K})...")

        # === Step 1: generate all samples ===
        all_sample_ids = []   # shape [B*K, S]
        all_continuations = []  # detok last G tokens

        for k in range(K):
            sample_ids, _ = get_samples(
                self.model,
                prompts=self.actor_prompts,
                gen_len=self.args.gen_len,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                prepend_bos=self.args.prepend_bos,
            )
            gen_only = sample_ids[:, -self.args.gen_len:]
            
            # [FIX (Comment 1)] Use batched to_string
            continuations = self.model.to_string(gen_only)

            all_sample_ids.append(sample_ids)
            all_continuations.extend(continuations)

            if accelerator.is_main_process and k == 0:
                print(f"[rollout k={k}] sample continuation: {repr(continuations[0][:160])}")

        all_sample_ids = t.cat(all_sample_ids, dim=0)  # [B*K, S]

        # === Step 2: judge ===
        if accelerator.is_main_process:
            print(f"[phase {self.phase}] calling judge for {len(all_continuations)} samples ...")
        all_user_prompts = self.judge_user_prompts * K

        syco_scores = asyncio.run(
            self.local_json_judge.score_batch_syco(
                system_prompt=self.judge_scorer_system_prompt,
                user_prompts=all_user_prompts,
                assistant_replies=all_continuations,
            )
        )
        syco = t.tensor(syco_scores, dtype=t.bfloat16, device=device)
        syco = t.nan_to_num(syco, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        rewards = syco  # [B*K]

        if accelerator.is_main_process:
            print("[judge] sycophancy (first 5):", [round(float(s), 3) for s in syco[:5]])
            print("[judge] rewards (first 5):", [round(float(r), 3) for r in rewards[:5]])

        # === Step 3: per-prompt RLOO + std-norm ===
        # This is the CORRECT RLOO baseline calculation
        rewards_per_prompt = rewards.view(B, K)  # [B, K]
        # per prompt leave-one-out baseline:
        sum_r = rewards_per_prompt.sum(dim=1, keepdim=True)
        K_f = rewards_per_prompt.shape[1]
        
        # Ensure K_f-1 is not zero if K=1
        loo_baseline = (sum_r - rewards_per_prompt) / max(1, K_f - 1) 
        
        advantages_seq = rewards_per_prompt - loo_baseline  # [B, K]
        
        # per-prompt std norm BUT NEEDS NORMALISING ACROSS ALL ADV:
        global_adv_std = advantages_seq.std().clamp_min(1e-6) # [scalar]
        advantages_seq = advantages_seq / global_adv_std

        # broadcast to tokens:
        advantages = einops.repeat(
            advantages_seq, "b k -> (b k) g", g=self.args.gen_len
        )  # [B*K, G]

        # reshape sample_ids back to [B*K, S]:
        sample_ids_flat = all_sample_ids.detach().clone()  # already flattened

        # === Step 4: pack memory ===
        new_args = dataclasses.replace(self.args, batch_size=sample_ids_flat.shape[0])
        new_args.minibatch_size = new_args.batch_size // new_args.num_minibatches
        memory = ReplayMemory(args=new_args, sample_ids=sample_ids_flat, advantages=advantages)

        # metrics:
        self._last_mean_syco = float(syco.mean().item())
        self._last_std_syco = float(syco.std().item()) # Stash for logging
        self._last_mean_reward = float(rewards_per_prompt.mean().item())
        self._last_std_reward = float(rewards_per_prompt.std().item()) # Stash for logging

        # CSV logging:
        if accelerator.is_main_process:
            with open(self.args.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(["phase", "prompt_idx", "sample_idx", "input_prompt_raw",
                                "actor_continuation", "sycophancy_0_1", "reward"])
                for b in range(B):
                    for k in range(K):
                        idx = b*K + k
                        w.writerow([self.phase, b, k,
                                    self.judge_user_prompts[b],
                                    all_continuations[idx],
                                    float(syco[idx].item()),
                                    float(syco[idx].item())])
            print(f"[phase {self.phase}] wrote rollout data -> {self.args.csv_path}")

        del all_sample_ids, all_continuations, syco, rewards, rewards_per_prompt
        t.cuda.empty_cache()
        return memory



    # ---------- Training objective / rollout ----------
    def compute_rlhf_objective(self, minibatch: ReplayMinibatch):
        """
        RL objective:
        E[ A * log pi ] + ent_coef * H - kl_coef * KL,
        all terms averaged per token over generated slice.
        Returns: (objective, metrics_dict)
        """
        B, S = minibatch.sample_ids.shape
        G = self.args.gen_len
        
        # [FIX (Comment 8)] We need prefix_len to call get_logprobs correctly
        prefix_len = S - G
        assert prefix_len > 0, "Prefix length must be > 0"

        # Forward actor + ref
        logits_actor = self.model.forward_with_steering(minibatch.sample_ids)  # [B,S,V]
        with t.no_grad():
            logits_ref = self.ref_model(minibatch.sample_ids)                  # [B,S,V]

        # Token logprobs
        # [FIX (Comment 8)] Pass prefix_len to get correct slice
        logprobs_gen     = get_logprobs(logits_actor, minibatch.sample_ids, prefix_len=prefix_len) # [B,G]
        ref_logprobs_gen = get_logprobs(logits_ref,   minibatch.sample_ids, prefix_len=prefix_len) # [B,G]
        advantages       = minibatch.advantages[:, -G:]                                            # [B,G]

        # Per-token entropy (actor)
        logp_actor = logits_actor.log_softmax(-1)     # [B,S,V]
        p_actor    = logp_actor.exp()                 # [B,S,V]
        H_tok      = -(p_actor * logp_actor).sum(dim=-1)     # [B,S]
        ent_gen    = H_tok[:, -G:].mean()             # scalar

        # Per-token KL(actor||ref)
        logp_ref = logits_ref.log_softmax(-1)         # [B,S,V]
        kl_tok   = (p_actor * (logp_actor - logp_ref)).sum(dim=-1)  # [B,S]
        kl_gen_tok = kl_tok[:, -G:]
        kl_gen = t.nan_to_num(kl_gen_tok, nan=0.0, posinf=1.0, neginf=0.0).mean()

        # Policy term (sequence mean of token-sum)
        policy_obj = (advantages * logprobs_gen).sum(dim=-1).mean()

        # Phase-aware entropy coef (annealed)
        ent_coef_used = self._current_ent_coef(self.phase)

        # Combine
        objective = policy_obj + ent_coef_used * ent_gen - self.args.kl_coef * kl_gen

        # Metrics to aggregate/log this phase
        metrics = {
            "kl_per_token": float(kl_gen.detach().item()),
            "entropy_per_token": float(ent_gen.detach().item()),
            "actor_logprob_per_token": float(logprobs_gen.mean().detach().item()),
            "ref_logprob_per_token": float(ref_logprobs_gen.mean().detach().item()),
            "ent_coef_used": float(ent_coef_used),
            "kl_coef_used": float(self.args.kl_coef),
        }

        # Free big tensors early
        del logits_actor, logits_ref, logp_actor, p_actor, logp_ref, H_tok, kl_tok
        return objective, metrics


    def train(self):
        """
        [FIXED (Comment 0)] This is the new, correct training loop.
        It calls the correct `rollout_phase` and logs metrics.
        """
        self.step = 0

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.init(project=self.args.wandb_project_name, entity=self.args.wandb_entity, name=self.run_name, config=self.args)

        import numpy as np

        for self.phase in range(self.args.total_phases):
            
            # 1. Call the CORRECT rollout function. 
            # This one function handles all K rollouts AND computes the correct advantages.
            big_memory = self.rollout_phase() 
            
            # 2. Get pooled metrics from this phase (stashed by rollout_phase)
            mean_syco = getattr(self, "_last_mean_syco", float('nan'))
            std_syco = getattr(self, "_last_std_syco", float('nan'))
            mean_reward = getattr(self, "_last_mean_reward", float('nan'))
            std_reward = getattr(self, "_last_std_reward", float('nan'))
            num_roll = self.args.rollouts_per_phase

            # 3. Do the learning on the concatenated batch
            loss = self.learning_phase(big_memory)

            # 4. Pull per-phase training metrics (KL, entropy, etc.)
            tm = getattr(self, "_last_train_metrics", {})
            mean_kl = tm.get("mean_kl_per_token", float('nan'))
            mean_ent = tm.get("mean_entropy_per_token", float('nan'))
            used_kl_coef = tm.get("kl_coef_used", self.args.kl_coef)
            used_ent_coef = tm.get("ent_coef_used", self._current_ent_coef(self.phase))

            # 5. Log everything
            if accelerator.is_main_process:
                print(f"[phase {self.phase+1}/{self.args.total_phases}] objective={loss:.6f}")
                print(f"[phase {self.phase}] pooled mean sycophancy={mean_syco:.4f} | mean reward={mean_reward:.4f} "
                      f"| KL/token={mean_kl:.4f} | H/token={mean_ent:.4f} | kl_coef={used_kl_coef:.3f} | ent_coef={used_ent_coef:.5f}")

                with open(self.args.train_metrics_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        self.phase,
                        mean_syco, std_syco,
                        mean_reward, std_reward,
                        mean_kl, mean_ent,
                        used_kl_coef, used_ent_coef,
                        num_roll,
                        self.args.batch_size,
                    ])

                if hasattr(self, "_log_steering_param_stats"):
                    self._log_steering_param_stats(tag=f"phase{self.phase+1}")
                if hasattr(self, "_steering_drift_report"):
                    self._steering_drift_report()

            if hasattr(self, "_log_steering_metrics_and_snapshot"):
                self._log_steering_metrics_and_snapshot(phase_idx=self.phase)

            # 6. Cleanup
            del big_memory
            t.cuda.empty_cache()

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.finish()

