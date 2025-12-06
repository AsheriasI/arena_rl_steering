import math
import time
from typing import Dict, List, Tuple

import torch as t
import wandb
from transformer_lens import HookedTransformer

from .config import RLHFArgs, accelerator, device
from .steering import HookedTransformerWithSteering


def get_optimizer(model: HookedTransformerWithSteering, base_lr: float, head_lr: float) -> t.optim.Optimizer:
    return t.optim.AdamW(
        [{"params": model.get_base_model_trainable_params(), "lr": base_lr}],
        eps=1e-6,  # stability for bf16
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


class RLHFTrainer:
    model: HookedTransformerWithSteering
    ref_model: HookedTransformer

    def __init__(self, args: RLHFArgs):
        t.manual_seed(args.seed)
        self.args = args
        self.run_name = f"{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"

        self.model = HookedTransformerWithSteering.from_pretrained(
            args.base_model,
            device_map="auto",
            dtype=t.bfloat16,
        ).to(device).train()

        self.ref_model = HookedTransformer.from_pretrained(
            args.base_model,
            device_map="auto",
            dtype=t.bfloat16,
        ).to(device).eval()

        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.args, self.model)
        self.model, self.optimizer, self.scheduler = accelerator.prepare(self.model, self.optimizer, self.scheduler)

        self.prefix_len = len(self.model.to_str_tokens(self.args.prefix, prepend_bos=self.args.prepend_bos))
        
        # [FIX (Comment 6)] Cache steering parameters
        self.steering_param_list = [
            p for name, p in self.model.named_parameters() 
            if "steering_hooks" in name and p.requires_grad
        ]
        if accelerator.is_main_process:
             print(f"[init] Caching {len(self.steering_param_list)} steering parameters for sanitization.")


    def compute_rlhf_objective(self, minibatch):
        raise NotImplementedError

    def learning_phase(self, memory):
        loss_val = 0.0
        minibatches = memory.get_minibatches()

        # collect metrics to average across minibatches
        mb_kl = []
        mb_ent = []
        mb_act_lp = []
        mb_ref_lp = []
        last_ent_coef_used = None
        last_kl_coef_used = None

        for minibatch in minibatches:
            self.optimizer.zero_grad()
            total_objective, mb_metrics = self.compute_rlhf_objective(minibatch)

            # --- Non-finite guard ---
            if not t.isfinite(total_objective):
                if accelerator.is_main_process:
                    print("[warn] objective is non-finite; skipping step")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            accelerator.backward(total_objective)

            if accelerator.is_main_process and (self.step % 50 == 0):
                gstats = []
                # [FIX (Comment 6)] Use cached list
                for p in self.steering_param_list:
                    if p.grad is not None and t.isfinite(p.grad).all():
                        gstats.append(float(p.grad.norm().item()))
                
                if gstats:
                    print(f"[steering::grads] norms (mean): {sum(gstats)/len(gstats):.6f}")

            accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # --- Parameter sanitization: kill NaN/Inf & clamp steering vectors ---
            # [FIXED (Comment 6)] Optimized loop
            with t.no_grad():
                for p in self.steering_param_list: # Use the cached list
                    mask = ~t.isfinite(p)
                    if mask.any():
                        p[mask] = 0.0
                    p.clamp_(min=-5.0, max=5.0)

            self.step += 1
            loss_val += float(total_objective.item())

            # collect
            mb_kl.append(mb_metrics["kl_per_token"])
            mb_ent.append(mb_metrics["entropy_per_token"])
            mb_act_lp.append(mb_metrics["actor_logprob_per_token"])
            mb_ref_lp.append(mb_metrics["ref_logprob_per_token"])
            last_ent_coef_used = mb_metrics["ent_coef_used"]
            last_kl_coef_used = mb_metrics["kl_coef_used"]

        # average
        n = max(1, len(minibatches))
        loss_val /= n
        mean_kl = float(sum(mb_kl)/n) if mb_kl else float('nan')
        mean_ent = float(sum(mb_ent)/n) if mb_ent else float('nan')
        mean_act_lp = float(sum(mb_act_lp)/n) if mb_act_lp else float('nan')
        mean_ref_lp = float(sum(mb_ref_lp)/n) if mb_ref_lp else float('nan')

        # step scheduler
        self.scheduler.step()

        # ===== Adaptive KL update (uses averaged KL for this phase step) =====
        if math.isfinite(mean_kl):
            self._maybe_update_kl_coef(mean_kl)

        # Stash for train() logging
        self._last_train_metrics = {
            "mean_kl_per_token": mean_kl,
            "mean_entropy_per_token": mean_ent,
            "mean_actor_logprob_per_token": mean_act_lp,
            "mean_ref_logprob_per_token": mean_ref_lp,
            "kl_coef_used": float(self.args.kl_coef),
            "ent_coef_used": float(last_ent_coef_used if last_ent_coef_used is not None else self._current_ent_coef(self.phase)),
        }
        return loss_val


    def train(self):
        self.step = 0

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.init(project=self.args.wandb_project_name, entity=self.args.wandb_entity, name=self.run_name, config=self.args)

        for self.phase in range(self.args.total_phases):
            # By default subclasses will implement rollout + learning.
            memory = self.rollout_phase()
            loss = self.learning_phase(memory)
            if accelerator.is_main_process:
                print(f"[phase {self.phase+1}/{self.args.total_phases}] objective={loss:.6f}")
                if hasattr(self, "_log_steering_param_stats"):
                    self._log_steering_param_stats(tag=f"phase{self.phase+1}")
                if hasattr(self, "_steering_drift_report"):
                    self._steering_drift_report()
            if hasattr(self, "_log_steering_metrics_and_snapshot"):
                self._log_steering_metrics_and_snapshot(phase_idx=self.phase)

        if self.args.use_wandb and accelerator.is_main_process:
            wandb.finish()

