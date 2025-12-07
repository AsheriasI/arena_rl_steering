import asyncio

from .config import RAW_PROMPTS_PATH, RLOOArgs, accelerator
from .plots import _maybe_plot_steering_csv, _maybe_plot_training_csv
from .trainers_rloo import RLOOTrainer
from .utils import load_raw_prompts


if __name__ == "__main__":
    raw_prompts = load_raw_prompts(str(RAW_PROMPTS_PATH))

    # Exact Llama 3.1 chat format (actor input; do NOT prepend BOS separately)
    formatted_prompts = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + p
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        for p in raw_prompts
    ]

    rloo_args = RLOOArgs(
        use_wandb=False,
        total_phases=15,

        # batch = number of prompts
        batch_size=len(formatted_prompts),  # must equal number of prompts
        num_minibatches=3,               # keep as-is; you asked not to use this as a knob
        batches_per_learning_phase=12,       # <-- your chosen OOM/stability knob

        gen_len=120,
        temperature=0.7,
        top_k=None,
        prepend_bos=False,

        steering_layer_indices=None,       # all layers
        steering_init_scale=0.2,

        # Actor vs judge prompts
        actor_prompts_inline=formatted_prompts,
        judge_user_prompts_inline=raw_prompts,

        # Multi-rollout
        rollouts_per_phase=12,               # <-- 8 rollouts each phase

        # Opt (stabilized)
        base_lr=5e-4,
        max_grad_norm=2.0,
        # ent_coef=0.001,
        # kl_coef=0.6,


        use_adaptive_kl=True,       # or False
        kl_target_nats=0.05,
        kl_coef=0.6,                # initial
        kl_coef_min=0.05,
        kl_coef_max=3.0,
        kl_up=1.05,
        kl_down=0.97,

        use_entropy_anneal=True,     # or False
        ent_coef_start=0.005,
        ent_coef_end=0.0,
        ent_warmup_phases=2,
    )    

    trainer = RLOOTrainer(rloo_args)
    
    try:
        trainer.train()
    except (KeyboardInterrupt, Exception) as e:
        print(f"Training interrupted or failed: {e}")
    finally:
        # [FIX (Comment 2)] Ensure client is closed
        if accelerator.is_main_process:
            print("Cleaning up judge client...")
            try:
                asyncio.run(trainer.local_json_judge.aclose())
                print("Cleanup complete.")
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    print("Event loop already closed, skipping client cleanup.")
                else:
                    print(f"Error during cleanup: {e}")

    if accelerator.is_main_process and rloo_args.plot_steering_after_training:
        _maybe_plot_steering_csv(rloo_args.steering_log_csv, out_png=rloo_args.steering_plot_png)
        _maybe_plot_training_csv(rloo_args.train_metrics_csv, out_png=rloo_args.plot_train_png)

