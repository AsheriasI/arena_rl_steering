import asyncio
from pathlib import Path

from .config import RAW_PROMPTS_PATH, RLOOArgs, accelerator
from .plots import _maybe_plot_steering_csv, _maybe_plot_training_csv
from .trainers_rloo import RLOOTrainer
from .utils import load_raw_prompts, load_jsonc


if __name__ == "__main__":
    raw_prompts = load_raw_prompts(str(RAW_PROMPTS_PATH))
    config_path = Path(__file__).resolve().parent / "configs" / "rloo_default.jsonc"
    cfg_dict = load_jsonc(config_path)

    # Exact Llama 3.1 chat format (actor input; do NOT prepend BOS separately)
    formatted_prompts = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + p
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        for p in raw_prompts
    ]

    # Merge JSONC config with runtime-derived values
    cfg_dict["batch_size"] = len(formatted_prompts)
    cfg_dict["actor_prompts_inline"] = formatted_prompts
    cfg_dict["judge_user_prompts_inline"] = raw_prompts

    rloo_args = RLOOArgs(**cfg_dict)

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
                asyncio.run(trainer.judge.aclose())
                print("Cleanup complete.")
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    print("Event loop already closed, skipping client cleanup.")
                else:
                    print(f"Error during cleanup: {e}")

    if accelerator.is_main_process and rloo_args.plot_steering_after_training:
        _maybe_plot_steering_csv(rloo_args.steering_log_csv, out_png=rloo_args.steering_plot_png)
        _maybe_plot_training_csv(rloo_args.train_metrics_csv, out_png=rloo_args.plot_train_png)

