import csv
import os


def _maybe_plot_steering_csv(csv_path: str, out_png: str = "steering_metrics.png"):
    try:
        import matplotlib.pyplot as plt
        xs, total_norm, total_drift = [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                xs.append(int(float(row["phase"])))
                total_norm.append(float(row["total_norm"]))
                total_drift.append(float(row["total_drift"]))
        if not xs:
            print("[plot] no rows to plot")
            return
        plt.figure(figsize=(8,4.5))
        plt.plot(xs, total_norm, label="total_norm")
        plt.plot(xs, total_drift, label="total_drift")
        plt.xlabel("phase"); plt.ylabel("value")
        plt.title("Steering vectors: total norm & total drift")
        plt.legend(); plt.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[plot] saved -> {out_png}")
    except Exception as e:
        print(f"[plot] skipped (matplotlib missing or error): {e}")


def _maybe_plot_training_csv(csv_path: str, out_png: str = "training_metrics.png"):
    """Create one PNG with 2 stacked subplots (no overlays): mean sycophancy, mean reward"""
    try:
        import matplotlib.pyplot as plt
        xs, mean_syco, mean_reward = [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                xs.append(int(float(row["phase"])))
                mean_syco.append(float(row["mean_sycophancy"]))
                mean_reward.append(float(row["mean_reward"]))
        if not xs:
            print("[plot-train] no rows to plot"); return
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(xs, mean_syco, marker="o")
        axes[0].set_ylabel("mean sycophancy")
        axes[0].set_title("Training metrics per phase (sycophancy-only reward)")

        axes[1].plot(xs, mean_reward, marker="o")
        axes[1].set_ylabel("mean reward")
        axes[1].set_xlabel("phase")

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"[plot-train] saved -> {out_png}")
    except Exception as e:
        print(f"[plot-train] skipped: {e}")

