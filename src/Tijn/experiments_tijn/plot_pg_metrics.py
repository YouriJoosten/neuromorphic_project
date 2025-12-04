import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(ax, df, metric, window):
    for agent, sub in df.groupby("agent"):
        x = sub["run"]
        y = moving_average(sub[metric], window)
        ax.plot(x, y, marker="o", label=str(agent))
    ax.set_title(metric)
    ax.set_xlabel("episode")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)


def main():
    prs = argparse.ArgumentParser(description="Plot policy gradient learning curves from training_log.csv.")
    prs.add_argument("--exp-dir", required=True, help="Experiment directory containing training_log.csv.")
    prs.add_argument(
        "--metrics",
        default="episode_return,discounted_return,mean_return,R_bar,theta_norm,theta_change",
        help="Comma-separated list of metrics to plot.",
    )
    prs.add_argument("--smooth", type=int, default=1, help="Moving-average window size.")
    prs.add_argument("--outfile", default="pg_learning_curves.png", help="Output plot filename (saved inside exp-dir).")
    args = prs.parse_args()

    exp_dir = Path(args.exp_dir)
    log_path = exp_dir / "training_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"No training_log.csv found at {log_path}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError(f"{log_path} is empty.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics requested to plot.")

    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        plot_metric(ax, df, metric, args.smooth)

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    fig.tight_layout()
    out_path = exp_dir / args.outfile
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
