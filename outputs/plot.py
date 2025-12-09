import argparse
import glob
import os
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set(
    style="darkgrid",
    rc={
        "figure.figsize": (7.2, 4.45),
        "text.usetex": False,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.size": 15,
        "figure.autolayout": True,
        "axes.titlesize": 16,
        "axes.labelsize": 17,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 15,
    },
)
colors = sns.color_palette("colorblind", 4)
dashes_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(colors)
colors = cycle(colors)


def moving_average(interval, window_size):
    if window_size <= 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")  # convert NaN string to NaN value

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean.values, ma)
        std = moving_average(std.values, ma)
        x = np.array(sorted(df.groupby(xaxis)[xaxis].mean().keys().values))
    else:
        x = df.groupby(xaxis)[xaxis].mean().keys().values

    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)


def plot_episode_summary(combined_df, ma=5, out_folder: Path = None, use_delta_waiting: bool = False):
    """Aggregate per-episode waiting-time metrics and plot them (sum and mean).

    If use_delta_waiting=True, we first compute the per-step difference of
    system_total_waiting_time *per episode* and aggregate those deltas.
    This is closer to the shaped reward (-Δwaiting) often used in RL.
    """
    if "episode" not in combined_df.columns:
        raise ValueError("combined dataframe must contain an 'episode' column to create episode summary.")

    combined_df["episode"] = combined_df["episode"].astype(int)

    # ---- optionally compute per-step delta of total waiting time ----
    df = combined_df.sort_values(["episode", "step"]).copy()
    if use_delta_waiting:
        # per-episode difference of total waiting time
        df["delta_total_wait"] = df.groupby("episode")["system_total_waiting_time"].diff().fillna(0.0)

        agg = df.groupby("episode").agg(
            steps=("step", "count"),
            mean_waiting=("delta_total_wait", "mean"),
            sum_waiting=("delta_total_wait", "sum"),
            max_waiting=("delta_total_wait", "max"),
        )
        ylabel_sum = "sum Δ system_total_waiting_time"
        ylabel_mean = "mean Δ system_total_waiting_time"
        sum_label = "sum_delta_waiting"
        mean_label = "mean_delta_waiting"
    else:
        agg = df.groupby("episode").agg(
            steps=("step", "count"),
            mean_waiting=("system_mean_waiting_time", "mean"),
            sum_waiting=("system_total_waiting_time", "sum"),
            max_waiting=("system_total_waiting_time", "max"),
        )
        ylabel_sum = "sum system_total_waiting_time"
        ylabel_mean = "system_mean_waiting_time"
        sum_label = "sum_waiting"
        mean_label = "mean_waiting"

    agg = agg.reset_index()

    # smoothing
    roll = lambda arr: moving_average(arr, ma) if ma > 1 else arr

    # ---- sum_waiting / sum_delta_waiting plot ----
    plt.figure(figsize=(9, 4))
    plt.plot(agg["episode"], agg["sum_waiting"], alpha=0.3, label=sum_label)
    sm = roll(agg["sum_waiting"].values)
    x_sm = agg["episode"].values
    if len(sm) < len(x_sm):  # adjust x for 'same' convolution length
        offset = (len(x_sm) - len(sm)) // 2
        x_sm = x_sm[offset : offset + len(sm)]
    plt.plot(x_sm, sm, label=f"{sum_label} (ma={ma})")
    plt.xlabel("episode")
    plt.ylabel(ylabel_sum)
    plt.legend()
    plt.grid(True)
    if out_folder:
        fname = "episode_sum_delta_waiting.png" if use_delta_waiting else "episode_sum_waiting.png"
        plt.savefig(out_folder / fname, dpi=150)
    plt.tight_layout()

    # ---- mean_waiting / mean_delta_waiting plot ----
    plt.figure(figsize=(9, 4))
    plt.plot(agg["episode"], agg["mean_waiting"], alpha=0.3, label=mean_label)
    sm2 = roll(agg["mean_waiting"].values)
    x_sm2 = agg["episode"].values
    if len(sm2) < len(x_sm2):
        offset = (len(x_sm2) - len(sm2)) // 2
        x_sm2 = x_sm2[offset : offset + len(sm2)]
    plt.plot(x_sm2, sm2, label=f"{mean_label} (ma={ma})")
    plt.xlabel("episode")
    plt.ylabel(ylabel_mean)
    plt.legend()
    plt.grid(True)
    if out_folder:
        fname = "episode_mean_delta_waiting.png" if use_delta_waiting else "episode_mean_waiting.png"
        plt.savefig(out_folder / fname, dpi=150)
    plt.tight_layout()

    return agg



def plot_training_log(training_log_path: Path, ma=5, out_folder: Path = None):
    """Plot basic diagnostics from training_log.csv if present."""
    if not training_log_path.exists():
        print("training_log not found at", training_log_path)
        return None

    tl = pd.read_csv(training_log_path)
    # ensure numeric
    for c in ["episode_return", "discounted_return", "mean_return", "R_bar", "theta_norm", "theta_change"]:
        if c in tl.columns:
            tl[c] = pd.to_numeric(tl[c], errors="coerce")

    # smoothing helper
    roll = lambda arr: moving_average(arr, ma) if ma > 1 else arr

    # theta diagnostics
    plt.figure(figsize=(9, 3))
    if "theta_norm" in tl.columns:
        plt.plot(tl.index + 1, tl["theta_norm"], alpha=0.3, label="theta_norm")
        sm = roll(tl["theta_norm"].fillna(0).values)
        x_sm = np.arange(1, len(tl) + 1)
        if len(sm) < len(x_sm):
            offset = (len(x_sm) - len(sm)) // 2
            x_sm = x_sm[offset : offset + len(sm)]
        plt.plot(x_sm, sm, label=f"theta_norm (ma={ma})")
    if "theta_change" in tl.columns:
        plt.plot(tl.index + 1, tl["theta_change"], alpha=0.3, label="theta_change")
        smc = roll(tl["theta_change"].fillna(0).values)
        x_smc = np.arange(1, len(tl) + 1)
        if len(smc) < len(x_smc):
            offset = (len(x_smc) - len(smc)) // 2
            x_smc = x_smc[offset : offset + len(smc)]
        plt.plot(x_smc, smc, label=f"theta_change (ma={ma})")
    plt.xlabel("episode")
    plt.yscale("symlog")
    plt.legend()
    plt.grid(True)
    if out_folder:
        plt.savefig(out_folder / "training_theta_diag.png", dpi=150)
    plt.tight_layout()

    # R_bar and returns
    plt.figure(figsize=(9, 3))
    if "episode_return" in tl.columns:
        plt.plot(tl.index + 1, tl["episode_return"], alpha=0.3, label="episode_return")
        smr = roll(tl["episode_return"].fillna(0).values)
        x_smr = np.arange(1, len(tl) + 1)
        if len(smr) < len(x_smr):
            offset = (len(x_smr) - len(smr)) // 2
            x_smr = x_smr[offset : offset + len(smr)]
        plt.plot(x_smr, smr, label=f"episode_return (ma={ma})")
    if "R_bar" in tl.columns:
        plt.plot(tl.index + 1, tl["R_bar"], alpha=0.8, label="R_bar")
    plt.xlabel("episode")
    plt.legend()
    plt.grid(True)
    if out_folder:
        plt.savefig(out_folder / "training_return_Rbar.png", dpi=150)
    plt.tight_layout()

    return tl


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
    )
    prs.add_argument(
    "--delta-waiting",
    action="store_true",
    help=(
        "When used with --episode-summary, aggregate per-episode statistics on the "
        "per-step difference of system_total_waiting_time instead of the raw value. "
        "This approximates the shaped RL reward and makes learning trends clearer."
    ),
)
    prs.add_argument("-f", nargs="+", required=True, help="Measures files (glob prefix allowed)\n")
    prs.add_argument("-l", nargs="+", default=None, help="File's legends\n")
    prs.add_argument("-t", type=str, default="", help="Plot title\n")
    prs.add_argument("-yaxis", type=str, default="system_total_waiting_time", help="The column to plot.\n")
    prs.add_argument("-xaxis", type=str, default="step", help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
    prs.add_argument("-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n")
    prs.add_argument("-ylabel", type=str, default="Total waiting time (s)", help="Y axis label.\n")
    prs.add_argument("-output", type=str, default=None, help="PDF output filename.\n")
    prs.add_argument(
        "--episode-summary",
        action="store_true",
        help="If set and input is combined per-step CSV (contains 'episode'), produce per-episode waiting-time summary plots.",
    )
    prs.add_argument(
        "--training-log",
        type=str,
        default=None,
        help="Optional explicit path to training_log.csv to plot diagnostics (if present). If omitted the script will look in the same folder as input files.",
    )

    args = prs.parse_args()
    labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(args.f))])

    plt.figure()

    for file in args.f:
        main_df = pd.DataFrame()
        for fpath in glob.glob(file + "*"):
            df = pd.read_csv(fpath, sep=args.sep)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        # If user requests episode summary and dataframe has 'episode' column -> produce summaries and diagnostics
        try:
            if args.episode_summary and "episode" in main_df.columns:
                # save outputs in same folder as first matched file
                first_match = glob.glob(file + "*")[0]
                out_folder = Path(first_match).resolve().parent
                agg = plot_episode_summary(
                    main_df,
                    ma=args.ma,
                    out_folder=out_folder,
                    use_delta_waiting=args.delta_waiting,
                )

                # try to find training_log automatically if not provided
                if args.training_log:
                    training_log_path = Path(args.training_log)
                else:
                    training_log_path = out_folder / "training_log.csv"
                plot_training_log(training_log_path, ma=args.ma, out_folder=out_folder)

                # also write per-episode summary csv
                if out_folder:
                    agg.to_csv(out_folder / "per_episode_summary.csv", index=False)
                    print("Saved per_episode_summary.csv to", out_folder)
                # continue to next input file (do not also call plot_df)
                continue
        except Exception as e:
            print("Episode-summary plotting failed:", e)

        # default behaviour: plot time-series grouped by xaxis
        try:
            plot_df(main_df, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)
        except Exception as e:
            print("Plotting failed for file pattern", file, "error:", e)

    plt.title(args.t)
    plt.ylabel(args.ylabel)
    plt.xlabel(args.xlabel)
    plt.ylim(bottom=0)

    if args.output is not None:
        plt.savefig(args.output + ".pdf", bbox_inches="tight")

    plt.show()
