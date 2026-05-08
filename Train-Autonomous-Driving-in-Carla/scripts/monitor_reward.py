"""
Plan-06 reward training monitor.

Reads episode_log.csv (written by continuous_driver.py during training),
prints summary stats, and plots reward curve + steps to PNG.

Usage:
    python monitor_reward.py                           # default: baseline/seed0
    python monitor_reward.py --csv <path>              # specify other CSV
    python monitor_reward.py --window 100              # change rolling window
    python monitor_reward.py --output <path.png>       # custom output path
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",
                   default="/root/autodl-tmp/runs/baseline/seed0/episode_log.csv",
                   help="path to episode_log.csv")
    p.add_argument("--output", default=None,
                   help="output PNG path (default: <csv-dir>/reward_curve.png)")
    p.add_argument("--window", type=int, default=50,
                   help="rolling-mean window size in episodes (default 50)")
    p.add_argument("--last", type=int, default=10,
                   help="show last N episodes' raw rewards in summary (default 10)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print("[ERR] CSV not found: " + args.csv)
        sys.exit(1)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv)
    n = len(df)
    if n == 0:
        print("[ERR] CSV is empty (no episodes yet)")
        sys.exit(1)

    df = df.sort_values("episode").reset_index(drop=True)
    df["rolling_mean"] = df["total_reward"].rolling(args.window, min_periods=1).mean()

    last_w = df.tail(args.window)
    overall_max = df["total_reward"].max()
    overall_max_ep = int(df.loc[df["total_reward"].idxmax(), "episode"])

    # === Print summary ===
    print("=" * 60)
    print("Reward Monitor: " + args.csv)
    print("=" * 60)
    print("Total episodes: " + str(n))
    print("Last %d rewards: %s" % (
        args.last,
        [round(r, 2) for r in df["total_reward"].tail(args.last).tolist()]
    ))
    print("")
    print("Last %d episodes:" % len(last_w))
    print("  rolling_mean = %.2f" % last_w["total_reward"].mean())
    print("  std          = %.2f" % last_w["total_reward"].std())
    print("  min          = %.2f" % last_w["total_reward"].min())
    print("  max          = %.2f" % last_w["total_reward"].max())
    print("  avg steps    = %.1f" % last_w["total_steps"].mean())
    print("")
    print("Overall:")
    print("  mean reward  = %.2f" % df["total_reward"].mean())
    print("  max reward   = %.2f (at episode %d)" % (overall_max, overall_max_ep))
    print("  total tsteps = %d" % df["timestep_at_done"].iloc[-1])
    print("")
    print("Failure mode distribution (last %d episodes):" % len(last_w))
    fm = last_w["done_reason"].value_counts(normalize=True)
    for k, v in fm.items():
        print("  %-22s  %5.1f%%" % (k, v * 100))
    print("")

    # === Suggest reward threshold ===
    if len(df) >= args.window * 2:
        # Compare last window mean to median of all earlier windows
        recent = last_w["total_reward"].mean()
        earlier = df.iloc[:-args.window]["total_reward"]
        earlier_recent = earlier.tail(args.window).mean() if len(earlier) >= args.window else earlier.mean()
        delta = recent - earlier_recent
        print("Trend (last %d vs prior %d): %+.2f reward delta" % (args.window, args.window, delta))
        if abs(delta) < 5.0 and len(df) > 200:
            suggested = recent * 0.85
            print("  -> looks like plateau. Suggested REWARD_THRESHOLD = %.0f (85%% of plateau mean)" % suggested)
        else:
            print("  -> still climbing/changing, NOT plateau yet")
    print("")

    # === Plot ===
    out = args.output or os.path.join(os.path.dirname(args.csv), "reward_curve.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Top: reward
    axes[0].plot(df["episode"], df["total_reward"], alpha=0.25, color="steelblue", label="raw")
    axes[0].plot(df["episode"], df["rolling_mean"], linewidth=2, color="navy",
                 label="rolling mean (w=%d)" % args.window)
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Plan-06 Baseline Training (n=%d episodes)" % n)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # Mid: steps per episode
    axes[1].plot(df["episode"], df["total_steps"], alpha=0.5, color="orange", label="steps")
    axes[1].plot(df["episode"], df["total_steps"].rolling(args.window, min_periods=1).mean(),
                 linewidth=2, color="darkred", label="rolling steps")
    axes[1].set_ylabel("Steps / episode")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # Bottom: failure mode stacked area
    fm_pivot = df.copy()
    fm_pivot["bucket"] = (fm_pivot["episode"] // max(1, n // 30)).astype(int)
    fm_counts = fm_pivot.groupby(["bucket", "done_reason"]).size().unstack(fill_value=0)
    fm_pct = fm_counts.div(fm_counts.sum(axis=1), axis=0)
    bucket_eps = fm_pivot.groupby("bucket")["episode"].mean()
    fm_pct.index = bucket_eps[fm_pct.index].values
    fm_pct.plot.area(ax=axes[2], alpha=0.7, stacked=True)
    axes[2].set_ylabel("Failure mode share")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()
    print("[OK] saved figure to: " + out)


if __name__ == "__main__":
    main()
