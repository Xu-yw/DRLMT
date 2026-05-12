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
    p.add_argument("--budget-tsteps", type=int, default=1000000,
                   help="Plan-06 Phase 1 budget stop in environment timesteps (default 1000000)")
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

    # === Spawn randomization audit ===
    print("Spawn randomization (last %d episodes):" % len(last_w))
    if "spawn_idx" in df.columns:
        spawn_series = last_w["spawn_idx"].dropna()
        if len(spawn_series) > 0:
            unique_spawns = int(spawn_series.nunique())
            print("  unique spawn_idx = %d / %d" % (unique_spawns, len(spawn_series)))
            top_counts = spawn_series.astype(int).value_counts().head(5)
            print("  top spawn_idx    = " + ", ".join("%s:%d" % (idx, cnt) for idx, cnt in top_counts.items()))
        else:
            print("  spawn_idx column exists but has no values")
    else:
        print("  spawn_idx column missing (old log schema)")
    if "start_x" in df.columns and "start_y" in df.columns:
        sx = last_w["start_x"].dropna()
        sy = last_w["start_y"].dropna()
        if len(sx) > 0 and len(sy) > 0:
            print("  start_x range    = %.2f .. %.2f" % (sx.min(), sx.max()))
            print("  start_y range    = %.2f .. %.2f" % (sy.min(), sy.max()))
    print("")

    # === Plan-06 Phase 1 stop conditions ===
    print("Plan-06 Stop Conditions:")
    total_tsteps = int(df["timestep_at_done"].iloc[-1])
    stop_a = False
    stop_b = total_tsteps >= args.budget_tsteps
    stop_c = False
    threshold_basis = None
    threshold_source = None

    if len(df) >= args.window * 3:
        recent_windows = []
        for start in [len(df) - args.window * 3, len(df) - args.window * 2, len(df) - args.window]:
            chunk = df.iloc[start:start + args.window]
            recent_windows.append({
                "start_episode": int(chunk["episode"].iloc[0]),
                "end_episode": int(chunk["episode"].iloc[-1]),
                "mean": float(chunk["total_reward"].mean()),
            })

        adj_changes = []
        for i in range(1, len(recent_windows)):
            prev = recent_windows[i - 1]["mean"]
            cur = recent_windows[i]["mean"]
            denom = abs(prev) if abs(prev) > 1e-9 else 1.0
            adj_changes.append(abs(cur - prev) / denom)
        first = recent_windows[0]["mean"]
        last = recent_windows[-1]["mean"]
        span_change = abs(last - first) / (abs(first) if abs(first) > 1e-9 else 1.0)
        stop_a = all(x < 0.10 for x in adj_changes) and span_change < 0.10

        print("  A plateau stop: %s" % ("YES" if stop_a else "NO"))
        for idx, win in enumerate(recent_windows, 1):
            print("    W%d ep %d-%d mean=%.2f" % (
                idx, win["start_episode"], win["end_episode"], win["mean"]
            ))
        print("    adjacent changes: %s" % ", ".join("%.2f%%" % (x * 100) for x in adj_changes))
        print("    W1->W3 change: %.2f%%" % (span_change * 100))

        rolling_full = df["total_reward"].rolling(args.window, min_periods=args.window).mean()
        best_idx = rolling_full.idxmax()
        best_mean = float(rolling_full.loc[best_idx])
        best_ep = int(df.loc[best_idx, "episode"])
        latest_mean = recent_windows[-1]["mean"]
        below_best = (best_mean - latest_mean) / (abs(best_mean) if abs(best_mean) > 1e-9 else 1.0)
        descending = recent_windows[0]["mean"] > recent_windows[1]["mean"] > recent_windows[2]["mean"]
        windows_after_best = recent_windows[0]["end_episode"] > best_ep
        stop_c = windows_after_best and descending and below_best > 0.20

        print("  C degradation stop: %s" % ("YES" if stop_c else "NO"))
        print("    best rolling%d mean=%.2f at episode %d" % (args.window, best_mean, best_ep))
        print("    latest W3 is %.2f%% below best; descending_3_windows=%s; after_best=%s" % (
            below_best * 100, str(descending), str(windows_after_best)
        ))

        if stop_a:
            threshold_basis = sum(w["mean"] for w in recent_windows) / len(recent_windows)
            threshold_source = "plateau average of latest 3 windows"
        elif stop_c:
            threshold_basis = best_mean
            threshold_source = "historical best rolling%d" % args.window
    else:
        print("  A plateau stop: NO (need at least %d episodes, have %d)" % (args.window * 3, len(df)))
        print("  C degradation stop: NO (need at least %d episodes, have %d)" % (args.window * 3, len(df)))

    print("  B budget stop: %s" % ("YES" if stop_b else "NO"))
    print("    total_tsteps=%d / budget_tsteps=%d (%.1f%%)" % (
        total_tsteps, args.budget_tsteps, total_tsteps * 100.0 / max(1, args.budget_tsteps)
    ))
    if stop_b and threshold_basis is None and len(df) >= args.window:
        rolling_full = df["total_reward"].rolling(args.window, min_periods=args.window).mean()
        best_idx = rolling_full.idxmax()
        threshold_basis = float(rolling_full.loc[best_idx])
        threshold_source = "historical best rolling%d" % args.window

    should_stop = stop_a or stop_b or stop_c
    print("  Overall: %s" % ("STOP Phase 1" if should_stop else "CONTINUE training"))
    if threshold_basis is not None:
        print("  Threshold candidate: %.0f (85%% of %s mean %.2f)" % (
            threshold_basis * 0.85, threshold_source, threshold_basis
        ))
    else:
        print("  Threshold candidate: not ready")
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
