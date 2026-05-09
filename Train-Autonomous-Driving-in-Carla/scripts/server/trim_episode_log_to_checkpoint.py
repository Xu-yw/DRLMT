#!/usr/bin/env python3
"""Trim episode_log.csv rows beyond the latest checkpoint episode.

When training is killed after writing episode rows but before the next checkpoint,
resuming from the checkpoint will replay those episode numbers. This utility keeps
episode logs consistent by dropping rows with episode > checkpoint_episode.
"""
import argparse
import csv
import os
import pickle
import re
import sys
import tempfile


def checkpoint_index(path):
    m = re.search(r"checkpoint_ppo_(\d+)\.pickle$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.startswith("checkpoint_ppo_") and name.endswith(".pickle")
    ]
    return max(files, key=checkpoint_index) if files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    ckpt = latest_checkpoint(args.checkpoint_dir)
    if ckpt is None or not os.path.exists(args.csv):
        return 0

    with open(ckpt, "rb") as f:
        data = pickle.load(f)
    checkpoint_episode = int(data.get("episode", -1))
    if checkpoint_episode < 0:
        return 0

    with open(args.csv, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0

    header = rows[0]
    try:
        ep_idx = header.index("episode")
    except ValueError:
        ep_idx = 0

    kept = [header]
    dropped = 0
    for row in rows[1:]:
        try:
            ep = int(float(row[ep_idx]))
        except Exception:
            kept.append(row)
            continue
        if ep <= checkpoint_episode:
            kept.append(row)
        else:
            dropped += 1

    if dropped == 0:
        print(f"trim: no rows dropped; checkpoint_episode={checkpoint_episode}")
        return 0

    directory = os.path.dirname(args.csv) or "."
    fd, tmp = tempfile.mkstemp(prefix=".episode_log_trim_", suffix=".csv", dir=directory)
    os.close(fd)
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows(kept)
    os.replace(tmp, args.csv)
    print(f"trim: dropped {dropped} rows newer than checkpoint_episode={checkpoint_episode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
