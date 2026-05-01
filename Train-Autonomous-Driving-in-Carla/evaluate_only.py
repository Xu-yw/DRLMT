"""
Evaluate a saved PPO policy on Town07 without training.

Loads any .pth (state_dict) into the current ActorCritic network with strict=False,
runs N episodes, records per-episode metrics to CSV.

Designed to NOT touch any training artifacts:
- does not write to preTrained_models/ or checkpoints/
- writes only to the --output-csv path you specify
- connects to the CARLA port you specify (default 2002, NOT 2000)
"""
import os
import sys
import time
import csv
import argparse
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import settings as sim_settings
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import LATENT_DIM, ACTION_STD_INIT, EPISODE_LENGTH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight-path", required=True, help="path to a .pth state_dict")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    p.add_argument("--town", type=str, default="Town07")
    p.add_argument("--port", type=int, default=2002, help="CARLA RPC port (NOT 2000 = training)")
    p.add_argument("--output-csv", required=True, help="per-episode CSV output path")
    p.add_argument("--label", type=str, default="", help="candidate label for CSV column")
    p.add_argument("--success-threshold", type=float, default=2000.0,
                   help="reward >= this counts as success (TFR numerator)")
    return p.parse_args()


def load_weights_into_agent(agent, weight_path):
    state = torch.load(weight_path, map_location="cpu")
    res_old = agent.old_policy.load_state_dict(state, strict=False)
    res_new = agent.policy.load_state_dict(state, strict=False)
    print(f"[LOAD] {weight_path}")
    print(f"[LOAD] old_policy missing={res_old.missing_keys} unexpected={res_old.unexpected_keys}")
    print(f"[LOAD] policy     missing={res_new.missing_keys} unexpected={res_new.unexpected_keys}")
    if res_old.unexpected_keys or res_new.unexpected_keys:
        raise RuntimeError(f"unexpected keys in checkpoint: {res_old.unexpected_keys}")
    allowed_missing = {"cov_var", "cov_mat"}
    bad = set(res_old.missing_keys) - allowed_missing
    if bad:
        raise RuntimeError(f"unacceptable missing keys: {bad}")
    for p in agent.old_policy.parameters():
        p.requires_grad = False
    for p in agent.policy.parameters():
        p.requires_grad = False


def main():
    args = parse_args()

    sim_settings.PORT = args.port
    print(f"[CFG] CARLA port={sim_settings.PORT} town={args.town} episodes={args.episodes}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    label = args.label or os.path.basename(os.path.dirname(args.weight_path)) or "unnamed"

    client, world = ClientConnection(args.town).setup()
    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None)
    encode = EncodeState(LATENT_DIM)

    agent = PPOAgent(args.town, ACTION_STD_INIT)
    load_weights_into_agent(agent, args.weight_path)

    rows = []
    t_start = datetime.now()

    for ep in range(args.episodes):
        observation = env.reset()
        if observation is None:
            print(f"[EP {ep}] reset returned None, skipping")
            continue
        observation = encode.process(observation)

        ep_reward = 0.0
        steps = 0
        ep_distance = 0.0
        ep_deviation = 0.0
        crashed = False
        ep_t0 = datetime.now()

        for t in range(args.episode_length):
            # signature: get_action(obs, flag, reward, done, train)
            # flag/reward/done are only consumed by the train=True branch (memory append)
            action = agent.get_action(observation, flag=0, reward=0.0, done=False, train=False)
            step_out = env.step(action)
            if step_out is None:
                crashed = True
                break
            observation, reward, done, info = step_out
            if observation is None:
                crashed = True
                break
            observation = encode.process(observation)
            ep_reward += reward
            steps += 1
            if done:
                if isinstance(info, (list, tuple)) and len(info) >= 2:
                    ep_distance = float(info[0])
                    ep_deviation = float(info[1])
                break

        ep_dt = (datetime.now() - ep_t0).total_seconds()
        success = ep_reward >= args.success_threshold
        rows.append({
            "candidate": label,
            "episode": ep,
            "reward": round(ep_reward, 4),
            "steps": steps,
            "distance_m": round(ep_distance, 4),
            "deviation_m": round(ep_deviation, 4),
            "wall_seconds": round(ep_dt, 2),
            "crashed": int(crashed),
            "success": int(success),
        })
        print(f"[EP {ep:>3}] reward={ep_reward:8.2f} steps={steps:5d} "
              f"dist={ep_distance:7.2f} dev={ep_deviation:6.2f} "
              f"crashed={int(crashed)} success={int(success)} "
              f"wall={ep_dt:.1f}s")

    fieldnames = ["candidate", "episode", "reward", "steps", "distance_m",
                  "deviation_m", "wall_seconds", "crashed", "success"]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if rows:
        rewards = [r["reward"] for r in rows]
        crashes = sum(r["crashed"] for r in rows)
        successes = sum(r["success"] for r in rows)
        print(f"\n[SUMMARY] {label} | episodes={len(rows)} "
              f"avg_reward={np.mean(rewards):.2f} std={np.std(rewards):.2f} "
              f"min={min(rewards):.2f} max={max(rewards):.2f} "
              f"crashes={crashes}/{len(rows)} successes={successes}/{len(rows)}")
    print(f"[DONE] CSV written to {args.output_csv} "
          f"in {(datetime.now()-t_start).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
