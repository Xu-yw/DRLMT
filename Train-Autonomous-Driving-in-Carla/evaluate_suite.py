"""
Plan-06 evaluation suite: run a candidate (baseline/mutant) on N=500 test cases.

Suites: validation (ClearNoon) / sunny (ClearNoon) / rainy (MidRainyNoon) / foggy (Custom_Foggy)
Raw fail rule: episode does not reach final waypoint. Phase 6 computes relative_fail.
Mutant evaluation activates mutation hooks (mutation module exists only on mutation/runtime branch)

CSV schema: see plan-06 04-evaluation-spec.md section 5.1
"""
import os
import sys
import time
import csv
import json
import argparse
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla
from simulation import settings as sim_settings
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import LATENT_DIM, ACTION_STD_INIT, EPISODE_LENGTH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-name", required=True, help="candidate label")
    p.add_argument("--candidate-ckpt", required=True, help=".pth state_dict path")
    p.add_argument("--suite", required=True, choices=["validation", "sunny", "rainy", "foggy"])
    p.add_argument("--weather", required=True,
                   help="CARLA WeatherParameters name (ClearNoon / MidRainyNoon / Custom_Foggy)")
    p.add_argument("--port", type=int, default=2002, help="evaluation port (training uses 2000)")
    p.add_argument("--test-cases", default="/root/autodl-tmp/eval/test_cases_v1.json")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--max-steps", type=int, default=EPISODE_LENGTH)
    p.add_argument("--mutation-type", default="none")
    p.add_argument("--town", default="Town07")
    p.add_argument("--limit", type=int, default=0, help="Optional: limit to first N cases (0 = all)")
    return p.parse_args()


def build_weather(name):
    if name == "Custom_Foggy":
        return carla.WeatherParameters(
            cloudiness=80.0, precipitation=0.0, sun_altitude_angle=70.0,
            fog_density=80.0, fog_distance=10.0, fog_falloff=2.0
        )
    if hasattr(carla.WeatherParameters, name):
        return getattr(carla.WeatherParameters, name)
    raise ValueError("Unknown weather preset: " + name)


def load_weights_into_agent(agent, weight_path):
    state = torch.load(weight_path, map_location="cpu")
    res_old = agent.old_policy.load_state_dict(state, strict=False)
    res_new = agent.policy.load_state_dict(state, strict=False)
    print("[LOAD] " + weight_path)
    print("[LOAD] old_policy missing=" + str(res_old.missing_keys) + " unexpected=" + str(res_old.unexpected_keys))
    print("[LOAD] policy     missing=" + str(res_new.missing_keys) + " unexpected=" + str(res_new.unexpected_keys))
    if res_old.unexpected_keys or res_new.unexpected_keys:
        raise RuntimeError("unexpected keys in checkpoint: " + str(res_old.unexpected_keys))
    allowed_missing = {"cov_var", "cov_mat"}
    bad = set(res_old.missing_keys) - allowed_missing
    if bad:
        raise RuntimeError("unacceptable missing keys: " + str(bad))
    for p in agent.old_policy.parameters():
        p.requires_grad = False
    for p in agent.policy.parameters():
        p.requires_grad = False


def main():
    args = parse_args()

    os.environ["MUTATION_TYPE"] = args.mutation_type
    try:
        from mutation import config as _mutation_config
        _mutation_config.init()
    except ImportError:
        if args.mutation_type != "none":
            print("[WARN] mutation_type=" + args.mutation_type + " but mutation module not found; expected on mutation/runtime branch")

    sim_settings.PORT = args.port
    print("[CFG] port=" + str(args.port) + " town=" + args.town + " suite=" + args.suite + " weather=" + args.weather)
    print("[CFG] candidate=" + args.candidate_name + " ckpt=" + args.candidate_ckpt)
    print("[CFG] mutation_type=" + args.mutation_type)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    with open(args.test_cases) as f:
        cases = json.load(f)
    if args.limit > 0:
        cases = cases[:args.limit]
    print("[CFG] Loaded " + str(len(cases)) + " test cases from " + args.test_cases)

    client, world = ClientConnection(args.town).setup()
    weather = build_weather(args.weather)
    world.set_weather(weather)

    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None)
    n_spawn_points = len(env.map.get_spawn_points())
    bad_cases = [c for c in cases if int(c.get("spawn_idx", -1)) < 0 or int(c.get("spawn_idx", -1)) >= n_spawn_points]
    if bad_cases:
        sample = bad_cases[:5]
        raise ValueError("test cases contain spawn_idx outside 0..%d: %s" % (n_spawn_points - 1, sample))
    print("[CFG] Town spawn points=" + str(n_spawn_points))
    encode = EncodeState(LATENT_DIM)
    agent = PPOAgent(args.town, ACTION_STD_INIT)
    load_weights_into_agent(agent, args.candidate_ckpt)

    fieldnames = [
        "case_id", "candidate", "suite", "spawn_idx", "heading_offset_deg",
        "total_reward", "total_steps", "distance_m", "done_reason", "raw_fail",
        "progress_ratio", "final_waypoint_idx", "route_length", "wall_time_s", "mutation_type",
    ]
    t_global = datetime.now()

    with open(args.output_csv, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        csv_f.flush()

        n_pass = 0
        n_fail = 0

        for case in cases:
            t0 = time.time()
            env._eval_spawn_idx = case["spawn_idx"]
            env._eval_heading_offset_deg = case["heading_offset_deg"]

            obs = env.reset()
            if obs is None:
                writer.writerow({
                    "case_id": case["case_id"], "candidate": args.candidate_name,
                    "suite": args.suite, "spawn_idx": case["spawn_idx"],
                    "heading_offset_deg": round(case["heading_offset_deg"], 4),
                    "total_reward": 0.0, "total_steps": 0, "distance_m": 0.0,
                    "done_reason": "reset_failure", "raw_fail": 1,
                    "progress_ratio": 0.0, "final_waypoint_idx": 0, "route_length": 0,
                    "wall_time_s": round(time.time() - t0, 2),
                    "mutation_type": args.mutation_type,
                })
                csv_f.flush()
                n_fail += 1
                continue

            obs = encode.process(obs)
            total_reward = 0.0
            steps = 0
            done = False
            done_reason = "max_steps_reached"
            info = None

            for step_i in range(args.max_steps):
                action = agent.get_action(obs, step_i, total_reward, done, train=False)
                step_out = env.step(action)
                if step_out is None or step_out[0] is None:
                    done_reason = "step_failure"
                    break
                obs, reward, done, info = step_out
                obs = encode.process(obs)
                if torch.any(torch.isnan(obs)):
                    obs = torch.zeros_like(obs)
                total_reward += float(reward)
                steps += 1
                if done:
                    done_reason = env.get_last_done_reason() or "unknown_done"
                    break

            final_idx = getattr(env, "current_waypoint_index", 0)
            route_len = len(env.route_waypoints) if getattr(env, "route_waypoints", None) else 0
            raw_fail = 0 if done_reason == "route_completed" else 1
            progress_ratio = max(0.0, min(1.0, float(final_idx) / max(1, route_len - 1)))

            writer.writerow({
                "case_id": case["case_id"], "candidate": args.candidate_name,
                "suite": args.suite, "spawn_idx": case["spawn_idx"],
                "heading_offset_deg": round(case["heading_offset_deg"], 4),
                "total_reward": round(total_reward, 4), "total_steps": steps,
                "distance_m": round(info[0], 2) if info else 0.0,
                "done_reason": done_reason, "raw_fail": raw_fail,
                "progress_ratio": round(progress_ratio, 6),
                "final_waypoint_idx": final_idx, "route_length": route_len,
                "wall_time_s": round(time.time() - t0, 2),
                "mutation_type": args.mutation_type,
            })
            csv_f.flush()

            if raw_fail:
                n_fail += 1
            else:
                n_pass += 1

            if (case["case_id"] + 1) % 50 == 0:
                tfr = n_fail / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0
                elapsed = (datetime.now() - t_global).total_seconds() / 60.0
                print("[PROGRESS] " + str(case["case_id"]+1) + "/" + str(len(cases)) +
                      " | pass=" + str(n_pass) + " fail=" + str(n_fail) +
                      " TFR=" + ("%.3f" % tfr) + " | elapsed=" + ("%.1f" % elapsed) + "min")

    tfr = n_fail / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0
    elapsed = (datetime.now() - t_global).total_seconds() / 60.0
    print("[DONE] " + args.candidate_name + " on " + args.suite +
          ": pass=" + str(n_pass) + " fail=" + str(n_fail) + " TFR=" + ("%.3f" % tfr))
    print("[DONE] CSV: " + args.output_csv)
    print("[DONE] elapsed: " + ("%.1f" % elapsed) + " min")


if __name__ == "__main__":
    main()
