import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def generate_cases(n_cases=500, n_spawn_points=120, seed=42):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_cases):
        cases.append({
            "case_id": i,
            "spawn_idx": int(rng.integers(0, n_spawn_points)),
            "heading_offset_deg": float(rng.uniform(-15.0, 15.0)),
        })
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/autodl-tmp/eval/test_cases_v1.json",
                        help="Output JSON path (will mkdir parent)")
    parser.add_argument("--n-cases", type=int, default=500)
    parser.add_argument("--n-spawn-points", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detect-spawn-points", action="store_true",
                        help="Connect to CARLA and use len(world.get_map().get_spawn_points())")
    parser.add_argument("--town", default="Town07")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    n_spawn_points = args.n_spawn_points
    if args.detect_spawn_points:
        from simulation import settings as sim_settings
        sim_settings.PORT = args.port
        from simulation.connection import ClientConnection
        client, world = ClientConnection(args.town).setup()
        n_spawn_points = len(world.get_map().get_spawn_points())
        print(f"[INFO] detected {n_spawn_points} spawn points in {args.town}")
    cases = generate_cases(args.n_cases, n_spawn_points, args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"[OK] generated {len(cases)} cases to {args.output} (n_spawn_points={n_spawn_points}, seed={args.seed})")
    print("First 3:")
    for c in cases[:3]:
        print(c)


if __name__ == "__main__":
    main()
