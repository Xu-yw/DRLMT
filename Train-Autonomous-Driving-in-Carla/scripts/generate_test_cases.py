import argparse
import json
import os

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
    parser.add_argument("--output", default="eval/test_cases_v1.json",
                        help="Output JSON path (will mkdir parent)")
    parser.add_argument("--n-cases", type=int, default=500)
    parser.add_argument("--n-spawn-points", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = generate_cases(args.n_cases, args.n_spawn_points, args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"[OK] generated {len(cases)} cases to {args.output}")
    print("First 3:")
    for c in cases[:3]:
        print(c)


if __name__ == "__main__":
    main()
