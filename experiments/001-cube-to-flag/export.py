"""Roll out every checkpoint and write the replay + training summary for the site.

    uv run experiments/001-cube-to-flag/export.py [--variant shaped|sparse]
Writes site/public/data/001-cube-to-flag/<variant>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from cube_env import Config, CubeArena
from train import VARIANTS

HERE = Path(__file__).parent
RUNS = HERE / "runs"
OUT = HERE.parents[1] / "site" / "public" / "data" / HERE.name
RUNS_PER_CHECKPOINT = 4
TRAINING_BINS = 20


def ckpt_steps(path: Path) -> int:
    return int(path.stem.split("_")[1])


def rollout(
    model: PPO,
    env: CubeArena,
    seed: int,
    options: dict | None = None,
    deterministic: bool = False,
) -> dict:
    obs, _ = env.reset(seed=seed, options=options)
    frames = [round(float(v), 3) for v in env.pos]
    coins, total = [], 0.0
    while True:
        # Sample from the policy (not argmax) so early checkpoints show what they actually explore.
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        frames += [round(float(v), 3) for v in env.pos]
        if info["coin"] is not None:
            coins.append([info["coin"], len(frames) // 2 - 1])
        if terminated or truncated:
            return {
                "outcome": info["outcome"],
                "return": round(total, 2),
                "xy": frames,
                "coins": coins,
            }


def training_summary(episodes: list[list], total: int) -> list[dict]:
    """Bin episodes by training step and count outcomes per bin."""
    width = total / TRAINING_BINS
    bins = [
        {"from": int(i * width), "to": int((i + 1) * width), "flag": 0, "lava": 0, "timeout": 0}
        for i in range(TRAINING_BINS)
    ]
    returns: list[list[float]] = [[] for _ in bins]
    for steps, ret, _length, outcome in episodes:
        i = min(int(steps // width), TRAINING_BINS - 1)
        bins[i][outcome] += 1
        returns[i].append(ret)
    for b, r in zip(bins, returns, strict=True):
        b["meanReturn"] = round(float(np.mean(r)), 2) if r else None
    return bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="shaped")
    args = ap.parse_args()
    src = RUNS / args.variant

    env = CubeArena(Config(rewards=VARIANTS[args.variant]))
    checkpoints = []
    for path in sorted(src.glob("ckpt_*.zip"), key=ckpt_steps):
        steps = int(path.stem.split("_")[1])
        model = PPO.load(path, device="cpu")
        model.set_random_seed(steps)
        runs = [rollout(model, env, seed=1000 + i) for i in range(RUNS_PER_CHECKPOINT)]
        checkpoints.append({"steps": steps, "runs": runs})
        print(f"{steps:>8} steps: " + ", ".join(r["outcome"] for r in runs))

    log = json.loads((src / "train_log.json").read_text())
    data = {
        "experiment": HERE.name,
        "variant": args.variant,
        "arena": env.describe(),
        "checkpoints": checkpoints,
        "training": {
            "totalSteps": log["total_steps"],
            "seconds": log["seconds"],
            "episodes": len(log["episodes"]),
            "bins": training_summary(log["episodes"], log["total_steps"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.variant}.json"
    dest.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
