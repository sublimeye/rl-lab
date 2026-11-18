"""Evaluate every checkpoint of the long-training runs and export the report data.

    uv run experiments/001-cube-to-flag/evaluate.py long-s1 long-s2 long-s3

For each checkpoint: 200 episodes from two kinds of start position, with two ways of acting.
  start "usual":    the training distribution (bottom-left, ±0.3 jitter)
  start "anywhere": uniformly anywhere below the lava band (never seen as a start in training)
  act "best":       always take the most likely action (how you'd deploy it)
  act "sampled":    sample actions from the policy (how it behaves during training)

Writes site/public/data/001-cube-to-flag/long-eval.json plus replay files for the report.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from stable_baselines3 import PPO

from cube_env import CubeArena, _overlaps
from export import OUT, RUNS, ckpt_steps, rollout

EPISODES = 200
CAPABLE = 0.99  # success rate that counts as "capable"


def starts(kind: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    env = CubeArena()
    lay, half = env.cfg.layout, env.cfg.physics.half
    if kind == "usual":
        j = env.cfg.start_jitter
        return np.array(lay.start) + rng.uniform(-j, j, size=(n, 2))
    out = []
    while len(out) < n:
        x, y = rng.uniform(0.3, 9.7), rng.uniform(0.3, 3.8)
        if not any(_overlaps(x, y, half, w) for w in lay.walls):
            out.append((x, y))
    return np.array(out)


def evaluate(model: PPO, start_xy: np.ndarray, deterministic: bool) -> dict:
    """Run one episode per start position, all in lockstep so the policy sees one batch."""
    n = len(start_xy)
    envs = [CubeArena() for _ in range(n)]
    obs = np.stack(
        [
            e.reset(seed=i, options={"start": s})[0]
            for i, (e, s) in enumerate(zip(envs, start_xy, strict=True))
        ]
    )
    done = np.zeros(n, dtype=bool)
    outcome = [""] * n
    ret = np.zeros(n)
    while not done.all():
        actions, _ = model.predict(obs, deterministic=deterministic)
        for i in np.flatnonzero(~done):
            o, r, term, trunc, info = envs[i].step(int(actions[i]))
            obs[i] = o
            ret[i] += r
            if term or trunc:
                done[i] = True
                outcome[i] = info["outcome"]
    outcome_arr = np.array(outcome)
    flag = outcome_arr == "flag"
    secs = np.array([e.steps * e.cfg.physics.dt for e in envs])
    coins = np.array([int((~e.coins_left).sum()) for e in envs])
    return {
        "success": round(float(flag.mean()), 4),
        "lava": round(float((outcome_arr == "lava").mean()), 4),
        "timeout": round(float((outcome_arr == "timeout").mean()), 4),
        "secondsToFlag": round(float(secs[flag].mean()), 3) if flag.any() else None,
        "coins": round(float(coins.mean()), 3),
        "return": round(float(ret.mean()), 3),
    }


def eval_run(name: str) -> dict:
    import torch

    torch.set_num_threads(1)
    ckpts = sorted((RUNS / name).glob("ckpt_*.zip"), key=ckpt_steps)
    start_sets = {k: starts(k, EPISODES, seed=7) for k in ("usual", "anywhere")}
    rows = []
    for path in ckpts:
        steps = int(path.stem.split("_")[1])
        cache = path.with_name(f"eval_{steps:07d}_{EPISODES}.json")
        if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
            rows.append(json.loads(cache.read_text()))
            continue
        model = PPO.load(path, device="cpu")
        model.set_random_seed(0)
        row = {"steps": steps}
        for kind, xy in start_sets.items():
            for mode, det in (("best", True), ("sampled", False)):
                row[f"{kind}_{mode}"] = evaluate(model, xy, det)
        cache.write_text(json.dumps(row))
        rows.append(row)
        u = row["usual_best"]
        print(
            f"{name} {steps:>9,}: usual {u['success']:.0%}, "
            f"anywhere {row['anywhere_best']['success']:.0%}",
            flush=True,
        )
    return {"name": name, "checkpoints": rows}


def first_capable(rows: list[dict], key: str = "usual_best") -> int | None:
    """First checkpoint from which success stays >= CAPABLE for every later checkpoint."""
    for i, r in enumerate(rows):
        if all(x[key]["success"] >= CAPABLE for x in rows[i:]):
            return r["steps"]
    return None


def first_hit(rows: list[dict], key: str = "usual_best") -> int | None:
    """First checkpoint where success reaches CAPABLE at all (it may drop again later)."""
    return next((r["steps"] for r in rows if r[key]["success"] >= CAPABLE), None)


def load(run: str, steps: int) -> PPO:
    model = PPO.load(RUNS / run / f"ckpt_{steps:07d}.zip", device="cpu")
    model.set_random_seed(1)
    return model


def replay_ck(run: str, steps: int, label: str, xy: np.ndarray, deterministic: bool) -> dict:
    env = CubeArena()
    model = load(run, steps)
    runs = [
        rollout(model, env, seed=i, options={"start": tuple(s)}, deterministic=deterministic)
        for i, s in enumerate(xy)
    ]
    return {"steps": steps, "label": label, "runs": runs}


def start_map(run: str, steps: int, label: str, res: float = 0.25) -> dict:
    """Deterministic outcome from a grid of start positions below the lava band."""
    env = CubeArena()
    half = env.cfg.physics.half
    grid = [
        (x, y)
        for y in np.arange(0.375, 3.8, res)
        for x in np.arange(0.375, 9.7, res)
        if not any(_overlaps(x, y, half, w) for w in env.cfg.layout.walls)
    ]
    model = load(run, steps)
    envs = [CubeArena() for _ in grid]
    obs = np.stack(
        [e.reset(seed=0, options={"start": xy})[0] for e, xy in zip(envs, grid, strict=True)]
    )
    done = np.zeros(len(grid), dtype=bool)
    outcome = [""] * len(grid)
    while not done.all():
        actions, _ = model.predict(obs, deterministic=True)
        for i in np.flatnonzero(~done):
            o, _r, term, trunc, info = envs[i].step(int(actions[i]))
            obs[i] = o
            if term or trunc:
                done[i] = True
                outcome[i] = info["outcome"]
    cells = [[round(x, 3), round(y, 3), o[0]] for (x, y), o in zip(grid, outcome, strict=True)]
    success = sum(o == "flag" for o in outcome) / len(outcome)
    print(f"start map {run}@{steps:,}: {success:.0%} of {len(grid)} starts reach the flag")
    return {
        "label": label,
        "run": run,
        "steps": steps,
        "res": res,
        "success": success,
        "cells": cells,
    }


def export_report(long_runs: list[dict], rand_runs: list[dict]) -> None:
    """Replays and start maps for the report post (uses seed 1 of each study)."""
    long1, rand1 = long_runs[0], rand_runs[0]
    name, final = long1["name"], long1["checkpoints"][-1]["steps"]
    hit = first_hit(long1["checkpoints"])
    steps_all = [c["steps"] for c in long1["checkpoints"]]
    halfway = min(steps_all, key=lambda s: abs(s - hit / 2))
    rand_final = rand1["checkpoints"][-1]["steps"]
    env = CubeArena()
    usual = starts("usual", 4, seed=1000)
    anywhere = starts("anywhere", 6, seed=2024)

    stages = {
        "arena": env.describe(),
        "checkpoints": [
            replay_ck(name, 0, "Knows nothing", usual, False),
            replay_ck(name, halfway, "Halfway", usual, False),
            replay_ck(name, final, "Final", usual, False),
        ],
    }
    (OUT / "stages.json").write_text(json.dumps(stages, separators=(",", ":")))

    anywhere_replay = {
        "arena": env.describe(),
        "checkpoints": [
            replay_ck(name, 1_000_000, "1M steps", anywhere, True),
            replay_ck(name, final, "10M steps", anywhere, True),
            replay_ck(rand1["name"], rand_final, "Random starts", anywhere, True),
        ],
    }
    (OUT / "anywhere.json").write_text(json.dumps(anywhere_replay, separators=(",", ":")))

    maps = {
        "arena": env.describe(),
        "maps": [
            start_map(name, 1_000_000, "1M steps"),
            start_map(name, final, f"{final // 1_000_000}M steps"),
            start_map(
                rand1["name"], rand_final, f"Random starts, {rand_final // 1_000_000}M steps"
            ),
        ],
    }
    (OUT / "start-maps.json").write_text(json.dumps(maps, separators=(",", ":")))
    print(f"report: first hit={hit:,}, halfway={halfway:,}, final={final:,}, rand={rand_final:,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, e.g. long-s1 long-s2 rand-s1")
    args = ap.parse_args()

    with ProcessPoolExecutor(max_workers=min(len(args.runs), 6)) as pool:
        results = list(pool.map(eval_run, args.runs))

    groups: dict[str, list[dict]] = {}
    for r in results:
        r["firstHit"] = first_hit(r["checkpoints"])
        r["capableAt"] = first_capable(r["checkpoints"])
        r["capableAtAnywhere"] = first_capable(r["checkpoints"], "anywhere_best")
        print(
            f"{r['name']}: first 99% at {r['firstHit']}, stays capable from {r['capableAt']}, "
            f"anywhere-capable from {r['capableAtAnywhere']}"
        )
        groups.setdefault(r["name"].split("-s")[0], []).append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    for prefix, runs in groups.items():
        data = {"episodes": EPISODES, "capableThreshold": CAPABLE, "runs": runs}
        (OUT / f"{prefix}-eval.json").write_text(json.dumps(data, separators=(",", ":")))

    if "long" in groups and "rand" in groups:
        export_report(groups["long"], groups["rand"])


if __name__ == "__main__":
    main()
