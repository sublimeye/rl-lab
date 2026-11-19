"""Train a chaser that only sees along lines of sight (post 006).

uv run experiments/003-tag/seek.py --sight map --seed 1 --name seek-map-s1

--sight picks what the chaser knows (see sight.py): xray, none, last or map. Its runners are
a mix: post 5's best runner (40%) and another post 5 runner (15%), both seeing everything and
picking moves with some randomness; the scripted hider (30%); and a runner standing still out
of sight (15%). Random maps, chaser speed drawn from 85-100% of the runner's per episode.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from selfplay import PPO_KW, RUNS, Outcomes, net_of
from sight import SIGHTS, Hider, SeekArena, Still
from tag_env import Config, NetEvader

SAVE_AT = [250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 7_500_000]
RUNNERS = {"msp2-s1/evader_r20": 0.40, "msp-s1/evader_r20": 0.15}  # post 5's 40 / 20 rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sight", choices=SIGHTS, required=True)
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ratio", nargs=2, type=float, default=[0.85, 1.0])
    ap.add_argument("--arch", nargs="+", type=int, default=[64, 64])
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    nets = [
        (NetEvader(net_of(PPO.load(RUNS / f"{p}.zip", device="cpu")), sample=True), w)
        for p, w in RUNNERS.items()
    ]
    hider, still = Hider(), Still()

    def runner(rng):
        r = rng.random()
        for prey, w in nets:
            if r < w:
                return prey
            r -= w
        return hider if r < 0.30 else still

    cfg = Config(evaders=runner, chaser_ratio=tuple(args.ratio), maps="random", rays=16)
    env = make_vec_env(
        SeekArena, n_envs=8, seed=args.seed, env_kwargs={"config": cfg, "sight": args.sight}
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": args.arch},
        **PPO_KW,
    )
    t0 = time.time()
    done = 0
    log = {"sight": args.sight, "seed": args.seed, "ratio": args.ratio, "arch": args.arch}
    log["phases"] = []
    model.save(out / "ckpt_0000000")
    for target in [s for s in SAVE_AT if s < args.steps] + [args.steps]:
        cb = Outcomes()
        model.learn(target - done, callback=cb, reset_num_timesteps=False)
        done = target
        model.save(out / f"ckpt_{target:07d}")
        row = {"steps": target, "caught": round(cb.caught / max(cb.total, 1), 3)}
        row["minutes"] = round((time.time() - t0) / 60, 1)
        log["phases"].append(row)
        print(json.dumps(row), flush=True)
        (out / "seek_log.json").write_text(json.dumps(log))
    log["seconds"] = round(time.time() - t0, 1)
    (out / "seek_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
