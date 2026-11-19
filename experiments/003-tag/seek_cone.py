"""Train a seeker with a field of view (post 007).

uv run experiments/003-tag/seek_cone.py --seed 1 --name cone-any-s1          # all settings
uv run experiments/003-tag/seek_cone.py --human --seed 1 --name cone-human-s1  # HUMAN only

Same runners as post 6 (seek.py): post 5's best runner 40%, another post 5 runner 15%, the
scripted hider 30%, a still runner out of sight 15%. Random maps, chaser speed 85-100%.
Without --human, every episode draws new senses (cone.sample_senses).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from cone import HUMAN, ConeArena
from seek import RUNNERS, SAVE_AT
from selfplay import PPO_KW, RUNS, Outcomes, net_of
from sight import Hider, Still
from tag_env import Config, NetEvader

SAVE_AT = [*SAVE_AT, 10_000_000, 15_000_000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true", help="train on HUMAN senses only")
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=1)
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

    cfg = Config(evaders=runner, chaser_ratio=(0.85, 1.0), maps="random", rays=16)
    senses = HUMAN if args.human else None
    env = make_vec_env(
        ConeArena, n_envs=8, seed=args.seed, env_kwargs={"config": cfg, "senses": senses}
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
    log = {"senses": "human" if args.human else "any", "seed": args.seed, "arch": args.arch}
    log["phases"] = []
    model.save(out / "ckpt_00000000")
    for target in [s for s in SAVE_AT if s < args.steps] + [args.steps]:
        cb = Outcomes()
        model.learn(target - done, callback=cb, reset_num_timesteps=False)
        done = target
        model.save(out / f"ckpt_{target:08d}")
        row = {"steps": target, "caught": round(cb.caught / max(cb.total, 1), 3)}
        row["minutes"] = round((time.time() - t0) / 60, 1)
        log["phases"].append(row)
        print(json.dumps(row), flush=True)
        (out / "cone_log.json").write_text(json.dumps(log))
    log["seconds"] = round(time.time() - t0, 1)
    (out / "cone_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
