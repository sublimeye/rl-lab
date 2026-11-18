"""Train the evader against chasers built to beat it (post 004, last stage).

uv run experiments/003-tag/psro.py --seed 1

Each iteration:
1. train a fresh chaser from scratch against the current evader (best actions, as in the
   game) until it has found its weak spots;
2. add that chaser to the pool;
3. train the evader against the whole pool (and a few scripted chasers).

This is a small version of PSRO (policy-space response oracles). Unlike plain self-play,
every chaser in the pool is one that actually beat some version of the evader.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from selfplay import PPO_KW, RUNS, Outcomes, net_of
from tag_env import ChaserArena, Config, NetChaser, NetEvader, TagArena, sample_chaser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--exploit-steps", type=int, default=750_000)
    ap.add_argument("--evader-steps", type=int, default=1_500_000)
    ap.add_argument("--evader-init", default="sp-s3/evader_r20")
    ap.add_argument("--pool", nargs="*", default=[], help="chaser checkpoints to start the pool")
    ap.add_argument("--ratio", nargs=2, type=float, default=[0.8, 0.95])
    ap.add_argument("--scripted", type=float, default=0.1)
    ap.add_argument("--name")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    out = RUNS / (args.name or f"psro-s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = args.ratio
    pool = [net_of(PPO.load(RUNS / f"{p}.zip", device="cpu")) for p in args.pool]

    def evader_opponent(rng):
        if not pool or rng.random() < args.scripted:
            return sample_chaser("mixed", rng)
        net = pool[-1] if rng.random() < 0.4 else pool[int(rng.integers(len(pool)))]
        return NetChaser(net, ratio=float(rng.uniform(lo, hi)))

    e_env = make_vec_env(
        TagArena, n_envs=8, seed=args.seed, env_kwargs={"config": Config(opponents=evader_opponent)}
    )
    evader = PPO.load(RUNS / f"{args.evader_init}.zip", env=e_env, device="cpu", **PPO_KW)
    evader.set_random_seed(args.seed)  # not the seed saved in the file
    evader.save(out / "evader_i00")

    log = {
        "seed": args.seed,
        "evaderInit": args.evader_init,
        "pool": args.pool,
        "ratio": [lo, hi],
        "iters": [],
    }
    t0 = time.time()
    for it in range(1, args.iters + 1):
        current = NetEvader(net_of(evader))
        cfg = Config(evaders=lambda rng, e=current: e, chaser_ratio=(lo, hi))
        c_env = make_vec_env(
            ChaserArena, n_envs=8, seed=args.seed * 1000 + it, env_kwargs={"config": cfg}
        )
        chaser = PPO(
            "MlpPolicy", c_env, seed=args.seed * 1000 + it, device="cpu", verbose=0, **PPO_KW
        )
        chaser.learn(args.exploit_steps * 3 // 4)
        late = Outcomes()  # how often the exploiter catches the evader once it has learned
        chaser.learn(args.exploit_steps // 4, callback=late, reset_num_timesteps=False)
        chaser.save(out / f"chaser_i{it:02d}")
        pool.append(net_of(chaser))

        cb = Outcomes()
        evader.learn(args.evader_steps, callback=cb, reset_num_timesteps=False)
        evader.save(out / f"evader_i{it:02d}")
        row = {
            "iter": it,
            "exploiterCatchRate": round(late.caught / max(late.total, 1), 3),
            "evaderEscapeRate": round(1 - cb.caught / max(cb.total, 1), 3),
            "pool": len(pool),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        log["iters"].append(row)
        print(json.dumps(row), flush=True)
        (out / "psro_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
