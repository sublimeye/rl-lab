"""Self-play tag: an evader and a chaser network train in alternating rounds (post 004).

uv run experiments/003-tag/selfplay.py --seed 1

The evader starts from the post-003 mixed-chaser model; the chaser starts from scratch.
Each round the chaser trains against the current evader (and a few older ones), then the
evader trains against the current chaser, older chasers and the scripted bots from 003.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from tag_env import MLP, ChaserArena, Config, NetChaser, NetEvader, TagArena, sample_chaser

HERE = Path(__file__).parent
RUNS = HERE / "runs"
PPO_KW = dict(n_steps=256, batch_size=512, gamma=0.995, ent_coef=0.01, learning_rate=3e-4)


def net_of(model: PPO) -> MLP:
    """The actor network of an SB3 PPO model as a numpy MLP (full precision)."""
    lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    lin.append(model.policy.action_net)
    return MLP(
        [{"w": m.weight.detach().cpu().numpy(), "b": m.bias.detach().cpu().numpy()} for m in lin]
    )


class Outcomes(BaseCallback):
    """Counts how episodes end during one training phase."""

    def __init__(self):
        super().__init__()
        self.caught = 0
        self.total = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.total += 1
                self.caught += info.get("outcome") == "caught"
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--first-chaser-steps", type=int, default=1_000_000)
    ap.add_argument("--chaser-steps", type=int, default=300_000)
    ap.add_argument("--evader-steps", type=int, default=500_000)
    ap.add_argument("--warm", help="003 run to start the evader from (default mixed-s<seed>)")
    ap.add_argument("--evader-init", help="evader checkpoint under runs/ (overrides --warm)")
    ap.add_argument("--chaser-init", help="chaser checkpoint under runs/ (default: from scratch)")
    ap.add_argument(
        "--pool", nargs="*", default=[], help="frozen chaser checkpoints the evader also faces"
    )
    ap.add_argument("--ratio", nargs=2, type=float, default=[0.8, 1.0], help="chaser speed range")
    ap.add_argument("--scripted", type=float, default=0.3, help="share of scripted chasers")
    ap.add_argument("--name", help="run folder (default sp-s<seed>)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    out = RUNS / (args.name or f"sp-s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    if args.evader_init:
        warm = RUNS / f"{args.evader_init}.zip"
    else:
        warm = RUNS / (args.warm or f"mixed-s{args.seed}") / "ckpt_5000000.zip"
    lo, hi = args.ratio

    evaders: list[MLP] = []
    chasers: list[MLP] = []  # this run's chaser, one snapshot per round
    fixed = [net_of(PPO.load(RUNS / f"{p}.zip", device="cpu")) for p in args.pool]

    def evader_opponent(rng):
        if not (chasers or fixed) or rng.random() < args.scripted:
            return sample_chaser("mixed", rng)
        pool = fixed + chasers
        if chasers and rng.random() < 0.6:
            net = chasers[-1]
        else:
            net = pool[int(rng.integers(len(pool)))]
        return NetChaser(net, ratio=float(rng.uniform(lo, hi)), sample=bool(rng.random() < 0.5))

    def chaser_opponent(rng):
        i = len(evaders) - 1 if rng.random() < 0.7 else int(rng.integers(len(evaders)))
        return NetEvader(evaders[i], sample=bool(rng.random() < 0.3))

    # DummyVecEnv keeps the envs in this process, so they see the pools as they grow.
    e_env = make_vec_env(
        TagArena, n_envs=8, seed=args.seed, env_kwargs={"config": Config(opponents=evader_opponent)}
    )
    c_env = make_vec_env(
        ChaserArena,
        n_envs=8,
        seed=args.seed + 100,
        env_kwargs={"config": Config(evaders=chaser_opponent, chaser_ratio=(lo, hi))},
    )
    evader = PPO.load(warm, env=e_env, device="cpu", **PPO_KW)
    if args.chaser_init:
        chaser = PPO.load(RUNS / f"{args.chaser_init}.zip", env=c_env, device="cpu", **PPO_KW)
    else:
        chaser = PPO("MlpPolicy", c_env, seed=args.seed, device="cpu", verbose=0, **PPO_KW)
    if args.evader_init or args.chaser_init:
        # PPO.load re-seeds with the seed saved in the file; use this run's seed instead.
        # (Stage 1 runs skip this: their warm start was trained with the same seed.)
        evader.set_random_seed(args.seed)
        chaser.set_random_seed(args.seed + 100)
    evaders.append(net_of(evader))
    evader.save(out / "evader_r00")
    chaser.save(out / "chaser_r00")

    log = {
        "seed": args.seed,
        "warm": str(warm.relative_to(HERE)),
        "chaserInit": args.chaser_init,
        "pool": args.pool,
        "ratio": [lo, hi],
        "scripted": args.scripted,
        "rounds": [],
    }
    t0 = time.time()
    for r in range(1, args.rounds + 1):
        cb = Outcomes()
        steps = args.first_chaser_steps if r == 1 else args.chaser_steps
        chaser.learn(steps, callback=cb, reset_num_timesteps=False)
        chasers.append(net_of(chaser))
        chaser.save(out / f"chaser_r{r:02d}")
        c_rate = cb.caught / max(cb.total, 1)

        cb = Outcomes()
        evader.learn(args.evader_steps, callback=cb, reset_num_timesteps=False)
        evaders.append(net_of(evader))
        evader.save(out / f"evader_r{r:02d}")
        e_rate = 1 - cb.caught / max(cb.total, 1)

        row = {
            "round": r,
            "chaserSteps": int(chaser.num_timesteps),
            "evaderSteps": int(evader.num_timesteps),
            "chaserCatchRate": round(c_rate, 3),  # while the chaser trained
            "evaderEscapeRate": round(e_rate, 3),  # while the evader trained (mixed opponents)
            "minutes": round((time.time() - t0) / 60, 1),
        }
        log["rounds"].append(row)
        print(json.dumps(row), flush=True)
        (out / "selfplay_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
