"""Train PPO to run away from scripted chasers, saving checkpoints and a training log.

uv run experiments/003-tag/train.py --chasers mixed --steps 3000000 --seed 1 --name mixed-s1
uv run experiments/003-tag/train.py --maps random --rays 16 --steps 10000000 --name maps-s1
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

from tag_env import Config, Rewards, TagArena

HERE = Path(__file__).parent
RUNS = HERE / "runs"  # git-ignored
REWARDS = {"default": Rewards(), "nodistance": Rewards(distance=0.0)}
SAVE_AT = [
    *range(25_000, 200_001, 25_000),
    *range(250_000, 1_000_001, 125_000),
    1_500_000,
    2_000_000,
    3_000_000,
    5_000_000,
    7_500_000,
    10_000_000,
    15_000_000,
]


class Recorder(BaseCallback):
    def __init__(self, save_at: list[int], out: Path):
        super().__init__()
        self.save_at = sorted(save_at)
        self.out = out
        self.episodes: list[list] = []

    def _on_training_start(self) -> None:
        self.model.save(self.out / "ckpt_0000000")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                ep = info["episode"]
                self.episodes.append([self.num_timesteps, round(ep["r"], 3), ep["l"]])
        while self.save_at and self.num_timesteps >= self.save_at[0]:
            self.model.save(self.out / f"ckpt_{self.save_at.pop(0):07d}")
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chasers", choices=["simple", "mixed"], default="mixed")
    ap.add_argument("--rewards", choices=REWARDS, default="default")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True, help="run folder under runs/")
    ap.add_argument("--threads", type=int, default=2, help="torch threads")
    ap.add_argument("--maps", help="post 005: random maps ('random' or one family)")
    ap.add_argument("--rays", type=int, default=0, help="wall-distance sensors")
    ap.add_argument("--arch", nargs="+", type=int, default=[64, 64], help="hidden layer sizes")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("ckpt_*.zip"):
        old.unlink()

    cfg = Config(
        rewards=REWARDS[args.rewards], chasers=args.chasers, maps=args.maps, rays=args.rays
    )
    env = make_vec_env(TagArena, n_envs=args.envs, seed=args.seed, env_kwargs={"config": cfg})
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=512,
        gamma=0.995,
        ent_coef=0.01,
        learning_rate=3e-4,
        seed=args.seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": args.arch},
    )
    save_at = [s for s in SAVE_AT if s < args.steps] + [args.steps]
    rec = Recorder(save_at, out)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=rec, progress_bar=False)
    elapsed = time.time() - t0

    log = {
        "chasers": args.chasers,
        "rewards": args.rewards,
        "maps": args.maps,
        "rays": args.rays,
        "arch": args.arch,
        "seed": args.seed,
        "total_steps": args.steps,
        "seconds": round(elapsed, 1),
        "episodes": rec.episodes,  # [steps, return, length]
    }
    (out / "train_log.json").write_text(json.dumps(log))
    recent = rec.episodes[-200:]
    full = sum(e[2] >= cfg.max_steps for e in recent)
    print(f"{args.name}: {elapsed / 60:.1f} min; last {len(recent)} episodes: {full} survived")


if __name__ == "__main__":
    main()
