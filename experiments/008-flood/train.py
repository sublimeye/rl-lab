"""Train PPO to keep the house dry, saving checkpoints and a training log.

uv run experiments/008-flood/train.py --steps 10000000 [--seed 1] [--name full-s1]
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
from stable_baselines3.common.vec_env import SubprocVecEnv

from flood_env import Config, FloodWorld, World

HERE = Path(__file__).parent
RUNS = HERE / "runs"  # git-ignored: runs/<name>/ checkpoints + raw log
FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


class Recorder(BaseCallback):
    """Saves checkpoints at fixed step counts and logs every finished episode."""

    def __init__(self, save_at: list[int], out: Path):
        super().__init__()
        self.save_at = sorted(save_at)
        self.out = out
        self.episodes: list[list] = []
        self.t0 = time.time()

    def _on_training_start(self) -> None:
        self._save(0)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                ep = info["episode"]
                self.episodes.append(
                    [
                        self.num_timesteps,
                        round(float(ep["r"]), 3),
                        int(ep["l"]),
                        info.get("outcome", "?"),
                        info.get("dropped", 0),
                    ]
                )
        while self.save_at and self.num_timesteps >= self.save_at[0]:
            self._save(self.save_at.pop(0))
        return True

    def _save(self, steps: int) -> None:
        self.model.save(self.out / f"ckpt_{steps:08d}")
        recent = self.episodes[-500:]
        if recent:
            dry = sum(e[3] == "dry" for e in recent) / len(recent)
            drops = sum(e[4] for e in recent) / len(recent)
            mins = (time.time() - self.t0) / 60
            print(f"{steps:>10,} steps {mins:6.1f} min  dry {dry:5.1%}  drops/ep {drops:5.1f}")
            (self.out / "train_log.json").write_text(json.dumps({"episodes": self.episodes}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default=None, help="run folder under runs/ (default: s<seed>)")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=2, help="torch threads")
    ap.add_argument(
        "--house-wall",
        type=int,
        default=None,
        help="round 2: sand against the house spills above floor + this (default: no limit)",
    )
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    out = RUNS / (args.name or f"s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("ckpt_*.zip"):
        old.unlink()
    for old in out.glob("eval_*.json"):
        old.unlink()
    (out / "config.json").write_text(json.dumps({"house_wall": args.house_wall}))
    cfg = Config(world=World(house_wall=args.house_wall))

    env = make_vec_env(
        FloodWorld,
        n_envs=args.envs,
        seed=args.seed,
        env_kwargs={"config": cfg},
        vec_env_cls=SubprocVecEnv,
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=2048,
        n_epochs=5,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent,
        learning_rate=3e-4,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",  # small MLP: CPU beats the M-series GPU here
        verbose=0,
    )
    rec = Recorder([int(args.steps * f) for f in FRACTIONS], out)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=rec)
    elapsed = time.time() - t0

    log = {
        "total_steps": args.steps,
        "seconds": round(elapsed, 1),
        "args": vars(args),
        "episodes": rec.episodes,
    }
    (out / "train_log.json").write_text(json.dumps(log))
    recent = rec.episodes[-500:]
    dry = sum(e[3] == "dry" for e in recent)
    print(f"done in {elapsed / 60:.1f} min; last {len(recent)} episodes: {dry} dry")


if __name__ == "__main__":
    main()
