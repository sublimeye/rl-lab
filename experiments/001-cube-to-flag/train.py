"""Train PPO on the cube arena, saving checkpoints and a training log.

uv run experiments/001-cube-to-flag/train.py [--variant shaped|sparse] [--steps 500000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from cube_env import Config, CubeArena, Rewards

HERE = Path(__file__).parent
RUNS = HERE / "runs"  # git-ignored: runs/<variant>/ checkpoints + raw log
VARIANTS = {"shaped": Rewards(), "sparse": Rewards(shaping=0.0)}


class Recorder(BaseCallback):
    """Saves checkpoints at fixed step counts and logs every finished episode."""

    def __init__(self, save_at: list[int], out: Path):
        super().__init__()
        self.save_at = sorted(save_at)
        self.out = out
        self.episodes: list[list] = []

    def _on_training_start(self) -> None:
        self._save(0)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                ep = info["episode"]
                self.episodes.append(
                    [self.num_timesteps, round(ep["r"], 3), ep["l"], info.get("outcome", "?")]
                )
        while self.save_at and self.num_timesteps >= self.save_at[0]:
            self._save(self.save_at.pop(0))
        return True

    def _save(self, steps: int) -> None:
        self.model.save(self.out / f"ckpt_{steps:07d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="shaped")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = RUNS / args.variant
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("ckpt_*.zip"):
        old.unlink()

    cfg = Config(rewards=VARIANTS[args.variant])
    env = make_vec_env(CubeArena, n_envs=args.envs, seed=args.seed, env_kwargs={"config": cfg})
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=512,
        gamma=0.995,
        ent_coef=0.01,
        learning_rate=3e-4,
        seed=args.seed,
        device="cpu",  # tiny MLP: CPU beats the M-series GPU here
        verbose=0,
    )
    fractions = [0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0]
    rec = Recorder([int(args.steps * f) for f in fractions], out)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=rec, progress_bar=True)
    elapsed = time.time() - t0

    log = {
        "variant": args.variant,
        "total_steps": args.steps,
        "seconds": round(elapsed, 1),
        "episodes": rec.episodes,
    }
    (out / "train_log.json").write_text(json.dumps(log))

    recent = rec.episodes[-200:]
    flags = sum(e[3] == "flag" for e in recent)
    print(f"done in {elapsed / 60:.1f} min; last {len(recent)} episodes: {flags} reached the flag")


if __name__ == "__main__":
    main()
