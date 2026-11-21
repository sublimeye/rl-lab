"""Train a pilot with PPO or SAC, saving checkpoints and a log of every flight.

uv run experiments/010-drone/train.py --algo ppo --steps 10000000 --seed 1
uv run experiments/010-drone/train.py --algo ppo --smooth 0.1 --name smooth-s1
uv run experiments/010-drone/train.py --algo ppo --weather rough --name rand-s1
uv run experiments/010-drone/train.py --algo sac --steps 1000000 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_env import CALM, HEAVY, ROUGH, WINDY, Config, DroneFlight, Physics, Reward

HERE = Path(__file__).parent
RUNS = HERE / "runs"  # git-ignored: runs/<name>/ checkpoints + raw log
FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
WEATHER = {"calm": CALM, "windy": WINDY, "heavy": HEAVY, "rough": ROUGH}


def actor(model) -> dict:
    """The actor as plain lists {act, squash, layers: [{w, b}]} (deterministic action)."""
    from stable_baselines3 import SAC

    def r(m):
        return {
            "w": m.weight.detach().cpu().numpy().tolist(),
            "b": m.bias.detach().cpu().numpy().tolist(),
        }

    if isinstance(model, SAC):
        lin = [m for m in model.policy.actor.latent_pi if hasattr(m, "weight")]
        lin.append(model.policy.actor.mu)
        return {"act": "relu", "squash": True, "layers": [r(m) for m in lin]}
    lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    lin.append(model.policy.action_net)
    return {"act": "tanh", "squash": False, "layers": [r(m) for m in lin]}


def load_model(ckpt: str):
    from stable_baselines3 import PPO, SAC

    path = RUNS / f"{ckpt}.zip"
    cfg = json.loads((path.parent / "config.json").read_text())
    return (SAC if cfg["algo"] == "sac" else PPO).load(path, device="cpu")


class Recorder(BaseCallback):
    """Saves checkpoints at fixed step counts and logs every finished flight."""

    def __init__(self, save_at: list[int], out: Path):
        super().__init__()
        self.save_at = sorted(save_at)
        self.out = out
        self.flights: list[list] = []
        self.marks: list[list] = []  # [steps, seconds] at each checkpoint
        self.t0 = time.time()

    def _on_training_start(self) -> None:
        self._save(0)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                self.flights.append([self.num_timesteps, info["outcome"], round(info["time"], 2)])
        while self.save_at and self.num_timesteps >= self.save_at[0]:
            self._save(self.save_at.pop(0))
        return True

    def _save(self, steps: int) -> None:
        self.model.save(self.out / f"ckpt_{steps:08d}")
        secs = time.time() - self.t0
        self.marks.append([steps, round(secs, 1)])
        recent = self.flights[-1000:]
        if recent:
            landed = sum(f[1] == "landed" for f in recent) / len(recent)
            crashed = sum(f[1] in ("crashed", "water", "lost") for f in recent) / len(recent)
            print(
                f"{steps:>10,} steps {secs / 60:6.1f} min  landed {landed:5.1%}  "
                f"failed {crashed:5.1%}",
                flush=True,
            )
        (self.out / "train_log.json").write_text(
            json.dumps({"marks": self.marks, "flights": self.flights})
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=("ppo", "sac"), default="ppo")
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=None, help="default: 16 for PPO, 1 for SAC")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default=None, help="run folder under runs/ (default: <algo>-s<seed>)")
    ap.add_argument("--weather", choices=tuple(WEATHER), default="calm")
    ap.add_argument("--smooth", type=float, default=0.0, help="penalty on stick changes")
    ap.add_argument("--stick-rate", type=float, default=0.0, help="max stick move per update")
    ap.add_argument("--arch", nargs="+", type=int, default=[64, 64])
    ap.add_argument("--threads", type=int, default=1, help="torch threads")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    from stable_baselines3 import PPO, SAC

    torch.set_num_threads(args.threads)
    out = RUNS / (args.name or f"{args.algo}-s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("ckpt_*.zip")) + list(out.glob("eval_*.json")):
        old.unlink()
    (out / "config.json").write_text(json.dumps(vars(args)))
    cfg = Config(
        physics=Physics(stick_rate=args.stick_rate),
        weather=WEATHER[args.weather],
        reward=Reward(smooth=args.smooth),
    )
    n_envs = args.envs or (16 if args.algo == "ppo" else 1)
    env = make_vec_env(
        DroneFlight,
        n_envs=n_envs,
        seed=args.seed,
        env_kwargs={"config": cfg},
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
    )
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.0,
            policy_kwargs={"net_arch": {"pi": args.arch, "vf": args.arch}},
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            buffer_size=1_000_000,
            batch_size=256,
            learning_starts=10_000,
            gamma=0.99,
            learning_rate=3e-4,
            policy_kwargs={"net_arch": args.arch},
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
    rec = Recorder([int(args.steps * f) for f in FRACTIONS], out)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=rec)
    elapsed = time.time() - t0
    log = {"seconds": round(elapsed, 1), "marks": rec.marks, "flights": rec.flights}
    (out / "train_log.json").write_text(json.dumps(log))
    recent = rec.flights[-1000:]
    landed = sum(f[1] == "landed" for f in recent)
    print(f"done in {elapsed / 60:.1f} min; last {len(recent)} flights: {landed} landed")


if __name__ == "__main__":
    main()
