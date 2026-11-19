"""Train either side with masked PPO, saving checkpoints and a training log.

uv run experiments/009-beaver/train.py --side beaver --steps 20000000 --seed 1
uv run experiments/009-beaver/train.py --side river --vs b-s1/ckpt_20000000 --name r-vs-b1
uv run experiments/009-beaver/train.py --side beaver --vs bots r-vs-b1/ckpt_10000000 --name b2-s1

--vs lists the opponents, one drawn per game: 'bots' (the scripted river mix), 'warden' (the
scripted beaver) or a checkpoint under runs/ (a trained network of the other side, which samples
its moves). Default: bots for the beaver, warden for the river.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from beaver_env import (
    BeaverGame,
    Config,
    Net,
    NetBeaver,
    NetRiver,
    RiverConfig,
    RiverGame,
    default_river,
    warden_action,
)

HERE = Path(__file__).parent
RUNS = HERE / "runs"  # git-ignored: runs/<name>/ checkpoints + raw log
FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def policy_net(model, digits: int | None = None) -> dict:
    """The actor MLP as plain lists: [{w: out x in, b: out}], tanh between layers."""
    lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    lin.append(model.policy.action_net)

    def r(a):
        a = a.detach().cpu().numpy().astype(float)
        return np.round(a, digits).tolist() if digits else a.tolist()

    return {"layers": [{"w": r(m.weight), "b": r(m.bias)} for m in lin]}


def load_spec(ckpt: str) -> dict:
    """A checkpoint's actor: {"convs": [...] (CNN runs only), "layers": [...]}."""
    model = MaskablePPO.load(RUNS / f"{ckpt}.zip", device="cpu")
    spec = policy_net(model)
    if hasattr(model.policy.pi_features_extractor, "convs"):
        from cnn import conv_layers

        spec["convs"] = conv_layers(model)
    return spec


class Mix:
    """Draws one opponent per game: 'bots' / 'warden' or a network's layers."""

    def __init__(self, side: str, specs: list):
        self.side = side
        self.specs = specs

    def __call__(self, rng: np.random.Generator):
        s = self.specs[int(rng.integers(len(self.specs)))]
        if s == "bots":
            return default_river(rng)
        if s == "warden":
            return warden_action
        net = Net(s, sample=True, rng=np.random.default_rng(int(rng.integers(2**31))))
        return NetRiver(net) if self.side == "beaver" else NetBeaver(net)


class Recorder(BaseCallback):
    """Saves checkpoints at fixed step counts and logs every finished game."""

    def __init__(self, save_at: list[int], out: Path, win: str):
        super().__init__()
        self.save_at = sorted(save_at)
        self.out = out
        self.win = win
        self.games: list[list] = []
        self.t0 = time.time()

    def _on_training_start(self) -> None:
        self._save(0)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                self.games.append([self.num_timesteps, info["outcome"], info["dams"], info["dug"]])
        while self.save_at and self.num_timesteps >= self.save_at[0]:
            self._save(self.save_at.pop(0))
        return True

    def _save(self, steps: int) -> None:
        self.model.save(self.out / f"ckpt_{steps:08d}")
        recent = self.games[-1000:]
        if recent:
            won = sum(e[1] == self.win for e in recent) / len(recent)
            dams = sum(e[2] for e in recent) / len(recent)
            mins = (time.time() - self.t0) / 60
            print(
                f"{steps:>10,} steps {mins:6.1f} min  won {won:5.1%}  dams/game {dams:4.1f}",
                flush=True,
            )
            (self.out / "train_log.json").write_text(json.dumps({"games": self.games}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("beaver", "river"), default="beaver")
    ap.add_argument("--vs", nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default=None, help="run folder under runs/ (default: <side>-s<seed>)")
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--arch", nargs="+", type=int, default=[128, 128])
    ap.add_argument("--threads", type=int, default=2, help="torch threads")
    ap.add_argument("--cnn", action="store_true", help="conv front end (cnn.py), --arch heads")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    out = RUNS / (args.name or f"{args.side[0]}-s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("ckpt_*.zip")) + list(out.glob("eval_*.json")):
        old.unlink()
    vs = args.vs or (["bots"] if args.side == "beaver" else ["warden"])
    (out / "config.json").write_text(
        json.dumps({"side": args.side, "vs": vs, "arch": args.arch, "cnn": args.cnn})
    )
    specs = [s if s in ("bots", "warden") else load_spec(s) for s in vs]
    mix = Mix(args.side, specs)
    if args.side == "beaver":
        env_cls, kw, win = BeaverGame, {"config": Config(river=mix)}, "dry"
    else:
        env_cls, kw, win = RiverGame, {"config": RiverConfig(beaver=mix)}, "flooded"

    policy_kwargs: dict = {"net_arch": args.arch}
    if args.cnn:
        from cnn import BoardCNN

        policy_kwargs = {
            "features_extractor_class": BoardCNN,
            "net_arch": {"pi": args.arch, "vf": args.arch},
        }
    env = make_vec_env(
        env_cls, n_envs=args.envs, seed=args.seed, env_kwargs=kw, vec_env_cls=SubprocVecEnv
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=2048,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.ent,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device="cpu",  # small MLP: CPU beats the M-series GPU here
        verbose=0,
    )
    rec = Recorder([int(args.steps * f) for f in FRACTIONS], out, win)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=rec)
    elapsed = time.time() - t0
    log = {
        "total_steps": args.steps,
        "seconds": round(elapsed, 1),
        "args": vars(args),
        "games": rec.games,
    }
    (out / "train_log.json").write_text(json.dumps(log))
    recent = rec.games[-1000:]
    won = sum(e[1] == win for e in recent)
    print(f"done in {elapsed / 60:.1f} min; last {len(recent)} games: {won} won")


if __name__ == "__main__":
    main()
