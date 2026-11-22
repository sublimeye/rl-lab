"""Train one pilot for the whole gauntlet with PPO (post 14).

uv run experiments/014-gauntlet/train.py --task school --dials --name main-s1   # the recipe
uv run experiments/014-gauntlet/train.py --task school --name nodials-s1        # no ramps
uv run experiments/014-gauntlet/train.py --task test --dials --name long-s1     # no drills

"school" is the training mix: courses from 15 to 130 m (short drills as well as full-length
flights) with waypoints closer together than on the test. --dials starts every threat easy
and makes it harder as the pilot copes (see Dials).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from gauntlet_env import FULL, MISSILE_SPEED, ROUGH, Config, FleetEnv, Physics, Reward

HERE = Path(__file__).parent
RUNS = HERE / "runs"
FRACTIONS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
SCHOOL = replace(FULL, route=(15.0, 130.0), first=(6.0, 12.0), spacing=(12.0, 20.0))
CALM_MIX = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DODGE = replace(SCHOOL, route=(20.0, 50.0), first=(4.0, 8.0), spacing=(8.0, 14.0),
                route_frog=0.0, guard=0.0)  # fmt: skip
FROGS = replace(SCHOOL, route=(15.0, 35.0), mix=CALM_MIX, route_frog=0.5, guard=1.0)
TASKS = {
    "frogs-predict": replace(FROGS, predict=True),  # the frog drills alone (reward A/B)
    "flock-only": replace(DODGE, predict=True, mix=(0.0, 0, 0, 0, 0, 1.0)),  # diagnostics
    "dodge": DODGE,  # the dodge drills alone (an A/B for the closest-approach inputs)
    "dodge-predict": replace(DODGE, predict=True),
    "drills-predict": replace(SCHOOL, predict=True),
    "school": SCHOOL,
    "school-none": replace(SCHOOL, memory="none"),
    "test": FULL,
    "drills": SCHOOL,  # + MIXES["drills"]
}
# drills: 40% school courses (presence dialled), 30% dodge drills (short, dense waypoints, no
# frogs), 30% frog drills (short, calm waypoints, always a frog on the pad)
MIXES = {
    "drills": [(0.4, SCHOOL), (0.3, DODGE), (0.3, FROGS)],
    "drills-predict": [
        (0.4, replace(SCHOOL, predict=True)),
        (0.3, replace(DODGE, predict=True)),
        (0.3, replace(FROGS, predict=True)),
    ],
}


def actor(model) -> dict:
    lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    lin.append(model.policy.action_net)

    def r(m):
        return {
            "w": m.weight.detach().cpu().numpy().tolist(),
            "b": m.bias.detach().cpu().numpy().tolist(),
        }

    return {"act": "tanh", "squash": False, "layers": [r(m) for m in lin]}


def load_model(ckpt: str):
    return PPO.load(RUNS / f"{ckpt}.zip", device="cpu")


class Recorder(BaseCallback):
    def __init__(self, save_at: list[int], out: Path):
        super().__init__()
        self.save_at = sorted(save_at)
        self.out = out
        self.flights: list[list] = []
        self.levels: list[list] = []
        self.marks: list[list] = []
        self.t0 = time.time()

    def _on_training_start(self) -> None:
        self._save(0)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                self.flights.append(
                    [self.num_timesteps, info["outcome"], round(info["time"], 2),
                     round(info["length"], 1)]
                )  # fmt: skip
        while self.save_at and self.num_timesteps >= self.save_at[0]:
            self._save(self.save_at.pop(0))
        return True

    def _save(self, steps: int) -> None:
        self.model.save(self.out / f"ckpt_{steps:09d}")
        secs = time.time() - self.t0
        self.marks.append([steps, round(secs, 1)])
        recent = self.flights[-3000:]
        if recent:
            share = {k: sum(f[1] == k for f in recent) / len(recent)
                     for k in ("landed", "bird", "missile", "frog", "timeout")}  # fmt: skip
            long = [f for f in recent if f[3] >= 60]
            ll = sum(f[1] == "landed" for f in long) / max(1, len(long))
            print(
                f"{steps:>11,} steps {secs / 60:6.1f} min  "
                + "  ".join(f"{k} {v:5.1%}" for k, v in share.items())
                + f"  | landed on 60 m+ {ll:5.1%}",
                flush=True,
            )
        (self.out / "train_log.json").write_text(
            json.dumps(
                {"marks": self.marks, "flights": self.flights[-20000:], "levels": self.levels}
            )  # fmt: skip
        )


class Dials(BaseCallback):
    """Difficulty dials, each turned up a step when the pilot copes with its threat:
    - missile: top speed from 6 m/s to the test's 11, +0.5 when it gets away from 60% of the
      last 300 missiles it faced;
    - birds: an aimed bird commits 2 s before contact at first (an early, easy-to-read line),
      down to the test's 1 s, -0.1 s when 80% of the last 500 birds it met missed;
    - frogs: the beat from one strike every 6-7 s to the test's 3-4 s, a tenth of the way
      each time half of the last 300 flights with frogs get past all of them.
    The bird and missile dials only move while the pilot lands GATE of its last 1000 flights:
    a flailing beginner makes early-committing birds miss by accident.
    And a fourth, presence: courses start with every waypoint calm and no frogs; the share
    that are real goes up a tenth each time the pilot lands 60% of its last 1000 school courses."""

    WINDOW = {"missile": 300, "bird": 500, "frog": 300, "presence": 1000}
    PASS = {"missile": 0.6, "bird": 0.8, "frog": 0.5, "presence": 0.6}
    GATE = 0.25

    def __init__(self, env: FleetEnv, rec: Recorder):
        super().__init__()
        self.env, self.rec = env, rec
        feat = env.config.features
        self.test = {"frog": feat.frog_period, "bird": feat.bird_lock}
        self.level = {"missile": 0.0, "bird": 0.0, "frog": 0.0, "presence": 0.0}
        self.recent: dict[str, list] = {k: [] for k in self.level}
        self.landed: list[bool] = []
        self._apply()

    def _apply(self) -> None:
        lv = self.level
        top = 6.0 + (MISSILE_SPEED - 6.0) * lv["missile"]
        lock = 2.0 + (self.test["bird"] - 2.0) * lv["bird"]
        lo, hi = self.test["frog"]
        period = (6.0 + (lo - 6.0) * lv["frog"], 7.0 + (hi - 7.0) * lv["frog"])
        cfg = self.env.config

        def dial(feat, presence=True):
            return replace(
                feat,
                missile_speed=(max(6.0, top - 2.0), top),
                bird_lock=round(lock, 3),
                frog_period=period,
                presence=lv["presence"] if presence else 1.0,
            )

        self.env.config = replace(cfg, features=dial(cfg.features))
        if self.env.mix:  # the drills are always real; only school courses ramp presence
            self.env.mix = [(w, dial(f, k == 0)) for k, (w, f) in enumerate(self.env.mix)]
        self.rec.levels.append([self.num_timesteps, dict(lv)])

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" not in info:
                continue
            (b_ok, m_ok), (b_hit, m_hit) = info["beaten"], info["caught"]
            b_open, m_open = info["open"]
            alive = info["outcome"] in ("landed", "timeout")
            self.recent["bird"] += [True] * (b_ok + (b_open if alive else 0)) + [False] * b_hit
            self.recent["missile"] += [True] * (m_ok + (m_open if alive else 0)) + [False] * m_hit
            self.landed.append(info["outcome"] == "landed")
            if info["drill"] == 0:  # school courses only (the drills are always real)
                self.recent["presence"].append(info["outcome"] == "landed")
            f_ok, f_fail = info["frogs"]
            self.recent["frog"] += [True] * f_ok + [False] * f_fail
        changed = False
        self.landed = self.landed[-1000:]
        landing = len(self.landed) == 1000 and np.mean(self.landed) >= self.GATE
        for k, rec in self.recent.items():
            w = self.WINDOW[k]
            if k in ("bird", "missile") and not landing:
                self.recent[k] = rec[-w:]
                continue
            if len(rec) >= w and self.level[k] < 1.0:
                if np.mean(rec[-w:]) >= self.PASS[k]:
                    step = 0.1
                    self.level[k] = min(1.0, round(self.level[k] + step, 2))
                    self.recent[k] = []
                    changed = True
                    print(f"{self.num_timesteps:>11,} steps  {k} level {self.level[k]}", flush=True)
                else:
                    self.recent[k] = rec[-w:]
        if changed:
            self._apply()
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), default="school")
    ap.add_argument("--steps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=32, help="steps per env per rollout")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", nargs="+", type=int, default=[256, 256])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=None, help="decay linearly to this")
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--dials", action="store_true")
    ap.add_argument("--near", type=float, default=0.0, help="near-miss cost (Reward.near)")
    ap.add_argument("--time-cost", type=float, default=0.0, help="per update (Reward.time)")
    ap.add_argument("--danger", type=float, default=0.0, help="danger shaping (Reward.danger)")
    ap.add_argument("--landed", type=float, default=10.0, help="landing bonus (Reward.landed)")
    ap.add_argument("--progress", type=float, default=0.3, help="per metre (Reward.progress)")
    ap.add_argument("--failed", type=float, default=-10.0, help="on failing (Reward.failed)")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from (runs/...)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=290.0, help="stop after this long")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    th.set_num_threads(args.threads)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("ckpt_*")) + list(out.glob("eval*.json")):
        old.unlink()
    (out / "config.json").write_text(json.dumps({**vars(args), "weather": "rough"}))
    cfg = Config(
        Physics(),
        ROUGH,
        TASKS[args.task],
        Reward(
            progress=args.progress,
            landed=args.landed,
            failed=args.failed,
            near=args.near,
            time=args.time_cost,
            danger=args.danger,
        ),
    )
    env = FleetEnv(args.envs, cfg, seed=args.seed, mix=MIXES.get(args.task))
    common = dict(
        n_steps=args.n_steps,
        batch_size=min(args.envs * args.n_steps, args.batch),
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        learning_rate=args.lr
        if args.lr_end is None
        else (lambda left: args.lr_end + (args.lr - args.lr_end) * left),
        ent_coef=args.ent,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    if args.resume:
        model = PPO.load(RUNS / args.resume, env=env, device="cpu", **common)
    else:
        kw = {"net_arch": {"pi": args.arch, "vf": args.arch}}
        model = PPO("MlpPolicy", env, policy_kwargs=kw, **common)
    rec = Recorder([int(args.steps * f) for f in FRACTIONS], out)

    class Deadline(BaseCallback):
        def _on_step(self) -> bool:
            return time.time() - rec.t0 < args.minutes * 60

    t0 = time.time()
    callbacks = [rec, Deadline()] + ([Dials(env, rec)] if args.dials else [])
    model.learn(total_timesteps=args.steps, callback=callbacks)
    if rec.marks[-1][0] != model.num_timesteps:
        rec._save(model.num_timesteps)
    elapsed = time.time() - t0
    log = json.loads((out / "train_log.json").read_text())
    log["seconds"] = round(elapsed, 1)
    (out / "train_log.json").write_text(json.dumps(log))
    print(f"done in {elapsed / 60:.1f} min, {model.num_timesteps:,} steps")


if __name__ == "__main__":
    main()
