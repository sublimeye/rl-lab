"""Train a pilot with PPO (optionally CAPS) on the vectorised fleet, or on post 10's
one-drone-at-a-time env for the speed comparison.

uv run experiments/011-drone2/train.py --sim basic --task v010 --name basic-s1      # post 10's way
uv run experiments/011-drone2/train.py --sim fleet --task v010 --name fleet-s1      # vectorised
uv run experiments/011-drone2/train.py --task state --steps 30000000 --name state-s1
uv run experiments/011-drone2/train.py --task vision --caps 0.1 0.05 --name vcaps-s1

CAPS (Mysore et al., ICRA 2021) adds two terms to the PPO loss: temporal, how far the policy's
action moves between one state and the next; spatial, how far it moves for a slightly
perturbed state. Both use the deterministic action (the mean), as in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from fleet_env import CALM, FULL, FULL_VISION, ROUGH, V010, Config, FleetEnv, Physics, Reward

HERE = Path(__file__).parent
RUNS = HERE / "runs"
FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
TASKS = {
    "v010": V010,
    "state": FULL,
    "vision": FULL_VISION,
    "vision1": replace(FULL_VISION, stack=1),  # no memory
}


class CapsBuffer(RolloutBuffer):
    """A rollout buffer that also hands out each sample's next observation (and whether there
    is one: not at an episode's end or the rollout's last step)."""

    def get(self, batch_size=None):
        if not self.generator_ready:
            nxt = np.zeros_like(self.observations)
            nxt[:-1] = self.observations[1:]
            valid = np.zeros((self.buffer_size, self.n_envs), np.float32)
            valid[:-1] = 1.0 - self.episode_starts[1:]
            self.next_observations = self.swap_and_flatten(nxt)
            self.next_valid = self.swap_and_flatten(valid)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        s = super()._get_samples(batch_inds, env)
        self._last = (
            self.to_torch(self.next_observations[batch_inds]),
            self.to_torch(self.next_valid[batch_inds].flatten()),
        )
        return s


class CapsPPO(PPO):
    """PPO with the CAPS smoothness terms added to the policy loss (lambda_t, lambda_s; sigma
    is the spatial perturbation, in observation units)."""

    def __init__(self, *args, caps=(0.0, 0.0), caps_sigma=0.05, **kwargs):
        self.caps = caps
        self.caps_sigma = caps_sigma
        kwargs["rollout_buffer_class"] = CapsBuffer
        super().__init__(*args, **kwargs)
        self.caps_log: list = []

    def _mean(self, obs):
        return self.policy.get_distribution(obs).distribution.mean

    def train(self) -> None:  # PPO.train (SB3 2.9.2) plus the two CAPS terms
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        lam_t, lam_s = self.caps
        temporal_log, spatial_log = [], []
        for _ in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, actions)
                values = values.flatten()
                adv = data.advantages
                if self.normalize_advantage and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = th.exp(log_prob - data.old_log_prob)
                pg1 = adv * ratio
                pg2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(pg1, pg2).mean()
                value_loss = F.mse_loss(data.returns, values)
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if lam_t or lam_s:
                    nxt, valid = self.rollout_buffer._last
                    mu = self._mean(data.observations)
                    if lam_t:
                        d = th.linalg.vector_norm(mu - self._mean(nxt), dim=1)
                        temporal = (d * valid).sum() / valid.sum().clamp(min=1.0)
                        loss = loss + lam_t * temporal
                        temporal_log.append(temporal.item())
                    if lam_s:
                        noisy = data.observations + self.caps_sigma * th.randn_like(
                            data.observations
                        )
                        spatial = th.linalg.vector_norm(mu - self._mean(noisy), dim=1).mean()
                        loss = loss + lam_s * spatial
                        spatial_log.append(spatial.item())
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
        explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        if temporal_log or spatial_log:
            self.caps_log.append(
                [
                    float(np.mean(temporal_log)) if temporal_log else 0.0,
                    float(np.mean(spatial_log)) if spatial_log else 0.0,
                ]
            )


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
    return CapsPPO.load(RUNS / f"{ckpt}.zip", device="cpu")


class Recorder(BaseCallback):
    def __init__(self, save_at: list[int], out: Path):
        super().__init__()
        self.save_at = sorted(save_at)
        self.out = out
        self.flights: list[list] = []
        self.marks: list[list] = []
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
        self.model.save(self.out / f"ckpt_{steps:09d}")
        secs = time.time() - self.t0
        self.marks.append([steps, round(secs, 1)])
        recent = self.flights[-2000:]
        if recent:
            landed = sum(f[1] == "landed" for f in recent) / len(recent)
            bird = sum(f[1] == "bird" for f in recent) / len(recent)
            print(
                f"{steps:>11,} steps {secs / 60:6.1f} min  landed {landed:5.1%}  bird {bird:5.1%}",
                flush=True,
            )
        (self.out / "train_log.json").write_text(
            json.dumps(
                {
                    "marks": self.marks,
                    "flights": self.flights[-20000:],
                    "caps": getattr(self.model, "caps_log", []),
                }
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=("fleet", "basic"), default="fleet")
    ap.add_argument("--task", choices=tuple(TASKS), default="state")
    ap.add_argument("--weather", choices=("calm", "rough"), default=None)
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=32, help="steps per env per rollout")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--caps", nargs=2, type=float, default=(0.0, 0.0), metavar=("LT", "LS"))
    ap.add_argument("--arch", nargs="+", type=int, default=None)
    ap.add_argument("--bird", type=float, default=None, help="penalty for hitting a bird")
    ap.add_argument("--min-birds", type=int, default=0, help="at least this many birds")
    ap.add_argument("--near", type=float, default=0.0, help="cost per update near a bird")
    ap.add_argument("--batch", type=int, default=None, help="minibatch (default 1024 / 4096)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=300.0, help="stop after this long")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    th.set_num_threads(args.threads)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("ckpt_*.zip")) + list(out.glob("eval*.json")):
        old.unlink()
    feat = TASKS[args.task]
    if args.min_birds:
        feat = replace(feat, min_birds=args.min_birds)
    weather = args.weather or ("calm" if args.task == "v010" else "rough")
    arch = args.arch or ([64, 64] if args.task == "v010" else [128, 128])
    (out / "config.json").write_text(json.dumps({**vars(args), "weather": weather, "arch": arch}))
    if args.sim == "basic":
        sys.path.insert(0, str(HERE.parent / "010-drone"))
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        from drone_env import DroneFlight

        assert args.task == "v010" and weather == "calm"
        env = make_vec_env(DroneFlight, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    else:
        cfg = Config(
            physics=Physics(),
            weather=CALM if weather == "calm" else ROUGH,
            features=feat,
            reward=Reward(bird=args.bird, near=args.near),
        )
        env = FleetEnv(args.envs, cfg, seed=args.seed)
    batch = args.envs * args.n_steps
    model = CapsPPO(
        "MlpPolicy",
        env,
        caps=tuple(args.caps),
        n_steps=args.n_steps,
        batch_size=args.batch or min(batch, 4096 if args.task != "v010" else 1024),
        n_epochs=args.epochs,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.0,
        policy_kwargs={"net_arch": {"pi": arch, "vf": arch}},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    rec = Recorder([int(args.steps * f) for f in FRACTIONS], out)

    class Deadline(BaseCallback):
        def _on_step(self) -> bool:
            return time.time() - rec.t0 < args.minutes * 60

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=[rec, Deadline()])
    rec._save(model.num_timesteps) if rec.marks[-1][0] != model.num_timesteps else None
    elapsed = time.time() - t0
    log = json.loads((out / "train_log.json").read_text())
    log["seconds"] = round(elapsed, 1)
    (out / "train_log.json").write_text(json.dumps(log))
    recent = rec.flights[-2000:]
    landed = sum(f[1] == "landed" for f in recent)
    print(
        f"done in {elapsed / 60:.1f} min, {model.num_timesteps:,} steps;"
        f" last {len(recent)}: {landed} landed"
    )


if __name__ == "__main__":
    main()
