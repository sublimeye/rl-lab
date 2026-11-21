"""Train a pilot past air-frogs with PPO (post 13).

uv run experiments/013-frogs/train.py --task none --name none-s1        # 0.15 s of memory
uv run experiments/013-frogs/train.py --task hist --name hist-s1        # + 3 s of frog history
uv run experiments/013-frogs/train.py --task hist --aux 1 --name aux-s1 # + countdown head
uv run experiments/013-frogs/train.py --task lstm --lstm --name lstm-s1 # an LSTM's own memory

The last K_FROGS observation columns are the true time to each frog's next strike: the
feature extractor drops them, and with --aux a small head on the actor's last hidden layer
learns to predict them (drawn on the site as a countdown).
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
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from frog_env import FULL, K_FROGS, NONE, ROUGH, Config, FleetEnv, Physics

HERE = Path(__file__).parent
RUNS = HERE / "runs"
FRACTIONS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
FROGS = replace(FULL, obs="frog", threats=NONE, frogs=(1, 2), pad_move=0.0)
TASKS = {
    "none": FROGS,
    "hist": replace(FROGS, memory="hist"),
    "tell": replace(FROGS, tell=True),
    "lstm": replace(FROGS, stack=1),  # the LSTM keeps its own memory
    "guard": replace(FROGS, memory="hist", guard=True, frog_period=(3.0, 4.0)),
    "guard-none": replace(FROGS, guard=True, frog_period=(3.0, 4.0)),
    "guard-tell": replace(FROGS, guard=True, frog_period=(3.0, 4.0), tell=True),
    "guard-lstm": replace(FROGS, guard=True, frog_period=(3.0, 4.0), stack=1),
    "slow": replace(FROGS, memory="hist", frog_period=(4.0, 5.0)),  # diagnostics
    "slow-none": replace(FROGS, frog_period=(4.0, 5.0)),
}


class DropLabels(BaseFeaturesExtractor):
    """The pilot's input: the observation without its last K_FROGS columns (the labels)."""

    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0] - K_FROGS)

    def forward(self, obs):
        return obs[:, :-K_FROGS]


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

    def __init__(self, *args, caps=(0.0, 0.0), caps_sigma=0.05, aux=0.0, **kwargs):
        self.caps = caps
        self.caps_sigma = caps_sigma
        self.aux = aux
        kwargs["rollout_buffer_class"] = CapsBuffer
        super().__init__(*args, **kwargs)
        self.caps_log: list = []
        self.aux_log: list = []
        if hasattr(self, "policy"):  # (not when SB3 is loading a checkpoint)
            self.add_head()

    def add_head(self) -> None:
        """The countdown head, kept beside the policy (SB3 rebuilds the policy on loading)
        and trained only with aux > 0."""
        torch_seed = th.random.get_rng_state()
        self.aux_head = th.nn.Linear(self.policy.mlp_extractor.latent_dim_pi, K_FROGS)
        th.random.set_rng_state(torch_seed)
        self.aux_opt = th.optim.Adam(self.aux_head.parameters(), lr=3e-4)

    def countdown(self, obs):
        feats = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        return self.aux_head(self.policy.mlp_extractor.forward_actor(feats))

    def _mean(self, obs):
        return self.policy.get_distribution(obs).distribution.mean

    def train(self) -> None:  # PPO.train (SB3 2.9.2) plus the two CAPS terms
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        lam_t, lam_s = self.caps
        temporal_log, spatial_log, aux_log = [], [], []
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
                if self.aux:
                    want = data.observations[:, -K_FROGS:]
                    err = F.mse_loss(self.countdown(data.observations), want)
                    loss = loss + self.aux * err
                    aux_log.append(err.item())
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
                if self.aux:
                    self.aux_opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                if self.aux:
                    self.aux_opt.step()
            self._n_updates += 1
        explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        if aux_log:
            self.aux_log.append(float(np.mean(aux_log)))
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

    out = {"act": "tanh", "squash": False, "layers": [r(m) for m in lin]}
    if hasattr(model, "aux_head"):
        out["aux"] = r(model.aux_head)
    return out


def load_model(ckpt: str):
    """A checkpoint (the countdown head is saved separately: SB3 rebuilds the policy)."""
    path = RUNS / f"{ckpt}.zip"
    m = CapsPPO.load(path, device="cpu")
    head = path.with_suffix(".aux.pt")
    if head.exists():
        m.add_head()
        m.aux_head.load_state_dict(th.load(head))
    return m


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
                    [self.num_timesteps, info["outcome"], round(info["time"], 2), info["arena"]]
                )
        while self.save_at and self.num_timesteps >= self.save_at[0]:
            self._save(self.save_at.pop(0))
        return True

    def _save(self, steps: int) -> None:
        self.model.save(self.out / f"ckpt_{steps:09d}")
        if hasattr(self.model, "aux_head"):
            head = self.model.aux_head.state_dict()
            th.save(head, self.out / f"ckpt_{steps:09d}.aux.pt")
        secs = time.time() - self.t0
        self.marks.append([steps, round(secs, 1)])
        recent = self.flights[-2000:]
        if recent:
            landed = sum(f[1] == "landed" for f in recent) / len(recent)
            frog = sum(f[1] == "frog" for f in recent) / len(recent)
            t = np.mean([f[2] for f in recent if f[1] == "landed"] or [0])
            print(
                f"{steps:>11,} steps {secs / 60:6.1f} min  landed {landed:5.1%}"
                f"  frog {frog:5.1%}  time to land {t:4.1f} s",
                flush=True,
            )
        (self.out / "train_log.json").write_text(
            json.dumps(
                {
                    "marks": self.marks,
                    "flights": self.flights[-20000:],
                    "caps": getattr(self.model, "caps_log", []),
                    "aux": getattr(self.model, "aux_log", []),
                    "levels": self.levels,
                }
            )
        )


class Levels(BaseCallback):
    """Frog rhythm dial: strikes start every 6-7 s (long gaps to land in) and come a step
    faster each time the pilot lands PASS of its last WINDOW flights, down to the task's own
    period."""

    STEP, PASS, WINDOW = 0.1, 0.5, 500

    def __init__(self, env: FleetEnv, rec: Recorder):
        super().__init__()
        self.env, self.rec = env, rec
        self.test = env.config.features.frog_period
        self.level = 0.0
        self.recent: list[bool] = []
        self._apply()

    def _apply(self) -> None:
        lv, (lo, hi) = self.level, self.test
        period = (6.0 + (lo - 6.0) * lv, 7.0 + (hi - 7.0) * lv)
        cfg = self.env.config
        self.env.config = replace(cfg, features=replace(cfg.features, frog_period=period))
        self.rec.levels.append([self.num_timesteps, round(lv, 2)])

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                self.recent.append(info["outcome"] == "landed")
        if len(self.recent) >= self.WINDOW and self.level < 1.0:
            if np.mean(self.recent[-self.WINDOW :]) >= self.PASS:
                self.level = min(1.0, round(self.level + self.STEP, 2))
                self.recent = []
                self._apply()
                print(f"{self.num_timesteps:>11,} steps  level now {self.level}", flush=True)
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), default="hist")
    ap.add_argument("--steps", type=int, default=40_000_000)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=32, help="steps per env per rollout")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", nargs="+", type=int, default=[128, 128])
    ap.add_argument("--batch", type=int, default=None, help="minibatch (default 4096)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--aux", type=float, default=0.0, help="weight of the countdown loss")
    ap.add_argument("--lstm", action="store_true", help="RecurrentPPO (sb3-contrib)")
    ap.add_argument("--levels", action="store_true", help="frog rhythm dial (Levels)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=300.0, help="stop after this long")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    th.set_num_threads(args.threads)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("ckpt_*")) + list(out.glob("eval*.json")):
        old.unlink()
    (out / "config.json").write_text(json.dumps({**vars(args), "weather": "rough"}))
    cfg = Config(physics=Physics(), weather=ROUGH, features=TASKS[args.task])
    env = FleetEnv(args.envs, cfg, seed=args.seed)
    batch = args.envs * args.n_steps
    common = dict(
        n_steps=args.n_steps,
        batch_size=args.batch or min(batch, 4096),
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.0,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    kw = {"net_arch": {"pi": args.arch, "vf": args.arch}, "features_extractor_class": DropLabels}
    if args.lstm:
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=kw, **common)
    else:
        model = CapsPPO("MlpPolicy", env, aux=args.aux, policy_kwargs=kw, **common)
    rec = Recorder([int(args.steps * f) for f in FRACTIONS], out)

    class Deadline(BaseCallback):
        def _on_step(self) -> bool:
            return time.time() - rec.t0 < args.minutes * 60

    t0 = time.time()
    callbacks = [rec, Deadline()] + ([Levels(env, rec)] if args.levels else [])
    model.learn(total_timesteps=args.steps, callback=callbacks)
    rec._save(model.num_timesteps) if rec.marks[-1][0] != model.num_timesteps else None
    elapsed = time.time() - t0
    log = json.loads((out / "train_log.json").read_text())
    log["seconds"] = round(elapsed, 1)
    (out / "train_log.json").write_text(json.dumps(log))
    print(f"done in {elapsed / 60:.1f} min, {model.num_timesteps:,} steps")


if __name__ == "__main__":
    main()
