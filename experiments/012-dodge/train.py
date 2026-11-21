"""Train a pilot with PPO on the vectorised fleet with threats (post 12).

uv run experiments/012-dodge/train.py --task mix --name mix-s1    # the test mix (post 11's way)
uv run experiments/012-dodge/train.py --task school --curriculum --name school-s1  # + arenas
uv run experiments/012-dodge/train.py --task school --warm <post 11 checkpoint .zip>

--warm starts from a post 11 exact-position pilot: its inputs keep their weights (post 11's two
nearest-bird slots become the first two threat slots), the new inputs start at zero, so the
starting pilot flies exactly like post 11's. CAPS is kept from post 11 (off by default).
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
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from dodge_env import (
    ARENA,
    FULL,
    FULL_VISION,
    MISSILE_SPEED,
    MIX,
    N_STATE,
    ONLY,
    ROUGH,
    SCHOOL,
    Config,
    FleetEnv,
    Physics,
)

HERE = Path(__file__).parent
RUNS = HERE / "runs"
FRACTIONS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
TASKS = {
    "mix": replace(FULL, threats=MIX),
    "school": replace(FULL, threats=SCHOOL),
    "vmix": replace(FULL_VISION, threats=MIX),
    "vschool": replace(FULL_VISION, threats=SCHOOL),
    **{f"{k}-arena": replace(FULL, threats=v) for k, v in ONLY.items()},  # specialists
    "arena": replace(FULL, threats=ARENA),
    "school2": replace(FULL, threats=MIX),  # + MIXES["school2"]
    "vschool2": replace(FULL_VISION, threats=MIX),
}
# school 2: half test courses, a quarter aimed-bird arenas, a quarter missile arenas
SCHOOL2 = [(0.5, MIX), (0.25, ONLY["aimed"]), (0.25, ONLY["missile"])]
MIXES = {"school2": SCHOOL2, "vschool2": SCHOOL2}


def warm_start(model, path: str) -> None:
    """Copies a post 11 state pilot into model (see the module docstring)."""
    from stable_baselines3.common.save_util import load_from_zip_file

    _, params, _ = load_from_zip_file(path, device="cpu")
    old = params["policy"]
    new = model.policy.state_dict()
    w_old, w_new = 26, N_STATE
    stack = new["mlp_extractor.policy_net.0.weight"].shape[1] // w_new
    cols = []
    for c in range(w_old):
        cols.append(c if c < 16 else 18 + 8 * ((c - 16) // 5) + (c - 16) % 5)
    for key in new:
        if key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
            w = th.zeros_like(new[key])
            for fr in range(stack):
                for c, m in enumerate(cols):
                    w[:, fr * w_new + m] = old[key][:, fr * w_old + c]
            new[key] = w
        else:
            assert old[key].shape == new[key].shape, key
            new[key] = old[key].clone()
    model.policy.load_state_dict(new)


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
        self.levels: list[list] = []  # [steps, missile top speed] (curriculum)
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
        secs = time.time() - self.t0
        self.marks.append([steps, round(secs, 1)])
        recent = self.flights[-2000:]
        if recent:
            course = [f for f in recent if not f[3]] or [[0, ""]]
            arena = [f for f in recent if f[3]] or [[0, ""]]
            landed = sum(f[1] == "landed" for f in course) / len(course)
            held = sum(f[1] == "timeout" for f in arena) / len(arena)
            hits = sum(f[1] in ("bird", "rock", "missile") for f in recent) / len(recent)
            print(
                f"{steps:>11,} steps {secs / 60:6.1f} min  landed {landed:5.1%}"
                f"  arena held {held:5.1%}  threat hits {hits:5.1%}",
                flush=True,
            )
        (self.out / "train_log.json").write_text(
            json.dumps(
                {
                    "marks": self.marks,
                    "flights": self.flights[-20000:],
                    "caps": getattr(self.model, "caps_log", []),
                    "levels": self.levels,
                }
            )
        )


class Curriculum(BaseCallback):
    """Missile speed: starts at SLOW m/s; each time the pilot gets away from PASS of the last
    WINDOW missiles it faced, the top speed goes up a step, until it's the test missile's (the
    slowest a course draws stays 2 m/s below the top)."""

    SLOW, STEP, PASS, WINDOW = 6.0, 0.5, 0.6, 500

    def __init__(self, env: FleetEnv, rec: Recorder):
        super().__init__()
        self.env, self.rec = env, rec
        self.top = self.SLOW
        self.recent: list[bool] = []
        self._apply()

    def _apply(self) -> None:
        cfg = self.env.config
        speed = (max(self.SLOW, self.top - 2.0), self.top)
        tr = replace(cfg.features.threats, missile_speed=speed)
        self.env.config = replace(cfg, features=replace(cfg.features, threats=tr))
        if self.env.mix:
            self.env.mix = [(w, replace(t, missile_speed=speed)) for w, t in self.env.mix]
        self.rec.levels.append([self.num_timesteps, self.top])

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            # only flights that faced their missile: hit, outlasted it, or landed/held after
            # it launched (a flight that ended before launch, or by something else, says nothing)
            m, oc = info.get("missile", -1), info.get("outcome")
            if oc == "missile":
                self.recent.append(False)
            elif m == 2 or (m == 1 and oc in ("landed", "timeout")):
                self.recent.append(True)
        if len(self.recent) >= self.WINDOW and self.top < MISSILE_SPEED:
            if np.mean(self.recent[-self.WINDOW :]) >= self.PASS:
                self.top = min(MISSILE_SPEED, self.top + self.STEP)
                self.recent = []
                self._apply()
                print(f"{self.num_timesteps:>11,} steps  missile speed now {self.top}", flush=True)
        return True


class Levels(BaseCallback):
    """One difficulty dial for arena training, 0 (easy) to 1 (the test arenas), up a step each
    time the pilot holds PASS of its last WINDOW arenas: birds from 30% of their speed to full,
    rocks from one every 6-7 s to the test's beat, missiles from 6 m/s to the test's."""

    STEP, PASS, WINDOW = 0.1, 0.6, 300

    def __init__(self, env: FleetEnv, rec: Recorder):
        super().__init__()
        self.env, self.rec = env, rec
        self.test = env.config.features.threats
        self.level = 0.0
        self.recent: list[bool] = []
        self._apply()

    def _apply(self) -> None:
        lv, t = self.level, self.test
        lerp = lambda a, b: a + (b - a) * lv  # noqa: E731
        tr = replace(
            t,
            bird_scale=lerp(0.3, 1.0),
            rock_every=(lerp(6.0, t.rock_every[0]), lerp(7.0, t.rock_every[1])),
            missile_speed=(lerp(6.0, t.missile_speed[0]), lerp(6.0, t.missile_speed[1])),
        )
        cfg = self.env.config
        self.env.config = replace(cfg, features=replace(cfg.features, threats=tr))
        self.rec.levels.append([self.num_timesteps, round(lv, 2)])

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if info.get("arena"):
                self.recent.append(info["outcome"] == "timeout")
        if len(self.recent) >= self.WINDOW and self.level < 1.0:
            if np.mean(self.recent[-self.WINDOW :]) >= self.PASS:
                self.level = min(1.0, round(self.level + self.STEP, 2))
                self.recent = []
                self._apply()
                print(f"{self.num_timesteps:>11,} steps  level now {self.level}", flush=True)
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), default="school")
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=32, help="steps per env per rollout")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--caps", nargs=2, type=float, default=(0.0, 0.0), metavar=("LT", "LS"))
    ap.add_argument("--arch", nargs="+", type=int, default=[128, 128])
    ap.add_argument("--batch", type=int, default=None, help="minibatch (default 4096)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--warm", default=None, help="post 11 checkpoint to start from")
    ap.add_argument("--ent", type=float, default=0.0, help="entropy bonus")
    ap.add_argument(
        "--sde",
        type=int,
        default=0,
        help="gSDE: exploration noise held for this many updates (0: fresh noise every update)",
    )
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=300.0, help="stop after this long")
    ap.add_argument("--levels", action="store_true", help="arena difficulty dial (Levels)")
    ap.add_argument(
        "--curriculum",
        action="store_true",
        help="missiles start slow and speed up as the pilot gets away from them",
    )
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    th.set_num_threads(args.threads)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("ckpt_*.zip")) + list(out.glob("eval*.json")):
        old.unlink()
    (out / "config.json").write_text(json.dumps({**vars(args), "weather": "rough"}))
    cfg = Config(physics=Physics(), weather=ROUGH, features=TASKS[args.task])
    env = FleetEnv(args.envs, cfg, seed=args.seed, mix=MIXES.get(args.task))
    batch = args.envs * args.n_steps
    model = CapsPPO(
        "MlpPolicy",
        env,
        caps=tuple(args.caps),
        n_steps=args.n_steps,
        batch_size=args.batch or min(batch, 4096),
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=args.ent,
        use_sde=args.sde > 0,
        sde_sample_freq=args.sde,
        policy_kwargs={"net_arch": {"pi": args.arch, "vf": args.arch}},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    if args.warm:
        assert args.task in ("mix", "school")
        warm_start(model, args.warm)
    rec = Recorder([int(args.steps * f) for f in FRACTIONS], out)
    callbacks = [rec]
    if args.curriculum:
        callbacks.append(Curriculum(env, rec))
    if args.levels:
        callbacks.append(Levels(env, rec))

    class Deadline(BaseCallback):
        def _on_step(self) -> bool:
            return time.time() - rec.t0 < args.minutes * 60

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=[*callbacks, Deadline()])
    rec._save(model.num_timesteps) if rec.marks[-1][0] != model.num_timesteps else None
    elapsed = time.time() - t0
    log = json.loads((out / "train_log.json").read_text())
    log["seconds"] = round(elapsed, 1)
    (out / "train_log.json").write_text(json.dumps(log))
    print(f"done in {elapsed / 60:.1f} min, {model.num_timesteps:,} steps")


if __name__ == "__main__":
    main()
