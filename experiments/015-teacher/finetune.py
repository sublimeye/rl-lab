"""Fine-tune the taught network with PPO, kept close to its teacher (post 15).

Starting from a DAgger network (dagger.py), PPO trains on full gauntlet courses with post 14's
reward. Three things keep it from wrecking what it was taught:
- the value network learns alone first (the actor is frozen for --warmup updates), so the first
  policy updates don't follow a random critic;
- the exploration noise starts small (--std), not SB3's default of 1;
- every update also pays --stay x the squared distance between its sticks and the taught
  network's (a frozen copy) on the same states: the policy may move, but it pays to move.
With --school it trains on post 14's drill mix with post 14's difficulty dials (each threat
starts easy and gets harder as the pilot copes); without, on full test courses. Optionally,
--dodge pays a bonus for each aimed bird or missile got away from, and --bc fits the planner's
own sticks on the states DAgger collected at every update (as DAPG does with demonstrations,
Rajeswaran et al. 2018): kept close to the teacher itself, not only to its copy.

uv run experiments/015-teacher/finetune.py --init dagger-s1/round_03 --name ft-school --school \
    --stay 1 --bc 0.5 --bc-data dagger-s1
"""

from __future__ import annotations

import argparse
import copy
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))

from dagger import RUNS, actor_layers, stick_loss, test  # noqa: E402
from gauntlet_env import FULL, ROUGH, Config, FleetEnv, Physics, Reward  # noqa: E402

FEAT = replace(FULL, predict=True)


class DodgeEnv(FleetEnv):
    """FleetEnv + a bonus for each aimed bird or missile beaten (passed or outlasted)."""

    dodge = 0.0

    def step_wait(self):
        before = self.fleet.beaten.sum(1).copy()
        obs, r, dones, infos = super().step_wait()
        if self.dodge:
            after = self.fleet.beaten.sum(1)
            for i in np.nonzero(dones)[0]:  # restarted: its count is in the info
                after[i] = sum(infos[i]["beaten"])
            r = r + (self.dodge * (after - before)).astype(np.float32)
        return obs, r, dones, infos


class TaughtPPO(PPO):
    """PPO with a frozen teacher: loss += stay x |mean sticks - the teacher's|^2, and the actor
    frozen for the first `warmup` updates (the critic learns alone)."""

    teacher = None
    stay = 0.0
    warmup = 0
    bc_X = bc_A = None  # the planner's labelled states (DAgger data), if any
    bc = 0.0
    bc_batch = 1024

    def train(self) -> None:  # SB3 2.x PPO.train, plus the two additions
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions, stay_losses = [], [], [], [], []
        bc_losses = []
        frozen = self._n_updates < self.warmup * self.n_epochs
        for _epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())
                if frozen:
                    loss = self.vf_coef * value_loss
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    if self.bc_X is not None and self.bc:
                        # the planner's own sticks on states from the DAgger rounds (DAPG-style)
                        j = th.randint(len(self.bc_X), (self.bc_batch,))
                        pol = self.policy
                        mean = pol.action_net(pol.mlp_extractor.policy_net(self.bc_X[j]))
                        bc = stick_loss(mean, self.bc_A[j]).mean()
                        bc_losses.append(bc.item())
                        loss = loss + self.bc * bc
                    if self.teacher is not None and self.stay:
                        obs = rollout_data.observations
                        pol = self.policy
                        mean = pol.action_net(pol.mlp_extractor.policy_net(obs))
                        with th.no_grad():
                            target = self.teacher(obs)
                        stay = ((mean - target) ** 2).sum(1).mean()
                        stay_losses.append(stay.item())
                        loss = loss + self.stay * stay
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_divs.append(
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                self.policy.optimizer.zero_grad()
                loss.backward()
                if frozen:  # only the critic moves
                    for p in self._actor_params():
                        p.grad = None
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
        ev = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/stay", np.mean(stay_losses) if stay_losses else 0.0)
        self.logger.record("train/explained_variance", ev)
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.last_stats = {
            "value_loss": float(np.mean(value_losses)),
            "stay": float(np.mean(stay_losses)) if stay_losses else 0.0,
            "bc": float(np.mean(bc_losses)) if bc_losses else 0.0,
            "ev": float(ev),
            "std": float(th.exp(self.policy.log_std).mean().item()),
            "frozen": bool(frozen),
        }

    def _actor_params(self):
        pol = self.policy
        return (
            list(pol.mlp_extractor.policy_net.parameters())
            + list(pol.action_net.parameters())
            + [pol.log_std]
        )


class Teacher(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.net = copy.deepcopy(policy.mlp_extractor.policy_net)
        self.head = copy.deepcopy(policy.action_net)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, obs):
        return self.head(self.net(obs))


class Log(BaseCallback):
    """Prints training flights' outcomes, saves checkpoints and tests them on the 200 test
    courses every `every` steps."""

    def __init__(self, out: Path, every: int, t0: float, minutes: float, before=None):
        super().__init__()
        self.out, self.every, self.t0, self.minutes = out, every, t0, minutes
        self.flights: list = []
        self.points: list = list(before or [])  # a resumed run's earlier points
        self.offset = self.points[-1]["minutes"] if self.points else 0.0
        self.next = self.points[-1]["steps"] + every if self.points else 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                self.flights.append(info["outcome"])
        if self.num_timesteps >= self.next:
            self.save()
            self.next += self.every
        return time.time() - self.t0 < self.minutes * 60

    def save(self) -> None:
        steps = self.num_timesteps
        self.model.save(self.out / f"ckpt_{steps:09d}")
        ev = test(actor_layers(self.model))["gauntlet"]
        recent = self.flights[-2000:]
        share = {k: round(recent.count(k) / max(1, len(recent)), 3) for k in set(recent)}
        st = getattr(self.model, "last_stats", {})
        mins = self.offset + (time.time() - self.t0) / 60
        self.points.append(
            {"steps": steps, "minutes": round(mins, 1), "test": ev, "train": share, **st}
        )
        print(
            f"{steps / 1e6:7.2f}M {mins:6.1f} min  test landed {ev['landed']}/200 "
            f"{ev['outcomes']}  train {share}  {st}",
            flush=True,
        )
        (self.out / "points.json").write_text(json.dumps(self.points))
        if getattr(self, "levels", None) is not None:
            (self.out / "levels.json").write_text(json.dumps(self.levels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="a DAgger round (runs/<name>/round_NN)")
    ap.add_argument("--resume", help="continue a fine-tune (runs/<name>): its last checkpoint, "
                    "noise and dial levels; --minutes and --steps are then extra")  # fmt: skip
    ap.add_argument("--name")
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--minutes", type=float, default=240.0)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--std", type=float, default=0.15, help="starting exploration noise")
    ap.add_argument("--stay", type=float, default=1.0, help="cost of moving off the teacher")
    ap.add_argument("--warmup", type=int, default=30, help="updates with the actor frozen")
    ap.add_argument("--dodge", type=float, default=0.0, help="bonus per threat beaten")
    ap.add_argument("--bc", type=float, default=0.0, help="weight of the planner's labels")
    ap.add_argument("--bc-data", default=None, help="a DAgger run whose data_*.npz to use")
    ap.add_argument("--time-cost", type=float, default=0.01)
    ap.add_argument("--landed", type=float, default=20.0)
    ap.add_argument("--every", type=int, default=5_000_000, help="test every this many steps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr-end", type=float, default=None, help="decay linearly to this")
    ap.add_argument("--school", action="store_true",
                    help="post 14's recipe: the drill mix and the difficulty dials")  # fmt: skip
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    before, levels = None, None
    if args.resume:
        run = RUNS / args.resume
        old = json.loads((run / "config.json").read_text())
        args.name, args.school, args.dodge = args.resume, old["school"], old["dodge"]
        args.init = str(sorted(run.glob("ckpt_*.zip"))[-1].relative_to(RUNS))[:-4]
        before = json.loads((run / "points.json").read_text())
        if (run / "levels.json").exists():
            levels = json.loads((run / "levels.json").read_text())
        args.warmup = 0
    assert args.init and args.name, "--init and --name, or --resume"

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    th.set_num_threads(args.threads)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    conf = "config.json" if not args.resume else f"config-resume{len(before)}.json"
    (out / conf).write_text(json.dumps(vars(args)))
    cfg = Config(
        Physics(), ROUGH, FEAT, Reward(landed=args.landed, time=args.time_cost)
    )  # progress 0.3/m, failing -10: post 14's
    mix = None
    if args.school:  # post 14's final recipe (train.py --task drills-predict --dials)
        from train import MIXES, TASKS

        cfg = replace(cfg, features=TASKS["drills-predict"])
        mix = MIXES["drills-predict"]
    env = DodgeEnv(args.envs, cfg, seed=args.seed, mix=mix)
    env.dodge = args.dodge
    model = TaughtPPO.load(
        RUNS / args.init,
        env=env,
        device="cpu",
        n_steps=args.n_steps,
        batch_size=args.batch,
        n_epochs=args.epochs,
        learning_rate=args.lr
        if args.lr_end is None
        else (lambda left: args.lr_end + (args.lr - args.lr_end) * left),
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=0.0,
        seed=args.seed,
        verbose=0,
    )
    if not args.resume:
        with th.no_grad():
            model.policy.log_std.fill_(float(np.log(args.std)))
    model.teacher = Teacher(model.policy)
    if args.bc_data:
        X, A = [], []
        for p in sorted((RUNS / args.bc_data).glob("data_*.npz")):
            z = np.load(p)
            X.append(z["obs"])
            A.append(z["act"])
        model.bc_X = th.from_numpy(np.concatenate(X).astype(np.float32))
        model.bc_A = th.from_numpy(np.concatenate(A))
        model.bc = args.bc
        print(f"planner labels: {len(model.bc_X):,} states from {args.bc_data}")
    model.stay = args.stay
    model.warmup = args.warmup
    t0 = time.time()
    cb = Log(out, args.every, t0, args.minutes, before)
    callbacks = [cb]
    if args.school:
        from types import SimpleNamespace

        from train import Dials

        cb.levels = list(levels or [])
        dials = Dials(env, SimpleNamespace(levels=cb.levels))
        if levels:  # pick up where the run left its dials
            dials.level = dict(levels[-1][1])
            dials._apply()
            del cb.levels[len(levels) :]  # the two _apply records: nothing changed
        callbacks.append(dials)
    model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=not args.resume)
    cb.save()
    print(f"done in {(time.time() - t0) / 60:.1f} min, {model.num_timesteps:,} steps")


if __name__ == "__main__":
    main()
