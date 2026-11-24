"""The gauntlet flown in moves, with no planner anywhere (post 17).

Post 16's network picked one of the fair planner's 59 moves (a cruise height and speed, a jink,
a landing run, a wait beside the pad) and post 15's controller flew it. It was trained to copy
the planner's choice. Here the same pilot learns from its own flights instead (PPO), so this
file is post 14's FleetEnv with three changes:
- the action is a move, held for `hold` updates (0.05 s each), flown by expert.control;
- the pilot gets, besides post 14's 318 inputs, a memory of each frog's strikes (when the next
  two could come, as fair_planner.interval works it out from the strikes seen so far) and the
  move it made last. The same numbers post 16's network had, computed for all drones at once;
- nothing here knows the air: the controller assumes still air and the average mass (post 16's
  network was given the planner's estimate, which replays every update on a copy of the
  simulator; this pilot is to use no copies at all).
Moves that need the pad (land, wait) are only offered within 25 m of it, as in the planner.

Plain numpy: CI imports it without torch.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
for _p in ("014-gauntlet", "015-teacher", "016-flawless"):
    sys.path.insert(0, str(HERE.parent / _p))

import fair_planner as fp  # noqa: E402  (adds post 16's moves and its gentle touchdown)

import expert  # noqa: E402
from gauntlet_env import (  # noqa: E402
    FLYING,
    FROG_JITTER,
    FULL,
    K_FROGS,
    LANDED,
    OUTCOMES,
    ROUGH,
    SIGHT,
    Config,
    FleetEnv,
    Physics,
    Reward,
)

NM = len(expert.MOVES)  # 59
N_OBS = 318
N_FROG_MEM = 8
N_PREV = 7
DIM = N_OBS + K_FROGS * N_FROG_MEM + N_PREV
PH45 = replace(Physics(), tilt_max=float(np.radians(45.0)))
FEAT = replace(FULL, predict=True)
PAD_MOVES = np.isin(np.arange(NM), expert.PAD_MOVES)
PAD_NEAR = expert.PAD_NEAR
# post 14's reward: 0.3 a metre, +20 landed, -10 failed, 0.2 a second in the air
REWARD = Reward(progress=0.3, landed=20.0, failed=-10.0, time=0.01)


class FrogMemory:
    """What a pilot who remembers every strike it has seen knows about each frog's next one:
    fair_planner.FairPlanner.observe and .interval, for all drones at once."""

    def __init__(self, n: int, periods=(3.0, 4.0)):
        self.n = n
        self.p_lo, self.p_hi = periods
        self.seen_n = np.zeros((n, K_FROGS), np.int64)  # the simulator's strike count, last seen
        self.count = np.zeros((n, K_FROGS), np.int64)  # strikes seen (the frog in sight)
        self.s_first = np.zeros((n, K_FROGS))
        self.s_last = np.zeros((n, K_FROGS))
        self.first_seen = np.full((n, K_FROGS), np.nan)

    def reset(self, idx) -> None:
        self.seen_n[idx] = 0
        self.count[idx] = 0
        self.s_first[idx] = self.s_last[idx] = 0.0
        self.first_seen[idx] = np.nan

    def observe(self, f) -> None:
        t = (f.ticks * f.ph.dt)[:, None]
        px, py = f.perch()
        sight = f.frog_on & (np.hypot(px - f.x[:, None], py - f.y[:, None]) < SIGHT)
        new = (f.frog_n != self.seen_n) & sight
        self.count += new
        self.s_first = np.where(new & (self.count == 1), f.frog_s, self.s_first)
        self.s_last = np.where(new, f.frog_s, self.s_last)
        self.seen_n[:] = f.frog_n
        first = sight & np.isnan(self.first_seen)
        self.first_seen = np.where(first, t, self.first_seen)

    def interval(self, f):
        """(lo, hi, b_lo, b_hi), each (n, frogs): the next strike's start and the beat."""
        jit, sense = FROG_JITTER, fp.SENSE
        now = (f.ticks * f.ph.dt)[:, None]
        c = self.count
        many = np.maximum(c - 1, 1)
        est = (self.s_last - self.s_first) / many
        d = (2 * jit + 2 * sense) / many
        b1, b2 = np.maximum(self.p_lo, est - d), np.minimum(self.p_hi, est + d)
        b_lo = np.where(c >= 2, np.minimum(b1, b2), self.p_lo)
        b_hi = np.where(c >= 2, np.maximum(b1, b2), self.p_hi)
        lo = self.s_last - sense + b_lo - 2 * jit
        hi = self.s_last + sense + b_hi + 2 * jit
        start = np.where(np.isfinite(self.first_seen), self.first_seen, now)
        lo = np.where(c == 0, now, lo)
        hi = np.where(c == 0, np.maximum(now, start) + self.p_hi + 2 * jit, hi)
        lo = np.maximum(lo, now)
        return lo, np.maximum(hi, lo), b_lo, b_hi

    def inputs(self, f) -> np.ndarray:
        """(n, frogs x 8), as pilot.beliefs' frog columns."""
        now = (f.ticks * f.ph.dt)[:, None]
        lo, hi, b_lo, b_hi = self.interval(f)
        jit = 2 * FROG_JITTER
        dx = f.frog_p[..., 0] - f.x[:, None]
        on = f.frog_on & ~f.frog_past & np.isfinite(self.first_seen)
        win = np.stack([lo, hi + fp.LEAPING, lo + b_lo - jit, hi + b_hi + jit + fp.LEAPING], 2)
        out = np.zeros((f.n, K_FROGS, N_FROG_MEM))
        out[..., 0] = on
        out[..., 1] = np.where(on, np.clip(dx / 10, -5, 5), 5.0)
        out[..., 2:6] = np.where(on[..., None], np.clip((win - now[..., None]) / 3, 0, 2), 2.0)
        out[..., 6] = np.where(on, np.minimum(self.count, 5) / 5, 0.0)
        out[..., 7] = on & (np.abs(dx) < fp.ZONE)
        return out.reshape(f.n, -1)


def prev_inputs(prev: np.ndarray) -> np.ndarray:
    """(n, 7): the move made last (its kind, one-hot, and its two numbers)."""
    out = np.zeros((len(prev), N_PREV))
    out[np.arange(len(prev)), expert.MK[prev]] = 1.0
    out[:, 5] = expert.MA[prev] / 10
    out[:, 6] = expert.MB[prev] / 10
    return out


class SoloEnv:
    """n drones, each on a course of its own; a finished flight is replaced at once. step()
    takes a move per drone and returns (inputs, rewards, done, finished flights)."""

    def __init__(self, n: int, features=FEAT, mix=None, seed: int = 0, hold: int = 2,
                 physics: Physics = PH45, reward: Reward = REWARD, weather=ROUGH):  # fmt: skip
        self.env = FleetEnv(n, Config(physics, weather, features, reward), seed=seed, mix=mix)
        self.n, self.hold = n, hold
        self.mem = FrogMemory(n, features.frog_period)
        self.prev = np.full(n, expert.DEFAULT)
        self.obs = None
        self.tape: list[list] | None = None  # with start(record=True): each drone's sticks

    @property
    def fleet(self):
        return self.env.fleet

    def _inputs(self) -> np.ndarray:
        f = self.fleet
        self.mem.observe(f)
        return np.concatenate(
            [self.obs, self.mem.inputs(f), prev_inputs(self.prev)], 1, dtype=np.float32
        )

    def mask(self) -> np.ndarray:
        """(n, 59): the moves on offer (land and wait only near the pad)."""
        f = self.fleet
        near = np.abs(f.goal()[0] - f.x) < PAD_NEAR
        return near[:, None] | ~PAD_MOVES[None]

    def reset(self) -> np.ndarray:
        self.obs = self.env.reset()
        self.mem.reset(slice(None))
        self.prev[:] = expert.DEFAULT
        return self._inputs()

    def start(self, seeds, record: bool = False) -> np.ndarray:
        """Fixed courses, no restarts (tests). With record, the sticks flown are kept."""
        self.tape = [[] for _ in seeds] if record else None
        self.obs = self.env.start(seeds)
        self.mem.reset(slice(None))
        self.prev[:] = expert.DEFAULT
        return self._inputs()

    def step(self, moves: np.ndarray):
        env, f = self.env, self.fleet
        m = np.asarray(moves).copy()
        total = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        ended: list[dict] = []
        for k in range(self.hold):
            if k:
                self.mem.observe(f)
            # the sticks as float32, so the browser port can replay a recorded flight exactly
            a = expert.control(fp.nominal(f), m).astype(np.float32).astype(np.float64)
            if self.tape is not None:
                for i in np.nonzero(f.outcome == FLYING)[0]:
                    self.tape[i].append(a[i])
            self.obs, r, d, infos = env.step(a)
            total += np.where(done, 0.0, r)  # a new course in this slot earns nothing yet
            new = np.nonzero(d & ~done)[0]
            for i in new:
                ended.append(infos[i])
            if len(new) and env.auto_reset:
                self.mem.reset(new)
                m[new] = expert.DEFAULT  # the new course starts under the default cruise
            done |= d
            if not env.auto_reset and not (f.outcome == FLYING).any():
                break
        self.prev = m
        return self._inputs(), total, done, ended


def fly(policy, seeds, features=FEAT, hold: int = 2, physics: Physics = PH45, cap: int = 1300,
        record: bool = False):  # fmt: skip
    """`policy(inputs, mask) -> moves` flies fixed courses once: the outcome of each and the
    fleet (and, with record, each drone's sticks, update by update)."""
    env = SoloEnv(len(seeds), features, hold=hold, physics=physics, seed=7)
    x = env.start(seeds, record)
    f = env.fleet
    t = 0
    while (f.outcome == FLYING).any() and t < cap:
        x, _, _, _ = env.step(policy(x, env.mask()))
        t += hold
    out = [OUTCOMES[o] for o in f.outcome]
    return (out, f, env.tape) if record else (out, f)


def scripted(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A pilot that ignores every threat: cruise at 7 m and 10 m/s, land once near the pad."""
    return np.where(mask[:, expert.move_id(expert.LAND, 5.0)], expert.move_id(expert.LAND, 5.0),
                    expert.CANON)  # fmt: skip


__all__ = ["DIM", "FEAT", "LANDED", "NM", "PH45", "REWARD", "FrogMemory", "SoloEnv", "fly",
           "prev_inputs", "scripted"]  # fmt: skip
