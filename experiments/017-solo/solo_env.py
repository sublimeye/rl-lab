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
from collections import deque
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
    make_course,
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


class PracticeEnv(FleetEnv):
    """FleetEnv for practice: a share `replay` of the new courses are ones the pilot failed the
    last time it flew them (kept until it lands them), so what kills it comes up more often."""

    replay = 0.0
    keep = 20_000

    def _new_course(self, idx, seeds=None) -> None:
        if seeds is not None or not self.replay:
            super()._new_course(idx, seeds)
            return
        if not hasattr(self, "hard"):
            self.hard: list[tuple[int, int]] = []  # (seed, which of the mix), failed last time
            self.place: dict[tuple[int, int], int] = {}
        cfg, f = self.config, self.fleet
        for i in idx:
            key = (int(self.seeds[i]), int(self.drill[i]))
            if f.outcome[i] == LANDED:
                self._forget(key)
            elif f.outcome[i] != FLYING and key not in self.place:
                if len(self.hard) >= self.keep:
                    self._forget(self.hard[int(self.rng.integers(len(self.hard)))])
                self.place[key] = len(self.hard)
                self.hard.append(key)
            if self.hard and self.rng.random() < self.replay:
                seed, drill = self.hard[int(self.rng.integers(len(self.hard)))]
            else:
                seed, drill = int(self.rng.integers(0, 2**31)), 0
                if self.mix:
                    w = np.array([m[0] for m in self.mix])
                    drill = int(self.rng.choice(len(w), p=w / w.sum()))
            feat = self.mix[drill][1] if self.mix else cfg.features
            self.seeds[i], self.drill[i] = seed, drill
            f.set_course(int(i), make_course(seed, cfg.weather, feat, cfg.physics))
        self.noise_scale[idx] = self.rng.uniform(0.5, 1.5, len(idx))

    def _forget(self, key) -> None:
        j = self.place.pop(key, None)
        if j is None:
            return
        last = self.hard.pop()
        if j < len(self.hard):
            self.hard[j] = last
            self.place[last] = j


class SoloEnv:
    """n drones, each on a course of its own; a finished flight is replaced at once. step()
    takes a move per drone and returns (inputs, rewards, done, finished flights). `replay`:
    see PracticeEnv; `timeout`: what running out of time (60 s) pays (post 14's reward: 0);
    `again`: the share of new flights that begin 2 to 6 s before one of this fleet's recent
    failures (the same drone, course and moment, flown on from there) instead of at a start."""

    def __init__(self, n: int, features=FEAT, mix=None, seed: int = 0, hold: int = 2,
                 physics: Physics = PH45, reward: Reward = REWARD, weather=ROUGH,
                 replay: float = 0.0, timeout: float = 0.0, again: float = 0.0):  # fmt: skip
        self.env = PracticeEnv(n, Config(physics, weather, features, reward), seed=seed, mix=mix)
        self.env.replay = replay
        self.timeout = timeout
        self.again = again
        self._clock = 0  # updates so far
        self._born = np.zeros(n, np.int64)  # the update each slot's flight began at
        self._snaps: deque = deque(maxlen=6)  # (update, a copy of every drone), one a second
        self._pool: deque = deque(maxlen=1024)  # failures wound back: one-drone copies
        self._rng = np.random.default_rng(seed + 1)
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

    def inputs(self) -> np.ndarray:
        """The inputs as they stand (after start, step or fork)."""
        return self._inputs()

    def fork(self, i: int, k: int, seed: int = 0) -> SoloEnv:
        """k copies of drone i as it is now, mid-flight, each with sensor noise of its own from
        here on: rewinding a flight to see how else it could have gone (rewind.py)."""
        return self._copy(np.full(k, i), seed)

    def _copy(self, rows: np.ndarray, seed: int = 0) -> SoloEnv:
        cfg, s, k = self.env.config, self.env, len(rows)
        new = SoloEnv(k, cfg.features, None, seed, self.hold, cfg.physics, cfg.reward, cfg.weather)
        e = new.env
        e.fleet = s.fleet.take(rows)
        e.auto_reset = False
        e.noise_scale = s.noise_scale[rows].copy()
        e.pending = [p[rows].copy() for p in s.pending]
        e.frames, e.hist = s.frames[rows].copy(), s.hist[rows].copy()
        e.seeds, e.drill = s.seeds[rows].copy(), s.drill[rows].copy()
        for name in ("seen_n", "count", "s_first", "s_last", "first_seen"):
            setattr(new.mem, name, getattr(self.mem, name)[rows].copy())
        new.prev, new.obs = self.prev[rows].copy(), self.obs[rows].copy()
        return new

    def _adopt(self, i: int, rec: SoloEnv) -> None:
        """Slot i becomes the one drone of rec, mid-flight (the inverse of fork)."""
        e, s = self.env, rec.env
        for key, v in vars(s.fleet).items():
            if isinstance(v, np.ndarray) and v.shape[:1] == (1,):
                getattr(e.fleet, key)[i] = v[0]
        e.fleet.courses[i] = s.fleet.courses[0]
        e.noise_scale[i] = s.noise_scale[0]
        for mine, theirs in zip(e.pending, s.pending, strict=True):
            mine[i] = theirs[0]
        e.frames[i], e.hist[i] = s.frames[0], s.hist[0]
        e.seeds[i], e.drill[i] = s.seeds[0], s.drill[0]
        for name in ("seen_n", "count", "s_first", "s_last", "first_seen"):
            getattr(self.mem, name)[i] = getattr(rec.mem, name)[0]
        self.obs[i] = rec.obs[0]

    def _restart(self, new: np.ndarray, infos: list, m: np.ndarray) -> None:
        """Flights that just ended in slots `new`: a failed one is kept as it was 2 to 6 s
        before its end, and each new flight may begin from one of those kept."""
        for i in new:
            if infos[i]["outcome"] != "landed":
                ok = [c for t, c in self._snaps if self._born[i] <= t <= self._clock - 40]
                if ok:
                    self._pool.append(ok[int(self._rng.integers(len(ok)))].fork(int(i), 1))
            self._born[i] = self._clock
            if self._pool and self._rng.random() < self.again:
                rec = self._pool[int(self._rng.integers(len(self._pool)))]
                self._adopt(int(i), rec)
                m[i] = rec.prev[0]

    def step(self, moves: np.ndarray):
        env, f = self.env, self.fleet
        m = np.asarray(moves).copy()
        total = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        ended: list[dict] = []
        for k in range(self.hold):
            if k:
                self.mem.observe(f)
            if self.again and self._clock % 20 == 0:
                self._snaps.append((self._clock, self._copy(np.arange(self.n))))
            # the sticks as float32, so the browser port can replay a recorded flight exactly
            a = expert.control(fp.nominal(f), m).astype(np.float32).astype(np.float64)
            if self.tape is not None:
                for i in np.nonzero(f.outcome == FLYING)[0]:
                    self.tape[i].append(a[i])
            self.obs, r, d, infos = env.step(a)
            self._clock += 1
            total += np.where(done, 0.0, r)  # a new course in this slot earns nothing yet
            new = np.nonzero(d & ~done)[0]
            for i in new:
                ended.append(infos[i])
                if infos[i]["outcome"] == "timeout":
                    total[i] += self.timeout
            if len(new) and env.auto_reset:
                self.mem.reset(new)
                m[new] = expert.DEFAULT  # the new course starts under the default cruise
                if self.again:
                    self._restart(new, infos, m)
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


__all__ = ["DIM", "FEAT", "LANDED", "NM", "OUTCOMES", "PH45", "REWARD", "FrogMemory",
           "PracticeEnv", "SoloEnv", "fly", "prev_inputs", "scripted"]  # fmt: skip
