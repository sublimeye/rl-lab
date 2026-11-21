"""Post 10's drone, many at once: every copy's state lives in numpy arrays and one call steps
them all. With every feature off it is the same simulator as experiments/010-drone/drone_env.py
(same physics, same courses, same numbers; test_fleet_env.py checks it), only vectorised.

New in post 11, each switchable (Features):

- a moving landing pad (slides left and right);
- birds crossing the course, slow and fast; touching one is a crash;
- a battery: charge drains with throttle; full thrust sags a little as it empties, and only
  near empty enough to notice (like a real LiPo pack);
- sensor noise and a 50 ms delay on everything the pilot sees;
- a short memory: the pilot sees its last 3 snapshots;
- two ways of seeing (obs): "state" (exact positions of pads and the two nearest birds, like a
  motion-capture room) or "vision" (its own motion sensors, a noisy GPS-style fix on the pad's
  beacon, and a 1D depth camera: 32 rays fanned all round except a 60-degree cone straight up,
  each giving distance and whether it hit the marked pad or a bird).

Plain numpy on purpose: CI runs the tests without torch, and site/src/lib/drone2-sim.js repeats
the physics for one drone (test/drone2-sim.test.mjs checks it against this file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np

G = 9.81
M32 = 0xFFFFFFFF
FEET = ((-0.22, -0.18), (0.22, -0.18))
TIPS = ((-0.35, 0.04), (0.35, 0.04))
POINTS = FEET + TIPS
LEG = 0.18
FOOT_BAND = 0.15
K_BIRDS = 3  # bird slots per course
BIRD_HIT = 0.55  # drone centre within this of a bird's centre: a collision
N_RAYS = 32
RAY_SPAN = math.radians(150)  # from straight down (in the drone's frame) to each side
RAY_RANGE = 15.0

OUTCOMES = ("flying", "landed", "crashed", "water", "lost", "timeout", "bird")
FLYING, LANDED, CRASHED, WATER, LOST, TIMEOUT, BIRD = range(7)
FAILED = (CRASHED, WATER, LOST, BIRD)


@dataclass(frozen=True)
class Physics:
    dt: float = 1 / 60
    substeps: int = 3
    t_max: float = 20.0
    thrust_max: float = 2.0
    motor_tau: float = 0.08
    tilt_max: float = 0.5236
    kp: float = 70.0
    kd: float = 14.0
    alpha_max: float = 40.0
    drag_lin: float = 0.15
    drag_quad: float = 0.04
    land_vy: float = 1.5
    land_vx: float = 1.5
    land_tilt: float = 0.35
    width: float = 24.0
    ceiling: float = 13.0
    stick_rate: float = 0.0


@dataclass(frozen=True)
class Weather:
    wind: float = 0.0
    gust: float = 0.0
    heavy: float = 0.0


CALM = Weather()
ROUGH = Weather(wind=1.5, gust=1.0, heavy=0.35)


@dataclass(frozen=True)
class Features:
    """What's switched on. All off reproduces post 10 exactly."""

    pad_move: float = 0.0  # chance a course's pad moves (amplitude up to 2.5 m)
    birds: bool = False  # 0-3 birds per course
    min_birds: int = 0  # at least this many (training curriculum)
    battery: bool = False
    noise: float = 0.0  # sensor noise scale (1 = the post's standard amount)
    delay: int = 0  # updates of sensor delay (1 = 50 ms)
    stack: int = 1  # snapshots the pilot sees (current + previous)
    obs: str = "v010"  # "v010" (post 10's 14 numbers), "state" or "vision"


V010 = Features()
FULL = Features(pad_move=0.7, birds=True, battery=True, noise=1.0, delay=1, stack=3, obs="state")
FULL_VISION = replace(FULL, obs="vision")


def mulberry32(seed: int):
    a = seed & M32

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


def make_course(seed: int, weather: Weather, feat: Features, ph: Physics) -> dict:
    """One course as plain floats. The first draws are post 10's, in the same order, so a seed
    gives post 10's pillars and weather; the new things are drawn after them."""
    rand = mulberry32(seed)

    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * rand()

    sx = u(2.0, 5.0)
    sh = u(0.5, 4.0)
    tw = u(1.6, 2.6)
    tx = u(sx + 8.0, ph.width - 2.5)
    th = u(0.5, 7.0)
    if rand() < 0.5:
        sx, tx = ph.width - sx, ph.width - tx
    wind = u(-1.0, 1.0) * weather.wind
    gust = u(0.0, 1.0) * weather.gust
    period = u(2.0, 5.0)
    phase = u(0.0, 2 * math.pi)
    mass = 1.0 + u(0.0, 1.0) * weather.heavy
    # post 11
    moves = rand() < feat.pad_move
    amp = u(0.8, 2.5)
    room = min(tx - tw / 2 - 1.0, ph.width - 1.0 - (tx + tw / 2), abs(tx - sx) - 1.0 - tw / 2 - 1.5)
    amp = max(0.0, min(amp, room)) if moves else 0.0
    pad_period = u(4.0, 9.0)
    pad_phase = u(0.0, 2 * math.pi)
    lo = feat.min_birds
    n_birds = lo + int(rand() * (K_BIRDS + 1 - lo)) if feat.birds else 0
    birds = []
    for k in range(K_BIRDS):
        b_y = u(2.0, 11.0)
        fast = rand() < 0.5
        speed = u(4.0, 7.0) if fast else u(1.0, 3.0)
        direction = 1.0 if rand() < 0.5 else -1.0
        b_x0 = u(0.0, ph.width + 4.0)
        bob = u(0.0, 0.5)
        bob_period = u(1.5, 3.0)
        bob_phase = u(0.0, 2 * math.pi)
        # don't start a bird on top of the drone
        x_now = -2.0 + b_x0 % (ph.width + 4.0)
        if abs(x_now - sx) < 3.0 and abs(b_y - (sh + LEG)) < 3.0:
            b_x0 += (ph.width + 4.0) / 2
        if k < n_birds:
            birds.append((b_y, speed * direction, b_x0, bob, bob_period, bob_phase))
    charge = u(0.06, 1.0) if feat.battery else 1.0
    return {
        "start": (sx - 1.0, sx + 1.0, sh),
        "target": (tx - tw / 2, tx + tw / 2, th),
        "wind": wind,
        "gust": gust,
        "period": period,
        "phase": phase,
        "mass": mass,
        "amp": amp,
        "pad_period": pad_period,
        "pad_phase": pad_phase,
        "birds": birds,
        "charge": charge,
    }


def thrust_scale(charge):
    """Full thrust vs a full pack: a slow 3% sag over the whole pack, and a knee near empty."""
    return 1.0 - 0.03 * (1.0 - charge) - 0.2 * np.exp(-charge / 0.06)


# State: one float64 array per field, one entry per drone.
FIELDS = (
    "x y vx vy th om motor stick0 stick1 charge "
    "s_x0 s_x1 s_top t_x0 t_x1 t_top wind gust period phase mass amp pad_period pad_phase "
    "pad_x0 pad_x1 pad_vx"
).split()
INTS = ("ticks", "outcome", "crash", "grounded", "ground_pad")  # crash: 0 none, 1 hard, 2 side


class Fleet:
    """N drones stepped together. Courses come from seeds (set_course)."""

    def __init__(self, n: int, ph: Physics | None = None, feat: Features = V010):
        self.n = n
        self.ph = ph or Physics()
        self.feat = feat
        for f in FIELDS:
            setattr(self, f, np.zeros(n))
        for f in INTS:
            setattr(self, f, np.zeros(n, np.int64))
        self.bird = np.zeros((n, K_BIRDS, 6))  # y, vx, x0, bob, bob_period, bob_phase
        self.bird[..., 4] = 1.0
        self.bird_on = np.zeros((n, K_BIRDS), bool)
        self.touch = np.zeros((n, 2))
        self.courses: list[dict | None] = [None] * n

    # --- courses -------------------------------------------------------------------------
    def set_course(self, i: int, c: dict) -> None:
        self.courses[i] = c
        (self.s_x0[i], self.s_x1[i], self.s_top[i]) = c["start"]
        (self.t_x0[i], self.t_x1[i], self.t_top[i]) = c["target"]
        for k in ("wind", "gust", "period", "phase", "mass", "amp", "pad_period", "pad_phase"):
            getattr(self, k)[i] = c[k]
        self.bird[i] = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # empty slot (bob period 1: no 0/0)
        self.bird_on[i] = False
        for k, b in enumerate(c["birds"]):
            self.bird[i, k] = b
            self.bird_on[i, k] = True
        self.x[i] = (self.s_x0[i] + self.s_x1[i]) / 2
        self.y[i] = self.s_top[i] + LEG
        for f in ("vx", "vy", "th", "om", "motor", "stick1"):
            getattr(self, f)[i] = 0.0
        self.stick0[i] = -1.0
        self.charge[i] = c["charge"]
        self.ticks[i] = 0
        self.outcome[i] = FLYING
        self.crash[i] = 0
        self.grounded[i] = 1
        self.ground_pad[i] = 0
        self.touch[i] = 0.0
        self._pad(np.array([i]))

    def _pad(self, idx: np.ndarray) -> None:
        """The target deck's edges and speed at the current tick."""
        t = self.ticks[idx] * self.ph.dt
        w = 2 * math.pi / self.pad_period[idx]
        off = self.amp[idx] * np.sin(w * t + self.pad_phase[idx])
        self.pad_x0[idx] = self.t_x0[idx] + off
        self.pad_x1[idx] = self.t_x1[idx] + off
        self.pad_vx[idx] = self.amp[idx] * w * np.cos(w * t + self.pad_phase[idx])

    def bird_pos(self, idx: np.ndarray | slice = slice(None)):
        """Birds' (x, y, vx, vy) at the current tick: they fly straight across, bobbing a
        little, and come back in from the other side."""
        b = self.bird[idx]
        t = (self.ticks[idx] * self.ph.dt)[:, None]
        span = self.ph.width + 4.0
        x = -2.0 + np.mod(b[..., 2] + b[..., 1] * t, span)
        wb = 2 * math.pi / b[..., 4]
        y = b[..., 0] + b[..., 3] * np.sin(wb * t + b[..., 5])
        vy = b[..., 3] * wb * np.cos(wb * t + b[..., 5])
        return x, y, b[..., 1] + 0.0 * x, vy

    # --- physics ---------------------------------------------------------------------------
    def _points(self, idx):
        c, s = np.cos(self.th[idx]), np.sin(self.th[idx])
        x, y = self.x[idx], self.y[idx]
        return [(x + lx * c + ly * s, y - lx * s + ly * c) for lx, ly in POINTS]

    def tick(self, idx: np.ndarray, throttle: np.ndarray, tilt: np.ndarray) -> None:
        """One physics step (1/60 s) for drones idx (all still flying)."""
        ph = self.ph
        dt = ph.dt
        cmd = (throttle + 1.0) / 2.0
        if self.feat.battery:
            cmd = np.where(self.charge[idx] > 0.0, cmd, 0.0)
        self.motor[idx] += (cmd - self.motor[idx]) * dt / ph.motor_tau
        acc = self.motor[idx] * ph.thrust_max * G
        if self.feat.battery:
            acc = acc * thrust_scale(self.charge[idx])
            self.charge[idx] = np.maximum(
                0.0, self.charge[idx] - 0.0085 * self.motor[idx] ** 1.5 * dt
            )
        acc = acc / self.mass[idx]
        sit = (self.grounded[idx] == 1) & (acc <= G)
        if sit.any():  # still sitting on a pad: friction slows any slide, a moving deck carries it
            j = idx[sit]
            self.vx[j] *= 0.85
            self.x[j] += self.vx[j] * dt
            old = self.pad_x0[j].copy()
            self.ticks[j] += 1
            self._pad(j)
            on_target = self.ground_pad[j] == 1
            self.x[j] += np.where(on_target, self.pad_x0[j] - old, 0.0)
            self._birds_hit(j)
            self._limits(j[self.outcome[j] == FLYING])
        fly = ~sit
        if not fly.any():
            return
        idx, acc = idx[fly], acc[fly]
        throttle, tilt = throttle[fly], tilt[fly]
        self.grounded[idx] = 0
        before = self._points(idx)
        alpha = ph.kp * (tilt * ph.tilt_max - self.th[idx]) - ph.kd * self.om[idx]
        alpha = np.clip(alpha, -ph.alpha_max, ph.alpha_max)
        alpha *= np.minimum(1.0, 0.3 + 1.4 * self.motor[idx])
        self.om[idx] += alpha * dt
        self.th[idx] += self.om[idx] * dt
        vx, vy = self.vx[idx], self.vy[idx]
        speed = np.sqrt(vx * vx + vy * vy)
        t = self.ticks[idx] * dt
        air = self.wind[idx] + self.gust[idx] * np.sin(
            2 * math.pi * t / self.period[idx] + self.phase[idx]
        )
        th = self.th[idx]
        ax = acc * np.sin(th) - ph.drag_lin * vx - ph.drag_quad * vx * speed
        ay = acc * np.cos(th) - G - ph.drag_lin * vy - ph.drag_quad * vy * speed
        ax += air / self.mass[idx]
        self.vx[idx] += ax * dt
        self.vy[idx] += ay * dt
        self.x[idx] += self.vx[idx] * dt
        self.y[idx] += self.vy[idx] * dt
        self.ticks[idx] += 1
        self._pad(idx)
        self._contacts(idx, before)
        live = idx[self.outcome[idx] == FLYING]
        self._birds_hit(live)
        self._limits(live[self.outcome[live] == FLYING])

    def _contacts(self, idx, before) -> None:
        ph = self.ph
        pts = self._points(idx)
        decided = np.zeros(len(idx), bool)
        landing = np.full(len(idx), -1)  # pad 0 (start) or 1 (target)
        pads = (
            (self.s_x0[idx], self.s_x1[idx], self.s_top[idx]),
            (self.pad_x0[idx], self.pad_x1[idx], self.t_top[idx]),
        )
        vy = self.vy[idx]
        for k, (px, py) in enumerate(pts):
            for p, (x0, x1, top) in enumerate(pads):
                inside = ~decided & (x0 < px) & (px < x1) & (py < top)
                if k < 2:
                    from_above = before[k][1] >= top - FOOT_BAND
                    landing = np.where(inside & from_above & (vy <= 0.0), p, landing)
                    side = inside & ~from_above
                else:
                    side = inside
                if side.any():
                    j = idx[side]
                    self.crash[j] = 2
                    self._end(j, CRASHED)
                    decided |= side
            wet = ~decided & (py < 0.0)
            if wet.any():
                self._end(idx[wet], WATER)
                decided |= wet
        down = ~decided & (landing >= 0)
        if not down.any():
            return
        j = idx[down]
        lp = landing[down]
        rel_vx = self.vx[j] - np.where(lp == 1, self.pad_vx[j], 0.0)
        gentle = (
            (self.vy[j] >= -ph.land_vy)
            & (np.abs(rel_vx) <= ph.land_vx)
            & (np.abs(self.th[j]) <= ph.land_tilt)
        )
        hard = j[~gentle]
        self.crash[hard] = 1
        self._end(hard, CRASHED)
        ok, lp = j[gentle], lp[gentle]
        over = (lp == 1) & (self.pad_x0[ok] <= self.x[ok]) & (self.x[ok] <= self.pad_x1[ok])
        self._end(ok[over], LANDED)
        rest, lp = ok[~over], lp[~over]
        self.grounded[rest] = 1
        self.ground_pad[rest] = lp
        self.th[rest] = 0.0
        self.om[rest] = 0.0
        self.y[rest] = np.where(lp == 1, self.t_top[rest], self.s_top[rest]) + LEG
        self.vy[rest] = 0.0
        self.vx[rest] *= 0.85

    def _birds_hit(self, idx) -> None:
        if not self.feat.birds or len(idx) == 0:
            return
        bx, by, _, _ = self.bird_pos(idx)
        dx = bx - self.x[idx][:, None]
        dy = by - self.y[idx][:, None]
        hit = (self.bird_on[idx] & (dx * dx + dy * dy < BIRD_HIT * BIRD_HIT)).any(1)
        self._end(idx[hit], BIRD)

    def _limits(self, idx) -> None:
        ph = self.ph
        lost = (self.x[idx] < -0.5) | (self.x[idx] > ph.width + 0.5) | (self.y[idx] > ph.ceiling)
        self.outcome[idx[lost]] = LOST
        rest = idx[~lost]
        self.outcome[rest[self.ticks[rest] >= round(ph.t_max / ph.dt)]] = TIMEOUT

    def _end(self, idx, outcome) -> None:
        self.touch[idx, 0] = self.vx[idx]
        self.touch[idx, 1] = self.vy[idx]
        self.outcome[idx] = outcome

    def step(self, actions: np.ndarray) -> None:
        """One stick update for every drone still flying."""
        a = np.clip(np.asarray(actions, float), -1.0, 1.0)
        thr, tilt = a[:, 0].copy(), a[:, 1].copy()
        r = self.ph.stick_rate
        if r > 0:
            thr = self.stick0 + np.clip(thr - self.stick0, -r, r)
            tilt = self.stick1 + np.clip(tilt - self.stick1, -r, r)
        start = np.nonzero(self.outcome == FLYING)[0]
        live = start
        for _ in range(self.ph.substeps):
            live = live[self.outcome[live] == FLYING]
            if len(live) == 0:
                break
            self.tick(live, thr[live], tilt[live])
        self.stick0[start] = thr[start]
        self.stick1[start] = tilt[start]

    # --- helpers ---------------------------------------------------------------------------
    def goal(self):
        return (self.pad_x0 + self.pad_x1) / 2, self.t_top + LEG

    def distance(self):
        gx, gy = self.goal()
        dx, dy = gx - self.x, gy - self.y
        return np.sqrt(dx * dx + dy * dy)

    def surface(self, x):
        on_s = (self.s_x0 < x) & (x < self.s_x1)
        on_t = (self.pad_x0 < x) & (x < self.pad_x1)
        return np.where(on_s, self.s_top, np.where(on_t, self.t_top, 0.0))


# --- what the pilot sees ---------------------------------------------------------------------
def obs_v010(f: Fleet) -> np.ndarray:
    """Post 10's 14 numbers (exact, no delay)."""
    gx, gy = f.goal()
    under = f.y - LEG - f.surface(f.x)
    half = (f.t_x1 - f.t_x0) / 2
    s_cx = (f.s_x0 + f.s_x1) / 2
    return np.stack(
        [
            (gx - f.x) / 10,
            (gy - f.y) / 10,
            f.vx / 5,
            f.vy / 5,
            f.th / f.ph.tilt_max,
            f.om / 5,
            f.motor * 2 - 1,
            half / 1.3,
            (s_cx - f.x) / 10,
            (f.s_top + LEG - f.y) / 10,
            f.y / 10,
            np.clip(under / 5, -1.0, 2.0),
            f.stick0,
            f.stick1,
        ],
        1,
    )


N_STATE = 14 + 2 + 2 * 5
N_VISION = 11 + 3 * N_RAYS


def nearest_birds(f: Fleet, n: int = 2) -> np.ndarray:
    """(N, n, 5): dx, dy, vx, vy of the n nearest birds and 1 (or far away and 0 if missing)."""
    out = np.zeros((f.n, n, 5))
    out[:, :, 0] = 2.0  # "none": 20 m away
    if not f.feat.birds:
        return out
    bx, by, bvx, bvy = f.bird_pos()
    dx, dy = bx - f.x[:, None], by - f.y[:, None]
    d2 = np.where(f.bird_on, dx * dx + dy * dy, np.inf)
    order = np.argsort(d2, axis=1, kind="stable")[:, :n]
    rows = np.arange(f.n)[:, None]
    have = np.isfinite(d2[rows, order])
    out[:, :, 0] = np.where(have, dx[rows, order] / 10, 2.0)
    out[:, :, 1] = np.where(have, dy[rows, order] / 10, 0.0)
    out[:, :, 2] = np.where(have, bvx[rows, order] / 5, 0.0)
    out[:, :, 3] = np.where(have, bvy[rows, order] / 5, 0.0)
    out[:, :, 4] = have
    return out


def obs_state(f: Fleet, noise: np.ndarray | None) -> np.ndarray:
    o = np.concatenate(
        [
            obs_v010(f),
            (f.pad_vx / 3)[:, None],
            (f.charge * 2 - 1)[:, None],
            nearest_birds(f).reshape(f.n, -1),
        ],
        1,
    )
    if noise is not None:
        o = o + noise * STATE_SIGMA
    return o


# Noise per input at noise scale 1 (in the input's own units, after scaling).
STATE_SIGMA = np.array(
    [0.005, 0.005, 0.02, 0.02, 0.04, 0.02, 0.02, 0, 0.005, 0.005, 0.005, 0.01, 0, 0, 0.02, 0.01]
    + [0.005, 0.005, 0.02, 0.02, 0] * 2
)


def rays(f: Fleet):
    """The depth camera: N_RAYS rays from the drone's centre, fixed to its body, fanned from
    RAY_SPAN to the left of straight down to RAY_SPAN to the right. Returns (depth 0..1, hit
    the marked pad, hit a bird), each (N, N_RAYS)."""
    a = np.linspace(-RAY_SPAN, RAY_SPAN, N_RAYS)[None, :] - f.th[:, None]
    dx, dy = np.sin(a), -np.cos(a)
    ox, oy = f.x[:, None], f.y[:, None]
    best = np.full(dx.shape, RAY_RANGE)
    kind = np.zeros(dx.shape, np.int64)  # 0 nothing/other, 1 marked pad, 2 bird
    with np.errstate(divide="ignore", invalid="ignore"):
        t_w = np.where(dy < 0, -oy / dy, np.inf)  # the water
        best = np.minimum(best, np.where(t_w > 0, t_w, np.inf))
        for x0, x1, top, k in (
            (f.s_x0, f.s_x1, f.s_top, 0),
            (f.pad_x0, f.pad_x1, f.t_top, 1),
        ):
            x0, x1, top = x0[:, None], x1[:, None], top[:, None]
            tx1, tx2 = (x0 - ox) / dx, (x1 - ox) / dx
            ty1, ty2 = (-1.0 - oy) / dy, (top - oy) / dy
            t_in = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
            t_out = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
            hit = (t_in <= t_out) & (t_in > 0) & (t_in < best)
            best = np.where(hit, t_in, best)
            kind = np.where(hit, k, kind)
        if f.feat.birds:
            bx, by, _, _ = f.bird_pos()
            r = 0.3
            for k in range(K_BIRDS):
                cx, cy = (bx[:, k] - f.x)[:, None], (by[:, k] - f.y)[:, None]
                b = dx * cx + dy * cy
                c = cx * cx + cy * cy - r * r
                disc = b * b - c
                t = b - np.sqrt(np.maximum(disc, 0.0))
                hit = f.bird_on[:, k : k + 1] & (disc >= 0) & (t > 0) & (t < best)
                best = np.where(hit, t, best)
                kind = np.where(hit, 2, kind)
    return best / RAY_RANGE, (kind == 1) * 1.0, (kind == 2) * 1.0


def obs_vision(f: Fleet, noise: np.ndarray | None, gps_err: np.ndarray) -> np.ndarray:
    gx, gy = f.goal()
    depth, pad, bird = rays(f)
    own = np.stack(
        [
            f.vx / 5,
            f.vy / 5,
            f.th / f.ph.tilt_max,
            f.om / 5,
            f.motor * 2 - 1,
            f.charge * 2 - 1,
            f.y / 10,
            f.stick0,
            f.stick1,
            (gx + gps_err[:, 0] - f.x) / 10,
            (gy + gps_err[:, 1] - f.y) / 10,
        ],
        1,
    )
    if noise is not None:
        own = own + noise[:, :11] * VISION_SIGMA
        depth = np.clip(depth + noise[:, 11 : 11 + N_RAYS] * 0.01, 0.0, 1.0)
    return np.concatenate([own, depth, pad, bird], 1)


VISION_SIGMA = np.array([0.02, 0.02, 0.04, 0.02, 0.02, 0.01, 0.005, 0, 0, 0.005, 0.005])


# --- reward ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reward:
    progress: float = 1.0
    landed: float = 10.0
    failed: float = -5.0
    smooth: float = 0.0
    bird: float | None = None  # a bird hit's penalty (None: same as any failure)
    near: float = 0.0  # per update, per bird within 2 m: near * (1 - distance / 2)


def rewards(f: Fleet, before: np.ndarray, sticks_before: np.ndarray, rw: Reward, was_flying):
    r = rw.progress * (before - f.distance())
    r = r + np.where(f.outcome == LANDED, rw.landed, 0.0)
    r = r + np.where(np.isin(f.outcome, FAILED), rw.failed, 0.0)
    if rw.bird is not None:
        r = r + np.where(f.outcome == BIRD, rw.bird - rw.failed, 0.0)
    if rw.near and f.feat.birds:
        bx, by, _, _ = f.bird_pos()
        dist = np.sqrt((bx - f.x[:, None]) ** 2 + (by - f.y[:, None]) ** 2)
        r = r - rw.near * (np.clip(1.0 - dist / 2.0, 0.0, 1.0) * f.bird_on).sum(1)
    if rw.smooth:
        d0 = f.stick0 - sticks_before[:, 0]
        d1 = f.stick1 - sticks_before[:, 1]
        r = r - rw.smooth * (d0 * d0 + d1 * d1)
    return np.where(was_flying, r, 0.0)


# --- Stable-Baselines3 vectorised env ----------------------------------------------------------
@dataclass(frozen=True)
class Config:
    physics: Physics = Physics()
    weather: Weather = CALM
    features: Features = V010
    reward: Reward = Reward()


try:  # SB3 is only installed for training; the tests and CI use Fleet directly
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:  # pragma: no cover
    VecEnv = object


class FleetEnv(VecEnv):
    """N drones as one Stable-Baselines3 VecEnv: a finished flight is replaced by a new course
    in the same slot at once (with the usual terminal_observation in its info)."""

    def __init__(self, n: int, config: Config | None = None, seed: int = 0):
        self.config = cfg = config or Config()
        self.fleet = Fleet(n, cfg.physics, cfg.features)
        self.rng = np.random.default_rng(seed)
        feat = cfg.features
        self.width = {"v010": 14, "state": N_STATE, "vision": N_VISION}[feat.obs]
        dim = self.width * feat.stack
        self.render_mode = None
        obs_space = gym.spaces.Box(-5.0, 5.0, (dim,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        if VecEnv is not object:
            super().__init__(n, obs_space, act_space)
        else:
            self.num_envs, self.observation_space, self.action_space = n, obs_space, act_space
        self.noise_scale = np.zeros(n)
        self.gps_err = np.zeros((n, 2))
        self.pending = [np.zeros((n, self.width)) for _ in range(feat.delay)]
        self.frames = np.zeros((n, feat.stack, self.width))
        self._actions = np.zeros((n, 2))
        self.auto_reset = True
        self.seeds = np.zeros(n, np.int64)

    # observation pipeline: sense -> delay -> stack
    def _sense(self) -> np.ndarray:
        f, feat = self.fleet, self.config.features
        noise = None
        if feat.noise:
            noise = (
                self.rng.standard_normal((f.n, self.width + N_RAYS))
                * (feat.noise * self.noise_scale)[:, None]
            )
        if feat.obs == "v010":
            return obs_v010(f)
        if feat.obs == "state":
            return obs_state(f, None if noise is None else noise[:, : self.width])
        return obs_vision(f, noise, self.gps_err)

    def _push(self) -> np.ndarray:
        """This update's reading goes through the delay line into the snapshot memory."""
        now = self._sense()
        if self.config.features.delay:
            self.pending.append(now)
            now = self.pending.pop(0)
        self.frames = np.concatenate([now[:, None, :], self.frames[:, :-1]], 1)
        return self._flat()

    def _fill(self, idx: np.ndarray) -> None:
        """Slots starting a course: delay line and memory hold the first reading."""
        now = self._sense()
        for p in self.pending:
            p[idx] = now[idx]
        self.frames[idx] = now[idx][:, None, :]

    def _flat(self) -> np.ndarray:
        return np.clip(self.frames.reshape(self.num_envs, -1), -5.0, 5.0).astype(np.float32)

    def _new_course(self, idx: np.ndarray, seeds=None) -> None:
        cfg = self.config
        for j, i in enumerate(idx):
            seed = int(self.rng.integers(0, 2**31)) if seeds is None else int(seeds[j])
            self.seeds[i] = seed
            self.fleet.set_course(int(i), make_course(seed, cfg.weather, cfg.features, cfg.physics))
        self.noise_scale[idx] = self.rng.uniform(0.5, 1.5, len(idx))
        self.gps_err[idx] = self.rng.normal(0.0, 0.3, (len(idx), 2)) * cfg.features.noise

    def reset(self):
        idx = np.arange(self.num_envs)
        self._new_course(idx)
        self._fill(idx)
        return self._flat()

    def step_async(self, actions):
        self._actions = np.asarray(actions, float)

    def step_wait(self):
        f, cfg = self.fleet, self.config
        before, sticks = f.distance(), np.stack([f.stick0, f.stick1], 1)
        flying = f.outcome == FLYING
        f.step(self._actions)
        r = rewards(f, before, sticks, cfg.reward, flying).astype(np.float32)
        done_idx = np.nonzero(f.outcome != FLYING)[0]
        dones = f.outcome != FLYING
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        obs = self._push()
        for i in done_idx:
            infos[i] = {
                "outcome": OUTCOMES[f.outcome[i]],
                "time": f.ticks[i] * f.ph.dt,
                "terminal_observation": obs[i].copy(),
                "TimeLimit.truncated": bool(f.outcome[i] == TIMEOUT),
            }
        if len(done_idx) and self.auto_reset:
            self._new_course(done_idx)
            self._fill(done_idx)
            obs[done_idx] = self._flat()[done_idx]
        return obs, r, dones, infos

    def start(self, seeds) -> np.ndarray:
        """Fixed courses, one per slot, no automatic restarts (for tests and evaluation)."""
        self.auto_reset = False
        idx = np.arange(self.num_envs)
        self._new_course(idx, seeds)
        self._fill(idx)
        return self._flat()

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        n = len(self._idx(indices))
        return [getattr(self, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * len(self._idx(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._idx(indices))

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
