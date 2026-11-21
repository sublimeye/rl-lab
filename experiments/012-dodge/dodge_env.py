"""Post 11's fleet of drones with things to dodge. Same physics, battery, sensor noise, delay,
moving pad and weather as experiments/011-drone2/fleet_env.py (test_dodge_env.py checks the
flying is identical with no threats); new in post 12:

- birds as in post 11, but they come back in from 6 m beyond the screen edge (post 11's came
  back 2 m out, so near an edge a bird could appear right next to the drone);
- falling rocks: each dropper lets a rock go from above the screen on a steady beat, aimed near
  wherever the drone is at that moment; rocks fall under gravity with a little drift;
- a missile: launched from one side at a set time, faster than the drone, steering by
  proportional navigation (turn rate = N x how fast the line of sight turns) with a turn-rate
  limit and 4 s of fuel;
- the dodge-school arena: no landing pad, hold a point in the air for 12 s under dense threats.

What the pilot sees (obs): "state" = exact positions and velocities of the 4 most urgent threats
(urgency: distance over closing speed), with their kind; "vision" = its own motion sensors, a
GPS-style fix on the goal, a missile-launch warning and a 1D depth camera whose 32 rays report
distance, pad, threat and shadow (a falling rock's shadow on the surface below it: the camera
has a blind cone straight up, as most real drones do).

Plain numpy on purpose: CI runs the tests without torch, and site/src/lib/dodge-sim.js repeats
the physics for one drone (test/dodge-sim.test.mjs checks it against this file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import gymnasium as gym
import numpy as np

G = 9.81
M32 = 0xFFFFFFFF
FEET = ((-0.22, -0.18), (0.22, -0.18))
TIPS = ((-0.35, 0.04), (0.35, 0.04))
POINTS = FEET + TIPS
LEG = 0.18
FOOT_BAND = 0.15

K_BIRDS = 4
K_ROCKS = 2
K_THREATS = K_BIRDS + K_ROCKS + 1  # birds, rocks, the missile
K_SEE = 4  # threat slots in the "state" input
KIND = np.array([0] * K_BIRDS + [1] * K_ROCKS + [2])  # 0 bird, 1 rock, 2 missile
KINDS = ("bird", "rock", "missile")
HIT = np.array([0.55] * K_BIRDS + [0.5] * K_ROCKS + [0.6])  # contact: centre within this
LOOK = np.array([0.3] * K_BIRDS + [0.3] * K_ROCKS + [0.25])  # size the camera sees

BIRD_MARGIN = 6.0  # birds re-enter this far beyond the screen edge
ROCK_TOP = 16.0  # drop height (the screen ends at the 13 m ceiling)
ROCK_SPREAD = 1.6  # a rock lands up to this far either side of where the drone was
GOLDEN = 0.6180339887498949  # spreads successive drops evenly
MISSILE_SPEED = 11.0
MISSILE_TURN = 0.45  # rad/s at most (a 24 m turning circle: it can be out-turned up close)
MISSILE_N = 3.0
MISSILE_FUEL = 4.0
ARENA_T = 12.0
HOLD_R = 1.5

N_RAYS = 32
RAY_SPAN = math.radians(150)
RAY_RANGE = 15.0

OUTCOMES = ("flying", "landed", "crashed", "water", "lost", "timeout", "bird", "rock", "missile")
FLYING, LANDED, CRASHED, WATER, LOST, TIMEOUT, BIRD, ROCK, MISSILE = range(9)
FAILED = (CRASHED, WATER, LOST, BIRD, ROCK, MISSILE)


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
class Threats:
    """How many of each threat a course gets (ranges are drawn per course)."""

    birds: tuple = (0, 3)
    rocks: tuple = (0, 2)  # droppers
    rock_every: tuple = (3.0, 6.0)  # seconds between one dropper's rocks
    missile: float = 0.5  # chance of a missile
    launch: tuple = (2.0, 7.0)  # when it's launched (s)
    missile_speed: tuple = (MISSILE_SPEED, MISSILE_SPEED)  # training curriculum: slower first
    bird_scale: float = 1.0  # training curriculum: birds at this fraction of their speed
    aim_birds: bool = False  # arenas: every bird flies through the point, at 2-10 s
    arena: float = 0.0  # chance the course is a dodge-school arena


NONE = Threats(birds=(0, 0), rocks=(0, 0), missile=0.0)
MIX = Threats()  # the test courses
SCHOOL = Threats(birds=(0, 4), rocks=(0, 2), rock_every=(2.0, 4.0), missile=0.7,
                 launch=(1.0, 6.0), arena=0.5)  # fmt: skip
ARENA = Threats(birds=(2, 4), rocks=(1, 2), rock_every=(2.0, 4.0), missile=0.8,
                launch=(1.0, 6.0), arena=1.0)  # fmt: skip
ONLY = {  # one kind at a time, in the arena
    "bird": Threats(birds=(3, 3), rocks=(0, 0), missile=0.0, arena=1.0),
    "aimed": Threats(birds=(1, 3), rocks=(0, 0), missile=0.0, arena=1.0, aim_birds=True),
    "rock": Threats(birds=(0, 0), rocks=(2, 2), rock_every=(2.0, 3.0), missile=0.0, arena=1.0),
    "missile": Threats(birds=(0, 0), rocks=(0, 0), missile=1.0, launch=(1.0, 4.0), arena=1.0),
}


@dataclass(frozen=True)
class Features:
    pad_move: float = 0.7
    battery: bool = True
    noise: float = 1.0
    delay: int = 1
    stack: int = 3
    obs: str = "state"  # "state" or "vision"
    threats: Threats = field(default=MIX)


FULL = Features()
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
    """One course as plain floats. The first draws are post 10's (pillars, weather) and post
    11's (the pad's slide), in the same order; post 12's come after them."""
    rand = mulberry32(seed)
    tr = feat.threats

    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * rand()

    def count(lohi) -> int:
        lo, hi = lohi
        return lo + int(rand() * (hi + 1 - lo))

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
    moves = rand() < feat.pad_move
    amp = u(0.8, 2.5)
    room = min(tx - tw / 2 - 1.0, ph.width - 1.0 - (tx + tw / 2), abs(tx - sx) - 1.0 - tw / 2 - 1.5)
    amp = max(0.0, min(amp, room)) if moves else 0.0
    pad_period = u(4.0, 9.0)
    pad_phase = u(0.0, 2 * math.pi)
    # post 12
    arena = rand() < tr.arena
    goal = (u(6.0, ph.width - 6.0), u(4.0, 9.0))
    span = ph.width + 2 * BIRD_MARGIN
    n_birds = count(tr.birds)
    birds = []
    for k in range(K_BIRDS):
        b_y = u(2.0, 11.0)
        fast = rand() < 0.5
        speed = u(4.0, 7.0) if fast else u(1.0, 3.0)
        direction = 1.0 if rand() < 0.5 else -1.0
        b_x0 = u(0.0, span)
        bob = u(0.0, 0.5)
        bob_period = u(1.5, 3.0)
        bob_phase = u(0.0, 2 * math.pi)
        x_now = -BIRD_MARGIN + b_x0 % span  # don't start a bird on top of the drone
        if abs(x_now - sx) < 3.0 and abs(b_y - (sh + LEG)) < 3.0:
            b_x0 += span / 2
        if tr.aim_birds and arena:  # through the point at t_c (from the same draws)
            v = speed * direction * tr.bird_scale
            t_c = 2.0 + 8.0 * (x_now + BIRD_MARGIN) / span
            b_y = goal[1] + (b_y - 6.5) / 4.5 * 0.3
            bob = 0.1 * bob
            b_x0 = (goal[0] + BIRD_MARGIN - v * t_c) % span
        if k < n_birds:
            birds.append((b_y, speed * direction * tr.bird_scale, b_x0, bob, bob_period, bob_phase))
    n_rocks = count(tr.rocks)
    rocks = []
    for k in range(K_ROCKS):
        every = u(*tr.rock_every)
        first = u(0.5, 0.5 + every)
        drift = u(-1.0, 1.0)
        c = rand()
        if k < n_rocks:
            rocks.append((every, first, drift, c))
    has_missile = rand() < tr.missile
    launch = u(*tr.launch)
    side = 1.0 if rand() < 0.5 else 0.0
    m_y = u(1.0, 8.0)
    m_speed = u(*tr.missile_speed)
    charge = u(0.06, 1.0) if feat.battery else 1.0
    if arena:  # no landing: the target pad leaves the world
        tx0, tx1, ttop, amp = -100.0, -100.0, 0.0, 0.0
    else:
        tx0, tx1, ttop = tx - tw / 2, tx + tw / 2, th
    return {
        "start": (sx - 1.0, sx + 1.0, sh),
        "target": (tx0, tx1, ttop),
        "wind": wind,
        "gust": gust,
        "period": period,
        "phase": phase,
        "mass": mass,
        "amp": amp,
        "pad_period": pad_period,
        "pad_phase": pad_phase,
        "arena": bool(arena),
        "goal": goal,
        "birds": birds,
        "rocks": rocks,
        "missile": (launch, side, m_y, m_speed) if has_missile else None,
        "charge": charge,
    }


def thrust_scale(charge):
    """Full thrust vs a full pack: a slow 3% sag over the whole pack, and a knee near empty."""
    return 1.0 - 0.03 * (1.0 - charge) - 0.2 * np.exp(-charge / 0.06)


FIELDS = (
    "x y vx vy th om motor stick0 stick1 charge "
    "s_x0 s_x1 s_top t_x0 t_x1 t_top wind gust period phase mass amp pad_period pad_phase "
    "pad_x0 pad_x1 pad_vx goal_x goal_y m_t0 m_side m_y0 m_speed m_x m_y m_psi"
).split()
INTS = ("ticks", "outcome", "crash", "grounded", "ground_pad", "arena", "end_tick", "m_on",
        "m_state")  # fmt: skip  # m_state: 0 waiting, 1 flying, 2 gone


class Fleet:
    """N drones stepped together. Courses come from seeds (set_course)."""

    def __init__(self, n: int, ph: Physics | None = None, feat: Features = FULL):
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
        self.rock_p = np.zeros((n, K_ROCKS, 4))  # every, first, drift, c
        self.rock_on = np.zeros((n, K_ROCKS), bool)
        self.rock_n = np.zeros((n, K_ROCKS), np.int64)  # rocks dropped so far
        self.rock_t = np.full((n, K_ROCKS), -1e3)  # when the current one was dropped
        self.rock_x = np.zeros((n, K_ROCKS))  # where
        self.touch = np.zeros((n, 2))
        self.courses: list[dict | None] = [None] * n

    def take(self, rows) -> Fleet:
        """A copy of drones `rows` (repeats allowed), mid-flight state and all."""
        rows = np.asarray(rows)
        new = Fleet.__new__(Fleet)
        new.n, new.ph, new.feat = len(rows), self.ph, self.feat
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray) and v.shape[:1] == (self.n,):
                setattr(new, k, v[rows].copy())
        new.courses = [self.courses[i] for i in rows]
        return new

    # --- courses -------------------------------------------------------------------------
    def set_course(self, i: int, c: dict) -> None:
        self.courses[i] = c
        (self.s_x0[i], self.s_x1[i], self.s_top[i]) = c["start"]
        (self.t_x0[i], self.t_x1[i], self.t_top[i]) = c["target"]
        for k in ("wind", "gust", "period", "phase", "mass", "amp", "pad_period", "pad_phase"):
            getattr(self, k)[i] = c[k]
        self.arena[i] = int(c.get("arena", False))
        self.goal_x[i], self.goal_y[i] = c.get("goal", (0.0, 0.0))
        self.end_tick[i] = round((ARENA_T if self.arena[i] else self.ph.t_max) / self.ph.dt)
        self.bird[i] = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # empty slot (bob period 1: no 0/0)
        self.bird_on[i] = False
        for k, b in enumerate(c["birds"]):
            self.bird[i, k] = b
            self.bird_on[i, k] = True
        self.rock_p[i] = (1.0, 0.0, 0.0, 0.0)
        self.rock_on[i] = False
        for k, r in enumerate(c.get("rocks", [])):
            self.rock_p[i, k] = r
            self.rock_on[i, k] = True
        self.rock_n[i] = 0
        self.rock_t[i] = -1e3
        self.rock_x[i] = 0.0
        m = c.get("missile")
        self.m_on[i] = m is not None
        m = m if m is not None else (0.0, 0.0, 0.0, MISSILE_SPEED)
        self.m_t0[i], self.m_side[i], self.m_y0[i], self.m_speed[i] = m
        self.m_state[i] = 0
        self.m_x[i] = self.m_y[i] = self.m_psi[i] = 0.0
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

    # --- threats -----------------------------------------------------------------------------
    def bird_pos(self, idx: np.ndarray | slice = slice(None)):
        b = self.bird[idx]
        t = (self.ticks[idx] * self.ph.dt)[:, None]
        span = self.ph.width + 2 * BIRD_MARGIN
        x = -BIRD_MARGIN + np.mod(b[..., 2] + b[..., 1] * t, span)
        wb = 2 * math.pi / b[..., 4]
        y = b[..., 0] + b[..., 3] * np.sin(wb * t + b[..., 5])
        vy = b[..., 3] * wb * np.cos(wb * t + b[..., 5])
        return x, y, b[..., 1] + 0.0 * x, vy

    def threats(self, idx: np.ndarray | slice = slice(None)):
        """Every threat slot's (x, y, vx, vy, on), each (N, K_THREATS): birds, rocks, missile."""
        bx, by, bvx, bvy = self.bird_pos(idx)
        t = (self.ticks[idx] * self.ph.dt)[:, None]
        p = self.rock_p[idx]
        fall = t - self.rock_t[idx]
        ry = ROCK_TOP - 0.5 * G * fall * fall
        rx = self.rock_x[idx] + p[..., 2] * fall
        rvx, rvy = p[..., 2] + 0.0 * fall, -G * fall
        r_on = self.rock_on[idx] & (self.rock_n[idx] > 0) & (ry > -1.0)
        psi, sp = self.m_psi[idx], self.m_speed[idx]
        mx, my = self.m_x[idx][:, None], self.m_y[idx][:, None]
        mvx = (sp * np.cos(psi))[:, None]
        mvy = (sp * np.sin(psi))[:, None]
        m_on = (self.m_state[idx] == 1)[:, None]
        cat = np.concatenate
        return (
            cat([bx, rx, mx], 1),
            cat([by, ry, my], 1),
            cat([bvx, rvx, mvx], 1),
            cat([bvy, rvy, mvy], 1),
            cat([self.bird_on[idx], r_on, m_on], 1),
        )

    def _threats_move(self, idx) -> None:
        """Rocks drop and the missile launches and steers (the tick's clock already moved on)."""
        if len(idx) == 0:
            return
        ph = self.ph
        t = self.ticks[idx] * ph.dt
        if self.rock_on[idx].any():
            p = self.rock_p[idx]
            due = self.rock_on[idx] & (t[:, None] >= p[..., 1] + self.rock_n[idx] * p[..., 0])
            if due.any():
                rows, ks = np.nonzero(due)
                j = idx[rows]
                n = self.rock_n[j, ks]
                self.rock_t[j, ks] = p[rows, ks, 1] + n * p[rows, ks, 0]
                spread = 2.0 * np.mod(n * GOLDEN + p[rows, ks, 3], 1.0) - 1.0
                self.rock_x[j, ks] = np.clip(self.x[j] + ROCK_SPREAD * spread, 0.5, ph.width - 0.5)
                self.rock_n[j, ks] = n + 1
        fl = idx[self.m_state[idx] == 1]
        if len(fl):
            psi, sp = self.m_psi[fl], self.m_speed[fl]
            rx, ry = self.x[fl] - self.m_x[fl], self.y[fl] - self.m_y[fl]
            rvx = self.vx[fl] - sp * np.cos(psi)
            rvy = self.vy[fl] - sp * np.sin(psi)
            ldot = (rx * rvy - ry * rvx) / np.maximum(rx * rx + ry * ry, 1e-6)
            psi = psi + np.clip(MISSILE_N * ldot, -MISSILE_TURN, MISSILE_TURN) * ph.dt
            self.m_psi[fl] = psi
            self.m_x[fl] += sp * np.cos(psi) * ph.dt
            self.m_y[fl] += sp * np.sin(psi) * ph.dt
            tf = self.ticks[fl] * ph.dt
            mx, my = self.m_x[fl], self.m_y[fl]
            gone = (
                (tf - self.m_t0[fl] > MISSILE_FUEL)
                | (my < 0.0)
                | (mx < -8.0)
                | (mx > ph.width + 8.0)
                | (my > ph.ceiling + 6.0)
            )
            self.m_state[fl[gone]] = 2
        go = idx[(self.m_on[idx] == 1) & (self.m_state[idx] == 0) & (t >= self.m_t0[idx])]
        if len(go):
            self.m_x[go] = np.where(self.m_side[go] > 0.5, ph.width + 3.0, -3.0)
            self.m_y[go] = self.m_y0[go]
            self.m_psi[go] = np.arctan2(self.y[go] - self.m_y[go], self.x[go] - self.m_x[go])
            self.m_state[go] = 1

    def _threats_hit(self, idx) -> None:
        if len(idx) == 0:
            return
        x, y, _, _, on = self.threats(idx)
        dx, dy = x - self.x[idx][:, None], y - self.y[idx][:, None]
        hit = on & (dx * dx + dy * dy < HIT * HIT)
        any_hit = hit.any(1)
        if any_hit.any():
            kind = KIND[np.argmax(hit[any_hit], 1)]
            j = idx[any_hit]
            for k, oc in enumerate((BIRD, ROCK, MISSILE)):
                self._end(j[kind == k], oc)

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
            self._threats_move(j)
            self._threats_hit(j)
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
        self._threats_move(idx)
        self._contacts(idx, before)
        live = idx[self.outcome[idx] == FLYING]
        self._threats_hit(live)
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

    def _limits(self, idx) -> None:
        ph = self.ph
        lost = (self.x[idx] < -0.5) | (self.x[idx] > ph.width + 0.5) | (self.y[idx] > ph.ceiling)
        self.outcome[idx[lost]] = LOST
        rest = idx[~lost]
        self.outcome[rest[self.ticks[rest] >= self.end_tick[rest]]] = TIMEOUT

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
        a = self.arena == 1
        gx = np.where(a, self.goal_x, (self.pad_x0 + self.pad_x1) / 2)
        gy = np.where(a, self.goal_y, self.t_top + LEG)
        return gx, gy

    def distance(self):
        gx, gy = self.goal()
        dx, dy = gx - self.x, gy - self.y
        return np.sqrt(dx * dx + dy * dy)

    def surface(self, x):
        on_s = (self.s_x0 < x) & (x < self.s_x1)
        on_t = (self.pad_x0 < x) & (x < self.pad_x1)
        return np.where(on_s, self.s_top, np.where(on_t, self.t_top, 0.0))


# --- what the pilot sees ---------------------------------------------------------------------
N_BASE = 18
N_SLOT = 8
N_STATE = N_BASE + K_SEE * N_SLOT
N_OWN = 13
N_VISION = N_OWN + 4 * N_RAYS


def base_inputs(f: Fleet) -> np.ndarray:
    """Post 11's 16 numbers (post 10's 14, the pad's speed, the charge), then arena and the
    missile-launch warning."""
    gx, gy = f.goal()
    under = f.y - LEG - f.surface(f.x)
    half = np.where(f.arena == 1, 0.0, (f.t_x1 - f.t_x0) / 2)
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
            f.pad_vx / 3,
            f.charge * 2 - 1,
            f.arena * 1.0,
            (f.m_state == 1) * 1.0,
        ],
        1,
    )


def threat_slots(f: Fleet, hide=()) -> np.ndarray:
    """(N, K_SEE, 8): the most urgent threats first (distance / closing speed, receding ones
    count double their distance): dx, dy, vx, vy, present, bird, rock, missile. An empty slot
    reads as 20 m away and absent."""
    x, y, vx, vy, on = f.threats()
    for k in hide:
        on = on & (KIND != KINDS.index(k))[None, :]
    dx, dy = x - f.x[:, None], y - f.y[:, None]
    r = np.sqrt(dx * dx + dy * dy)
    closing = -(dx * (vx - f.vx[:, None]) + dy * (vy - f.vy[:, None])) / np.maximum(r, 1e-6)
    key = np.where(on, r / np.maximum(closing, 0.5), np.inf)
    order = np.argsort(key, axis=1, kind="stable")[:, :K_SEE]
    rows = np.arange(f.n)[:, None]
    have = np.isfinite(key[rows, order])
    out = np.zeros((f.n, K_SEE, N_SLOT))
    out[..., 0] = np.where(have, dx[rows, order] / 10, 2.0)
    out[..., 1] = np.where(have, dy[rows, order] / 10, 0.0)
    out[..., 2] = np.where(have, vx[rows, order] / 5, 0.0)
    out[..., 3] = np.where(have, vy[rows, order] / 5, 0.0)
    out[..., 4] = have
    kind = KIND[order]
    for k in range(3):
        out[..., 5 + k] = have & (kind == k)
    return out


def obs_state(f: Fleet, noise: np.ndarray | None, hide=()) -> np.ndarray:
    o = np.concatenate([base_inputs(f), threat_slots(f, hide).reshape(f.n, -1)], 1)
    if noise is not None:
        o = o + noise * STATE_SIGMA
    return o


STATE_SIGMA = np.array(
    [0.005, 0.005, 0.02, 0.02, 0.04, 0.02, 0.02, 0, 0.005, 0.005, 0.005, 0.01, 0, 0, 0.02, 0.01]
    + [0, 0]
    + [0.005, 0.005, 0.02, 0.02, 0, 0, 0, 0] * K_SEE
)


def rays(f: Fleet, hide=()):
    """The depth camera: N_RAYS rays from the drone's centre, fixed to its body, fanned from
    RAY_SPAN to the left of straight down to RAY_SPAN to the right. Returns (depth 0..1, hit
    the marked pad, hit a threat, a rock's shadow where it hit a surface), each (N, N_RAYS)."""
    a = np.linspace(-RAY_SPAN, RAY_SPAN, N_RAYS)[None, :] - f.th[:, None]
    dx, dy = np.sin(a), -np.cos(a)
    ox, oy = f.x[:, None], f.y[:, None]
    best = np.full(dx.shape, RAY_RANGE)
    kind = np.zeros(dx.shape, np.int64)  # 0 nothing/surface, 1 marked pad, 2 threat
    tx, ty, _, _, on = f.threats()
    for k in hide:
        on = on & (KIND != KINDS.index(k))[None, :]
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
        for k in range(K_THREATS):
            cx, cy = (tx[:, k] - f.x)[:, None], (ty[:, k] - f.y)[:, None]
            b = dx * cx + dy * cy
            c = cx * cx + cy * cy - LOOK[k] * LOOK[k]
            disc = b * b - c
            t = b - np.sqrt(np.maximum(disc, 0.0))
            hit = on[:, k : k + 1] & (disc >= 0) & (t > 0) & (t < best)
            best = np.where(hit, t, best)
            kind = np.where(hit, 2, kind)
    # shadows: where a ray ends on a surface's top (water or a pad), under a falling rock
    hx, hy = ox + best * dx, oy + best * dy
    on_top = (kind != 2) & (best < RAY_RANGE)
    s_top = np.where(
        (f.s_x0[:, None] < hx) & (hx < f.s_x1[:, None]),
        f.s_top[:, None],
        np.where((f.pad_x0[:, None] < hx) & (hx < f.pad_x1[:, None]), f.t_top[:, None], 0.0),
    )
    on_top &= np.abs(hy - s_top) < 1e-6
    shadow = np.zeros(dx.shape, bool)
    for k in range(K_BIRDS, K_BIRDS + K_ROCKS):
        rk = on[:, k : k + 1] & (np.abs(tx[:, k : k + 1] - hx) < 0.45) & (ty[:, k : k + 1] > hy)
        shadow |= on_top & rk
    return best / RAY_RANGE, (kind == 1) * 1.0, (kind == 2) * 1.0, shadow * 1.0


def obs_vision(f: Fleet, noise: np.ndarray | None, gps_err: np.ndarray, hide=()) -> np.ndarray:
    gx, gy = f.goal()
    depth, pad, threat, shadow = rays(f, hide)
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
            f.arena * 1.0,
            (f.m_state == 1) * 1.0,
        ],
        1,
    )
    if noise is not None:
        own = own + noise[:, :N_OWN] * VISION_SIGMA
        depth = np.clip(depth + noise[:, N_OWN : N_OWN + N_RAYS] * 0.01, 0.0, 1.0)
    return np.concatenate([own, depth, pad, threat, shadow], 1)


VISION_SIGMA = np.array([0.02, 0.02, 0.04, 0.02, 0.02, 0.01, 0.005, 0, 0, 0.005, 0.005, 0, 0])


# --- reward ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reward:
    progress: float = 1.0
    landed: float = 10.0
    failed: float = -5.0
    hold: float = 0.05  # arena: per update within HOLD_R of the point


def rewards(f: Fleet, before: np.ndarray, rw: Reward, was_flying):
    dist = f.distance()
    r = rw.progress * (before - dist)
    r = r + np.where(f.outcome == LANDED, rw.landed, 0.0)
    r = r + np.where(np.isin(f.outcome, FAILED), rw.failed, 0.0)
    if rw.hold:
        ok = (f.arena == 1) & ((f.outcome == FLYING) | (f.outcome == TIMEOUT)) & (dist < HOLD_R)
        r = r + np.where(ok, rw.hold, 0.0)
    return np.where(was_flying, r, 0.0)


# --- Stable-Baselines3 vectorised env ----------------------------------------------------------
@dataclass(frozen=True)
class Config:
    physics: Physics = Physics()
    weather: Weather = ROUGH
    features: Features = FULL
    reward: Reward = Reward()


try:  # SB3 is only installed for training; the tests and CI use Fleet directly
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:  # pragma: no cover
    VecEnv = object


class FleetEnv(VecEnv):
    """N drones as one Stable-Baselines3 VecEnv: a finished flight is replaced by a new course
    in the same slot at once (with the usual terminal_observation in its info). `hide` keeps
    threat kinds out of what the pilot sees (they still fly and still hit): the blindfold test."""

    def __init__(self, n: int, config: Config | None = None, seed: int = 0, hide=(), mix=None):
        self.config = cfg = config or Config()
        self.mix = mix  # [(weight, Threats)]: each new course draws its threats from one
        self.fleet = Fleet(n, cfg.physics, cfg.features)
        self.rng = np.random.default_rng(seed)
        self.hide = tuple(hide)
        feat = cfg.features
        self.width = {"state": N_STATE, "vision": N_VISION}[feat.obs]
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
        if feat.obs == "state":
            return obs_state(f, None if noise is None else noise[:, : self.width], self.hide)
        return obs_vision(f, noise, self.gps_err, self.hide)

    def _push(self) -> np.ndarray:
        now = self._sense()
        if self.config.features.delay:
            self.pending.append(now)
            now = self.pending.pop(0)
        self.frames = np.concatenate([now[:, None, :], self.frames[:, :-1]], 1)
        return self._flat()

    def _fill(self, idx: np.ndarray) -> None:
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
            feat = cfg.features
            if self.mix:
                w = np.array([m[0] for m in self.mix], float)
                feat = replace(feat, threats=self.mix[self.rng.choice(len(w), p=w / w.sum())][1])
            self.fleet.set_course(int(i), make_course(seed, cfg.weather, feat, cfg.physics))
        self.noise_scale[idx] = self.rng.uniform(0.5, 1.5, len(idx))
        self.gps_err[idx] = self.rng.normal(0.0, 0.3, (len(idx), 2)) * cfg.features.noise

    def reset(self):
        idx = np.arange(self.num_envs)
        self._new_course(idx)
        self._fill(idx)
        return self._flat()

    def step(self, actions):
        """Same as VecEnv.step (defined here too, so the env works without SB3 installed)."""
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self._actions = np.asarray(actions, float)

    def step_wait(self):
        f, cfg = self.fleet, self.config
        before = f.distance()
        flying = f.outcome == FLYING
        f.step(self._actions)
        r = rewards(f, before, cfg.reward, flying).astype(np.float32)
        done_idx = np.nonzero(f.outcome != FLYING)[0]
        dones = f.outcome != FLYING
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        obs = self._push()
        for i in done_idx:
            infos[i] = {
                "outcome": OUTCOMES[f.outcome[i]],
                "arena": bool(f.arena[i]),
                "missile": int(f.m_state[i]) if f.m_on[i] else -1,  # -1: none
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
