"""The gauntlet (post 14): one long flight past every threat from posts 12 and 13.

A course is 60-120 m from the start pillar to the landing pad (post 11's gap was 8-20 m). The
pilot knows which way the pad is but not how far: its distance reads as "at least 40 m" (the
sight range) until it comes into view. Along the way, at waypoints, the flight meets
encounters:
- an aimed bird: it homes on the drone (proportional navigation, agile but slow) until it is
  about a second from contact, then commits to a straight line;
- two aimed birds at once, timed to arrive together from different angles;
- the missile from post 12 (fast, a poor turner, 4 s of fuel), from ahead or behind;
- a bird and the missile together;
- a flock of 2-3 birds crossing at random heights, not aimed (a test of not over-reacting);
- or nothing.
An air-frog from post 13 guards the pad, and half the courses have another frog mid-route,
striking on a rhythm across its column. The pilot sees threats and frogs only within 40 m, and
with "hist" memory gets 3 s of each frog's offset from its perch.

Physics, weather and the battery are post 10-13's. Plain numpy: CI runs the tests without
torch, and site/src/lib/gauntlet-sim.js repeats it for one drone (test/gauntlet-sim.test.mjs
checks it against this file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

G = 9.81
M32 = 0xFFFFFFFF
FEET = ((-0.22, -0.18), (0.22, -0.18))
TIPS = ((-0.35, 0.04), (0.35, 0.04))
POINTS = FEET + TIPS
LEG = 0.18
FOOT_BAND = 0.15

SIGHT = 40.0  # threats, frogs and the pad are seen within this distance
MARGIN = 12.0  # world beyond the start pillar and the pad

# pursuers: 4 bird slots, then the missile
K_BIRDS = 6
K_MISSILES = 2
K_PURSUERS = K_BIRDS + K_MISSILES
P_KIND = np.array([0] * K_BIRDS + [1] * K_MISSILES)  # 0 bird, 1 missile
P_HIT = np.array([0.55] * K_BIRDS + [0.6] * K_MISSILES)
BIRD_TURN = 2.0  # rad/s at most
BIRD_N = 3.0
BIRD_LIFE = 14.0  # s
MISSILE_SPEED = 11.0
MISSILE_TURN = 0.45
MISSILE_N = 3.0
MISSILE_FUEL = 4.0
GONE_R = 50.0  # a pursuer this far from the drone leaves

ENCOUNTERS = ("calm", "bird", "two birds", "missile", "bird + missile", "flock")
CALM, BIRD1, BIRD2, MISSILE1, COMBO, FLOCK = range(6)
MAX_ENC = 16
N_ENC_RAND = 8

OUTCOMES = ("flying", "landed", "crashed", "water", "lost", "timeout", "bird", "missile", "frog")
FLYING, LANDED, CRASHED, WATER, LOST, TIMEOUT, BIRD, MISSILE, FROG = range(9)
FAILED = (CRASHED, WATER, LOST, BIRD, MISSILE, FROG)

K_FROGS = 2  # slot 0 mid-route, slot 1 guarding the pad
FROG_HIT = 0.6
FROG_REACH = 2.0
FROG_LEAD = 0.35
LEAP, HOLD, BACK = 0.35, 0.5, 1.0
CROUCH = 0.4
FROG_JITTER = 0.1
HIST, HIST_EVERY = 16, 4
GOLDEN = 0.6180339887498949


@dataclass(frozen=True)
class Physics:
    dt: float = 1 / 60
    substeps: int = 3
    t_max: float = 60.0
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
    ceiling: float = 13.0


@dataclass(frozen=True)
class Weather:
    wind: float = 0.0
    gust: float = 0.0
    heavy: float = 0.0


CALM_AIR = Weather()
ROUGH = Weather(wind=1.5, gust=1.0, heavy=0.35)

TEST_MIX = (0.1, 0.2, 0.2, 0.15, 0.2, 0.15)  # encounter odds, in ENCOUNTERS order


@dataclass(frozen=True)
class Features:
    battery: bool = True
    noise: float = 1.0
    delay: int = 1
    stack: int = 3
    memory: str = "hist"  # "hist": 3 s of each frog's offset from its perch; "none"
    route: tuple = (60.0, 120.0)  # start to pad (m)
    first: tuple = (8.0, 14.0)  # the first waypoint (m along the route)
    spacing: tuple = (18.0, 28.0)  # between waypoints
    mix: tuple = TEST_MIX
    route_frog: float = 0.5  # chance of a frog mid-route
    guard: float = 1.0  # chance of a frog guarding the pad
    frog_period: tuple = (3.0, 4.0)  # seconds between strikes (training dials: slower first)
    missile_speed: tuple = (MISSILE_SPEED, MISSILE_SPEED)
    bird_lock: float = 1.0  # an aimed bird commits this long (s) before contact
    presence: float = 1.0  # training dial: the share of waypoints and frogs that are real
    predict: bool = False  # each pursuer slot also gets its closest approach (see threat_slots)
    switch: tuple | None = None  # tests: (after strike n, frog period x ratio from then on)


FULL = Features()
CLEAR = Features(mix=(1.0, 0, 0, 0, 0, 0), route_frog=0.0, guard=0.0)  # no threats at all


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
    """One course as plain floats (the same draws, in the same order, as gauntlet-sim.js)."""
    rand = mulberry32(seed)

    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * rand()

    length = u(*feat.route)
    sh = u(0.5, 4.0)
    tw = u(1.6, 2.6)
    th = u(0.5, 7.0)
    direction = 1.0 if rand() < 0.5 else -1.0
    width = length + 2 * MARGIN
    sx = MARGIN if direction > 0 else width - MARGIN
    tx = sx + direction * length
    wind = u(-1.0, 1.0) * weather.wind
    gust = u(0.0, 1.0) * weather.gust
    period = u(2.0, 5.0)
    phase = u(0.0, 2 * math.pi)
    mass = 1.0 + u(0.0, 1.0) * weather.heavy
    charge = u(0.5, 1.0) if feat.battery else 1.0
    m_speed = u(*feat.missile_speed)
    # frogs: slot 0 mid-route, slot 1 above the pad
    frogs = []
    for k in range(K_FROGS):
        present = rand() < (feat.route_frog if k == 0 else feat.guard) * feat.presence
        frac = u(0.35, 0.65)
        y_perch = u(3.0, 9.0)
        f_period = u(*feat.frog_period)
        first = u(1.0, 1.0 + f_period)
        c = rand()
        bob_phase = u(0.0, 2 * math.pi)
        if k == 0:
            x_col = sx + direction * length * frac
        else:
            x_col = tx
            y_perch = th + 2.5 + (y_perch - 3.0) / 3.0
        frogs.append((x_col, y_perch, f_period, first, c, bob_phase) if present else None)
    # waypoints
    weights = np.asarray(feat.mix, float)
    cum = np.cumsum(weights / weights.sum())
    encounters = []
    at = u(*feat.first)
    while at < length - 12.0 and len(encounters) < MAX_ENC:
        r = rand()
        kind = int(min(np.searchsorted(cum, r, side="right"), len(cum) - 1))
        if rand() >= feat.presence:
            kind = CALM
        params = tuple(rand() for _ in range(N_ENC_RAND))
        if frogs[0] is not None:  # keep encounters clear of the mid-route frog's column
            col = abs(frogs[0][0] - sx)  # its distance along the route
            if col - 12.0 < at < col + 4.0:
                kind = CALM
        encounters.append((at, kind, params))
        at += u(*feat.spacing)
    return {
        "start": (sx - 1.0, sx + 1.0, sh),
        "target": (tx - tw / 2, tx + tw / 2, th),
        "dir": direction,
        "length": length,
        "world": (0.0, width),
        "wind": wind,
        "gust": gust,
        "period": period,
        "phase": phase,
        "mass": mass,
        "charge": charge,
        "m_speed": m_speed,
        "bird_lock": feat.bird_lock,
        "frogs": frogs,
        "encounters": encounters,
        "switch": feat.switch,
    }


def thrust_scale(charge):
    return 1.0 - 0.03 * (1.0 - charge) - 0.2 * np.exp(-charge / 0.06)


FIELDS = (
    "x y vx vy th om motor stick0 stick1 charge s_x0 s_x1 s_top t_x0 t_x1 t_top dir w_lo w_hi "
    "wind gust period phase mass m_speed lock"
).split()
INTS = ("ticks", "outcome", "crash", "grounded", "ground_pad", "end_tick", "enc_next", "n_enc",
        "hit_enc")  # fmt: skip


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
        self.enc_at = np.full((n, MAX_ENC), 1e9)
        self.enc_kind = np.zeros((n, MAX_ENC), np.int64)
        self.enc_p = np.zeros((n, MAX_ENC, N_ENC_RAND))
        self.enc_count = np.zeros((n, len(ENCOUNTERS)), np.int64)  # met so far, per kind
        # pursuers
        self.p_on = np.zeros((n, K_PURSUERS), bool)
        self.p_x = np.zeros((n, K_PURSUERS))
        self.p_y = np.zeros((n, K_PURSUERS))
        self.p_psi = np.zeros((n, K_PURSUERS))
        self.p_spd = np.ones((n, K_PURSUERS))
        self.p_steer = np.zeros((n, K_PURSUERS), bool)
        self.p_end = np.zeros((n, K_PURSUERS))  # leaves at this time
        self.p_hostile = np.zeros((n, K_PURSUERS), bool)  # aimed bird or missile
        self.p_done = np.zeros((n, K_PURSUERS), bool)  # counted as beaten
        self.p_enc = np.zeros((n, K_PURSUERS), np.int64)  # the encounter kind it came from
        self.beaten = np.zeros((n, 2), np.int64)  # aimed birds, missiles got away from
        # frogs
        self.frog_p = np.zeros((n, K_FROGS, 6))  # x_col, y_perch, period, first, c, bob_phase
        self.frog_p[..., 2] = 1.0
        self.frog_on = np.zeros((n, K_FROGS), bool)
        self.frog_n = np.zeros((n, K_FROGS), np.int64)
        self.frog_s = np.full((n, K_FROGS), -1e3)
        self.frog_sx = np.zeros((n, K_FROGS))
        self.frog_sy = np.zeros((n, K_FROGS))
        self.frog_tx = np.zeros((n, K_FROGS))
        self.frog_ty = np.zeros((n, K_FROGS))
        self.frog_past = np.zeros((n, K_FROGS), bool)  # got past it (the pad's: landed)
        self.sw_after = np.full(n, 10**6, np.int64)
        self.sw_ratio = np.ones(n)
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
        self.dir[i] = c["dir"]
        self.w_lo[i], self.w_hi[i] = c["world"]
        for k in ("wind", "gust", "period", "phase", "mass", "m_speed"):
            getattr(self, k)[i] = c[k]
        self.lock[i] = c["bird_lock"]
        self.end_tick[i] = round(self.ph.t_max / self.ph.dt)
        enc = c["encounters"]
        self.n_enc[i] = len(enc)
        self.enc_next[i] = 0
        self.enc_at[i] = 1e9
        self.enc_kind[i] = 0
        self.enc_p[i] = 0.0
        for k, (at, kind, params) in enumerate(enc):
            self.enc_at[i, k] = at
            self.enc_kind[i, k] = kind
            self.enc_p[i, k] = params
        self.enc_count[i] = 0
        self.hit_enc[i] = -1
        self.p_on[i] = False
        self.p_x[i] = self.p_y[i] = self.p_psi[i] = self.p_end[i] = 0.0
        self.p_spd[i] = 1.0
        self.p_steer[i] = self.p_hostile[i] = self.p_done[i] = False
        self.p_enc[i] = 0
        self.beaten[i] = 0
        self.frog_p[i] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        self.frog_on[i] = False
        for k, fr in enumerate(c["frogs"]):
            if fr is not None:
                self.frog_p[i, k] = fr
                self.frog_on[i, k] = True
        self.frog_n[i] = 0
        self.frog_s[i] = -1e3
        self.frog_sx[i] = self.frog_sy[i] = self.frog_tx[i] = self.frog_ty[i] = 0.0
        self.frog_past[i] = False
        sw = c.get("switch")
        self.sw_after[i], self.sw_ratio[i] = (sw[0], sw[1]) if sw else (10**6, 1.0)
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

    def progress(self, idx=slice(None)):
        """Metres flown along the route (from the start pillar's centre)."""
        sx = (self.s_x0[idx] + self.s_x1[idx]) / 2
        return self.dir[idx] * (self.x[idx] - sx)

    # --- encounters --------------------------------------------------------------------
    def _spawn(self, i: int, slot: int, x, y, psi, spd, steer, hostile, end, enc) -> None:
        self.p_on[i, slot] = True
        self.p_x[i, slot], self.p_y[i, slot] = x, y
        self.p_psi[i, slot], self.p_spd[i, slot] = psi, spd
        self.p_steer[i, slot], self.p_hostile[i, slot] = steer, hostile
        self.p_done[i, slot] = not hostile
        self.p_end[i, slot] = end
        self.p_enc[i, slot] = enc

    def _slot(self, i: int, kind: int) -> int:
        """A free slot for a bird (kind 0) or a missile (1); if all are taken (rare), the one
        leaving soonest."""
        ks = np.nonzero(P_KIND == kind)[0]
        free = ks[~self.p_on[i, ks]]
        return int(free[0]) if len(free) else int(ks[np.argmin(self.p_end[i, ks])])

    def _from(self, i: int, ang: float, dist: float, behind: bool):
        """A point `dist` from the drone at `ang` above the line ahead (or behind), kept
        between 1 and 12.5 m up (the angle mirrors first, then the height is clipped)."""
        fx = -self.dir[i] if behind else self.dir[i]
        y = self.y[i] + dist * math.sin(ang)
        if not 1.0 <= y <= 12.5:
            ang = -ang
            y = self.y[i] + dist * math.sin(ang)
        y = min(max(y, 1.0), 12.5)
        x = self.x[i] + fx * dist * math.cos(ang)
        return x, y

    def _closing(self, i: int, x: float, y: float, speed: float) -> float:
        """Rough closing speed of something at (x, y) flying at the drone at `speed`."""
        dx, dy = x - self.x[i], y - self.y[i]
        r = max(math.hypot(dx, dy), 1e-6)
        return speed + max(0.0, (self.vx[i] * dx + self.vy[i] * dy) / r)

    def _aim(self, i: int, x: float, y: float, speed: float) -> float:
        """Heading onto a collision course with the drone as it flies now (straight at it if
        there's none)."""
        rx, ry = self.x[i] - x, self.y[i] - y
        vx, vy = self.vx[i], self.vy[i]
        a = vx * vx + vy * vy - speed * speed
        b = 2 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        t = -1.0
        if abs(a) < 1e-9:
            t = -c / b if b < 0 else -1.0
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                q = math.sqrt(disc)
                ts = [s for s in ((-b - q) / (2 * a), (-b + q) / (2 * a)) if s > 0]
                t = min(ts) if ts else -1.0
        if t > 0:
            rx, ry = rx + vx * t, ry + vy * t
        return math.atan2(ry, rx)

    def _bird(self, i, ang, dist, speed, behind, enc, t) -> float:
        x, y = self._from(i, ang, dist, behind)
        slot = self._slot(i, 0)
        psi = self._aim(i, x, y, speed)
        self._spawn(i, slot, x, y, psi, speed, True, True, t + BIRD_LIFE, enc)
        return dist / self._closing(i, x, y, speed)

    def _missile(self, i, ang, dist, behind, enc, t) -> None:
        x, y = self._from(i, ang, dist, behind)
        spd = self.m_speed[i]
        psi = self._aim(i, x, y, spd)
        self._spawn(i, self._slot(i, 1), x, y, psi, spd, True, True, t + MISSILE_FUEL, enc)

    def _match(self, i, t_arrive, ang, speed, behind) -> float:
        """The distance at `ang` from which something at `speed` arrives in t_arrive."""
        x, y = self._from(i, ang, 20.0, behind)
        return min(max(t_arrive * self._closing(i, x, y, speed), 12.0), 45.0)

    def _encounter(self, i: int) -> None:
        k = self.enc_next[i]
        kind = int(self.enc_kind[i, k])
        c = self.enc_p[i, k]
        self.enc_next[i] = k + 1
        self.enc_count[i, kind] += 1
        t = self.ticks[i] * self.ph.dt
        deg = math.pi / 180
        if kind == BIRD1:
            self._bird(i, (2 * c[0] - 1) * 50 * deg, 28 + 10 * c[1], 4 + 3 * c[2], False, kind, t)
        elif kind == BIRD2:
            ta = self._bird(i, (10 + 40 * c[0]) * deg, 28 + 10 * c[1], 4 + 3 * c[2], False, kind, t)
            ang, spd, behind = -(10 + 40 * c[3]) * deg, 4 + 3 * c[4], c[5] < 0.3
            dist = self._match(i, ta, ang, spd, behind)
            self._bird(i, ang, dist, spd, behind, kind, t)
        elif kind == MISSILE1:
            self._missile(i, (2 * c[0] - 1) * 40 * deg, 30 + 8 * c[2], c[1] < 0.3, kind, t)
        elif kind == COMBO:
            tb = self._bird(i, (2 * c[0] - 1) * 50 * deg, 28 + 10 * c[1], 4 + 3 * c[2], False,
                            kind, t)  # fmt: skip
            ang, behind = (2 * c[3] - 1) * 40 * deg, c[4] < 0.3
            self._missile(i, ang, self._match(i, tb, ang, self.m_speed[i], behind), behind,
                          kind, t)  # fmt: skip
        elif kind == FLOCK:
            n = 2 + (c[0] < 0.5)
            spd = 4 + 3 * c[7]
            psi = math.pi if self.dir[i] > 0 else 0.0  # flying back along the route
            for j in range(n):
                y = 1.5 + 10.5 * c[1 + j]
                x = self.x[i] + self.dir[i] * (24 + 14 * c[4 + j])
                slot = self._slot(i, 0)
                self._spawn(i, slot, x, y, psi, spd, False, False, t + BIRD_LIFE, kind)

    def _encounters(self, idx) -> None:
        k = self.enc_next[idx]
        has = k < self.n_enc[idx]
        if not has.any():
            return
        at = self.enc_at[idx, np.minimum(k, MAX_ENC - 1)]
        due = has & (self.progress(idx) >= at)
        for i in idx[due]:
            self._encounter(int(i))

    # --- pursuers --------------------------------------------------------------------------
    def pursuers(self, idx=slice(None)):
        """(x, y, vx, vy, on), each (N, K_PURSUERS)."""
        psi, sp = self.p_psi[idx], self.p_spd[idx]
        return self.p_x[idx], self.p_y[idx], sp * np.cos(psi), sp * np.sin(psi), self.p_on[idx]

    def _pursuers_move(self, idx) -> None:
        on = self.p_on[idx]
        if not on.any():
            return
        dt = self.ph.dt
        t = (self.ticks[idx] * dt)[:, None]
        px, py, psi, sp = self.p_x[idx], self.p_y[idx], self.p_psi[idx], self.p_spd[idx]
        rx, ry = self.x[idx][:, None] - px, self.y[idx][:, None] - py
        rvx = self.vx[idx][:, None] - sp * np.cos(psi)
        rvy = self.vy[idx][:, None] - sp * np.sin(psi)
        r2 = np.maximum(rx * rx + ry * ry, 1e-6)
        r = np.sqrt(r2)
        ldot = (rx * rvy - ry * rvx) / r2
        closing = -(rx * rvx + ry * rvy) / r
        bird = P_KIND == 0
        steer = self.p_steer[idx]
        commit = on & steer & bird & (closing > 0) & (r < self.lock[idx][:, None] * closing)
        steer = steer & ~commit
        gain = np.where(bird, BIRD_N, MISSILE_N)
        turn = np.where(bird, BIRD_TURN, MISSILE_TURN)
        psi = psi + np.where(steer & on, np.clip(gain * ldot, -turn, turn) * dt, 0.0)
        px = np.where(on, px + sp * np.cos(psi) * dt, px)
        py = np.where(on, py + sp * np.sin(psi) * dt, py)
        self.p_steer[idx], self.p_psi[idx], self.p_x[idx], self.p_y[idx] = steer, psi, px, py
        # beaten: an aimed bird or the missile that has passed (receding, close) or leaves
        passed = on & (closing < 0) & (r < 6.0) & (~steer | ~bird)
        leaves = on & ((t >= self.p_end[idx]) | (py < -0.5) | (py > self.ph.ceiling + 4.0)
                       | (r > GONE_R))  # fmt: skip
        count = (passed | leaves) & ~self.p_done[idx]
        if count.any():
            rows, ks = np.nonzero(count)
            j = idx[rows]
            self.p_done[j, ks] = True
            np.add.at(self.beaten, (j, P_KIND[ks]), 1)
        if leaves.any():
            rows, ks = np.nonzero(leaves)
            self.p_on[idx[rows], ks] = False

    # --- frogs (post 13) ---------------------------------------------------------------------
    def strike_time(self, idx, n):
        p = self.frog_p[idx]
        after = self.sw_after[idx][:, None]
        beats = np.where(n <= after, n, after + (n - after) * self.sw_ratio[idx][:, None])
        jitter = FROG_JITTER * (2.0 * np.mod(n * GOLDEN + p[..., 4], 1.0) - 1.0)
        return p[..., 3] + p[..., 2] * beats + jitter

    def perch(self, idx=slice(None)):
        p = self.frog_p[idx]
        t = (self.ticks[idx] * self.ph.dt)[:, None]
        w = 2 * math.pi / 1.7
        return p[..., 0] + 0.0 * t, p[..., 1] + 0.3 * np.sin(w * t + p[..., 5])

    def frog_pos(self, idx=slice(None)):
        """Frogs' (x, y, vx, vy), each (N, K_FROGS)."""
        t = (self.ticks[idx] * self.ph.dt)[:, None]
        px, py = self.perch(idx)
        w = 2 * math.pi / 1.7
        pvy = 0.3 * w * np.cos(w * t + self.frog_p[idx][..., 5])
        u = t - self.frog_s[idx]
        sx, sy = self.frog_sx[idx], self.frog_sy[idx]
        tx, ty = self.frog_tx[idx], self.frog_ty[idx]
        a = np.clip(u / LEAP, 0.0, 1.0)
        b = np.clip((u - LEAP - HOLD) / BACK, 0.0, 1.0)
        leap, back = (u >= 0) & (u < LEAP), (u >= LEAP + HOLD) & (u < LEAP + HOLD + BACK)
        hold = (u >= LEAP) & (u < LEAP + HOLD)
        sa, sb = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
        da, db = 6 * a * (1 - a) / LEAP, 6 * b * (1 - b) / BACK
        x = np.where(
            leap, sx + (tx - sx) * sa, np.where(hold, tx, np.where(back, tx + (px - tx) * sb, px))
        )
        y = np.where(
            leap, sy + (ty - sy) * sa, np.where(hold, ty, np.where(back, ty + (py - ty) * sb, py))
        )
        vx = np.where(leap, (tx - sx) * da, np.where(back, (px - tx) * db, 0.0))
        vy = np.where(
            leap, (ty - sy) * da, np.where(back, (py - ty) * db, np.where(hold, 0.0, pvy))
        )
        return x, y, vx, vy

    def next_strike(self, idx=slice(None)):
        t = (self.ticks[idx] * self.ph.dt)[:, None]
        return self.strike_time(idx, self.frog_n[idx]) - t

    def _frogs_move(self, idx) -> None:
        if not self.frog_on[idx].any():
            return
        # the mid-route frog is behind the drone once it's past its reach
        ahead = self.dir[idx] * (self.x[idx] - self.frog_p[idx, 0, 0])
        past = self.frog_on[idx, 0] & (ahead > FROG_REACH + 1.0)
        self.frog_past[idx[past], 0] = True
        t = self.ticks[idx] * self.ph.dt
        due = self.frog_on[idx] & (t[:, None] >= self.strike_time(idx, self.frog_n[idx]))
        if not due.any():
            return
        rows, ks = np.nonzero(due)
        j = idx[rows]
        fx, fy, _, _ = self.frog_pos(idx)
        self.frog_s[j, ks] = self.strike_time(idx, self.frog_n[idx])[rows, ks]
        self.frog_sx[j, ks], self.frog_sy[j, ks] = fx[rows, ks], fy[rows, ks]
        col = self.frog_p[j, ks, 0]
        aim_x = self.x[j] + self.vx[j] * FROG_LEAD
        aim_y = self.y[j] + self.vy[j] * FROG_LEAD
        self.frog_tx[j, ks] = np.clip(aim_x, col - FROG_REACH, col + FROG_REACH)
        self.frog_ty[j, ks] = np.clip(aim_y, 0.5, self.ph.ceiling - 0.5)
        self.frog_n[j, ks] += 1

    def _threats_move(self, idx) -> None:
        if len(idx) == 0:
            return
        self._frogs_move(idx)
        self._encounters(idx)
        self._pursuers_move(idx)

    def _threats_hit(self, idx) -> None:
        if len(idx) == 0:
            return
        x, y, _, _, on = self.pursuers(idx)
        dx, dy = x - self.x[idx][:, None], y - self.y[idx][:, None]
        hit = on & (dx * dx + dy * dy < P_HIT * P_HIT)
        any_hit = hit.any(1)
        if any_hit.any():
            rows = np.nonzero(any_hit)[0]
            k = np.argmax(hit[rows], 1)
            j = idx[rows]
            self.hit_enc[j] = self.p_enc[j, k]
            kind = P_KIND[k]
            self._end(j[kind == 0], BIRD)
            self._end(j[kind == 1], MISSILE)
        if self.frog_on[idx].any():
            live = idx[self.outcome[idx] == FLYING]
            fx, fy, _, _ = self.frog_pos(live)
            dx, dy = fx - self.x[live][:, None], fy - self.y[live][:, None]
            hit = (self.frog_on[live] & (dx * dx + dy * dy < FROG_HIT * FROG_HIT)).any(1)
            self._end(live[hit], FROG)

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
        if sit.any():  # still sitting on a pillar: friction slows any slide
            j = idx[sit]
            self.vx[j] *= 0.85
            self.x[j] += self.vx[j] * dt
            self.ticks[j] += 1
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
            (self.t_x0[idx], self.t_x1[idx], self.t_top[idx]),
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
        gentle = (
            (self.vy[j] >= -ph.land_vy)
            & (np.abs(self.vx[j]) <= ph.land_vx)
            & (np.abs(self.th[j]) <= ph.land_tilt)
        )
        hard = j[~gentle]
        self.crash[hard] = 1
        self._end(hard, CRASHED)
        ok, lp = j[gentle], lp[gentle]
        over = (lp == 1) & (self.t_x0[ok] <= self.x[ok]) & (self.x[ok] <= self.t_x1[ok])
        self.frog_past[ok[over], 1] = self.frog_on[ok[over], 1]
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
        x = self.x[idx]
        lost = (x < self.w_lo[idx] - 0.5) | (x > self.w_hi[idx] + 0.5) | (self.y[idx] > ph.ceiling)
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
        return (self.t_x0 + self.t_x1) / 2, self.t_top + LEG

    def distance(self):
        gx, gy = self.goal()
        dx, dy = gx - self.x, gy - self.y
        return np.sqrt(dx * dx + dy * dy)

    def surface(self, x):
        on_s = (self.s_x0 < x) & (x < self.s_x1)
        on_t = (self.t_x0 < x) & (x < self.t_x1)
        return np.where(on_s, self.s_top, np.where(on_t, self.t_top, 0.0))

    def frog_failed(self) -> np.ndarray:
        """(N,): hit by a frog, or out of time near one it hadn't got past yet."""
        near = self.frog_on & ~self.frog_past & (np.abs(self.frog_p[..., 0] - self.x[:, None]) < 10)
        return (self.outcome == FROG) | ((self.outcome == TIMEOUT) & near.any(1))


# --- what the pilot sees ---------------------------------------------------------------------
N_BASE = 18
K_SEE = 4
N_SLOT = 7
N_PREDICT = 3
N_FROG = 8
N_FRAME = N_BASE + K_SEE * N_SLOT + K_FROGS * N_FROG  # without predictions


def slot_width(feat: Features) -> int:
    return N_SLOT + N_PREDICT * feat.predict


def frame_width(feat: Features) -> int:
    return N_BASE + K_SEE * slot_width(feat) + K_FROGS * N_FROG


N_HIST = 3
HIST_LEN = (HIST - 1) * HIST_EVERY + 1
FAR = SIGHT / 10  # an empty slot reads as this far away (x 10 m)


def base_inputs(f: Fleet, hide: bool = False) -> np.ndarray:
    """The pad's offset (its distance capped at the sight range; its height only once in
    sight), the drone's own state, the start pillar's offset (likewise), whether the pad is in
    sight, the charge, the missile warning (blanked with the threats by the blindfold) and the
    pad's direction."""
    gx, gy = f.goal()
    dx = gx - f.x
    seen = np.abs(dx) < SIGHT
    under = f.y - LEG - f.surface(f.x)
    half = (f.t_x1 - f.t_x0) / 2
    s_dx = (f.s_x0 + f.s_x1) / 2 - f.x
    s_seen = np.abs(s_dx) < SIGHT
    return np.stack(
        [
            np.clip(dx, -SIGHT, SIGHT) / 10,
            np.where(seen, (gy - f.y) / 10, 0.0),
            f.vx / 5,
            f.vy / 5,
            f.th / f.ph.tilt_max,
            f.om / 5,
            f.motor * 2 - 1,
            half / 1.3,
            np.clip(s_dx, -SIGHT, SIGHT) / 10,
            np.where(s_seen, (f.s_top + LEG - f.y) / 10, 0.0),
            f.y / 10,
            np.clip(under / 5, -1.0, 2.0),
            f.stick0,
            f.stick1,
            seen * 1.0,
            f.charge * 2 - 1,
            f.p_on[:, K_BIRDS:].any(1) * (not hide),
            f.dir,
        ],
        1,
    )


def threat_slots(f: Fleet, hide: bool = False, predict: bool = False) -> np.ndarray:
    """(N, K_SEE, 7 or 10): the most urgent pursuers in sight first (distance / closing speed):
    dx, dy, vx, vy, present, bird, missile. With predict, also the closest approach if neither
    changed course (a collision-warning sensor's numbers): in how long (/ 3 s) and the
    pursuer's offset from the drone then (/ 5 m); for one moving away, now and where it is."""
    x, y, vx, vy, on = f.pursuers()
    dx, dy = x - f.x[:, None], y - f.y[:, None]
    r = np.sqrt(dx * dx + dy * dy)
    on = on & (r < SIGHT) & (not hide)
    closing = -(dx * (vx - f.vx[:, None]) + dy * (vy - f.vy[:, None])) / np.maximum(r, 1e-6)
    key = np.where(on, r / np.maximum(closing, 0.5), np.inf)
    order = np.argsort(key, axis=1, kind="stable")[:, :K_SEE]
    rows = np.arange(f.n)[:, None]
    have = np.isfinite(key[rows, order])
    out = np.zeros((f.n, K_SEE, N_SLOT + N_PREDICT * predict))
    out[..., 0] = np.where(have, dx[rows, order] / 10, FAR)
    out[..., 1] = np.where(have, dy[rows, order] / 10, 0.0)
    out[..., 2] = np.where(have, vx[rows, order] / 5, 0.0)
    out[..., 3] = np.where(have, vy[rows, order] / 5, 0.0)
    out[..., 4] = have
    kind = P_KIND[order]
    out[..., 5] = have & (kind == 0)
    out[..., 6] = have & (kind == 1)
    if predict:
        rvx, rvy = vx - f.vx[:, None], vy - f.vy[:, None]
        v2 = np.maximum(rvx * rvx + rvy * rvy, 1e-6)
        t = np.clip(-(dx * rvx + dy * rvy) / v2, 0.0, 6.0)
        mx, my = dx + rvx * t, dy + rvy * t
        out[..., 7] = np.where(have, t[rows, order] / 3, 2.0)
        out[..., 8] = np.where(have, np.clip(mx[rows, order] / 5, -2.0, 2.0), 2.0)
        out[..., 9] = np.where(have, np.clip(my[rows, order] / 5, -2.0, 2.0), 0.0)
    return out


def frog_inputs(f: Fleet, hide: bool = False) -> np.ndarray:
    """(N, K_FROGS, 8), as post 13's (the crouch column is always 0: no tell), for frogs whose
    perch is in sight."""
    fx, fy, fvx, fvy = f.frog_pos()
    px, py = f.perch()
    x, y = f.x[:, None], f.y[:, None]
    on = f.frog_on & (np.hypot(px - x, py - y) < SIGHT) & (not hide)
    out = np.zeros((f.n, K_FROGS, N_FROG))
    out[..., 0] = np.where(on, (fx - x) / 10, FAR)
    out[..., 1] = np.where(on, (fy - y) / 10, 0.0)
    out[..., 2] = np.where(on, fvx / 10, 0.0)
    out[..., 3] = np.where(on, fvy / 10, 0.0)
    out[..., 4] = on
    out[..., 6] = np.where(on, (px - x) / 10, FAR)
    out[..., 7] = np.where(on, (py - y) / 10, 0.0)
    return out


def sigma(predict: bool) -> np.ndarray:
    """Sensor noise per input (x the course's noise scale)."""
    slot = [0.005, 0.005, 0.02, 0.02, 0, 0, 0] + [0.01, 0.01, 0.01] * predict
    return np.concatenate(
        [
            [0.005, 0.005, 0.02, 0.02, 0.04, 0.02, 0.02, 0, 0.005, 0.005, 0.005, 0.01, 0, 0, 0],
            [0.01, 0, 0],
            slot * K_SEE,
            [0.005, 0.005, 0.02, 0.02, 0, 0, 0.005, 0.005] * K_FROGS,
        ]
    )


SIGMA = sigma(False)


def observe(f: Fleet, noise: np.ndarray | None, hide=(), predict: bool = False) -> np.ndarray:
    """One frame of what the pilot senses. `hide` may hold "threats" and/or "frogs"."""
    o = np.concatenate(
        [
            base_inputs(f, "threats" in hide),
            threat_slots(f, "threats" in hide, predict).reshape(f.n, -1),
            frog_inputs(f, "frogs" in hide).reshape(f.n, -1),
        ],
        1,
    )
    if noise is not None:
        o = o + noise * sigma(predict)
    return o


def frog_offsets(seen: np.ndarray, predict: bool = False) -> np.ndarray:
    """From a sensed frame: each frog's offset from its perch and whether it's seen."""
    fr = seen[:, N_BASE + K_SEE * (N_SLOT + N_PREDICT * predict) :]
    fr = fr.reshape(len(seen), K_FROGS, N_FROG)
    on = fr[..., 4]
    out = np.stack([(fr[..., 0] - fr[..., 6]) * on, (fr[..., 1] - fr[..., 7]) * on, on], 2)
    return out.reshape(len(seen), -1)


# --- reward ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reward:
    progress: float = 0.3  # per metre closer to the pad
    landed: float = 10.0
    failed: float = -10.0
    near: float = 0.0  # per update, x how deep a bird or missile is inside NEAR_R (0..1, summed)
    time: float = 0.0  # per update in the air (waiting isn't free)
    danger: float = 0.0  # potential-based shaping on predicted near passes (see danger())


NEAR_R = 2.0
DANGER_MISS, DANGER_T = 2.0, 3.0


def danger(f: Fleet) -> np.ndarray:
    """(N,): how bad the predicted passes look, summed over the pursuers in sight that are
    still closing: (1 - miss / 2 m) x (1 - time to it / 3 s), each part clipped to 0..1. The
    shaping potential is -Reward.danger x this, so the pilot is paid at once for opening up a
    predicted miss (and charged for closing one), long before any hit. Potential-based, so the
    best policy is unchanged (Ng, Harada & Russell 1999)."""
    x, y, vx, vy, on = f.pursuers()
    dx, dy = x - f.x[:, None], y - f.y[:, None]
    rvx, rvy = vx - f.vx[:, None], vy - f.vy[:, None]
    v2 = np.maximum(rvx * rvx + rvy * rvy, 1e-6)
    t = -(dx * rvx + dy * rvy) / v2
    miss = np.hypot(dx + rvx * np.maximum(t, 0), dy + rvy * np.maximum(t, 0))
    on = on & (np.hypot(dx, dy) < SIGHT) & (t > 0)
    d = np.clip(1 - miss / DANGER_MISS, 0, 1) * np.clip(1 - t / DANGER_T, 0, 1)
    return np.where(on, d, 0.0).sum(1)


def rewards(f: Fleet, before: np.ndarray, rw: Reward, was_flying, d_before=None, gamma=0.995):
    r = rw.progress * (before - f.distance()) - rw.time
    if rw.danger:  # gamma x phi(next) - phi(now), phi = -danger x c; phi = 0 once it's over
        d_now = np.where(f.outcome == FLYING, danger(f), 0.0)
        r = r - rw.danger * (gamma * d_now - d_before)
    if rw.near:
        x, y, _, _, on = f.pursuers()
        d = np.hypot(x - f.x[:, None], y - f.y[:, None])
        r = r - rw.near * np.where(on, np.clip(1.0 - d / NEAR_R, 0.0, 1.0), 0.0).sum(1)
    r = r + np.where(f.outcome == LANDED, rw.landed, 0.0)
    r = r + np.where(np.isin(f.outcome, FAILED), rw.failed, 0.0)
    return np.where(was_flying, r, 0.0)


# --- Stable-Baselines3 vectorised env ----------------------------------------------------------
@dataclass(frozen=True)
class Config:
    physics: Physics = Physics()
    weather: Weather = ROUGH
    features: Features = field(default=FULL)
    reward: Reward = Reward()


try:  # SB3 is only installed for training; the tests and CI use Fleet directly
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:  # pragma: no cover
    VecEnv = object


class FleetEnv(VecEnv):
    """N drones as one Stable-Baselines3 VecEnv: a finished flight is replaced by a new course
    in the same slot at once. `hide` ("threats", "frogs") blanks those inputs: the blindfold."""

    def __init__(self, n: int, config: Config | None = None, seed: int = 0, hide=(), mix=None):
        self.config = cfg = config or Config()
        self.mix = mix
        self.fleet = Fleet(n, cfg.physics, cfg.features)
        self.rng = np.random.default_rng(seed)
        self.hide = tuple(hide)
        feat = cfg.features
        self.memory = feat.memory == "hist"
        self.predict = feat.predict
        self.width = frame_width(feat)
        dim = self.width * feat.stack + (HIST * K_FROGS * N_HIST if self.memory else 0)
        self.render_mode = None
        obs_space = gym.spaces.Box(-5.0, 5.0, (dim,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        if VecEnv is not object:
            super().__init__(n, obs_space, act_space)
        else:
            self.num_envs, self.observation_space, self.action_space = n, obs_space, act_space
        self.noise_scale = np.zeros(n)
        self.pending = [np.zeros((n, self.width)) for _ in range(feat.delay)]
        self.frames = np.zeros((n, feat.stack, self.width))
        self.hist = np.zeros((n, HIST_LEN, K_FROGS * N_HIST))
        self._actions = np.zeros((n, 2))
        self.auto_reset = True
        self.seeds = np.zeros(n, np.int64)
        self.drill = np.zeros(n, np.int64)  # which entry of the mix each course came from

    def _sense(self) -> np.ndarray:
        f, feat = self.fleet, self.config.features
        noise = None
        if feat.noise:
            noise = (
                self.rng.standard_normal((f.n, self.width))
                * (feat.noise * self.noise_scale)[:, None]
            )
        return observe(f, noise, self.hide, self.predict)

    def _push(self) -> np.ndarray:
        now = self._sense()
        if self.config.features.delay:
            self.pending.append(now)
            now = self.pending.pop(0)
        self.frames = np.concatenate([now[:, None, :], self.frames[:, :-1]], 1)
        if self.memory:
            off = frog_offsets(now, self.predict)
            self.hist = np.concatenate([off[:, None, :], self.hist[:, :-1]], 1)
        return self._flat()

    def _fill(self, idx: np.ndarray) -> None:
        now = self._sense()
        for p in self.pending:
            p[idx] = now[idx]
        self.frames[idx] = now[idx][:, None, :]
        if self.memory:
            self.hist[idx] = frog_offsets(now[idx], self.predict)[:, None, :]

    def _flat(self) -> np.ndarray:
        parts = [self.frames.reshape(self.num_envs, -1)]
        if self.memory:
            parts.append(self.hist[:, ::HIST_EVERY].reshape(self.num_envs, -1))
        return np.clip(np.concatenate(parts, 1), -5.0, 5.0).astype(np.float32)

    def _new_course(self, idx: np.ndarray, seeds=None) -> None:
        """New courses in slots idx: from the config's features, or (training, with a mix of
        (weight, features) set) from one of the mix's, picked at random each time."""
        cfg = self.config
        for j, i in enumerate(idx):
            seed = int(self.rng.integers(0, 2**31)) if seeds is None else int(seeds[j])
            self.seeds[i] = seed
            feat, self.drill[i] = cfg.features, 0
            if self.mix and seeds is None:
                w = np.array([m[0] for m in self.mix])
                self.drill[i] = int(self.rng.choice(len(w), p=w / w.sum()))
                feat = self.mix[self.drill[i]][1]
            self.fleet.set_course(int(i), make_course(seed, cfg.weather, feat, cfg.physics))
        self.noise_scale[idx] = self.rng.uniform(0.5, 1.5, len(idx))

    def reset(self):
        idx = np.arange(self.num_envs)
        self._new_course(idx)
        self._fill(idx)
        return self._flat()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self._actions = np.asarray(actions, float)

    def step_wait(self):
        f, cfg = self.fleet, self.config
        before = f.distance()
        flying = f.outcome == FLYING
        d_before = danger(f) if cfg.reward.danger else None
        f.step(self._actions)
        r = rewards(f, before, cfg.reward, flying, d_before).astype(np.float32)
        done_idx = np.nonzero(f.outcome != FLYING)[0]
        dones = f.outcome != FLYING
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        obs = self._push()
        if len(done_idx):
            frog_fail = f.frog_failed()
        for i in done_idx:
            hostile = f.p_hostile[i] & ~f.p_done[i] & f.p_on[i]  # still after it at the end
            infos[i] = {
                "outcome": OUTCOMES[f.outcome[i]],
                "time": f.ticks[i] * f.ph.dt,
                "length": f.courses[i]["length"],
                "drill": int(self.drill[i]),
                "beaten": f.beaten[i].tolist(),
                # aimed birds / missiles that got it (a hit counts against the dial)
                "caught": [int(f.outcome[i] == BIRD), int(f.outcome[i] == MISSILE)],
                "open": [int(hostile[:K_BIRDS].sum()), int(hostile[K_BIRDS:].sum())],
                # frogs: got past every frog on the course (the pad's: landed), or failed one
                "frogs": [
                    int(f.frog_on[i].any() and (f.frog_past[i] == f.frog_on[i]).all()),
                    int(frog_fail[i]),
                ],
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
        return [getattr(self, attr_name, None)] * len(self._idx(indices))

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
