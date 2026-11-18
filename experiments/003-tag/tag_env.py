"""Tag: a square learns to run away from a slower chaser.

Coordinates are arena units, y up, origin bottom-left. Two square bodies (the evader we
train and a scripted chaser) share the same momentum physics; the chaser's top speed is
`ratio` times the evader's. The arena has round pillars to hide behind.

The simulation uses plain Python floats (not numpy) so the browser port in
site/src/lib/tag-sim.js can reproduce it step for step.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Pillar = tuple[float, float, float]  # cx, cy, radius


@dataclass(frozen=True)
class Arena:
    width: float = 12.0
    height: float = 8.0
    pillars: tuple[Pillar, ...] = ((4.0, 4.0, 0.9), (8.0, 4.0, 0.9))


@dataclass(frozen=True)
class Physics:
    dt: float = 0.05
    accel: float = 14.0
    drag: float = 0.9  # velocity multiplier per step
    max_speed: float = 4.0  # the evader's; the chaser gets ratio * this
    half: float = 0.3  # half-size of both squares


@dataclass(frozen=True)
class Rewards:
    alive: float = 0.01  # every step not caught
    caught: float = -5.0
    # Per step, times min(distance, distance_cap) / distance_cap. 0 = no distance bonus.
    distance: float = 0.01
    distance_cap: float = 6.0


@dataclass(frozen=True)
class Chaser:
    """A scripted chaser. Every `reaction` steps it picks a direction and holds it."""

    kind: str = "pursuit"  # pursuit: run at the evader; intercept: aim where it is heading
    ratio: float = 0.85  # top speed relative to the evader
    reaction: int = 1  # steps between decisions (1 step = 0.05 s)
    dirs8: bool = False  # snap to 8 directions, like a keyboard player
    lead: float = 1.0  # intercept only: scales how far ahead it aims


def sample_chaser(pool: str, rng: np.random.Generator) -> Chaser:
    """Training opponents. 'simple' is one fixed chaser; 'mixed' varies everything."""
    if pool == "simple":
        return Chaser()
    if pool == "mixed":
        return Chaser(
            kind="pursuit" if rng.random() < 0.5 else "intercept",
            ratio=float(rng.uniform(0.75, 1.0)),
            reaction=int(rng.integers(1, 6)),
            dirs8=bool(rng.random() < 0.5),
            lead=float(rng.uniform(0.5, 1.5)),
        )
    raise ValueError(f"unknown chaser pool {pool!r}")


class MLP:
    """A policy network as numpy arrays: tanh hidden layers, linear action logits.

    Same format as the JSON the browser loads: [{"w": out x in, "b": out}, ...].
    """

    def __init__(self, layers: list[dict]):
        self.layers = [(np.asarray(L["w"], dtype=np.float64), np.asarray(L["b"])) for L in layers]

    def to_json(self, digits: int | None = 5) -> dict:
        def r(a):
            return np.round(a, digits).tolist() if digits else a.tolist()

        return {"layers": [{"w": r(w), "b": r(b)} for w, b in self.layers]}

    def logits(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def act(self, obs, rng: np.random.Generator | None = None) -> int:
        """Best action, or a sample from the softmax when rng is given."""
        z = self.logits(obs)
        if rng is None:
            return int(np.argmax(z))
        p = np.exp(z - z.max())
        return int(rng.choice(len(p), p=p / p.sum()))


@dataclass(frozen=True)
class NetChaser:
    """A chaser driven by a trained network instead of a script."""

    net: MLP
    ratio: float = 0.9
    sample: bool = False
    reaction: int = 1


@dataclass(frozen=True)
class NetEvader:
    net: MLP
    sample: bool = False


@dataclass
class Config:
    arena: Arena = field(default_factory=Arena)
    physics: Physics = field(default_factory=Physics)
    rewards: Rewards = field(default_factory=Rewards)
    max_steps: int = 400  # 20 seconds
    chasers: str = "simple"
    min_start_distance: float = 4.0
    # Optional opponent samplers (self-play). They override `chasers` for TagArena, and
    # supply the evader for ChaserArena.
    opponents: Callable[[np.random.Generator], Chaser | NetChaser] | None = None
    evaders: Callable[[np.random.Generator], NetEvader] | None = None
    chaser_ratio: tuple[float, float] = (0.8, 1.0)  # ChaserArena: own speed, drawn per episode


# 0 = stand still, then 8 directions counter-clockwise from east.
_D = math.sqrt(0.5)
ACTIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 0.0),
    (_D, _D),
    (0.0, 1.0),
    (-_D, _D),
    (-1.0, 0.0),
    (-_D, -_D),
    (0.0, -1.0),
    (_D, -_D),
)


class Body:
    __slots__ = ("x", "y", "vx", "vy")

    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy

    def state(self) -> list[float]:
        return [self.x, self.y, self.vx, self.vy]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def move(b: Body, ax: float, ay: float, max_speed: float, p: Physics, a: Arena) -> None:
    """One physics step: accelerate along (ax, ay), cap speed, move, collide."""
    b.vx = (b.vx + ax * p.accel * p.dt) * p.drag
    b.vy = (b.vy + ay * p.accel * p.dt) * p.drag
    speed = math.hypot(b.vx, b.vy)
    if speed > max_speed:
        b.vx *= max_speed / speed
        b.vy *= max_speed / speed
    b.x += b.vx * p.dt
    b.y += b.vy * p.dt
    _edges(b, p.half, a)
    for cx, cy, r in a.pillars:
        qx = _clamp(cx, b.x - p.half, b.x + p.half)  # closest point of the square to the centre
        qy = _clamp(cy, b.y - p.half, b.y + p.half)
        dx, dy = qx - cx, qy - cy
        d = math.hypot(dx, dy)
        if d >= r:
            continue
        if d < 1e-9:  # centre inside the square: push along centre-to-centre
            dx, dy = b.x - cx, b.y - cy
            d = math.hypot(dx, dy) or 1.0
            push = r + p.half - d
        else:
            push = r - d
        nx, ny = dx / d, dy / d
        b.x += nx * push
        b.y += ny * push
        vn = b.vx * nx + b.vy * ny
        if vn < 0:
            b.vx -= vn * nx
            b.vy -= vn * ny
    _edges(b, p.half, a)


def _edges(b: Body, h: float, a: Arena) -> None:
    if b.x < h or b.x > a.width - h:
        b.x = _clamp(b.x, h, a.width - h)
        b.vx = 0.0
    if b.y < h or b.y > a.height - h:
        b.y = _clamp(b.y, h, a.height - h)
        b.vy = 0.0


def steer(fx: float, fy: float, tx: float, ty: float, a: Arena, clearance: float):
    """Unit direction from (fx, fy) toward (tx, ty), going around the nearest blocking pillar."""
    gx, gy = tx - fx, ty - fy
    glen = math.hypot(gx, gy)
    if glen < 1e-9:
        return 0.0, 0.0
    gx, gy = gx / glen, gy / glen
    best = None
    for cx, cy, r in a.pillars:
        R = r + clearance
        px, py = cx - fx, cy - fy
        dist = math.hypot(px, py)
        along = px * gx + py * gy
        if along <= 0 or along - R > glen:  # pillar behind us or beyond the target
            continue
        if abs(px * gy - py * gx) >= R:  # path misses it
            continue
        if best is None or dist < best[0]:
            best = (dist, px, py, R)
    if best is None:
        return gx, gy
    dist, px, py, R = best
    ux, uy = px / dist, py / dist
    if dist <= R:
        alpha = math.pi / 2  # touching it: slide along the surface
    else:
        alpha = math.asin(R / dist)
    ca, sa = math.cos(alpha), math.sin(alpha)
    left = (ux * ca - uy * sa, ux * sa + uy * ca)
    right = (ux * ca + uy * sa, -ux * sa + uy * ca)
    if left[0] * gx + left[1] * gy >= right[0] * gx + right[1] * gy:
        return left
    return right


def chaser_direction(c: Chaser, me: Body, prey: Body, p: Physics, a: Arena):
    tx, ty = prey.x, prey.y
    if c.kind == "intercept":
        dist = math.hypot(prey.x - me.x, prey.y - me.y)
        t = min(c.lead * dist / (p.max_speed * c.ratio), 1.5)
        tx = _clamp(prey.x + prey.vx * t, p.half, a.width - p.half)
        ty = _clamp(prey.y + prey.vy * t, p.half, a.height - p.half)
    dx, dy = steer(me.x, me.y, tx, ty, a, clearance=p.half * 1.5)
    if c.dirs8 and (dx or dy):
        ang = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
        dx, dy = math.cos(ang), math.sin(ang)
    return dx, dy


def caught(e: Body, c: Body, h: float) -> bool:
    return abs(e.x - c.x) < 2 * h and abs(e.y - c.y) < 2 * h


def observe(e: Body, c: Body, p: Physics, a: Arena) -> list[float]:
    """9 numbers, all roughly in [-1, 1]. Shared with the browser port."""
    dx, dy = c.x - e.x, c.y - e.y
    return [
        e.x / a.width * 2 - 1,
        e.y / a.height * 2 - 1,
        e.vx / p.max_speed,
        e.vy / p.max_speed,
        dx / a.width,
        dy / a.height,
        c.vx / p.max_speed,
        c.vy / p.max_speed,
        math.hypot(dx, dy) / math.hypot(a.width, a.height),
    ]


class TagArena(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.evader = Body(0, 0)
        self.chaser = Body(0, 0)
        self.opponent = Chaser()
        self.steps = 0
        self._dir = (0.0, 0.0)

    def _free_point(self) -> tuple[float, float]:
        a, h = self.cfg.arena, self.cfg.physics.half
        while True:
            x = float(self.np_random.uniform(h, a.width - h))
            y = float(self.np_random.uniform(h, a.height - h))
            if all(math.hypot(x - cx, y - cy) > r + h * 1.5 for cx, cy, r in a.pillars):
                return x, y

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.opponent = options.get("chaser") or (
            self.cfg.opponents(self.np_random)
            if self.cfg.opponents
            else sample_chaser(self.cfg.chasers, self.np_random)
        )
        self._spawn(options)
        return self._obs(), {"chaser": self.opponent}

    def _spawn(self, options: dict) -> None:
        if "evader" in options:
            self.evader = Body(*options["evader"])
            self.chaser = Body(*options["chaser_pos"])
        else:
            self.evader = Body(*self._free_point())
            while True:
                self.chaser = Body(*self._free_point())
                d = math.hypot(self.chaser.x - self.evader.x, self.chaser.y - self.evader.y)
                if d >= self.cfg.min_start_distance:
                    break
        self.steps = 0
        self._dir = (0.0, 0.0)

    def step(self, action: int):
        p, a, rw = self.cfg.physics, self.cfg.arena, self.cfg.rewards
        if isinstance(self.opponent, NetChaser):
            obs_c = observe(self.chaser, self.evader, p, a)
            rng = self.np_random if self.opponent.sample else None
            self._dir = ACTIONS[self.opponent.net.act(obs_c, rng)]
        elif self.steps % self.opponent.reaction == 0:
            self._dir = chaser_direction(self.opponent, self.chaser, self.evader, p, a)
        self.steps += 1
        ax, ay = ACTIONS[int(action)]
        move(self.evader, ax, ay, p.max_speed, p, a)
        move(self.chaser, self._dir[0], self._dir[1], p.max_speed * self.opponent.ratio, p, a)

        dist = math.hypot(self.chaser.x - self.evader.x, self.chaser.y - self.evader.y)
        info: dict = {}
        if caught(self.evader, self.chaser, p.half):
            info["outcome"] = "caught"
            return self._obs(), rw.caught, True, False, info
        reward = rw.alive + rw.distance * min(dist, rw.distance_cap) / rw.distance_cap
        truncated = self.steps >= self.cfg.max_steps
        if truncated:
            info["outcome"] = "escaped"
        return self._obs(), float(reward), False, truncated, info

    def _obs(self) -> np.ndarray:
        return np.array(
            observe(self.evader, self.chaser, self.cfg.physics, self.cfg.arena), dtype=np.float32
        )

    def describe(self) -> dict:
        a, p = self.cfg.arena, self.cfg.physics
        return {
            "width": a.width,
            "height": a.height,
            "pillars": [list(x) for x in a.pillars],
            "half": p.half,
            "dt": p.dt,
            "accel": p.accel,
            "drag": p.drag,
            "maxSpeed": p.max_speed,
            "maxSteps": self.cfg.max_steps,
        }


class ChaserArena(TagArena):
    """The same game from the chaser's side: the agent chases a network-driven evader.

    Zero-sum: the chaser's reward is the evader's reward with the sign flipped.
    """

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        self.prey = options.get("prey") or self.cfg.evaders(self.np_random)
        self.ratio = options.get("ratio") or float(self.np_random.uniform(*self.cfg.chaser_ratio))
        self._spawn(options)
        return self._obs(), {}

    def step(self, action: int):
        p, a, rw = self.cfg.physics, self.cfg.arena, self.cfg.rewards
        rng = self.np_random if self.prey.sample else None
        ea = self.prey.net.act(observe(self.evader, self.chaser, p, a), rng)
        self.steps += 1
        move(self.evader, *ACTIONS[ea], p.max_speed, p, a)
        cx, cy = ACTIONS[int(action)]
        move(self.chaser, cx, cy, p.max_speed * self.ratio, p, a)

        info: dict = {}
        if caught(self.evader, self.chaser, p.half):
            info["outcome"] = "caught"
            return self._obs(), -rw.caught, True, False, info
        dist = math.hypot(self.chaser.x - self.evader.x, self.chaser.y - self.evader.y)
        reward = -(rw.alive + rw.distance * min(dist, rw.distance_cap) / rw.distance_cap)
        truncated = self.steps >= self.cfg.max_steps
        if truncated:
            info["outcome"] = "escaped"
        return self._obs(), float(reward), False, truncated, info

    def _obs(self) -> np.ndarray:
        return np.array(
            observe(self.chaser, self.evader, self.cfg.physics, self.cfg.arena), dtype=np.float32
        )


def flee_action(e: Body, c: Body) -> int:
    """Hand-written baseline: accelerate straight away from the chaser."""
    dx, dy = e.x - c.x, e.y - c.y
    best, best_dot = 0, -2.0
    n = math.hypot(dx, dy) or 1.0
    for i, (ax, ay) in enumerate(ACTIONS[1:], start=1):
        dot = (ax * dx + ay * dy) / n
        if dot > best_dot:
            best, best_dot = i, dot
    return best
