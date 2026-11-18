"""Cube to flag: a tiny 2D arena with momentum, walls, lava and coins.

Coordinates are in arena units with y pointing up, origin bottom-left.
The agent is a square cube that accelerates in one of four directions.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass(frozen=True)
class Layout:
    width: float = 10.0
    height: float = 10.0
    start: tuple[float, float] = (1.0, 1.0)
    flag: tuple[float, float] = (1.5, 8.6)
    flag_radius: float = 0.45
    walls: tuple[Rect, ...] = ((4.8, 0.0, 5.2, 2.4),)
    lava: tuple[Rect, ...] = (
        (0.0, 4.2, 7.5, 5.2),  # band across the middle, gap on the right
        (3.5, 7.2, 5.5, 10.0),  # pool at the top, corridor below it
    )
    coins: tuple[tuple[float, float], ...] = (
        (3.5, 3.3),
        (8.7, 2.8),
        (8.7, 6.2),
        (6.5, 6.2),
        (2.5, 6.2),
    )


@dataclass(frozen=True)
class Physics:
    dt: float = 0.05
    accel: float = 14.0
    drag: float = 0.9  # velocity multiplier per step
    max_speed: float = 4.0
    half: float = 0.25  # half-size of the cube
    lava_margin: float = 0.1  # how far the cube may overlap lava before it counts


@dataclass(frozen=True)
class Rewards:
    step: float = -0.005
    coin: float = 1.0
    flag: float = 10.0
    lava: float = -3.0
    # Potential-based shaping: reward for reducing the walking distance to the flag
    # (around walls and lava, not a straight line). 0 = sparse rewards only.
    shaping: float = 0.5


@dataclass
class Config:
    layout: Layout = field(default_factory=Layout)
    physics: Physics = field(default_factory=Physics)
    rewards: Rewards = field(default_factory=Rewards)
    max_steps: int = 500
    start_jitter: float = 0.3


ACTIONS = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.float64)
ACTION_NAMES = ("none", "up", "down", "left", "right")


def _overlaps(cx: float, cy: float, half: float, r: Rect) -> bool:
    return cx + half > r[0] and cx - half < r[2] and cy + half > r[1] and cy - half < r[3]


def distance_field(layout: Layout, half: float, res: float = 0.1) -> np.ndarray:
    """Walking distance from every grid cell to the flag (Dijkstra, 8-connected).

    Cells the cube cannot occupy (walls or lava, inflated by the cube size) are inf.
    """
    nx, ny = int(round(layout.width / res)), int(round(layout.height / res))
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    blocked = np.zeros((nx, ny), dtype=bool)
    for r in layout.walls + layout.lava:
        blocked |= (gx + half > r[0]) & (gx - half < r[2]) & (gy + half > r[1]) & (gy - half < r[3])

    dist = np.full((nx, ny), np.inf)
    fx = min(int(layout.flag[0] / res), nx - 1)
    fy = min(int(layout.flag[1] / res), ny - 1)
    dist[fx, fy] = 0.0
    heap = [(0.0, fx, fy)]
    steps = [(dx, dy, res * np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for dx, dy, cost in steps:
            a, b = i + dx, j + dy
            if 0 <= a < nx and 0 <= b < ny and not blocked[a, b] and d + cost < dist[a, b]:
                dist[a, b] = d + cost
                heapq.heappush(heap, (d + cost, a, b))
    return dist


class CubeArena(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        n_coins = len(self.cfg.layout.coins)
        # x, y, vx, vy (normalised) + one flag per coin (1 = still there)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4 + n_coins,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.coins_left = np.ones(n_coins, dtype=bool)
        self.steps = 0
        # Inflate obstacles a little less than the cube so every live position has a finite value.
        self._res = 0.1
        self._dist = distance_field(self.cfg.layout, half=0.1, res=self._res)
        self._dist[np.isinf(self._dist)] = self._dist[np.isfinite(self._dist)].max()
        self._d = 0.0

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        j = self.cfg.start_jitter
        self.pos = np.array(self.cfg.layout.start) + self.np_random.uniform(-j, j, size=2)
        self.vel = np.zeros(2)
        self.coins_left[:] = True
        self.steps = 0
        self._d = self.walk_distance()
        return self._obs(), {}

    def step(self, action: int):
        p, rw, lay = self.cfg.physics, self.cfg.rewards, self.cfg.layout
        self.steps += 1

        self.vel = (self.vel + ACTIONS[int(action)] * p.accel * p.dt) * p.drag
        speed = float(np.hypot(*self.vel))
        if speed > p.max_speed:
            self.vel *= p.max_speed / speed
        self._move_axis(0, self.vel[0] * p.dt)
        self._move_axis(1, self.vel[1] * p.dt)

        d = self.walk_distance()
        reward = rw.step + rw.shaping * (self._d - d)
        self._d = d
        info: dict = {"coin": None}
        terminated = False

        for i, (cx, cy) in enumerate(lay.coins):
            if self.coins_left[i] and abs(self.pos[0] - cx) < 0.45 and abs(self.pos[1] - cy) < 0.45:
                self.coins_left[i] = False
                reward += rw.coin
                info["coin"] = i

        inner = p.half - p.lava_margin
        if any(_overlaps(self.pos[0], self.pos[1], inner, r) for r in lay.lava):
            reward += rw.lava
            terminated = True
            info["outcome"] = "lava"
        elif (
            np.hypot(self.pos[0] - lay.flag[0], self.pos[1] - lay.flag[1])
            < lay.flag_radius + p.half
        ):
            reward += rw.flag
            terminated = True
            info["outcome"] = "flag"

        truncated = not terminated and self.steps >= self.cfg.max_steps
        if truncated:
            info["outcome"] = "timeout"
        return self._obs(), float(reward), terminated, truncated, info

    # -- internals ---------------------------------------------------------
    def walk_distance(self) -> float:
        i = min(int(self.pos[0] / self._res), self._dist.shape[0] - 1)
        j = min(int(self.pos[1] / self._res), self._dist.shape[1] - 1)
        return float(self._dist[i, j])

    def _move_axis(self, axis: int, delta: float) -> None:
        """Move along one axis and stop against walls or the arena edge."""
        p, lay = self.cfg.physics, self.cfg.layout
        new = self.pos.copy()
        new[axis] += delta
        limit = lay.width if axis == 0 else lay.height
        if new[axis] - p.half < 0 or new[axis] + p.half > limit:
            new[axis] = np.clip(new[axis], p.half, limit - p.half)
            self.vel[axis] = 0.0
        for r in lay.walls:
            if _overlaps(new[0], new[1], p.half, r):
                lo, hi = (r[0], r[2]) if axis == 0 else (r[1], r[3])
                new[axis] = lo - p.half - 1e-6 if delta > 0 else hi + p.half + 1e-6
                self.vel[axis] = 0.0
        self.pos = new

    def _obs(self) -> np.ndarray:
        lay, p = self.cfg.layout, self.cfg.physics
        base = [
            self.pos[0] / lay.width * 2 - 1,
            self.pos[1] / lay.height * 2 - 1,
            self.vel[0] / p.max_speed,
            self.vel[1] / p.max_speed,
        ]
        return np.array(base + self.coins_left.astype(float).tolist(), dtype=np.float32)

    def describe(self) -> dict:
        """Arena description for the browser replay viewer."""
        lay, p = self.cfg.layout, self.cfg.physics
        return {
            "width": lay.width,
            "height": lay.height,
            "start": list(lay.start),
            "flag": {"x": lay.flag[0], "y": lay.flag[1], "r": lay.flag_radius},
            "walls": [list(r) for r in lay.walls],
            "lava": [list(r) for r in lay.lava],
            "coins": [list(c) for c in lay.coins],
            "cube": p.half * 2,
            "dt": p.dt,
        }
