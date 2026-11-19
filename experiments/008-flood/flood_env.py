"""Flood defence: a digger has to keep a house dry while the tide comes in.

A side view of a strip of coast, `width` columns by `height` rows. The sea is off the left
edge; the house stands on the right edge. Both sand and water are height fields:

- sand: an integer height per column (no holes, no overhangs). A column more than `max_step`
  cells above a neighbour slides one grain down to it per settling pass (the angle of repose).
- water: a depth per column on top of the sand, moved by "virtual pipes" between neighbours:
  a surface difference pushes water through the pipe, the flow keeps some momentum (so water
  arrives as a wave), and no column sends more than it holds. The sea is an endless column
  left of column 0 whose level rises on a schedule (the tide).

The digger stands on top of the sand in one column (it can wade), walks one column per step
and digs or drops one grain in the column to its left or right. It climbs steps of at most
`max_step` cells and carries up to `carry` grains. The house's columns are fixed: no digging,
dropping or walking into them. In round 2 (`house_wall`), sand leaning on the house can't rise
above its wall: the excess spills over the roof and is lost. The episode ends when water
reaches the house floor (flooded) or after `max_steps` steps (dry).

Reward: 1 / max_steps for every step the house is dry, and `flood` once when it floods.
Nothing for moving sand or for walls: building has to pay for itself by keeping the water out.

Plain numpy, elementwise only, so site/src/lib/flood-sim.js reproduces it step for step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Actions: wait, walk left / right, dig left / right, drop left / right.
ACTIONS = ("wait", "left", "right", "dig-left", "dig-right", "drop-left", "drop-right")
WAIT, LEFT, RIGHT, DIG_L, DIG_R, DROP_L, DROP_R = range(7)


@dataclass(frozen=True)
class World:
    width: int = 40
    height: int = 20
    house_width: int = 5  # the house's columns, at the right edge
    house_floor: tuple[int, int] = (6, 9)  # inclusive range, drawn per map
    tide_over: tuple[float, float] = (1.0, 3.0)  # peak tide above the house floor
    tide_start: float = 2.5  # sea level at step 0 (beach sand is 2)
    rise: float = 0.7  # the tide reaches its peak at this fraction of the episode
    beach: tuple[int, int] = (3, 6)  # columns of flat beach at the left, inclusive
    flow: float = 0.2  # how hard a surface difference pushes water through a pipe, per sub-step
    damping: float = 0.95  # how much of its flow a pipe keeps from the last sub-step
    substeps: int = 4  # water sub-steps per agent step
    settle_passes: int = 2  # sand settling passes per agent step
    flood_depth: float = 0.05  # water this deep on any house column floods it
    carry: int = 3  # grains the digger can hold
    max_step: int = 2  # steepest stable sand step, and the highest step the digger can climb
    # Round 2: the house's wall is this many cells tall, and sand piled against it higher than
    # that spills over the roof and is gone. None (round 1): the house is a wall of any height.
    house_wall: int | None = None


@dataclass(frozen=True)
class Rewards:
    dry: float = 1.0  # spread over the episode: dry / max_steps per step
    flood: float = -1.0


@dataclass
class Config:
    world: World = field(default_factory=World)
    rewards: Rewards = field(default_factory=Rewards)
    max_steps: int = 500
    # Curriculum: scales the tide's height above the house floor (1 = full tide).
    tide_scale: float = 1.0


@dataclass
class Map:
    sand: np.ndarray  # int64, per column
    house_floor: int
    peak: float  # the tide's highest level
    start_x: int
    seed: int = -1  # the map seed, if generated


M32 = 0xFFFFFFFF


def mulberry32(seed: int):
    """Small PRNG with the same sequence in site/src/lib/flood-sim.js: floats in [0, 1)."""
    a = seed & M32

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


def make_map(seed: int, w: World, tide_scale: float = 1.0) -> Map:
    """Random coast from a seed: flat beach, a few bumps and dips, the house on the right.

    The terrain in front of the house always stays below the peak tide, so doing nothing
    floods. Plain Python floats, so the browser builds the same map from the same seed.
    """
    rand = mulberry32(seed)

    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * rand()

    def i(lo: int, hi: int) -> int:  # inclusive
        return lo + math.floor(rand() * (hi - lo + 1))

    n = w.width - w.house_width  # the columns the digger can use
    hf = i(w.house_floor[0], w.house_floor[1])
    peak = hf + u(w.tide_over[0], w.tide_over[1]) * tide_scale
    peak = max(peak, hf + 0.3)
    cap = max(2, math.floor(peak - 0.5))
    beach = i(w.beach[0], w.beach[1])

    span = max(1, n - 1 - beach)
    prof = [2.0 + (hf - 2.0) * min(1.0, max(0.0, (x - beach) / span)) for x in range(n)]
    for _ in range(i(1, 3)):
        c = u(beach + 2, n - 2)
        width = u(1.5, 5.0)
        amp = u(-4.0, 5.0)
        for x in range(n):
            z = (x - c) / width
            prof[x] += amp * math.exp(-0.5 * z * z)
    sand = [2 if x < beach else min(cap, max(2, math.floor(prof[x] + 0.5))) for x in range(n)]
    # Lower anything steeper than one cell (generation only removes sand).
    changed = True
    while changed:
        changed = False
        for x in range(n):
            lo = sand[x]
            if x > 0:
                lo = min(lo, sand[x - 1] + 1)
            if x < n - 1:
                lo = min(lo, sand[x + 1] + 1)
            if lo < sand[x]:
                sand[x] = lo
                changed = True
    land = [x for x in range(beach + 1, n) if sand[x] > w.tide_start]
    start_x = land[math.floor(rand() * len(land))] if land else n - 1
    full = np.array(sand + [hf] * w.house_width, dtype=np.int64)
    return Map(full, hf, peak, start_x, seed)


def sea_level(t: int, m: Map, w: World, max_steps: int) -> float:
    frac = min(1.0, t / (w.rise * max_steps))
    return w.tide_start + (m.peak - w.tide_start) * frac


def initial_water(sand: np.ndarray, level: float) -> np.ndarray:
    """Fill from the sea inwards until the first column at or above the sea level."""
    water = np.zeros(len(sand))
    for i, s in enumerate(sand):
        if s >= level:
            break
        water[i] = level - s
    return water


def settle(sand: np.ndarray, n: int, max_step: int) -> None:
    """One settling pass over columns [0, n): a grain slides off any step steeper than
    max_step, in place."""
    s = sand[:n]
    d = s[1:] - s[:-1]
    to_right = (d < -max_step).astype(np.int64)  # left column is higher: one grain slides right
    to_left = (d > max_step).astype(np.int64)
    s[:-1] += to_left - to_right
    s[1:] += to_right - to_left


def flow_water(
    sand: np.ndarray, water: np.ndarray, flux: np.ndarray, sea: float, w: World
) -> tuple[np.ndarray, np.ndarray]:
    """One water sub-step ("virtual pipes"). Returns the new depths and pipe flows.

    Pipe p joins column p-1 (the sea for p = 0) to column p; flux[p] > 0 flows rightwards.
    Flows keep some momentum, so water travels as a wave instead of creeping. A column never
    sends more than it holds; the right edge is closed.
    """
    surf = sand + water
    left = np.empty_like(surf)
    left[0] = sea
    left[1:] = surf[:-1]
    q = flux * w.damping + w.flow * (left - surf)
    out = np.maximum(-q, 0.0)  # column p sends left through pipe p
    out[:-1] += np.maximum(q[1:], 0.0)  # and right through pipe p + 1
    scale = np.where(out > water, water / np.where(out > 0, out, 1.0), 1.0)
    src = np.ones_like(scale)  # scale of the column each pipe draws from, if q > 0
    src[1:] = scale[:-1]
    q = np.where(q > 0, q * src, q * scale)
    new = water + q
    new[:-1] -= q[1:]
    return np.maximum(new, 0.0), q


class FloodWorld(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        w = self.cfg.world
        self.n_obs = 3 * w.width + 4
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(self.n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.n = w.width - w.house_width
        self.map: Map | None = None
        self.sand = np.zeros(w.width, dtype=np.int64)
        self.water = np.zeros(w.width)
        self.flux = np.zeros(w.width)
        self.x = 0
        self.carrying = 0
        self.t = 0
        self.dug = 0
        self.dropped = 0
        self.spilled = 0

    # --- state --------------------------------------------------------------------------
    def load(self, m: Map) -> None:
        w = self.cfg.world
        self.map = m
        self.sand = m.sand.copy()
        self.water = initial_water(self.sand, w.tide_start)
        self.flux = np.zeros(w.width)
        self.x = m.start_x
        self.carrying = 0
        self.t = 0
        self.dug = 0
        self.dropped = 0
        self.spilled = 0

    def sea(self) -> float:
        return sea_level(self.t, self.map, self.cfg.world, self.cfg.max_steps)

    def flooded(self) -> bool:
        return bool(self.water[self.n :].max() > self.cfg.world.flood_depth)

    def observe(self) -> np.ndarray:
        return np.asarray(observe(self), dtype=np.float32)

    # --- gym ----------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        m = options.get("map")
        if m is None:
            seed = options.get("map_seed")
            if seed is None:
                seed = int(self.np_random.integers(2**31))
            m = make_map(seed, self.cfg.world, self.cfg.tide_scale)
        self.load(m)
        return self.observe(), {}

    def step(self, action: int):
        apply_action(self, int(action))
        physics(self)
        T = self.cfg.max_steps
        flooded = self.flooded()
        info = {"dug": self.dug, "dropped": self.dropped, "spilled": self.spilled}
        if flooded:
            info["outcome"] = "flooded"
            return self.observe(), self.cfg.rewards.flood, True, False, info
        truncated = self.t >= T
        if truncated:
            info["outcome"] = "dry"
        return self.observe(), self.cfg.rewards.dry / T, False, truncated, info


def apply_action(env: FloodWorld, a: int) -> None:
    w = env.cfg.world
    n, x = env.n, env.x
    if a in (LEFT, RIGHT):
        nx = x - 1 if a == LEFT else x + 1
        if 0 <= nx < n and env.sand[nx] - env.sand[x] <= w.max_step:
            env.x = nx
    elif a in (DIG_L, DIG_R):
        tx = x - 1 if a == DIG_L else x + 1
        if 0 <= tx < n and env.carrying < w.carry and env.sand[tx] > 1:
            env.sand[tx] -= 1
            env.carrying += 1
            env.dug += 1
    elif a in (DROP_L, DROP_R):
        tx = x - 1 if a == DROP_L else x + 1
        if 0 <= tx < n and env.carrying > 0 and env.sand[tx] < w.height - 1:
            env.sand[tx] += 1
            env.carrying -= 1
            env.dropped += 1


def spill(sand: np.ndarray, n: int, top: int) -> int:
    """Round 2: a grain above the house wall's top, against the wall, spills over the roof.
    Returns how many grains were lost (0 or 1)."""
    if sand[n - 1] > top:
        sand[n - 1] -= 1
        return 1
    return 0


def physics(env: FloodWorld) -> None:
    w = env.cfg.world
    for _ in range(w.settle_passes):
        settle(env.sand, env.n, w.max_step)
        if w.house_wall is not None:
            env.spilled += spill(env.sand, env.n, env.map.house_floor + w.house_wall)
    env.t += 1
    sea = env.sea()
    for _ in range(w.substeps):
        env.water, env.flux = flow_water(env.sand, env.water, env.flux, sea, w)


def observe(env: FloodWorld) -> list[float]:
    """Sand heights, water depths, where the digger is (one-hot), then 4 numbers. Same in JS."""
    w = env.cfg.world
    H = float(w.height)
    pos = [0.0] * w.width
    pos[env.x] = 1.0
    return (
        [float(s) / H for s in env.sand]
        + [min(float(d), H) / H for d in env.water]
        + pos
        + [
            env.carrying / w.carry,
            env.sea() / H,
            env.map.peak / H,
            env.t / env.cfg.max_steps,
        ]
    )


# --- baselines -------------------------------------------------------------------------------
def wall_site(env: FloodWorld) -> int:
    """The scripted builder's wall: the column that needs the fewest grains to stand above the
    peak tide (a pile of height h on flat ground takes about h * h grains)."""
    need = math.floor(env.map.peak) + 1
    s = env.sand[: env.n - 2].astype(np.float64)
    cost = np.maximum(need - s, 0.0) ** 2 + 0.01 * (env.n - np.arange(len(s)))  # ties: near house
    return int(np.argmin(cost))


def builder_action(env: FloodWorld, c: int) -> int:
    """A hand-written builder: digs well away from column c (beyond the foot of the finished
    pile, on whichever side has sand) and drops onto c until it stands above the peak tide.
    A reference for what a simple strategy scores."""
    w = env.cfg.world
    need = math.floor(env.map.peak) + 1
    x = env.x

    def go(target: int) -> int:
        return WAIT if target == x else (RIGHT if target > x else LEFT)

    if env.sand[c] >= need and env.carrying == 0:
        return WAIT
    if env.carrying == w.carry or (env.carrying > 0 and env.sand[c] >= need):
        stand = c + 1 if c + 1 < env.n else c - 1  # drop from the house side if there's room
        if x == stand:
            return DROP_L if stand > c else DROP_R
        return go(stand)
    lo = int(env.map.sand[max(0, c - 4) : c + 5].min())
    k = max(3, (need - lo + 1) // w.max_step + 2)  # beyond the foot of the finished pile
    spots = [d for d in range(c + k, env.n) if env.sand[d] > 1 and env.water[d] < 0.5]
    spots += [d for d in range(c - k, -1, -1) if env.sand[d] > 1]
    if not spots:
        return WAIT
    d = min(spots, key=lambda d: abs(d - x))
    stand = d - 1 if d > c else d + 1
    if x == stand:
        return DIG_R if d > stand else DIG_L
    return go(stand)
