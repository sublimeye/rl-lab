"""Hide and seek (post 006): a chaser that only sees along clear lines of sight.

The runner still sees everything (like a person playing). The chaser sees the runner only
when the segment between their centres misses every obstacle. What else it gets depends on
`sight`:

  xray  sees the runner through walls (the reference: post 5's chaser view)
  none  sees it only in line of sight, remembers nothing
  last  + where and when it last saw the runner
  map   + a "possibility map": every cell the runner could have reached since then without
        being seen. It starts as everything out of sight, grows at the runner's top speed and
        is wiped wherever the chaser can see. Each cell is weighted by how long ago the
        chaser last saw it (a spot it looked at a second ago is unlikely, one it never saw is
        fully likely). The network gets it summarised by direction: for each of the 8 first
        steps on the grid, the share of the weight that lies that way and the best spot
        (likely and close) that way.

The possibility map is plain code (bookkeeping, like a person keeping track); the network
learns what to do with it. site/src/lib/tag-sight.js is the browser port.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import maps
from tag_env import (
    ACTIONS,
    Body,
    Chaser,
    ChaserArena,
    NetEvader,
    caught,
    chaser_direction,
    move,
    observe,
)

SIGHTS = ("xray", "none", "last", "map")
NEAR_SCALE = 3.0  # units: nearness feature = exp(-path length / NEAR_SCALE)
FORGET = 2.0  # seconds: a spot it just looked at counts as unlikely, recovering over ~this
NEVER = 10_000  # age (steps) of a cell it has never seen
_BIG = 30000  # grid distances at or above this are unreachable


def obs_size(sight: str, rays: int = 16) -> int:
    return 4 + rays + 6 + (6 if sight in ("last", "map") else 0) + (17 if sight == "map" else 0)


# ---------------------------------------------------------------- line of sight


def sees(ax: float, ay: float, bx: float, by: float, a) -> bool:
    """Does the segment from (ax, ay) to (bx, by) miss every obstacle? (grazing counts as
    blocked). Squared distances only, so the browser gets bit-identical answers."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    for cx, cy, r in a.pillars:
        t = 0.0 if L2 == 0 else min(1.0, max(0.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
        qx, qy = ax + t * dx - cx, ay + t * dy - cy
        if qx * qx + qy * qy < r * r:
            return False
    for x0, y0, x1, y1 in a.blocks:
        lo, hi = 0.0, 1.0
        for o, d, mn, mx in ((ax, dx, x0, x1), (ay, dy, y0, y1)):
            if abs(d) < 1e-12:
                if o < mn or o > mx:
                    lo, hi = 1.0, 0.0
                    break
                continue
            t1, t2 = (mn - o) / d, (mx - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            lo, hi = max(lo, t1), min(hi, t2)
        if lo <= hi:
            return False
    return True


class Grid:
    """The bot grid of a map (maps.NavGrid) plus what sight and tracking need."""

    def __init__(self, nav: maps.NavGrid):
        self.nav = nav
        nx, ny = nav.nx, nav.ny
        self.free = nav.free.reshape(-1).copy()
        k = np.arange(nx * ny)
        self.cx = (k % nx + 0.5) * nav.cell
        self.cy = (k // nx + 0.5) * nav.cell
        self.oks = [nav._move_ok(di, dj) for di, dj in maps._NB]
        # cells every other cell can be reached from (the open area a square walks in)
        reach = nav.dist < _BIG
        self.main = self.free & (reach.sum(axis=1) == reach.sum(axis=1).max())
        # first-step neighbours of every cell, in ACTIONS order; -1 where the move is blocked
        nb = np.full((nx * ny, 8), -1, dtype=np.int64)
        for c in np.flatnonzero(self.free):
            j, i = divmod(int(c), nx)
            for d, (di, dj) in enumerate(maps._NB):
                ii, jj = i + di, j + dj
                if not (0 <= ii < nx and 0 <= jj < ny) or not nav.free[jj, ii]:
                    continue
                if di and dj and not (nav.free[j, ii] and nav.free[jj, i]):
                    continue
                nb[c, d] = jj * nx + ii
        self.nb = nb
        P, B = maps._arrays(nav.a)
        self.P, self.B = P, B
        # Look points: where a runner's centre can be inside each open cell (the quarter
        # centres and the centre, where a whole square fits). A cell counts as seen only when
        # all of them are in sight, so a runner tucked into a corner of a cell isn't missed.
        a, h, q = nav.a, maps.BODY_HALF, nav.cell / 4
        px, py, owner = [], [], []
        for c in np.flatnonzero(self.free):
            c = int(c)
            pts = [
                (self.cx[c] + ox, self.cy[c] + oy)
                for ox, oy in ((-q, -q), (q, -q), (-q, q), (q, q), (0.0, 0.0))
            ]
            pts = [
                (x, y)
                for x, y in pts
                if h <= x <= a.width - h
                and h <= y <= a.height - h
                and not maps.square_hits(x, y, h, a)
            ] or [(self.cx[c], self.cy[c])]
            for x, y in pts:
                px.append(x)
                py.append(y)
                owner.append(c)
        self.px, self.py = np.array(px), np.array(py)
        self.owner = np.array(owner, dtype=np.int64)

    def visible(self, x: float, y: float) -> np.ndarray:
        """Open cells whose look points are all in line of sight from (x, y)."""
        hidden = ~self._sees_points(x, y)
        vis = self.free.copy()
        vis[self.owner[hidden]] = False
        return vis

    def _sees_points(self, x: float, y: float) -> np.ndarray:
        """sees() from (x, y) to every look point, vectorised with the same maths."""
        X, Y = self.px, self.py
        dx, dy = X - x, Y - y
        L2 = dx * dx + dy * dy
        ok = np.ones(len(X), dtype=bool)
        for cx, cy, r in self.P:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(L2 == 0, 0.0, ((cx - x) * dx + (cy - y) * dy) / L2)
            t = np.minimum(1.0, np.maximum(0.0, t))
            qx, qy = x + t * dx - cx, y + t * dy - cy
            ok &= ~(qx * qx + qy * qy < r * r)
        small_x, small_y = np.abs(dx) < 1e-12, np.abs(dy) < 1e-12
        for x0, y0, x1, y1 in self.B:
            lo = np.zeros_like(X)
            hi = np.ones_like(X)
            for o, d, small, mn, mx in ((x, dx, small_x, x0, x1), (y, dy, small_y, y0, y1)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1, t2 = (mn - o) / d, (mx - o) / d
                tl, th = np.minimum(t1, t2), np.maximum(t1, t2)
                inside = mn <= o <= mx
                tl = np.where(small, -np.inf if inside else np.inf, tl)
                th = np.where(small, np.inf if inside else -np.inf, th)
                lo, hi = np.maximum(lo, tl), np.minimum(hi, th)
            ok &= ~(lo <= hi)
        return ok

    def dilate(self, m: np.ndarray) -> np.ndarray:
        """Every cell one grid step (no corner cutting) from a cell in m, plus m."""
        ny, nx = self.nav.ny, self.nav.nx
        P = m.reshape(ny, nx)
        new = P.copy()
        for (di, dj), ok in zip(maps._NB, self.oks, strict=True):
            src = P[max(0, -dj) : ny - max(0, dj), max(0, -di) : nx - max(0, di)]
            dst = (slice(max(0, dj), ny - max(0, -dj)), slice(max(0, di), nx - max(0, -di)))
            new[dst] |= src & ok[dst]
        return new.reshape(-1)


_GRIDS: dict[int, Grid] = {}


def grid_for(nav: maps.NavGrid) -> Grid:
    g = _GRIDS.get(id(nav))
    if g is None or g.nav is not nav:
        if len(_GRIDS) > 256:
            _GRIDS.clear()
        g = _GRIDS[id(nav)] = Grid(nav)
    return g


# ---------------------------------------------------------------- what the chaser knows


class Tracker:
    """The chaser's knowledge during one episode: sight, last sighting, possibility map."""

    def __init__(self, grid: Grid, speed: float, dt: float):
        self.g, self.step_len, self.cell, self.dt = grid, speed * dt, grid.nav.cell, dt

    def reset(self, me: Body, prey: Body) -> None:
        self.ever, self.since = False, 0
        self.last = (0.0, 0.0, 0.0, 0.0)
        self.budget = 0.0
        self.possible = self.g.main.copy()
        self.age = np.full(len(self.possible), NEVER, dtype=np.int64)
        self._look(me, prey)

    def update(self, me: Body, prey: Body) -> None:
        self.since += 1
        self.age = np.minimum(self.age + 1, NEVER)
        self.budget += self.step_len
        while self.budget >= self.cell:  # the runner could have moved one more cell
            self.budget -= self.cell
            self.possible = self.g.dilate(self.possible) & self.g.main
        self._look(me, prey)

    def _look(self, me: Body, prey: Body) -> None:
        self._view(me, prey)
        self.age[self.vis] = 0
        if self.seen:
            self.ever, self.since = True, 0
            self.last = (prey.x, prey.y, prey.vx, prey.vy)
            self.possible = np.zeros_like(self.possible)
            self.possible[self.g.nav.cell_of(prey.x, prey.y)] = True
            # it may be anywhere in that cell, so it can be in a neighbour almost at once
            self.possible = self.g.dilate(self.possible) & self.g.main
            self.budget = 0.0
        else:
            self.possible &= ~self.vis
            if not self.possible.any():  # grid rounding lost it: anywhere out of sight
                self.possible = self.g.main & ~self.vis
        f = np.exp(-self.age * (self.dt / FORGET))
        self.weight = np.where(self.possible, 1.0 - f, 0.0)

    def _view(self, me: Body, prey: Body) -> None:
        """Sets vis (cells checked now), los (cells in line of sight) and seen."""
        self.vis = self.los = self.g.visible(me.x, me.y)
        self.seen = sees(me.x, me.y, prey.x, prey.y, self.g.nav.a)

    def by_direction(self, me: Body) -> tuple[list[float], list[float]]:
        """For each first step (8 directions, ACTIONS order): the share of the weight best
        reached that way, and the best spot that way: max of weight * exp(-path / NEAR_SCALE)."""
        g = self.g
        c = g.nav.cell_of(me.x, me.y)
        cells = np.flatnonzero(self.weight > 0)
        mass, near = [0.0] * 8, [0.0] * 8
        if not len(cells):
            return mass, near
        nb = g.nb[c]
        D = np.full((len(cells), 8), _BIG, dtype=np.int64)
        for d in range(8):
            if nb[d] >= 0:
                D[:, d] = g.nav.dist[cells, nb[d]]
        first = D.argmin(axis=1)
        md = D.min(axis=1)
        w = self.weight[cells]
        keep = md < _BIG
        total = float(w[keep].sum())
        score = w * np.exp(-(1 + md) * (self.cell / NEAR_SCALE))
        for d in range(8):
            sel = keep & (first == d)
            if sel.any():
                mass[d] = float(w[sel].sum()) / total
                near[d] = float(score[sel].max())
        return mass, near

    def best_spot(self, me: Body) -> int:
        """The cell the search bot heads for: max of weight * exp(-path / NEAR_SCALE)."""
        nav = self.g.nav
        L = nav.dist[:, nav.cell_of(me.x, me.y)].astype(np.int64)
        score = np.where(L < _BIG, self.weight * np.exp(-L * (self.cell / NEAR_SCALE)), 0.0)
        return int(np.argmax(score))


def seek_observe(me: Body, prey: Body, tr: Tracker, sight: str, p, a, rays: int = 16):
    """The chaser's inputs. Same order and scaling in tag-sight.js."""
    ms = p.max_speed
    out = [me.x / a.width * 2 - 1, me.y / a.height * 2 - 1, me.vx / ms, me.vy / ms]
    out += maps.rays(me.x, me.y, a, rays)
    if sight == "xray" or tr.seen:
        dx, dy = prey.x - me.x, prey.y - me.y
        out += [1.0, dx / a.width, dy / a.height, prey.vx / ms, prey.vy / ms]
        out.append(math.hypot(dx, dy) / math.hypot(a.width, a.height))
    else:
        out += [0.0] * 6
    if sight in ("last", "map"):
        if tr.ever:
            lx, ly, lvx, lvy = tr.last
            out += [1.0, (lx - me.x) / a.width, (ly - me.y) / a.height, lvx / ms, lvy / ms]
            out.append(min(tr.since * p.dt / 20.0, 1.0))
        else:
            out += [0.0] * 6
    if sight == "map":
        mass, near = tr.by_direction(me)
        out.append(float(tr.weight.sum()) / float(tr.g.main.sum()))
        for m, n in zip(mass, near, strict=True):
            out += [m, n]
    return out


# ---------------------------------------------------------------- scripted players


@dataclass(frozen=True)
class Hider:
    """A scripted runner: runs to the farthest spot out of the chaser's sight that it can
    reach first, then stands still until it is seen."""

    replan: int = 10


@dataclass(frozen=True)
class Still:
    """Stands still all episode, starting out of the chaser's sight (a search test)."""


def hider_direction(env: SeekArena) -> tuple[float, float]:
    g, tr, e, c = env.grid, env.tracker, env.evader, env.chaser
    nav = g.nav
    st = env._hide
    if tr.seen or st["target"] is None or (not st["arrived"] and env.steps % env.prey.replan == 0):
        hidden = np.flatnonzero(g.main & ~tr.los)  # out of line of sight
        st["arrived"] = False
        if len(hidden):
            ck, hk = nav.cell_of(c.x, c.y), nav.cell_of(e.x, e.y)
            dc = nav.dist[hidden, ck].astype(np.int64)
            dh = nav.dist[hidden, hk].astype(np.int64)
            score = (dc - dh) * 1000 + dc  # reach it first, then as far from the chaser as can be
            st["target"] = int(hidden[int(np.argmax(score))])
        else:
            st["target"] = None
    if st["target"] is None:
        dx, dy = e.x - c.x, e.y - c.y
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n
    tx, ty = nav.aim[st["target"]]
    if st["arrived"] or math.hypot(tx - e.x, ty - e.y) < 0.15:
        st["arrived"] = True
        return 0.0, 0.0
    return nav.direction(e.x, e.y, tx, ty, cl=0.36)


@dataclass(frozen=True)
class SeekBot:
    """Scripted chasers for comparison. All intercept the runner while they can see it.
    xray: always sees it (post 5's bot). lastseen: goes to where it last saw it, then waits.
    search: heads for the likeliest close spot on the possibility map (weight times
    exp(-path / 3)), so it peeks the nearest corner it hasn't just checked."""

    kind: str = "search"


def seekbot_direction(bot: SeekBot, env: SeekArena) -> tuple[float, float]:
    tr, c, e, p = env.tracker, env.chaser, env.evader, env.cfg.physics
    if bot.kind == "xray" or tr.seen:
        return chaser_direction(Chaser("intercept", env.ratio), c, e, p, env.arena, env.grid.nav)
    nav = env.grid.nav
    if bot.kind == "lastseen":
        if not tr.ever:
            return 0.0, 0.0
        tx, ty = tr.last[0], tr.last[1]
        if math.hypot(tx - c.x, ty - c.y) < 0.2:
            return 0.0, 0.0
        return nav.direction(c.x, c.y, tx, ty, cl=0.36)
    if not tr.weight.any():
        return 0.0, 0.0
    tx, ty = nav.aim[tr.best_spot(c)]
    return nav.direction(c.x, c.y, tx, ty, cl=0.36)


# ---------------------------------------------------------------- the environment


class SeekArena(ChaserArena):
    """The chaser's side of tag with line of sight. cfg.evaders returns a NetEvader, a Hider
    or a Still runner. options "bot": SeekBot plays the chaser instead of the agent."""

    def __init__(self, config, sight: str = "map"):
        super().__init__(config)
        assert sight in SIGHTS
        self.sight = sight
        n = obs_size(sight, self.cfg.rays)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(n,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        self.prey = options.get("prey") or self.cfg.evaders(self.np_random)
        self.bot = options.get("bot")
        self.ratio = options.get("ratio") or float(self.np_random.uniform(*self.cfg.chaser_ratio))
        self._pick_map(options, fresh=seed is not None)
        self.grid = grid_for(self.nav)
        p = self.cfg.physics
        self.tracker = Tracker(self.grid, p.max_speed, p.dt)
        for _ in range(100):
            self._spawn(options)
            if "evader" in options or not isinstance(self.prey, Still):
                break
            if not sees(self.chaser.x, self.chaser.y, self.evader.x, self.evader.y, self.arena):
                break
        self._hide = {"target": None, "arrived": False}
        self.tracker.reset(self.chaser, self.evader)
        self.seen_steps = 0
        return self._obs(), {}

    def _prey_dir(self):
        p, a = self.cfg.physics, self.arena
        if isinstance(self.prey, NetEvader):
            rng = self.np_random if self.prey.sample else None
            obs = observe(self.evader, self.chaser, p, a, self.cfg.rays)
            return ACTIONS[self.prey.net.act(obs, rng)]
        if isinstance(self.prey, Hider):
            return hider_direction(self)
        return 0.0, 0.0

    def step(self, action: int):
        p, a, rw = self.cfg.physics, self.arena, self.cfg.rewards
        ex, ey = self._prey_dir()
        cx, cy = seekbot_direction(self.bot, self) if self.bot else ACTIONS[int(action)]
        self.steps += 1
        move(self.evader, ex, ey, p.max_speed, p, a)
        move(self.chaser, cx, cy, p.max_speed * self.ratio, p, a)
        self.tracker.update(self.chaser, self.evader)
        self.seen_steps += self.tracker.seen
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
        o = seek_observe(
            self.chaser,
            self.evader,
            self.tracker,
            self.sight,
            self.cfg.physics,
            self.arena,
            self.cfg.rays,
        )
        return np.array(o, dtype=np.float32)
