"""Random tag maps (post 005): generator, navigability check, ray sensors, bot navigation.

Every map is the 12 x 8 arena plus a few obstacles. Each obstacle is an island: a round
pillar, a block, or a straight or L-shaped wall, kept at least GAP away from every other
obstacle and from the outer walls. GAP is two squares wide, so a square can always go around
any obstacle, no area is closed off, and there are no dead-end corridors. `navigable()` then
checks it with a flood fill at the real square size.

The generator uses its own small PRNG (mulberry32) so site/src/lib/tag-sim.js builds the
exact same map from the same seed; site/test/tag-sim.test.mjs checks that.
"""

from __future__ import annotations

import math
from collections import deque
from functools import lru_cache

import numpy as np

from tag_env import Arena

FAMILIES = ("pillars", "blocks", "walls", "maze", "mixed")
GAP = 1.2  # clear space around every obstacle (two squares)
WALL = 0.3  # wall thickness
TRIES = 60  # placement attempts per obstacle
BODY_HALF = 0.3
M32 = 0xFFFFFFFF


def mulberry32(seed: int):
    """Same sequence as the JavaScript version: floats in [0, 1)."""
    a = seed & M32

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


# A shape is a list of primitives: ("c", cx, cy, r) or ("r", x0, y0, x1, y1).


def _prim_gap(p, q) -> float:
    """Clear distance between two primitives (negative or 0 when they overlap)."""
    if p[0] == "c" and q[0] == "c":
        return math.hypot(p[1] - q[1], p[2] - q[2]) - p[3] - q[3]
    if p[0] == "r" and q[0] == "r":
        dx = max(0.0, q[1] - p[3], p[1] - q[3])
        dy = max(0.0, q[2] - p[4], p[2] - q[4])
        return math.hypot(dx, dy)
    c, r = (p, q) if p[0] == "c" else (q, p)
    dx = max(0.0, r[1] - c[1], c[1] - r[3])
    dy = max(0.0, r[2] - c[2], c[2] - r[4])
    return math.hypot(dx, dy) - c[3]


def _bounds(p) -> tuple[float, float, float, float]:
    if p[0] == "c":
        return p[1] - p[3], p[2] - p[3], p[1] + p[3], p[2] + p[3]
    return p[1], p[2], p[3], p[4]


def _make(kind: str, rand, w: float, h: float):
    def u(lo: float, hi: float) -> float:
        return lo + rand() * (hi - lo)

    if kind == "pillar":
        r = u(0.5, 1.2)
        return [("c", u(GAP + r, w - GAP - r), u(GAP + r, h - GAP - r), r)]
    if kind == "block":
        bw = u(0.8, 2.4)
        bh = bw if rand() < 0.5 else u(0.8, 2.4)
        x0, y0 = u(GAP, w - GAP - bw), u(GAP, h - GAP - bh)
        return [("r", x0, y0, x0 + bw, y0 + bh)]
    # wall: straight or L-shaped. Maze walls are longer and bend more often.
    maze = kind == "maze"
    if rand() < (0.5 if maze else 0.65):
        n = u(2.0, 5.0) if maze else u(1.6, 4.0)
        if rand() < 0.5:
            x0, y0 = u(GAP, w - GAP - n), u(GAP, h - GAP - WALL)
            return [("r", x0, y0, x0 + n, y0 + WALL)]
        x0, y0 = u(GAP, w - GAP - WALL), u(GAP, h - GAP - n)
        return [("r", x0, y0, x0 + WALL, y0 + n)]
    hi = 3.6 if maze else 2.8
    a, b = u(1.2, hi), u(1.2, hi)  # horizontal and vertical arm lengths
    x0, y0 = u(GAP, w - GAP - a), u(GAP, h - GAP - b)
    flip_x, flip_y = rand() < 0.5, rand() < 0.5
    hy = y0 + b - WALL if flip_y else y0  # horizontal arm at the top or bottom
    vx = x0 + a - WALL if flip_x else x0  # vertical arm at the right or left
    return [("r", x0, hy, x0 + a, hy + WALL), ("r", vx, y0, vx + WALL, y0 + b)]


def _plan(family: str, rand) -> list[str]:
    if family == "pillars":
        return ["pillar"] * (2 + int(rand() * 3))
    if family == "blocks":
        return ["block"] * (2 + int(rand() * 3))
    if family == "walls":
        return ["wall"] * (4 + int(rand() * 4))
    if family == "maze":
        return ["maze"] * 12  # as many as fit
    if family == "mixed":
        extra = [("pillar", "block", "wall")[int(rand() * 3)] for _ in range(int(rand() * 3))]
        return ["pillar", "block", "wall", "wall", *extra]
    raise ValueError(f"unknown map family {family!r}")


def _fits(shape, placed, w: float, h: float) -> bool:
    for p in shape:
        x0, y0, x1, y1 = _bounds(p)
        if x0 < GAP or y0 < GAP or x1 > w - GAP or y1 > h - GAP:
            return False
    return all(_prim_gap(p, q) >= GAP for s in placed for p in shape for q in s)


def _to_arena(placed, w: float, h: float) -> Arena:
    prims = [p for s in placed for p in s]
    return Arena(
        width=w,
        height=h,
        pillars=tuple((p[1], p[2], p[3]) for p in prims if p[0] == "c"),
        blocks=tuple((p[1], p[2], p[3], p[4]) for p in prims if p[0] == "r"),
    )


@lru_cache(maxsize=1024)
def generate(seed: int, family: str, w: float = 12.0, h: float = 8.0) -> Arena:
    """The map for (seed, family). Same seed, same map, in Python and in the browser."""
    rand = mulberry32(seed)
    for _ in range(20):
        placed: list = []
        for kind in _plan(family, rand):
            for _ in range(TRIES):
                shape = _make(kind, rand, w, h)
                if _fits(shape, placed, w, h):
                    placed.append(shape)
                    break
        arena = _to_arena(placed, w, h)
        if placed and navigable(arena):
            return arena
    raise RuntimeError(f"no navigable map for seed {seed}")  # never seen in tests


# ---------------------------------------------------------------- collision helpers


def square_hits(x: float, y: float, h: float, a: Arena) -> bool:
    """Does a square of half-size h centred at (x, y) overlap any obstacle?"""
    for cx, cy, r in a.pillars:
        qx = min(max(cx, x - h), x + h)
        qy = min(max(cy, y - h), y + h)
        if math.hypot(qx - cx, qy - cy) < r:
            return True
    return any(
        x + h > x0 and x - h < x1 and y + h > y0 and y - h < y1 for x0, y0, x1, y1 in a.blocks
    )


def open_cells(a: Arena, cell: float, h: float) -> np.ndarray:
    """(ny, nx) grid: True where a square of half-size h fits at the cell centre."""
    nx, ny = round(a.width / cell), round(a.height / cell)
    X = ((np.arange(nx) + 0.5) * cell)[None, :]
    Y = ((np.arange(ny) + 0.5) * cell)[:, None]
    hit = np.zeros((ny, nx), dtype=bool)
    for cx, cy, r in a.pillars:
        qx = np.minimum(np.maximum(cx, X - h), X + h)
        qy = np.minimum(np.maximum(cy, Y - h), Y + h)
        hit |= np.hypot(qx - cx, qy - cy) < r
    for x0, y0, x1, y1 in a.blocks:
        hit |= (X + h > x0) & (X - h < x1) & (Y + h > y0) & (Y - h < y1)
    return ~hit


def navigable(a: Arena, cell: float = 0.1, h: float = BODY_HALF) -> bool:
    """Flood fill over the spots a square fits: every one of them must be reachable."""
    free = open_cells(a, cell, h)
    X = (np.arange(free.shape[1]) + 0.5) * cell
    Y = (np.arange(free.shape[0]) + 0.5) * cell
    free &= ((X >= h) & (X <= a.width - h))[None, :] & ((Y >= h) & (Y <= a.height - h))[:, None]
    ny, nx = free.shape
    flat = free.reshape(-1).tolist()
    start = flat.index(True) if True in flat else -1
    if start < 0:
        return False
    seen = bytearray(len(flat))
    seen[start] = 1
    q = deque([start])
    count = 1
    while q:
        k = q.popleft()
        i = k % nx
        for n in (k - 1 if i > 0 else -1, k + 1 if i < nx - 1 else -1, k - nx, k + nx):
            if 0 <= n < len(flat) and flat[n] and not seen[n]:
                seen[n] = 1
                count += 1
                q.append(n)
    return count == sum(flat)


def segment_clear(ax: float, ay: float, bx: float, by: float, a: Arena, cl: float) -> bool:
    """Is the segment at least `cl` away from every obstacle?"""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    for cx, cy, r in a.pillars:
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
        if math.hypot(ax + t * dx - cx, ay + t * dy - cy) < r + cl:
            return False
    for x0, y0, x1, y1 in a.blocks:
        lo, hi = 0.0, 1.0
        hit = True
        for o, d, mn, mx in ((ax, dx, x0 - cl, x1 + cl), (ay, dy, y0 - cl, y1 + cl)):
            if abs(d) < 1e-12:
                if o < mn or o > mx:
                    hit = False
                    break
                continue
            t1, t2 = (mn - o) / d, (mx - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            lo, hi = max(lo, t1), min(hi, t2)
            if lo > hi:
                hit = False
                break
        if hit:
            return False
    return True


# ---------------------------------------------------------------- ray sensors


@lru_cache(maxsize=8)
def ray_dirs(n: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n)
    return np.array([math.cos(2 * math.pi * i / n) for i in k]), np.array(
        [math.sin(2 * math.pi * i / n) for i in k]
    )


@lru_cache(maxsize=1024)
def _arrays(a: Arena):
    P = np.array(a.pillars, dtype=np.float64).reshape(-1, 3)
    B = np.array(a.blocks, dtype=np.float64).reshape(-1, 4)
    return P, B


def rays(x: float, y: float, a: Arena, n: int = 16, reach: float = 6.0) -> list[float]:
    """Distance to the nearest wall or obstacle in n directions, as a fraction of `reach`."""
    dx, dy = ray_dirs(n)
    P, B = _arrays(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 0, (a.width - x) / dx, np.where(dx < 0, -x / dx, np.inf))
        ty = np.where(dy > 0, (a.height - y) / dy, np.where(dy < 0, -y / dy, np.inf))
        best = np.minimum(tx, ty)
        if len(P):
            ox = x - P[:, 0][None, :]
            oy = y - P[:, 1][None, :]
            b = ox * dx[:, None] + oy * dy[:, None]
            c = ox * ox + oy * oy - P[:, 2][None, :] ** 2
            disc = b * b - c
            t = -b - np.sqrt(np.where(disc >= 0, disc, 0.0))
            t = np.where((disc >= 0) & (t >= 0), t, np.inf)
            best = np.minimum(best, t.min(axis=1))
        if len(B):
            t1 = (B[:, 0][None, :] - x) / dx[:, None]
            t2 = (B[:, 2][None, :] - x) / dx[:, None]
            t3 = (B[:, 1][None, :] - y) / dy[:, None]
            t4 = (B[:, 3][None, :] - y) / dy[:, None]
            tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
            tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
            t = np.where((tmax >= tmin) & (tmin >= 0), tmin, np.inf)
            best = np.minimum(best, t.min(axis=1))
    return np.minimum(best / reach, 1.0).tolist()


# ---------------------------------------------------------------- bot navigation

# Neighbour order: the 8 directions of ACTIONS (east, then counter-clockwise).
_NB = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
UNREACHED = 32767


def _aim_keys(per: int) -> np.ndarray:
    """Squared distance (in half-steps) of each spot in a per x per patch from its centre."""
    k = np.arange(per * per)
    return (2 * (k % per) - per + 1) ** 2 + (2 * (k // per) - per + 1) ** 2


class NavGrid:
    """Grid path planning for the scripted chasers on any map.

    Cells are 0.5 units. A cell is open if a 0.2 half-size square fits at its centre (a bit
    smaller than a real square, so gaps between obstacles always have an open cell).
    `dist[t, c]` is the number of grid steps from cell c to cell t, for every pair.
    Bots steer toward each cell's `aim` point: the spot nearest the centre, within the cell,
    where a real square fits (so they don't press into a corner the cell only half clears).
    """

    def __init__(self, a: Arena, cell: float = 0.5, h: float = 0.2):
        self.a, self.cell = a, cell
        self.nx, self.ny = nx, ny = round(a.width / cell), round(a.height / cell)
        self.free = free = open_cells(a, cell, h)
        n = nx * ny
        # Breadth-first search from every cell at once, as array shifts.
        dist = np.full((n, ny, nx), UNREACHED, dtype=np.int16)
        front = np.zeros((n, ny, nx), dtype=bool)
        idx = np.arange(n)
        front[idx, idx // nx, idx % nx] = free.reshape(-1)
        dist[front] = 0
        seen = front.copy()
        steps = 0
        oks = [self._move_ok(di, dj) for di, dj in _NB]
        while front.any():
            steps += 1
            new = np.zeros_like(front)
            for (di, dj), ok in zip(_NB, oks, strict=True):
                src = front[:, max(0, -dj) : ny - max(0, dj), max(0, -di) : nx - max(0, di)]
                new[:, max(0, dj) : ny - max(0, -dj), max(0, di) : nx - max(0, -di)] |= (
                    src & ok[max(0, dj) : ny - max(0, -dj), max(0, di) : nx - max(0, -di)]
                )
            new &= ~seen
            seen |= new
            dist[new] = steps
            front = new
        self.dist = dist.reshape(n, n)
        # Nearest open cell for every cell (bodies pressed against an obstacle); ties go to
        # the first open cell in row order.
        flat = free.reshape(-1)
        opens = np.flatnonzero(flat)
        jj, ii = np.divmod(np.arange(n), nx)
        d = (jj[:, None] - jj[opens][None, :]) ** 2 + (ii[:, None] - ii[opens][None, :]) ** 2
        self.near = np.where(flat, np.arange(n), opens[np.argmin(d, axis=1)])
        # Aim points, from a 0.05 grid of spots where a real square fits: the first open spot
        # in order of distance from the centre (ties in row order).
        f = 0.05
        per = round(cell / f)
        fine = open_cells(a, f, BODY_HALF + 0.02)
        sub = fine.reshape(ny, per, nx, per).transpose(0, 2, 1, 3).reshape(n, per * per)
        order = np.argsort(_aim_keys(per), kind="stable")
        sub = sub[:, order]
        m = order[sub.argmax(axis=1)]
        sj, si = np.divmod(m, per)
        cj, ci = np.divmod(np.arange(n), nx)
        has = sub.any(axis=1)
        px = np.where(has, (ci * per + si + 0.5) * f, (ci + 0.5) * cell)
        py = np.where(has, (cj * per + sj + 0.5) * f, (cj + 0.5) * cell)
        self.aim = list(zip(px.tolist(), py.tolist(), strict=True))

    def _move_ok(self, di: int, dj: int) -> np.ndarray:
        """Destination cells reachable by step (di, dj): open, and no corner cutting."""
        ok = self.free.copy()
        if di and dj:
            ny, nx = self.free.shape
            side_x = np.zeros_like(ok)  # the cell (x - di, y) next to the destination
            side_y = np.zeros_like(ok)
            side_x[:, max(0, di) : nx - max(0, -di)] = self.free[:, max(0, -di) : nx - max(0, di)]
            side_y[max(0, dj) : ny - max(0, -dj), :] = self.free[max(0, -dj) : ny - max(0, dj), :]
            # Moving diagonally from (i, j): needs (i + di, j) and (i, j + dj) open, which
            # seen from the destination (i', j') are (i', j' - dj) and (i' - di, j').
            ok &= side_x & side_y
        return ok

    def cell_of(self, x: float, y: float) -> int:
        i = min(max(int(x / self.cell), 0), self.nx - 1)
        j = min(max(int(y / self.cell), 0), self.ny - 1)
        return int(self.near[j * self.nx + i])

    def center(self, k: int) -> tuple[float, float]:
        j, i = divmod(k, self.nx)
        return (i + 0.5) * self.cell, (j + 0.5) * self.cell

    def next_cell(self, c: int, t: int) -> int:
        best, bd = c, self.dist[t, c]
        j, i = divmod(c, self.nx)
        for di, dj in _NB:
            ii, jj = i + di, j + dj
            if not (0 <= ii < self.nx and 0 <= jj < self.ny) or not self.free[jj, ii]:
                continue
            if di and dj and not (self.free[j, ii] and self.free[jj, i]):
                continue
            k = jj * self.nx + ii
            if self.dist[t, k] < bd:
                best, bd = k, self.dist[t, k]
        return best

    def direction(self, fx: float, fy: float, tx: float, ty: float, cl: float):
        """Unit direction from (fx, fy) toward (tx, ty) along a shortest open path."""
        if segment_clear(fx, fy, tx, ty, self.a, cl):
            gx, gy = tx - fx, ty - fy
        else:
            c, t = self.cell_of(fx, fy), self.cell_of(tx, ty)
            n1 = self.next_cell(c, t)
            if n1 == c:  # already in the target's cell
                gx, gy = tx - fx, ty - fy
            else:
                n2 = self.next_cell(n1, t)
                px, py = self.aim[n2]
                if not segment_clear(fx, fy, px, py, self.a, cl * 0.5):
                    px, py = self.aim[n1]
                gx, gy = px - fx, py - fy
        n = math.hypot(gx, gy)
        return (gx / n, gy / n) if n > 1e-9 else (0.0, 0.0)


@lru_cache(maxsize=256)
def nav_for(a: Arena) -> NavGrid:
    return NavGrid(a)
