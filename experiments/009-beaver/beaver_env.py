"""Beaver vs river: a turn-based board game on a small grid, seen from above.

One side is the river: water comes out of a spring and spreads along low ground, one tile per
round. The river's player digs: each round it opens one tile next to the water (soil becomes
low ground in one dig, clay takes two). Its goal is to get water into the beaver's den.

The other side is the beaver. Each round it steps to a neighbouring tile or drops a log on a
neighbouring tile of low ground (wet or dry), which blocks water there for good. It has a few
logs; the current is too strong for a dam close to the spring, and it can't dam the tiles
right next to its den (its way in). The beaver wins if the den is still dry after the last
round.

Two limits keep the river from flooding everything:
- strength: water reaches at most `reach` tiles from the spring, counted along open ground
  (its shortest path). A dam that forces a long detour can starve the flow.
- time: one dig per round, and the water only moves one tile per round into what's dug.

Round order: the river digs, the beaver acts, then the water moves (cut-off water drains away,
then the water spreads one tile into open ground within reach). The den floods the moment
water enters it.

Plain Python on flat lists so site/src/lib/beaver-sim.js reproduces it exactly.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Tiles.
ROCK, TREE, SOIL, CLAY, LOW, LOG, SPRING, DEN = range(8)
TILE_NAMES = ("rock", "tree", "soil", "clay", "low", "log", "spring", "den")
WATER_OPEN = (LOW, SPRING, DEN)  # water can stand here
DIGGABLE = (SOIL, CLAY)
BLOCKED = (ROCK, TREE)  # nobody walks or digs here

# Beaver actions: stay, step N/E/S/W, drop a log N/E/S/W.
B_ACTIONS = ("stay", "up", "right", "down", "left", "log-up", "log-right", "log-down", "log-left")
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # N, E, S, W

M32 = 0xFFFFFFFF
INF = 10**6


@dataclass(frozen=True)
class Rules:
    size: int = 9  # the board is size x size tiles
    rounds: int = 24  # the beaver wins if the den is dry after this many rounds
    logs: int = 4  # the beaver's logs
    reach: int = 16  # water travels at most this many tiles from the spring
    calm: int = 3  # no dams on tiles closer than this to the spring (counted over open ground)


@dataclass
class Board:
    """A map: the tiles, the spring, the den and where the beaver starts."""

    tiles: list[int]
    spring: int
    den: int
    start: int
    seed: int = 0


# ----------------------------------------------------------------------------- random maps
def mulberry32(seed: int):
    """Small PRNG with the same sequence in the JS port: floats in [0, 1)."""
    a = seed & M32

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


def neighbours(n: int) -> list[list[int]]:
    """4-neighbours of every tile, in N, E, S, W order (missing ones left out)."""
    out = []
    for i in range(n * n):
        x, y = i % n, i // n
        out.append(
            [(y + dy) * n + x + dx for dx, dy in DIRS if 0 <= x + dx < n and 0 <= y + dy < n]
        )
    return out


_NB: dict[int, list[list[int]]] = {}


def nbrs(n: int) -> list[list[int]]:
    if n not in _NB:
        _NB[n] = neighbours(n)
    return _NB[n]


def step_to(i: int, d: int, n: int) -> int:
    """Tile one step from i in direction d (0..3), or -1 off the board."""
    x, y = i % n + DIRS[d][0], i // n + DIRS[d][1]
    return y * n + x if 0 <= x < n and 0 <= y < n else -1


def distances(tiles: list[int], n: int, src: int, open_tiles) -> list[int]:
    """BFS steps from src over tiles whose type is in open_tiles (src always counts)."""
    d = [INF] * (n * n)
    d[src] = 0
    q = deque([src])
    nb = nbrs(n)
    while q:
        i = q.popleft()
        for j in nb[i]:
            if d[j] == INF and tiles[j] in open_tiles:
                d[j] = d[i] + 1
                q.append(j)
    return d


def make_board(seed: int, r: Rules | None = None) -> Board:
    """A random map from a seed. Canonical layout: spring on the top edge near the left, den
    in the bottom-right area; then one of the 8 rotations/flips. Retries until the river has a
    route to the den that needs at least 3 digs and fits within reach."""
    r = r or Rules()
    n = r.size
    rand = mulberry32(seed)

    def ri(lo: int, hi: int) -> int:  # inclusive
        return lo + int(rand() * (hi - lo + 1))

    for _ in range(100):
        t = [SOIL] * (n * n)
        sx = ri(0, 3)
        spring = sx
        den = ri(n - 4, n - 2) + ri(n - 4, n - 2) * n
        # clay patches
        for _ in range(ri(2, 4)):
            cx, cy, rad = ri(0, n - 1), ri(0, n - 1), ri(1, 2)
            for i in range(n * n):
                if abs(i % n - cx) + abs(i // n - cy) <= rad and rand() < 0.75:
                    t[i] = CLAY
        # rocks and trees, in small clumps
        for _ in range(ri(5, 8)):
            i = ri(0, n * n - 1)
            kind = ROCK if rand() < 0.5 else TREE
            for _ in range(ri(1, 3)):
                t[i] = kind
                j = step_to(i, ri(0, 3), n)
                i = j if j >= 0 else i
        # the live stream from the spring, drifting towards the den
        i = spring
        for _ in range(ri(3, 6)):
            x, y = i % n, i // n
            dx, dy = den % n - x, den // n - y
            if rand() < 0.5 and dy != 0:
                d = 2 if dy > 0 else 0
            elif dx != 0:
                d = 1 if dx > 0 else 3
            else:
                d = ri(0, 3)
            if rand() < 0.25:
                d = ri(0, 3)
            j = step_to(i, d, n)
            if j >= 0 and j != den:
                i = j
                t[i] = LOW
        # old dry streambeds
        for _ in range(ri(1, 2)):
            i = ri(n, n * n - n - 1)
            d = ri(0, 3)
            for _ in range(ri(3, 5)):
                if i != den and i != spring:
                    t[i] = LOW
                if rand() < 0.3:
                    d = ri(0, 3)
                j = step_to(i, d, n)
                i = j if j >= 0 else i
        t[spring] = SPRING
        t[den] = DEN
        for j in nbrs(n)[spring]:  # the spring always has an outlet
            if t[j] in BLOCKED:
                t[j] = SOIL
        open_den = [j for j in nbrs(n)[den] if t[j] not in BLOCKED]
        if len(open_den) < 2:
            continue
        # the beaver starts next to its den
        starts = [j for j in nbrs(n)[den] if t[j] not in BLOCKED and t[j] != SPRING]
        start = starts[ri(0, len(starts) - 1)]
        # feasible but not free: some route within reach, and it takes digging
        reachable = distances(t, n, spring, (LOW, SPRING, DEN, SOIL, CLAY))
        if reachable[den] > r.reach - 2:
            continue
        b = Board(t, spring, den, start, seed)
        g = Game(b, r)
        if g.digs_needed()[den] < 3:
            continue
        return transform(b, ri(0, 7), n)
    raise RuntimeError(f"no board for seed {seed}")


def transform(b: Board, k: int, n: int) -> Board:
    """One of the 8 symmetries of the square (k = 0..7)."""

    def f(i: int) -> int:
        x, y = i % n, i // n
        if k & 1:
            x = n - 1 - x
        if k & 2:
            y = n - 1 - y
        if k & 4:
            x, y = y, x
        return y * n + x

    tiles = [0] * (n * n)
    for i, v in enumerate(b.tiles):
        tiles[f(i)] = v
    return Board(tiles, f(b.spring), f(b.den), f(b.start), b.seed)


# ----------------------------------------------------------------------------- the game
class Game:
    """State of one game. River and beaver call dig() / beaver() each round, then flow()."""

    def __init__(self, board: Board, rules: Rules | None = None):
        self.r = rules or Rules()
        self.board = board
        n = self.r.size
        self.n = n
        self.nb = nbrs(n)
        self.tiles = list(board.tiles)
        self.beaver_at = board.start
        self.logs = self.r.logs
        self.round = 0
        self.flooded = False
        self.done = False
        self.last_dig = -1
        self.last_log = -1
        self.dug = 0
        self.dams = 0
        # "calm" water: dams only at least `calm` steps from the spring over the board's
        # open ground (rock and trees excluded; fixed for the whole game).
        self.spring_dist = distances(
            self.tiles, n, board.spring, (LOW, SPRING, DEN, SOIL, CLAY, LOG)
        )
        # No dams in the fast water near the spring, or right next to the den (its way in).
        self.no_dam = [d < self.r.calm for d in self.spring_dist]
        for j in self.nb[board.den]:
            self.no_dam[j] = True
        self.wet = [False] * (n * n)
        self.settle()

    def copy(self) -> Game:
        g = object.__new__(Game)
        g.__dict__.update(self.__dict__)
        g.tiles = list(self.tiles)
        g.wet = list(self.wet)
        return g

    # --- water --------------------------------------------------------------------------
    def strength(self) -> list[int]:
        """Steps from the spring along open water ground right now."""
        return distances(self.tiles, self.n, self.board.spring, WATER_OPEN)

    def settle(self) -> None:
        """Start of game: water fills every open tile it can reach."""
        d = self.strength()
        self.wet = [d[i] <= self.r.reach for i in range(self.n * self.n)]

    def flow(self) -> None:
        """End of round: cut-off water drains, then water spreads one tile within reach."""
        n2 = self.n * self.n
        t = self.tiles
        sp = self.board.spring
        # drain: keep wet tiles still open and connected to the spring through wet tiles
        keep = [False] * n2
        keep[sp] = True
        q = deque([sp])
        while q:
            i = q.popleft()
            for j in self.nb[i]:
                if not keep[j] and self.wet[j] and t[j] in WATER_OPEN:
                    keep[j] = True
                    q.append(j)
        d = self.strength()
        new = list(keep)
        for i in range(n2):
            if keep[i]:
                for j in self.nb[i]:
                    if not keep[j] and t[j] in WATER_OPEN and d[j] <= self.r.reach:
                        new[j] = True
        self.wet = new
        self.round += 1
        if self.wet[self.board.den]:
            self.flooded = True
            self.done = True
        elif self.round >= self.r.rounds:
            self.done = True

    # --- river --------------------------------------------------------------------------
    def can_dig(self, i: int, d: list[int] | None = None) -> bool:
        """Soil or clay next to water that still has strength to push into it."""
        if not 0 <= i < self.n * self.n or self.tiles[i] not in DIGGABLE:
            return False
        d = d if d is not None else self.strength()
        return any(self.wet[j] and d[j] < self.r.reach for j in self.nb[i])

    def diggable(self) -> list[int]:
        d = self.strength()
        return [i for i in range(self.n * self.n) if self.can_dig(i, d)]

    def dig(self, i: int) -> bool:
        """The river's move: dig tile i (or pass with i = -1). Returns whether it dug."""
        self.last_dig = -1
        if i < 0 or not self.can_dig(i):
            return False
        self.tiles[i] = SOIL if self.tiles[i] == CLAY else LOW
        self.last_dig = i
        self.dug += 1
        return True

    # --- beaver -------------------------------------------------------------------------
    def walkable(self, i: int) -> bool:
        return self.tiles[i] not in BLOCKED

    def can_log(self, i: int) -> bool:
        return i >= 0 and self.logs > 0 and self.tiles[i] == LOW and not self.no_dam[i]

    def beaver_legal(self) -> list[bool]:
        """Which of the 9 beaver actions do something."""
        out = [True] + [False] * 8
        for d in range(4):
            j = step_to(self.beaver_at, d, self.n)
            out[1 + d] = j >= 0 and self.walkable(j)
            out[5 + d] = self.can_log(j)
        return out

    def beaver(self, a: int) -> None:
        """The beaver's move (B_ACTIONS). Illegal moves do nothing."""
        self.last_log = -1
        if 1 <= a <= 4:
            j = step_to(self.beaver_at, a - 1, self.n)
            if j >= 0 and self.walkable(j):
                self.beaver_at = j
        elif 5 <= a <= 8:
            j = step_to(self.beaver_at, a - 5, self.n)
            if self.can_log(j):
                self.tiles[j] = LOG
                self.wet[j] = False
                self.logs -= 1
                self.dams += 1
                self.last_log = j

    # --- analysis (bots and the beaver's observation) ------------------------------------
    def digs_needed(self) -> list[int]:
        """Fewest digs before water could reach each tile (0-1-2 BFS from the wet tiles;
        ignores time and reach). INF where it never can."""
        cost = {LOW: 0, DEN: 0, SPRING: 0, SOIL: 1, CLAY: 2}
        n2 = self.n * self.n
        best = [INF] * n2
        q: deque[int] = deque()
        for i in range(n2):
            if self.wet[i]:
                best[i] = 0
                q.append(i)
        # small integer costs: a Dijkstra with a bucket queue
        buckets: list[list[int]] = [list(q)]
        k = 0
        while k < len(buckets):
            for i in buckets[k]:
                if best[i] != k:
                    continue
                for j in self.nb[i]:
                    c = cost.get(self.tiles[j])
                    if c is None:
                        continue
                    v = k + c
                    if v < best[j]:
                        best[j] = v
                        while len(buckets) <= v:
                            buckets.append([])
                        buckets[v].append(j)
            k += 1
        return best


# ----------------------------------------------------------------------------- bots
def plan_route(g: Game, extra: list[float] | None = None, dig_w: float = 3.0) -> list[int]:
    """The river bot's route: tiles from the water to the den, cheapest by
    dig_w * digs + steps (+ extra per tile), keeping total steps from the spring within reach.
    Empty if there is none."""
    n2 = g.n * g.n
    cost = {LOW: 0, DEN: 0, SOIL: 1, CLAY: 2}
    sd = g.strength()
    dist = [float("inf")] * n2
    steps = [INF] * n2
    prev = [-1] * n2
    pq = []
    for i in range(n2):
        if g.wet[i]:
            dist[i] = 0.0
            steps[i] = sd[i]
            heapq.heappush(pq, (0.0, i))
    while pq:
        c, i = heapq.heappop(pq)
        if c > dist[i]:
            continue
        if i == g.board.den:
            break
        for j in g.nb[i]:
            k = cost.get(g.tiles[j])
            if k is None or g.wet[j]:
                continue
            s = steps[i] + 1
            if s > g.r.reach:
                continue
            v = c + dig_w * k + 1 + (extra[j] if extra else 0.0)
            if v < dist[j]:
                dist[j] = v
                steps[j] = s
                prev[j] = i
                heapq.heappush(pq, (v, j))
    if prev[g.board.den] < 0 and not g.wet[g.board.den]:
        return []
    route = []
    i = g.board.den
    while i >= 0 and not g.wet[i]:
        route.append(i)
        i = prev[i]
    return route[::-1]


class RiverBot:
    """Scripted river: follows its cheapest route to the den, digging the first tile on it
    that is next to water. Per-game noise on tile costs varies the route; `dodge` makes it
    avoid tiles near the beaver; `eps` digs a random diggable tile now and then."""

    def __init__(self, rng: np.random.Generator, noise=0.0, dodge=0.0, eps=0.0, dig_w=3.0):
        self.rng = rng
        self.noise = noise
        self.dodge = dodge
        self.eps = eps
        self.dig_w = dig_w
        self.extra: list[float] | None = None

    def reset(self, g: Game) -> None:
        n2 = g.n * g.n
        self.extra = (
            [float(self.rng.random() * self.noise) for _ in range(n2)] if self.noise else None
        )

    def __call__(self, g: Game) -> int:
        options = g.diggable()
        if not options:
            return -1
        if self.eps and self.rng.random() < self.eps:
            return int(options[self.rng.integers(len(options))])
        extra = list(self.extra) if self.extra else [0.0] * (g.n * g.n)
        if self.dodge:
            b = g.beaver_at
            for i in range(g.n * g.n):
                m = abs(i % g.n - b % g.n) + abs(i // g.n - b // g.n)
                extra[i] += self.dodge * max(0, 3 - m)
        route = plan_route(g, extra, self.dig_w)
        ok = set(options)
        for i in route:
            if i in ok:
                return i
        # route not dig-able yet (water still on its way): dig the option nearest the den
        if route:
            return -1
        den = g.board.den
        return min(options, key=lambda i: abs(i % g.n - den % g.n) + abs(i // g.n - den // g.n))


def beaver_path(g: Game, goal_tiles: set[int]) -> int:
    """First step (1..4 action) of the beaver's shortest walk to any goal tile; 0 if already
    there, -1 if unreachable."""
    if g.beaver_at in goal_tiles:
        return 0
    n2 = g.n * g.n
    first = [-1] * n2
    seen = [False] * n2
    seen[g.beaver_at] = True
    q = deque()
    for d in range(4):
        j = step_to(g.beaver_at, d, g.n)
        if j >= 0 and g.walkable(j) and not seen[j]:
            seen[j] = True
            first[j] = d
            q.append(j)
    while q:
        i = q.popleft()
        if i in goal_tiles:
            return 1 + first[i]
        for j in g.nb[i]:
            if not seen[j] and g.walkable(j):
                seen[j] = True
                first[j] = first[i]
                q.append(j)
    return -1


def warden_action(g: Game) -> int:
    """Scripted beaver: predict the river's route and dam the first low tile on it it can;
    with nothing to dam yet, walk next to the tile the river will dig next."""
    route = plan_route(g)
    sd = g.strength()
    # also the water's own leading edge: low tiles the water is about to enter
    damable = [i for i in route if g.tiles[i] == LOW and g.can_log(i)] if g.logs else []
    if not damable and g.logs:
        # the water front feeding the route: wet tiles next to the first route tile
        if route:
            damable = [j for j in g.nb[route[0]] if g.wet[j] and g.can_log(j)]
    for d in range(4):
        j = step_to(g.beaver_at, d, g.n)
        if j in damable:
            return 5 + d
    if damable:
        stands = {j for i in damable for j in g.nb[i] if g.walkable(j)}
        a = beaver_path(g, stands)
        if a > 0:
            return a
    if route:
        target = route[0]
        stands = {j for j in g.nb[target] if g.walkable(j)}
        a = beaver_path(g, stands)
        if a > 0:
            return a
    del sd
    return 0


# ----------------------------------------------------------------------------- one game
def play(board: Board, river, beaver, rules: Rules | None = None, record: bool = False):
    """Play a whole game. river(g) -> tile or -1; beaver(g) -> action. Returns the game (and
    the frames if record)."""
    g = Game(board, rules)
    if hasattr(river, "reset"):
        river.reset(g)
    frames = []
    while not g.done:
        g.dig(river(g))
        g.beaver(beaver(g))
        g.flow()
        if record:
            frames.append((g.last_dig, g.last_log, g.beaver_at))
    return (g, frames) if record else g


# ----------------------------------------------------------------------------- gym envs
N_CH = 12


def observe(g: Game) -> list[float]:
    """What both sides see: 12 planes of size x size, then logs left and rounds left.
    Planes: blocked, soil, clay, low (dry), log, wet, den, beaver, last dig, digs the river
    needs to get water there (1 / (1 + digs), 0 if never), no-dam zone, last log."""
    n2 = g.n * g.n
    t = g.tiles
    need = g.digs_needed()
    out = [0.0] * (N_CH * n2 + 2)
    for i in range(n2):
        v = t[i]
        out[i] = 1.0 if v in BLOCKED else 0.0
        out[n2 + i] = 1.0 if v == SOIL else 0.0
        out[2 * n2 + i] = 1.0 if v == CLAY else 0.0
        out[3 * n2 + i] = 1.0 if v == LOW and not g.wet[i] else 0.0
        out[4 * n2 + i] = 1.0 if v == LOG else 0.0
        out[5 * n2 + i] = 1.0 if g.wet[i] else 0.0
        out[6 * n2 + i] = 1.0 if v == DEN else 0.0
        out[9 * n2 + i] = 1.0 / (1.0 + need[i]) if need[i] < INF else 0.0
        out[10 * n2 + i] = 1.0 if g.no_dam[i] else 0.0
    out[7 * n2 + g.beaver_at] = 1.0
    if g.last_dig >= 0:
        out[8 * n2 + g.last_dig] = 1.0
    if g.last_log >= 0:
        out[11 * n2 + g.last_log] = 1.0
    out[N_CH * n2] = g.logs / g.r.logs
    out[N_CH * n2 + 1] = (g.r.rounds - g.round) / g.r.rounds
    return out


@dataclass
class Config:
    rules: Rules = field(default_factory=Rules)
    # The river opponent for a new game: rng -> callable(g) -> tile. Default: a noisy bot.
    river: object = None


def default_river(rng: np.random.Generator):
    kind = rng.random()
    if kind < 0.4:
        return RiverBot(rng, noise=float(rng.uniform(0, 3)), dig_w=float(rng.uniform(1, 4)))
    if kind < 0.8:
        return RiverBot(rng, noise=float(rng.uniform(0, 3)), dodge=float(rng.uniform(1, 4)))
    return RiverBot(rng, noise=2.0, eps=0.3)


class BeaverGame(gym.Env):
    """The beaver's side. Reward +1 if the den is dry at the end, -1 when it floods."""

    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        n2 = self.cfg.rules.size**2
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_CH * n2 + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(B_ACTIONS))
        self.g: Game | None = None
        self.river = None

    def action_masks(self) -> np.ndarray:
        return np.array(self.g.beaver_legal(), dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        board = options.get("board")
        if board is None:
            s = options.get("board_seed")
            board = make_board(
                int(self.np_random.integers(2**31)) if s is None else s, self.cfg.rules
            )
        self.g = Game(board, self.cfg.rules)
        make = self.cfg.river or default_river
        self.river = options.get("river") or make(self.np_random)
        if hasattr(self.river, "reset"):
            self.river.reset(self.g)
        self.g.dig(self.river(self.g))
        return np.asarray(observe(self.g), dtype=np.float32), {}

    def step(self, action: int):
        g = self.g
        g.beaver(int(action))
        g.flow()
        reward = 0.0
        if g.done:
            reward = -1.0 if g.flooded else 1.0
        else:
            g.dig(self.river(g))
        info = {}
        if g.done:
            info = {"outcome": "flooded" if g.flooded else "dry", "dams": g.dams, "dug": g.dug}
        return np.asarray(observe(g), dtype=np.float32), reward, g.done, False, info


class Net:
    """A policy network as numpy arrays (tanh hidden layers, linear logits), the same maths as
    the JS port. Illegal actions are masked out before choosing."""

    def __init__(self, spec, sample: bool = True, rng=None):
        """spec: {"convs": [...] or absent, "layers": [...]}, or just the layers list."""
        if isinstance(spec, list):
            spec = {"layers": spec}
        f = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
        self.convs = [(f(L["w"]), f(L["b"])) for L in spec.get("convs") or []]
        self.layers = [(f(L["w"]), f(L["b"])) for L in spec["layers"]]
        self.sample = sample
        self.rng = rng or np.random.default_rng()

    def features(self, x: np.ndarray) -> np.ndarray:
        """The conv front end (3x3, padded, ReLU) on the planes, flattened, plus the scalars."""
        if not self.convs:
            return x
        n = int(round(((len(x) - 2) / N_CH) ** 0.5))
        h = x[: N_CH * n * n].reshape(N_CH, n, n)
        for w, b in self.convs:
            p = np.pad(h, ((0, 0), (1, 1), (1, 1)))
            out = np.broadcast_to(b[:, None, None], (len(b), n, n)).copy()
            for dy in range(3):
                for dx in range(3):
                    out += np.einsum("oc,chw->ohw", w[:, :, dy, dx], p[:, dy : dy + n, dx : dx + n])
            h = np.maximum(out, 0.0)
        return np.concatenate([h.reshape(-1), x[N_CH * n * n :]])

    def logits(self, x) -> np.ndarray:
        x = self.features(np.asarray(x, dtype=np.float64))
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def choose(self, obs, legal) -> int:
        z = self.logits(obs)
        z = np.where(np.asarray(legal, dtype=bool), z, -np.inf)
        if not self.sample:
            return int(np.argmax(z))
        p = np.exp(z - z.max())
        return int(self.rng.choice(len(p), p=p / p.sum()))


class NetBeaver:
    def __init__(self, net: Net):
        self.net = net

    def __call__(self, g: Game) -> int:
        return self.net.choose(observe(g), g.beaver_legal())


class NetRiver:
    def __init__(self, net: Net):
        self.net = net

    def __call__(self, g: Game) -> int:
        a = self.net.choose(observe(g), river_legal(g))
        return -1 if a == g.n * g.n else a


def river_legal(g: Game) -> list[bool]:
    """Which tiles the river can dig, plus 'pass' (always) as the last action."""
    n2 = g.n * g.n
    out = [False] * (n2 + 1)
    for i in g.diggable():
        out[i] = True
    out[n2] = True
    return out


@dataclass
class RiverConfig:
    rules: Rules = field(default_factory=Rules)
    # The beaver opponent for a new game: rng -> callable(g) -> action. Default: the warden.
    beaver: object = None


class RiverGame(gym.Env):
    """The river's side: action = a tile to dig (or the last action, pass). Reward +1 when the
    den floods, -1 if it's dry at the end."""

    metadata = {"render_modes": []}

    def __init__(self, config: RiverConfig | None = None):
        self.cfg = config or RiverConfig()
        n2 = self.cfg.rules.size**2
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_CH * n2 + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(n2 + 1)
        self.g: Game | None = None
        self.opp = None

    def action_masks(self) -> np.ndarray:
        return np.array(river_legal(self.g), dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        board = options.get("board")
        if board is None:
            s = options.get("board_seed")
            board = make_board(
                int(self.np_random.integers(2**31)) if s is None else s, self.cfg.rules
            )
        self.g = Game(board, self.cfg.rules)
        make = self.cfg.beaver or (lambda rng: warden_action)
        self.opp = options.get("beaver") or make(self.np_random)
        return np.asarray(observe(self.g), dtype=np.float32), {}

    def step(self, action: int):
        g = self.g
        n2 = g.n * g.n
        g.dig(-1 if int(action) == n2 else int(action))
        g.beaver(self.opp(g))
        g.flow()
        reward = 0.0
        info = {}
        if g.done:
            reward = 1.0 if g.flooded else -1.0
            info = {"outcome": "flooded" if g.flooded else "dry", "dams": g.dams, "dug": g.dug}
        return np.asarray(observe(g), dtype=np.float32), reward, g.done, False, info
