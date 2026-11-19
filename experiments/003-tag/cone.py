"""Hide and seek with a field of view (post 007): the chaser faces somewhere.

Post 6's seeker saw all around itself at once. This one sees only inside a cone in front of
it, has to turn its head to look elsewhere, and turning takes time. Four settings make it
more or less human (`Senses`):

  fov        how wide it sees, in degrees (360 = all around, like post 6)
  turn       how fast it turns, in degrees per second
  reaction   how long something must stay in view before it notices (seconds); a spot it
             glanced at for less than that doesn't count as checked either
  backpedal  its top speed when moving straight away from where it faces, as a share of
             its forward speed (sideways is halfway between)

The network controls its legs and its head separately: one of 9 moves (as before) and one of
9 looks (hold, or turn toward one of the 8 directions at its turning speed). During training
every setting is drawn at random each episode and given to the network as inputs, so one
network plays them all. Everything else (the possibility map and its summary) is post 6's,
with "in sight" now meaning in the cone too. site/src/lib/tag-cone.js is the browser port.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from sight import (
    SeekArena,
    SeekBot,
    Tracker,
    hider_direction,
    seek_observe,
    seekbot_direction,
    sees,
)
from tag_env import ACTIONS, Body, NetEvader, caught, move, observe

LOOKS = 9  # 0 = hold, then turn toward ACTIONS[1..8]'s directions
BASE = 49  # post 6's map inputs come first, so post 6's seeker can read them
EXTRA = 2 + 4 + 8  # facing, the four settings, what it could see by looking each way
FOV_RANGE = (60.0, 200.0)
TURN_RANGE = (90.0, 1080.0)
REACTION_MAX = 0.4
BACKPEDAL_RANGE = (0.4, 1.0)
TOUCH = 0.75  # anything this close counts as in view whichever way it faces
_D = math.sqrt(0.5)
UX = np.array([1.0, _D, 0.0, -_D, -1.0, -_D, 0.0, _D])
UY = np.array([0.0, _D, 1.0, _D, 0.0, -_D, -1.0, -_D])


@dataclass(frozen=True)
class Senses:
    fov: float = 100.0
    turn: float = 300.0
    reaction: float = 0.25
    backpedal: float = 0.5


HUMAN = Senses()
ALLROUND = Senses(360.0, 1080.0, 0.0, 1.0)  # post 6's chaser, as far as these settings go


def sample_senses(rng: np.random.Generator) -> Senses:
    """Training draws: 15% all-round sight, otherwise a cone of 60-200 degrees; turning speed
    log-uniform 90-1080 degrees/s; reaction 0-0.4 s; backpedal 40-100%."""
    fov = 360.0 if rng.random() < 0.15 else float(rng.uniform(*FOV_RANGE))
    turn = float(np.exp(rng.uniform(math.log(TURN_RANGE[0]), math.log(TURN_RANGE[1]))))
    return Senses(
        fov, turn, float(rng.uniform(0, REACTION_MAX)), float(rng.uniform(*BACKPEDAL_RANGE))
    )


def obs_size(rays: int = 16) -> int:
    return BASE - 16 + rays + EXTRA


def wrap(a: float) -> float:
    """Angle into (-pi, pi]."""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def turn_toward(facing: float, target: float, step: float) -> float:
    d = wrap(target - facing)
    if abs(d) <= step:
        return wrap(target)
    return wrap(facing + (step if d > 0 else -step))


def look_angle(look: int) -> float:
    return (look - 1) * math.pi / 4


def speed_factor(dx: float, dy: float, facing: float, backpedal: float) -> float:
    """Top-speed multiplier when moving along (dx, dy): 1 straight ahead, `backpedal`
    straight back."""
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 1.0
    c = (dx * math.cos(facing) + dy * math.sin(facing)) / n
    return 1.0 - (1.0 - backpedal) * (1.0 - c) / 2.0


class ConeTracker(Tracker):
    """Post 6's tracker, but "in sight" also means inside the cone, and only after `reaction`
    seconds: the runner is noticed once it has been in view that long (at once again if it
    was seen a moment ago), and a cell is crossed off once it has been in view that long."""

    def __init__(self, grid, speed: float, dt: float, senses: Senses):
        super().__init__(grid, speed, dt)
        self.s = senses
        self.k = 1 + int(senses.reaction / dt + 1e-9)  # steps in view before it counts
        self.cos_half = math.cos(math.radians(senses.fov / 2))

    def reset(self, me: Body, prey: Body, facing: float = 0.0) -> None:
        self.facing = facing
        self.dwell = np.zeros(len(self.g.free), dtype=np.int64)
        self.streak = 0
        super().reset(me, prey)

    def in_cone(self, dx, dy):
        """Is the offset (dx, dy) inside the cone, or within touching distance? Works on
        floats and arrays."""
        if self.s.fov >= 360:
            return np.ones_like(dx, dtype=bool) if isinstance(dx, np.ndarray) else True
        fx, fy = math.cos(self.facing), math.sin(self.facing)
        d2 = dx * dx + dy * dy
        return (dx * fx + dy * fy >= self.cos_half * np.sqrt(d2)) | (d2 < TOUCH * TOUCH)

    def _view(self, me: Body, prey: Body) -> None:
        g = self.g
        los_pts = g._sees_points(me.x, me.y)
        view_pts = los_pts & self.in_cone(g.px - me.x, g.py - me.y)
        self.los = g.free.copy()
        self.los[g.owner[~los_pts]] = False
        in_view = g.free.copy()
        in_view[g.owner[~view_pts]] = False
        self.dwell = np.where(in_view, self.dwell + 1, 0)
        self.vis = in_view & (self.dwell >= self.k)
        dx, dy = prey.x - me.x, prey.y - me.y
        view = sees(me.x, me.y, prey.x, prey.y, g.nav.a) and bool(self.in_cone(dx, dy))
        self.streak = self.streak + 1 if view else 0
        self.seen = view and (self.streak >= self.k or (self.ever and self.since <= self.k))

    def lookable(self, me: Body) -> list[float]:
        """For each of 8 compass sectors: the share of the map's weight in line of sight that
        way, i.e. what it could check just by turning its head."""
        w = np.where(self.los, self.weight, 0.0)
        total = float(self.weight.sum())
        out = [0.0] * 8
        if total <= 0:
            return out
        cells = np.flatnonzero(w > 0)
        if not len(cells):
            return out
        dx = self.g.cx[cells] - me.x
        dy = self.g.cy[cells] - me.y
        sector = np.argmax(dx[:, None] * UX[None, :] + dy[:, None] * UY[None, :], axis=1)
        for d in range(8):
            out[d] = float(w[cells[sector == d]].sum()) / total
        return out


def cone_observe(me: Body, prey: Body, tr: ConeTracker, p, a, rays: int = 16) -> list[float]:
    """Post 6's 49 map inputs, then facing (cos, sin), the four settings, and lookable()."""
    out = seek_observe(me, prey, tr, "map", p, a, rays)
    s = tr.s
    out += [math.cos(tr.facing), math.sin(tr.facing)]
    out += [s.fov / 360.0, s.turn / TURN_RANGE[1], s.reaction / REACTION_MAX, s.backpedal]
    out += tr.lookable(me)
    return out


def auto_look(env: ConeArena) -> float | None:
    """Head for scripted chasers and post 6's seeker: look at the runner while it is seen;
    otherwise the search bot looks at the spot it is heading for, the last-seen bot at where
    it last saw the runner, and post 6's seeker where it is going. None = hold."""
    c, e, tr = env.chaser, env.evader, env.tracker
    if tr.seen:
        return math.atan2(e.y - c.y, e.x - c.x)
    tx = ty = None
    if env.bot and env.bot.kind == "search" and tr.weight.any():
        tx, ty = env.grid.nav.aim[tr.best_spot(c)]
    elif env.bot and env.bot.kind == "lastseen" and tr.ever:
        tx, ty = tr.last[0], tr.last[1]
    if tx is not None and math.hypot(tx - c.x, ty - c.y) > 0.3:
        return math.atan2(ty - c.y, tx - c.x)
    if math.hypot(c.vx, c.vy) > 0.5:
        return math.atan2(c.vy, c.vx)
    return None


class ConeArena(SeekArena):
    """The chaser's side of hide and seek with a field of view. Actions: [move, look].

    senses: fixed settings, or None to draw new ones every episode. Reset options may set
    "senses", "facing" (radians) and "head": "auto" (look like a scripted bot; the agent
    then only picks moves, so post 6's seeker can play)."""

    def __init__(self, config, senses: Senses | None = None):
        super().__init__(config, "map")
        self.fixed = senses
        n = obs_size(self.cfg.rays)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(n,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(ACTIONS), LOOKS])

    def reset(self, *, seed=None, options=None):
        options = options or {}
        super().reset(seed=seed, options=options)
        senses = options.get("senses") or self.fixed or sample_senses(self.np_random)
        facing = options.get("facing")
        if facing is None:
            facing = float(self.np_random.uniform(-math.pi, math.pi))
        self.head = options.get("head", "net")
        p = self.cfg.physics
        self.tracker = ConeTracker(self.grid, p.max_speed, p.dt, senses)
        self.tracker.reset(self.chaser, self.evader, wrap(facing))
        return self._obs(), {}

    def step(self, action):
        p, a, rw = self.cfg.physics, self.arena, self.cfg.rewards
        tr = self.tracker
        ex, ey = self._prey_dir()
        if self.bot:
            cx, cy = seekbot_direction(self.bot, self)
        else:
            cx, cy = ACTIONS[int(action[0])]
        if self.bot or self.head == "auto":
            target = auto_look(self)
        else:
            look = int(action[1])
            target = look_angle(look) if look else None
        if target is not None:
            tr.facing = turn_toward(tr.facing, target, math.radians(tr.s.turn) * p.dt)
        f = speed_factor(cx, cy, tr.facing, tr.s.backpedal)
        self.steps += 1
        move(self.evader, ex, ey, p.max_speed, p, a)
        move(self.chaser, cx, cy, p.max_speed * self.ratio * f, p, a)
        tr.update(self.chaser, self.evader)
        self.seen_steps += tr.seen
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
        if not isinstance(getattr(self, "tracker", None), ConeTracker):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        o = cone_observe(
            self.chaser, self.evader, self.tracker, self.cfg.physics, self.arena, self.cfg.rays
        )
        return np.array(o, dtype=np.float32)


def act2(net, obs) -> tuple[int, int]:
    """Best move and best look of a two-headed network (MLP with 9 + 9 outputs)."""
    z = net.logits(obs)
    return int(np.argmax(z[: len(ACTIONS)])), int(np.argmax(z[len(ACTIONS) :]))


__all__ = [
    "ALLROUND",
    "HUMAN",
    "ConeArena",
    "ConeTracker",
    "NetEvader",
    "SeekBot",
    "Senses",
    "act2",
    "cone_observe",
    "hider_direction",
    "observe",
    "obs_size",
    "sample_senses",
]
