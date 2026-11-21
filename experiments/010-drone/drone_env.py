"""A 2D drone seen from the side, flown with two analog sticks, from one platform to another.

The sticks work like a hobby drone's in "angle mode" (Mode 2, reduced to 2D):

- left stick, up/down: throttle. Centre is half power, which holds a level drone in the air.
- right stick, left/right: tilt. The flight controller turns the stick into a target angle (up to
  30 degrees) and spins the rotors to hold it; tilting points the thrust sideways, which is how
  the drone moves.

Physics per 1/60 s: motors spin up with a lag, the flight controller's torque is limited (and
weak
when the motors are slow), gravity, air drag (linear + quadratic), optional wind. The network
(or the person) moves the sticks 20 times a second.

A course: two pillars sticking out of water, random heights and gap, mirrored half the time.
The drone starts resting on one and has to touch down on the other gently (slow, nearly level,
centre over the pad). Hitting a pillar's side, landing hard, the water or leaving the area
ends it.

Plain Python floats on purpose: site/src/lib/drone-sim.js repeats every line in the same order,
so the browser drone flies the same as this one (see test/drone-sim.test.mjs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

G = 9.81
M32 = 0xFFFFFFFF

# Contact points in the drone's own frame (metres, y up): two feet, two rotor tips.
FEET = ((-0.22, -0.18), (0.22, -0.18))
TIPS = ((-0.35, 0.04), (0.35, 0.04))
LEG = 0.18  # centre height above the surface when resting level
FOOT_BAND = 0.15  # a foot this close under a deck's top came down on it (not into the side)

OUTCOMES = ("landed", "crashed", "water", "lost", "timeout")


@dataclass(frozen=True)
class Physics:
    dt: float = 1 / 60
    substeps: int = 3  # sticks move at 20 Hz
    t_max: float = 20.0  # seconds per flight
    thrust_max: float = 2.0  # full throttle, in units of the drone's (nominal) weight
    motor_tau: float = 0.08  # seconds for the motors to get ~63% of the way to the command
    tilt_max: float = 0.5236  # 30 degrees at full stick
    kp: float = 70.0  # flight controller: angle error -> angular acceleration
    kd: float = 14.0
    alpha_max: float = 40.0  # rad/s^2, the most the rotors can turn it
    drag_lin: float = 0.15
    drag_quad: float = 0.04
    land_vy: float = 1.5  # touchdown limits: falling speed, sideways speed, tilt
    land_vx: float = 1.5
    land_tilt: float = 0.35
    width: float = 24.0
    ceiling: float = 13.0
    stick_rate: float = 0.0  # most a stick can move per update (0: no limit; 2 is end to end)


@dataclass(frozen=True)
class Weather:
    """What a course can throw at the drone. Training uses calm air unless randomised."""

    wind: float = 0.0  # steady wind, up to this many m/s^2 either way
    gust: float = 0.0  # plus a slow sine gust up to this amplitude
    heavy: float = 0.0  # a payload: up to this fraction heavier (same motors)


CALM = Weather()
WINDY = Weather(wind=1.5, gust=1.0)
HEAVY = Weather(heavy=0.35)
ROUGH = Weather(wind=1.5, gust=1.0, heavy=0.35)


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


@dataclass
class Pad:
    x0: float
    x1: float
    top: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def half(self) -> float:
        return (self.x1 - self.x0) / 2


@dataclass
class Course:
    start: Pad
    target: Pad
    wind: float  # m/s^2, + is to the right
    gust: float
    gust_period: float
    gust_phase: float
    mass: float  # 1 = nominal

    @property
    def pads(self) -> tuple[Pad, Pad]:
        return (self.start, self.target)


def make_course(seed: int, weather: Weather = CALM, ph: Physics | None = None) -> Course:
    """The same seed gives the same pillars whatever the weather (the weather draws come last)."""
    ph = ph or Physics()
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
    return Course(
        Pad(sx - 1.0, sx + 1.0, sh),
        Pad(tx - tw / 2, tx + tw / 2, th),
        wind,
        gust,
        period,
        phase,
        mass,
    )


@dataclass
class Drone:
    """The full state of one flight."""

    course: Course
    ph: Physics = field(default_factory=Physics)
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    th: float = 0.0  # tilt, + leans right (thrust pushes right)
    om: float = 0.0
    motor: float = 0.0  # 0..1 of full thrust
    ticks: int = 0  # physics steps so far
    stick: tuple[float, float] = (-1.0, 0.0)  # last (throttle, tilt), each -1..1
    grounded: bool = True  # sitting on a pad: the legs hold it level until thrust beats weight
    outcome: str | None = None
    crash: str | None = None  # "hard" (a landing too fast or tilted) or "side" (hit a pillar)
    touch: tuple[float, float] = (0.0, 0.0)  # (vx, vy) at the final contact

    def __post_init__(self) -> None:
        self.x = self.course.start.cx
        self.y = self.course.start.top + LEG

    @property
    def done(self) -> bool:
        return self.outcome is not None

    @property
    def time(self) -> float:
        return self.ticks * self.ph.dt

    def goal(self) -> tuple[float, float]:
        t = self.course.target
        return t.cx, t.top + LEG

    def distance(self) -> float:
        gx, gy = self.goal()
        dx, dy = gx - self.x, gy - self.y
        return math.sqrt(dx * dx + dy * dy)

    def points(self) -> list[tuple[float, float]]:
        """Feet, then rotor tips, in world coordinates."""
        c, s = math.cos(self.th), math.sin(self.th)
        return [(self.x + lx * c + ly * s, self.y - lx * s + ly * c) for lx, ly in FEET + TIPS]

    def surface(self, x: float) -> float:
        """Height of whatever is under x: a pillar's top, else the water (0)."""
        for p in self.course.pads:
            if p.x0 < x < p.x1:
                return p.top
        return 0.0

    def tick(self, throttle: float, tilt: float) -> None:
        """One physics step (1/60 s) with the sticks at (throttle, tilt)."""
        ph, co = self.ph, self.course
        dt = ph.dt
        cmd = (throttle + 1.0) / 2.0
        self.motor += (cmd - self.motor) * dt / ph.motor_tau
        acc = self.motor * ph.thrust_max * G / co.mass
        if self.grounded and acc <= G:  # still sitting: friction slows any slide
            self.vx *= 0.85
            self.x += self.vx * dt
            self.ticks += 1
            self._limits()
            return
        self.grounded = False
        before = self.points()
        alpha = ph.kp * (tilt * ph.tilt_max - self.th) - ph.kd * self.om
        alpha = max(-ph.alpha_max, min(ph.alpha_max, alpha))
        alpha *= min(1.0, 0.3 + 1.4 * self.motor)
        self.om += alpha * dt
        self.th += self.om * dt
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        t = self.ticks * dt
        air = co.wind + co.gust * math.sin(2 * math.pi * t / co.gust_period + co.gust_phase)
        ax = acc * math.sin(self.th) - ph.drag_lin * self.vx - ph.drag_quad * self.vx * speed
        ay = acc * math.cos(self.th) - G - ph.drag_lin * self.vy - ph.drag_quad * self.vy * speed
        ax += air / co.mass
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.ticks += 1
        self._contacts(before)
        if self.outcome is None:
            self._limits()

    def _limits(self) -> None:
        ph = self.ph
        if self.x < -0.5 or self.x > ph.width + 0.5 or self.y > ph.ceiling:
            self.outcome = "lost"
        elif self.ticks >= round(ph.t_max / ph.dt):
            self.outcome = "timeout"

    def _contacts(self, before: list[tuple[float, float]]) -> None:
        ph = self.ph
        landing = None
        for k, (px, py) in enumerate(self.points()):
            for p in self.course.pads:
                if p.x0 < px < p.x1 and py < p.top:
                    if k < 2 and before[k][1] >= p.top - FOOT_BAND:
                        if self.vy <= 0.0:
                            landing = p  # a foot coming down on the deck
                        # rising: the foot just pivots on the deck as it lifts off
                    else:
                        self.crash = "side"  # a side or a rotor hit the pillar
                        self._end("crashed")
                        return
            if py < 0.0:
                self._end("water")
                return
        if landing is None:
            return
        gentle = (
            self.vy >= -ph.land_vy and abs(self.vx) <= ph.land_vx and abs(self.th) <= ph.land_tilt
        )
        if not gentle:
            self.crash = "hard"
            self._end("crashed")
        elif landing is self.course.target and landing.x0 <= self.x <= landing.x1:
            self._end("landed")
        else:  # sitting down on a pad: the legs level it and friction slows it
            self.grounded = True
            self.th = 0.0
            self.om = 0.0
            self.y = landing.top + LEG
            self.vy = 0.0
            self.vx *= 0.85

    def _end(self, outcome: str) -> None:
        self.touch = (self.vx, self.vy)
        self.outcome = outcome

    def step(self, throttle: float, tilt: float) -> None:
        """One stick update: hold the sticks for `substeps` physics steps."""
        throttle = max(-1.0, min(1.0, throttle))
        tilt = max(-1.0, min(1.0, tilt))
        r = self.ph.stick_rate
        if r > 0:  # a thumb can't flick a stick end to end in one update
            t0, s0 = self.stick
            throttle = t0 + max(-r, min(r, throttle - t0))
            tilt = s0 + max(-r, min(r, tilt - s0))
        for _ in range(self.ph.substeps):
            self.tick(throttle, tilt)
            if self.done:
                break
        self.stick = (throttle, tilt)


N_OBS = 14


def observe(d: Drone) -> list[float]:
    """What the pilot network sees (no wind, no mass: it has to feel those)."""
    gx, gy = d.goal()
    s = d.course.start
    under = d.y - LEG - d.surface(d.x)
    return [
        (gx - d.x) / 10,
        (gy - d.y) / 10,
        d.vx / 5,
        d.vy / 5,
        d.th / d.ph.tilt_max,
        d.om / 5,
        d.motor * 2 - 1,
        d.course.target.half / 1.3,
        (s.cx - d.x) / 10,
        (s.top + LEG - d.y) / 10,
        d.y / 10,
        max(-1.0, min(2.0, under / 5)),
        d.stick[0],
        d.stick[1],
    ]


@dataclass(frozen=True)
class Reward:
    progress: float = 1.0  # per metre closer to the landing spot
    landed: float = 10.0
    failed: float = -5.0  # crashed, water, lost
    smooth: float = 0.0  # per unit of squared stick change, per update


def reward(d: Drone, before: float, stick_before: tuple[float, float], rw: Reward) -> float:
    r = rw.progress * (before - d.distance())
    if d.outcome == "landed":
        r += rw.landed
    elif d.outcome in ("crashed", "water", "lost"):
        r += rw.failed
    if rw.smooth:
        da = d.stick[0] - stick_before[0]
        db = d.stick[1] - stick_before[1]
        r -= rw.smooth * (da * da + db * db)
    return r


def autopilot(d: Drone) -> tuple[float, float]:
    """A hand-written flight controller, the engineer's answer: climb above both pillars, fly
    across, stop over the pad, come down slowly. It sees what the network sees (no wind, no
    mass)."""
    ph = d.ph
    gx, gy = d.goal()
    s, t = d.course.start, d.course.target
    cruise = max(s.top, t.top) + LEG + 1.8
    dx = gx - d.x
    if abs(dx) > 0.35 or abs(d.vx) > 0.4:
        y_want = cruise
        vx_want = max(-4.0, min(4.0, 0.9 * dx)) if d.y > cruise - 1.2 else 0.0
        vy_want = max(-2.0, min(2.0, 1.5 * (y_want - d.y)))
    else:
        vx_want = 0.6 * dx
        vy_want = -max(0.5, min(1.2, 0.8 * (d.y - gy)))
    tilt = max(-1.0, min(1.0, 0.45 * (vx_want - d.vx)))
    hover = 1.0 / (ph.thrust_max * max(0.5, math.cos(d.th)))
    frac = hover + 0.18 * (vy_want - d.vy)
    return max(-1.0, min(1.0, 2 * frac - 1)), tilt


@dataclass(frozen=True)
class Config:
    physics: Physics = Physics()
    weather: Weather = CALM
    reward: Reward = Reward()


class DroneFlight(gym.Env):
    """Gymnasium wrapper: action = (throttle, tilt) in [-1, 1]^2, 20 updates per second."""

    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (N_OBS,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.drone: Drone | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        course_seed = (options or {}).get("course")
        if course_seed is None:
            course_seed = int(self.np_random.integers(0, 2**31))
        cfg = self.config
        self.drone = Drone(make_course(course_seed, cfg.weather, cfg.physics), cfg.physics)
        return np.asarray(observe(self.drone), np.float32), {}

    def step(self, action):
        d = self.drone
        before, stick_before = d.distance(), d.stick
        d.step(float(action[0]), float(action[1]))
        r = reward(d, before, stick_before, self.config.reward)
        obs = np.clip(np.asarray(observe(d), np.float32), -5.0, 5.0)
        info = {}
        terminated = d.done and d.outcome != "timeout"
        truncated = d.outcome == "timeout"
        if d.done:
            info = {"outcome": d.outcome, "time": d.time}
        return obs, r, terminated, truncated, info


def fly(d: Drone, pilot) -> Drone:
    """Flies to the end with pilot(drone) -> (throttle, tilt)."""
    while not d.done:
        d.step(*pilot(d))
    return d


class Net:
    """The exported actor: MLP layers [{w, b}], tanh (PPO) or relu (SAC) between them; SAC
    squashes the output with tanh, PPO's is clipped to the stick range."""

    def __init__(self, spec: dict):
        self.layers = [
            (np.asarray(L["w"], float), np.asarray(L["b"], float)) for L in spec["layers"]
        ]
        self.act = np.tanh if spec.get("act", "tanh") == "tanh" else lambda v: np.maximum(v, 0.0)
        self.squash = spec.get("squash", False)
        self.physics = spec.get("physics", {})

    def __call__(self, obs) -> np.ndarray:
        h = np.clip(np.asarray(obs, float), -5.0, 5.0)
        for i, (w, b) in enumerate(self.layers):
            h = w @ h + b
            if i < len(self.layers) - 1:
                h = self.act(h)
        return np.tanh(h) if self.squash else np.clip(h, -1.0, 1.0)

    def pilot(self, d: Drone) -> tuple[float, float]:
        a = self(observe(d))
        return float(a[0]), float(a[1])
