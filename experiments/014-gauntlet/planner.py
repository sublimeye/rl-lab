"""A planning pilot with the simulator as its crystal ball (a privileged expert, not a fair pilot).

Every few updates, each drone is copied once per candidate plan (a cruise height and speed, or
holding still at a height), every copy is flown ahead a few seconds in the real simulator, and
the drone follows the plan that went best. It sees what no network sees: where every bird will
steer and when every frog will strike. It answers one question: how well can these courses be
flown at all? If it lands nearly every course, a network can be taught from it.

uv run experiments/014-gauntlet/planner.py [--n 200] [--horizon 3.0] [--every 5]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from evaluate import TESTS
from gauntlet_env import FAILED, FLYING, LANDED, OUTCOMES, ROUGH, Fleet, make_course

HEIGHTS = (1.5, 3.0, 5.0, 7.0, 9.0, 11.0)
SPEEDS = (-4.0, 0.0, 4.0, 7.0, 10.0)
PLANS = np.array([(h, v) for h in HEIGHTS for v in SPEEDS])  # v == 0: hold still


def plan_pilot(f: Fleet, h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per drone: cruise at height h and speed v toward the pad (landing when over it), or
    hold still at height h when v == 0, or back off when v < 0."""
    gx, gy = f.goal()
    dx = gx - f.x
    go = v > 0
    near = (np.abs(dx) < 6.0) & go
    want_vx = np.where(near, np.clip(0.8 * dx, -2.5, 2.5), f.dir * v)
    want_y = np.where(near & (np.abs(dx) < 1.0), gy - 0.5, np.maximum(h, gy + 1.5))
    want_vy = np.clip(1.2 * (want_y - f.y), -2.5, 2.5)
    tilt = np.clip(0.35 * (want_vx - f.vx), -1, 1)
    thr = np.clip(0.05 + 0.6 * (want_vy - f.vy), -1, 1)
    return np.stack([thr, tilt], 1)


def score(sim: Fleet, start_dist: np.ndarray, survived: np.ndarray, horizon_u: int) -> np.ndarray:
    failed = np.isin(sim.outcome, FAILED)
    landed = sim.outcome == LANDED
    progress = start_dist - sim.distance()
    s = progress.copy()
    s = np.where(landed, 1000.0 - survived, s)
    s = np.where(failed, -1000.0 + 10.0 * survived / horizon_u, s)
    return s


def rollout(sim: Fleet, hp, vp, updates: int, alive_u: np.ndarray) -> None:
    for _ in range(updates):
        flying = sim.outcome == FLYING
        if not flying.any():
            break
        alive_u += flying
        sim.step(plan_pilot(sim, hp, vp))


def fly_planned(feat, seeds, horizon=3.5, every=5, first=1.0, second=None):
    """Two-step search: each plan is held for `first` s, then the best of the plans `second`
    (default: all of them) is followed to the end of the horizon. A plan scores as well as its
    best continuation, so a drone can trigger the next waypoint knowing it can dodge after."""
    f = Fleet(len(seeds), feat=feat)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(s, ROUGH, feat, f.ph))
    upd = f.ph.dt * f.ph.substeps
    horizon_u, first_u = int(round(horizon / upd)), int(round(first / upd))
    P = len(PLANS)
    P2 = PLANS if second is None else PLANS[second]
    Q = len(P2)
    choice = np.full(f.n, int(np.argmin(np.abs(PLANS - (7.0, 7.0)).sum(1))))
    t = 0
    while (f.outcome == FLYING).any() and t < 1300:
        if t % every == 0:
            live = np.nonzero(f.outcome == FLYING)[0]
            sim = f.take(np.repeat(live, P))
            d0 = sim.distance()
            alive = np.zeros(sim.n)
            rollout(
                sim,
                np.tile(PLANS[:, 0], len(live)),
                np.tile(PLANS[:, 1], len(live)),
                first_u,
                alive,
            )
            sim2 = sim.take(np.repeat(np.arange(sim.n), Q))
            alive2 = np.repeat(alive, Q)
            rollout(
                sim2,
                np.tile(P2[:, 0], sim.n),
                np.tile(P2[:, 1], sim.n),
                horizon_u - first_u,
                alive2,
            )
            sc = score(sim2, np.repeat(d0, Q), alive2, horizon_u).reshape(len(live), P, Q)
            sc = sc.max(2)
            # a little loyalty to the current plan, so the drone doesn't dither between equals
            sc[np.arange(len(live)), choice[live]] += 0.5
            choice[live] = sc.argmax(1)
        f.step(plan_pilot(f, PLANS[choice, 0], PLANS[choice, 1]))
        t += 1
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--horizon", type=float, default=3.5)
    ap.add_argument("--first", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--tests", nargs="+", default=["gauntlet"])
    args = ap.parse_args()
    seeds = range(20_000, 20_000 + args.n)
    for test in args.tests:
        t0 = time.time()
        f = fly_planned(TESTS[test][0], seeds, args.horizon, args.every, args.first)
        c = Counter(OUTCOMES[o] for o in f.outcome)
        print(
            f"{test}: landed {c['landed']}/{args.n}  {dict(c)}  "
            f"({time.time() - t0:.0f} s, horizon {args.horizon} s, every {args.every})"
        )


if __name__ == "__main__":
    main()
