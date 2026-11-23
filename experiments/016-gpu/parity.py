"""Does the torch step (torch_fleet.py) match the numpy Fleet's, as the planner's copies use it?

A few real drones fly courses (numpy, encounters on, so birds, missiles and frogs come) under
post 15's controller with random moves. At checkpoints, each is copied as the planner copies
it (waypoints not yet reached cleared; the air, mass and landing limits it assumes; with or
without ghost windows), and the copies fly ahead in numpy and in torch with the same sticks:
half the controller's for random moves (landing runs and waits by the pad among them), half
random sticks past full deflection. Every field is compared after every update.

uv run experiments/016-gpu/parity.py [--drones 4] [--copies 8] [--horizon 90]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
for d in (HERE, HERE.parent / "014-gauntlet", HERE.parent / "015-teacher",
          HERE.parent / "016-flawless"):  # fmt: skip
    sys.path.insert(0, str(d))

import fair_planner as fp  # noqa: E402
import torch_fleet as tf  # noqa: E402

from gauntlet_env import (  # noqa: E402
    FLYING,
    FULL,
    MAX_ENC,
    OUTCOMES,
    ROUGH,
    Fleet,
    Physics,
    make_course,
)

expert = fp.expert
PH45 = replace(Physics(), tilt_max=float(np.radians(45.0)))
NM = len(expert.MOVES)


def compare(f: Fleet, S: dict, tol: float):
    """Largest float difference, and the first field that differs past tol or exactly (ints,
    bools), or None."""
    worst = 0.0
    for k, kind in tf.STATE.items():
        a = np.asarray(getattr(f, k))
        b = S[k].cpu().numpy()
        if kind == "f":
            d = float(np.max(np.abs(a - b))) if a.size else 0.0
            worst = max(worst, d)
            if not d <= tol:
                return worst, f"{k}: max diff {d:.3g}"
        elif not np.array_equal(a.astype(np.int64), b.astype(np.int64)):
            rows = np.nonzero((a != b).reshape(len(a), -1).any(1))[0]
            return worst, f"{k}: rows {rows[:5].tolist()} numpy {a[rows[:2]]} torch {b[rows[:2]]}"
    return worst, None


def copies(f: Fleet, rows, rng, ghost: bool) -> Fleet:
    """The planner's kind of copies of drones `rows`: no pending waypoints, an assumed air and
    mass (off the truth by a little), stricter landing limits, maybe ghost windows."""
    g = f.take(rows)
    g.enc_at[:] = 1e9
    g.wind += rng.uniform(-0.5, 0.5, g.n)
    g.gust *= rng.uniform(0.5, 1.5, g.n)
    g.mass = np.clip(g.mass + rng.uniform(-0.1, 0.1, g.n), 1.0, 1.35)
    g.ph = replace(f.ph, land_vx=0.9 * f.ph.land_vx, land_vy=0.9 * f.ph.land_vy,
                   land_tilt=0.9 * f.ph.land_tilt)  # fmt: skip
    soon = rng.random(g.n) < 0.25  # some run out of time within the look-ahead
    g.end_tick[soon] = g.ticks[soon] + rng.integers(1, 270, int(soon.sum()))
    if ghost:
        t = g.ticks * g.ph.dt
        lo = t[:, None, None] + rng.uniform(-1.0, 3.0, (g.n, 2, 3))
        g.ghost = np.stack([lo, lo + rng.uniform(0.2, 1.5, lo.shape)], -1)
        g.ghost[rng.random((g.n, 2, 3)) < 0.3] = -1e9  # some frogs unknown
    return g


def fly_ahead(g: Fleet, horizon: int, rng, dtype, tol: float):
    S, C = tf.from_numpy(g, dtype=dtype)
    P = tf.params(g.ph, g.feat.battery)
    moves = rng.integers(0, NM, g.n)
    kind = rng.random(g.n)
    random, land = kind < 0.4, kind > 0.7  # random sticks; landing runs and waits
    pad_moves = expert.PAD_MOVES
    worst, events = 0.0, Counter()
    for u in range(horizon):
        if u % 12 == 0:
            moves = np.where(land, rng.choice(pad_moves, g.n), rng.integers(0, NM, g.n))
            moves = np.where(land & (u >= 36), expert.move_id(expert.LAND, 3.5), moves)
        a = expert.control(g, moves)
        a[random] = rng.uniform(-1.3, 1.3, (int(random.sum()), 2))
        n0, on0, sit0 = g.frog_n.copy(), g.p_on.copy(), (g.grounded == 1) & (g.outcome == FLYING)
        g.step(a)
        S = tf.step(S, C, torch.as_tensor(a, dtype=dtype), P)
        events["frog strikes"] += int((g.frog_n - n0).sum())
        events["pursuers gone"] += int((on0 & ~g.p_on).sum())
        events["sitting"] += int(sit0.sum())
        events["touchdowns"] += int(((g.grounded == 1) & ~sit0 & (g.outcome == FLYING)).sum())
        d, bad = compare(g, S, tol)
        worst = max(worst, d)
        if bad:
            return worst, events, f"update {u}: {bad}"
    events.update(OUTCOMES[o] for o in g.outcome)
    return worst, events, None


def run(drones=4, n_copies=8, horizon=90, seed=0, dtype=torch.float64, tol=1e-9, log=print):
    rng = np.random.default_rng(seed)
    f = Fleet(drones, PH45, feat=FULL)
    course = 20_000 + 50 * seed

    def new(i):
        nonlocal course
        f.set_course(i, make_course(course, ROUGH, FULL, f.ph))
        course += 1
        if i == 0:  # drone 0 starts next to the pad (landings, waits, the pad's frog)
            gx, gy = f.goal()
            f.x[0], f.y[0], f.grounded[0] = gx[0] - f.dir[0] * 4.0, gy[0] + 2.0, 0
            f.n_enc[0] = 0

    for i in range(drones):
        new(i)
    worst, events, checks = 0.0, Counter(), 0
    moves = np.full(drones, expert.DEFAULT)
    for u in range(400):
        if u % 40 == 0:  # a checkpoint: copies of every drone flying
            rows = np.repeat(np.nonzero(f.outcome == FLYING)[0], n_copies)
            if len(rows):
                g = copies(f, rows, rng, ghost=(u // 40) % 2 == 0)
                pend = g.enc_next < g.n_enc
                assert not (pend & (g.enc_at[np.arange(g.n), np.minimum(g.enc_next,
                            MAX_ENC - 1)] < 1e8)).any()  # fmt: skip
                d, ev, bad = fly_ahead(g, horizon, rng, dtype, tol)
                worst = max(worst, d)
                events += ev
                checks += 1
                if bad:
                    log(f"MISMATCH at checkpoint update {u}: {bad}")
                    return False, worst, events, checks
        if u % 10 == 0:  # the real drones: mostly cruising, now and then a random move
            moves = np.where(rng.random(drones) < 0.7, expert.CANON, rng.integers(0, NM, drones))
            moves[0] = rng.choice(expert.np.nonzero(expert.MK == expert.WAIT)[0])
        f.step(expert.control(f, moves))
        for i in np.nonzero(f.outcome != FLYING)[0]:
            new(i)
    return True, worst, events, checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", type=int, default=4)
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--float32", action="store_true", help="float32 (outcomes may flip)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.float32 else torch.float64
    tol = 1e-2 if args.float32 else 1e-9
    for s in args.seeds:
        t0 = time.time()
        ok, worst, ev, checks = run(args.drones, args.copies, args.horizon, s, dtype, tol)
        verdict = "MATCH" if ok else "FAIL"
        print(f"seed {s}: {verdict}  {checks} checkpoints, largest float difference "
              f"{worst:.2e}  ({time.time() - t0:.0f} s)\n  {dict(ev)}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
