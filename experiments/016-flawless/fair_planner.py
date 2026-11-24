"""Post 15's planner, limited to what a pilot can know (post 16).

Post 15's planner (015-teacher/expert.py) lands 200 of 200 gauntlet courses, but its copies of
the simulator know things no pilot can sense. Here they are taken away:
- waypoints: the copies know nothing of waypoints not yet reached (post 15's `fair`);
- frogs (--frogs seen): the copies' frogs strike when a pilot who remembers every strike it
  has seen would expect them to. From the strikes seen so far (only within sight, 40 m), the
  next one falls in an interval [lo, hi]: one seen, the beat is anywhere in the course's range
  (3-4 s on the test courses); two or more, it is measured, up to the jitter (each strike is
  up to 0.1 s off the beat) and the sensing (0.05 s). Each plan is flown under three
  hypotheses (the next strike at lo, the middle and hi) and scores by its worst;
- frogs (--frogs ghost): as "seen", plus a guarantee about the leap: in the copies, a frog
  whose next strike could start at any time in [lo, hi] owns its column (2.8 m either side)
  from lo until a leap started at hi would have landed (0.35 s later), for each of its next
  three strikes; a copy found there then is caught. Where the frog holds and how it goes back
  is still checked under the three hypotheses. Birds and missiles get a margin too: a copy
  passing within 0.25 m (birds) or 0.4 m (missiles) of being hit counts as hit, within 0.3 m
  of a frog likewise, and a copy lands only within 90% of the landing limits (speed, tilt);
- air (--air nominal): the copies and the controllers know neither the wind, nor the gusts,
  nor the drone's mass: no wind, the average mass. Re-planning every 0.2 s has to correct for
  them, as feedback;
- air (--air estimated): they are estimated from the drone's own motion, as a pilot that knows
  its own dynamics could. Each update is replayed on a copy in still air with a mass of 1; how
  far the real drone's climb rate differs gives the mass, how far its sideways speed differs
  the air's push. The push is fit as a steady wind plus a sine (the gusts), over the last 10 s.

uv run experiments/016-flawless/fair_planner.py --frogs seen --air estimated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))
sys.path.insert(0, str(HERE.parent / "015-teacher"))

import expert  # noqa: E402
from expert import Planner  # noqa: E402
from gauntlet_env import (  # noqa: E402
    BIRD,
    FLYING,
    FROG,
    FROG_HIT,
    FROG_JITTER,
    FROG_REACH,
    GOLDEN,
    K_BIRDS,
    K_MISSILES,
    LEAP,
    MISSILE,
    OUTCOMES,
    P_HIT,
    P_KIND,
    ROUGH,
    SIGHT,
    Fleet,
    G,
    make_course,
)

SENSE = 0.05  # a strike's start is seen to within one update
ZONE = FROG_REACH + FROG_HIT + 0.2  # a strike can reach this far from the frog's column
MARGIN_F = 0.3
STRICT = 0.9  # the copies land only within this share of the landing limits
MARGIN_P = np.array([0.25] * K_BIRDS + [0.4] * K_MISSILES)  # past the hit radius
LEAPING = LEAP + SENSE  # a strike's leap, which goes where the drone was when it started


WAITS_HIGH = (1.2, 2.5)  # m above the pad's landing height


def _more_moves() -> None:
    """Post 15's moves, plus waiting on the far side of the pad (a drone that overshot it, or
    arrived while the frog owned its column, can still set up a landing from there) and a
    faster run in to land (7 m/s: the 45-degree drone can brake from it). Appended, so post
    15's move numbers keep their meaning."""
    moves = [tuple(m) for m in expert.MOVES]
    extra = [(expert.WAIT, -off, b) for off in (3.2, 5.0) for b in (0.1, 0.45)]
    # and higher: a leap at a drone waiting there holds above the way in to the pad
    extra += [(expert.WAIT, s * off, b) for s in (1, -1) for off in (3.2, 5.0) for b in WAITS_HIGH]
    extra += [(expert.LAND, 7.0, 0.0)]
    moves += [m for m in extra if m not in moves]
    expert.MOVES = np.array(moves)
    expert.MK = expert.MOVES[:, 0].astype(int)
    expert.MA, expert.MB = expert.MOVES[:, 1], expert.MOVES[:, 2]
    expert.PAD_MOVES = np.nonzero((expert.MK == expert.LAND) | (expert.MK == expert.WAIT))[0]


_more_moves()

_control = expert.control


def _gentle_control(f: Fleet, m: np.ndarray) -> np.ndarray:
    """Post 15's controller, except that touching down (over the pad, under 0.6 m above it)
    tilts at most 80% of the touchdown limit. Post 15 capped it at 0.6 of the drone's tilt
    limit: 18 degrees at 30, but 27 at 45, past the 20 a landing allows."""
    a = _control(f, m)
    gx, _ = f.goal()
    half = (f.t_x1 - f.t_x0) / 2
    over = (np.abs(gx - f.x) < half + 0.3) & (f.y - expert.LEG - f.t_top < 0.6)
    landing = (expert.MK[m] == expert.LAND) & over  # touching down, not the run in
    cap = 0.8 * f.ph.land_tilt / f.ph.tilt_max
    a[:, 1] = np.where(landing, np.clip(a[:, 1], -cap, cap), a[:, 1])
    return a


expert.control = _gentle_control

_step = Fleet.step


def _ghost_step(self, actions):
    """Fleet.step, and for copies that carry frog windows (ghost: (n, frogs, strikes, 2)),
    a drone inside a frog's zone during one of its windows is caught."""
    _step(self, actions)
    gh = getattr(self, "ghost", None)
    if gh is None:
        return
    # a margin round birds and missiles: the copies' own flight is only as good as the air
    # estimate, so a plan that only just misses in a copy may not miss for real
    x, y, _, _, on = self.pursuers()
    near = np.hypot(x - self.x[:, None], y - self.y[:, None]) < P_HIT + MARGIN_P
    caught = (self.outcome == FLYING)[:, None] & on & near
    if caught.any():
        rows, ks = np.nonzero(caught)
        self.outcome[rows] = np.where(P_KIND[ks] == 1, MISSILE, BIRD)
    if self.frog_on.any():  # and round the frogs themselves
        fx, fy, _, _ = self.frog_pos()
        d = np.hypot(fx - self.x[:, None], fy - self.y[:, None])
        close = (self.frog_on & (d < FROG_HIT + MARGIN_F)).any(1)
        self.outcome[(self.outcome == FLYING) & close] = FROG
    t = (self.ticks * self.ph.dt)[:, None, None]
    inside = np.abs(self.x[:, None] - self.frog_p[..., 0]) < ZONE
    live = (self.outcome == FLYING)[:, None] & self.frog_on & ~self.frog_past & inside
    hit = (live[..., None] & (t >= gh[..., 0]) & (t <= gh[..., 1])).any((1, 2))
    self.outcome[hit] = FROG


Fleet.step = _ghost_step
MASS0 = 1.0 + 0.5 * ROUGH.heavy  # the average drone
FAR = 80.0  # frogs farther than this can't matter within a look-ahead (7.5 s at 10 m/s)


def nominal(f: Fleet) -> Fleet:
    """A copy of f whose air and mass are what a pilot would assume: still air, average
    mass."""
    g = f.take(np.arange(f.n))
    g.wind[:] = 0.0
    g.gust[:] = 0.0
    g.mass[:] = MASS0
    return g


PERIODS = np.arange(2.0, 5.001, 0.05)  # the gusts' periods, as the weather has them


class FairPlanner(Planner):
    def __init__(self, n: int, frogs: str = "ghost", air: str = "estimated", hyps: int = 3, **kw):
        super().__init__(n, **kw)
        self.frogs, self.air, self.hyps = frogs, air, hyps
        # the air and mass as estimated (start: still air, the average mass)
        self.est = {"wind": np.zeros(n), "gust": np.zeros(n), "period": np.full(n, 3.5),
                    "phase": np.zeros(n), "mass": np.full(n, MASS0)}  # fmt: skip
        self.inv_m: list[list[float]] = [[] for _ in range(n)]
        self.pushes: list[list[tuple]] = [[] for _ in range(n)]
        self._prev = None
        self.seen_n = np.zeros((n, 2), np.int64)  # strikes counted so far
        self.starts: list[list[list[float]]] = [[[], []] for _ in range(n)]  # seen ones
        self.first_seen = np.full((n, 2), np.nan)
        self._h = None  # the hypothesis the copies are being made under

    # --- what the pilot has seen ---------------------------------------------------------
    def observe(self, f: Fleet) -> None:
        t = f.ticks * f.ph.dt
        px, py = f.perch()
        sight = f.frog_on & (np.hypot(px - f.x[:, None], py - f.y[:, None]) < SIGHT)
        for i, k in zip(*np.nonzero(f.frog_n != self.seen_n), strict=True):
            if sight[i, k]:
                self.starts[i][k].append(float(f.frog_s[i, k]))
        self.seen_n[:] = f.frog_n
        new = sight & np.isnan(self.first_seen)
        self.first_seen[new] = np.broadcast_to(t[:, None], new.shape)[new]

    def interval(self, f: Fleet, i: int, k: int):
        """(lo, hi, beat): when frog k of drone i strikes next, as far as the pilot knows."""
        p_lo, p_hi = f.feat.frog_period
        jit = FROG_JITTER
        now = float(f.ticks[i] * f.ph.dt)
        s = self.starts[i][k]
        if len(s) >= 2:
            est = (s[-1] - s[0]) / (len(s) - 1)
            d = (2 * jit + 2 * SENSE) / (len(s) - 1)
            b_lo, b_hi = max(p_lo, est - d), min(p_hi, est + d)
            b_lo, b_hi = min(b_lo, b_hi), max(b_lo, b_hi)
            lo, hi = s[-1] - SENSE + b_lo - 2 * jit, s[-1] + SENSE + b_hi + 2 * jit
        elif len(s) == 1:
            lo, hi = s[0] - SENSE + p_lo - 2 * jit, s[0] + SENSE + p_hi + 2 * jit
            b_lo, b_hi = p_lo, p_hi
        else:  # none seen: it may strike at any moment, within a beat of first sighting
            start = self.first_seen[i, k] if np.isfinite(self.first_seen[i, k]) else now
            lo, hi = now, max(now, start) + p_hi + 2 * jit
            b_lo, b_hi = p_lo, p_hi
        lo = max(lo, now)  # in sight, a strike that came would have been seen
        self._beats = (b_lo, b_hi)
        return lo, max(hi, lo), 0.5 * (b_lo + b_hi)

    # --- the air, from the drone's own motion -------------------------------------------
    def assumed(self, f: Fleet) -> Fleet:
        """A copy of f with the air and mass the pilot assumes (all of f's drones), whose
        landings must be gentler than the real limits (a margin, as for the threats)."""
        g = nominal(f) if self.air == "nominal" else f.take(np.arange(f.n))
        ph = getattr(self, "ph_real", f.ph)  # (f may be a copy already)
        g.ph = replace(ph, land_vx=STRICT * ph.land_vx, land_vy=STRICT * ph.land_vy,
                       land_tilt=STRICT * ph.land_tilt)  # fmt: skip
        if self.air == "nominal":
            return g
        if self.air == "estimated":
            for k, v in self.est.items():
                getattr(g, k)[:] = v
        return g

    def remember(self, f: Fleet, a: np.ndarray) -> None:
        """Keep this update's start and sticks, to replay in still air at the next."""
        if self.air != "estimated":
            return
        g = f.take(np.arange(f.n))
        g.wind[:] = 0.0
        g.gust[:] = 0.0
        g.mass[:] = 1.0
        self._prev = (g, a.copy(), f.ticks.copy(), g.motor.copy(), g.th.copy())

    def estimate(self, f: Fleet) -> None:
        if self.air != "estimated" or self._prev is None:
            return
        g, a, t0, m0, th0 = self._prev
        flying0 = g.outcome == FLYING
        g.step(a)
        ph = f.ph
        span = 3 * ph.dt
        ok = flying0 & (f.outcome == FLYING) & (g.outcome == FLYING) & (f.ticks - t0 == 3)
        ok &= (f.grounded == 0) & (g.grounded == 0) & (g.ticks - t0 == 3)
        acc = 0.5 * (m0 + g.motor) * ph.thrust_max * G * expert.thrust_scale(g.charge)
        th = 0.5 * (th0 + g.th)
        ky, kx = span * acc * np.cos(th), span * acc * np.sin(th)
        ok &= ky > 0.5
        for i in np.nonzero(ok)[0]:
            inv = 1.0 + (f.vy[i] - g.vy[i]) / ky[i]
            self.inv_m[i].append(float(np.clip(inv, 0.5, 1.5)))
            inv_m = float(np.mean(self.inv_m[i][-200:]))
            push = (f.vx[i] - g.vx[i] - kx[i] * (inv_m - 1.0)) / (span * inv_m)
            self.pushes[i].append(((t0[i] + 1.5) * ph.dt, float(push)))
            self.est["mass"][i] = float(np.clip(1.0 / inv_m, 1.0, 1.0 + ROUGH.heavy))
            if len(self.pushes[i]) % 4 == 0:
                self.fit(i)

    def fit(self, i: int) -> None:
        tp = np.array(self.pushes[i][-200:])  # the last 10 s
        t, y = tp[:, 0], tp[:, 1]
        e = self.est
        if t[-1] - t[0] < 1.5:  # too short to see a gust: a steady push
            e["wind"][i], e["gust"][i] = float(np.mean(y[-20:])), 0.0
            return
        best = None
        for p in PERIODS:
            w = 2 * np.pi / p
            X = np.stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)], 1)
            c, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            err = float(((X @ c - y) ** 2).sum())
            if best is None or err < best[0]:
                best = (err, p, c)
        _, p, (c0, ca, cb) = best
        e["wind"][i] = float(np.clip(c0, -ROUGH.wind, ROUGH.wind))
        e["gust"][i] = float(min(np.hypot(ca, cb), ROUGH.gust))
        e["period"][i], e["phase"][i] = p, float(np.arctan2(cb, ca))

    def believe(self, f: Fleet, idx, h: int) -> Fleet:
        """A copy of f whose frogs (of drones idx) strike next at hypothesis h of H."""
        g = self.assumed(f)
        if self.frogs == "true":
            return g
        if self.frogs == "ghost":
            g.ghost = np.full((g.n, 2, 3, 2), -1e9)
        for i in idx:
            for k in range(2):
                if not g.frog_on[i, k] or g.frog_past[i, k]:
                    continue
                lo, hi, beat = self.interval(f, i, k)
                when = lo + (hi - lo) * (h / max(self.hyps - 1, 1))
                n = g.frog_n[i, k]
                g.frog_p[i, k, 2] = beat
                g.frog_p[i, k, 3] = when - beat * n
                g.frog_p[i, k, 4] = np.mod(0.5 - n * GOLDEN, 1.0)  # no jitter on that one
                g.sw_after[i] = 10**6
                if self.frogs == "ghost":
                    b_lo, b_hi = self._beats
                    jit = 2 * FROG_JITTER
                    for j in range(3):
                        g.ghost[i, k, j] = (lo + j * (b_lo - jit), hi + j * (b_hi + jit) + LEAPING)
        return g

    # --- planning ----------------------------------------------------------------------
    def evaluate(self, f: Fleet, idx: np.ndarray):
        hyps = self.hyps if (self.frogs in ("seen", "ghost") and self._near(f, idx)) else 1
        if self.frogs == "true" and self.air == "true":
            return super().evaluate(f, idx)
        out, best = None, None
        for h in range(hyps):
            hh = h if hyps > 1 else (self.hyps - 1) // 2
            groups, sc, sim = super().evaluate(self.believe(f, idx, hh), idx)
            best = sc if best is None else np.minimum(best, sc)
            out = (groups, sim)
        return out[0], best, out[1]

    def _near(self, f: Fleet, idx) -> bool:
        live = f.frog_on[idx] & ~f.frog_past[idx]
        close = np.abs(f.frog_p[idx, :, 0] - f.x[idx][:, None]) < FAR
        return bool((live & close).any())

    def act(self, f: Fleet) -> np.ndarray:
        self.ph_real = f.ph
        self.observe(f)
        self.estimate(f)
        flying = np.nonzero(f.outcome == FLYING)[0]
        due = flying[self.fresh[flying] | (self.age[flying] >= self.every)]
        if self.frogs in ("seen", "ghost"):  # drones near a frog: several hypotheses, apart
            live = f.frog_on[due] & ~f.frog_past[due]
            close = np.abs(f.frog_p[due, :, 0] - f.x[due][:, None]) < FAR
            near = (live & close).any(1)
            self.decide(f, due[near])
            self.decide(f, due[~near])
        else:
            self.decide(f, due)
        a = expert.control(self.assumed(f) if self.air != "true" else f, self.current())
        self.remember(f, a)
        self.age += 1
        return a

    def decide(self, f: Fleet, idx: np.ndarray) -> None:
        # with nominal air, everything the decision copies or compares with assumes still air
        # and the average mass (the copies made from it too)
        self._decide(self.assumed(f) if self.air != "true" else f, idx)

    def _decide(self, f: Fleet, idx: np.ndarray) -> None:
        """Post 15's decide, with one change: the band's plain cruise heads toward the pad, not
        along the route. Past the pad (an overshoot), "along the route" pointed away from it,
        and the band kept the drone hovering at the world's edge."""
        if len(idx) == 0:
            return
        groups, sc, _ = self.evaluate(f, idx)
        act0 = self.act0
        g = f.take(idx)
        gx, _ = g.goal()
        g.dir = np.where(gx >= g.x, 1.0, -1.0)
        canon = expert.control(g, np.full(len(idx), expert.CANON))
        k = 0
        for q, (grp, i) in enumerate(zip(groups, idx, strict=True)):
            s = sc[k : k + len(grp)].copy()
            if not self.fresh[i]:
                s[0] += self.loyal
            j = int(np.argmax(s))
            if self.band and s[j] > -500:
                near = np.nonzero(s >= s[j] - self.band)[0]
                d = ((act0[k + near] - canon[q]) ** 2).sum(1)
                j = int(near[np.lexsort((-s[near], d))[0]])
            self.plan[i] = grp[j]
            k += len(grp)
        self.age[idx] = 0
        self.fresh[idx] = False
        self.decisions += len(idx)


def fly(feat, seeds, weather=ROUGH, ph=None, **kw):
    f = Fleet(len(seeds), ph, feat=feat)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(int(s), weather, feat, f.ph))
    pl = FairPlanner(f.n, **kw)
    t = 0
    while (f.outcome == FLYING).any() and t < 1300:
        f.step(pl.act(f))
        t += 1
    return f, pl


def _job(args):
    test, seeds, kw = args
    os.environ["OMP_NUM_THREADS"] = "1"
    from dataclasses import replace

    from evaluate import TESTS
    from gauntlet_env import Physics

    kw = dict(kw)
    ph = replace(Physics(), **kw.pop("physics", {}))
    f, pl = fly(TESTS[test][0], seeds, ph=ph, **kw)
    out = []
    for i, s in enumerate(seeds):
        out.append({"seed": int(s), "outcome": OUTCOMES[f.outcome[i]],
                    "time": round(float(f.ticks[i] * f.ph.dt), 2), "x": float(f.x[i]),
                    "togo": float(abs(f.goal()[0][i] - f.x[i])), "hit_by": int(f.hit_enc[i]),
                    "met": f.enc_count[i].tolist()})  # fmt: skip
    return out, pl.decisions


def run(test, seeds, workers=10, chunk=5, **kw):
    from multiprocessing import Pool

    seeds = list(seeds)
    jobs = [(test, seeds[i : i + chunk], kw) for i in range(0, len(seeds), chunk)]
    with Pool(workers) as p:
        parts = p.map(_job, jobs)
    return sum((r for r, _ in parts), []), sum(d for _, d in parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=20_000)
    ap.add_argument("--tests", nargs="+", default=["gauntlet"])
    ap.add_argument("--frogs", choices=("true", "seen", "ghost"), default="ghost")
    ap.add_argument("--air", choices=("true", "nominal", "estimated"), default="estimated")
    ap.add_argument("--hyps", type=int, default=3)
    ap.add_argument("--deep", type=float, default=3.0, help="deepening searches this long (s)")
    ap.add_argument("--band", type=float, default=16.0, help="post 15's copyability band")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="these test courses only")
    ap.add_argument("--tilt", type=float, default=None, help="the drone's tilt limit (degrees)")
    ap.add_argument("--thrust", type=float, default=None, help="thrust at full throttle (g)")
    args = ap.parse_args()
    kw = {
        "frogs": args.frogs,
        "air": args.air,
        "hyps": args.hyps,
        "deep": args.deep,
        "band": args.band,
    }
    phys = {}
    if args.tilt is not None:
        phys["tilt_max"] = float(np.radians(args.tilt))
    if args.thrust is not None:
        phys["thrust_max"] = args.thrust
    if phys:
        kw["physics"] = phys
    saved = {}
    for test in args.tests:
        t0 = time.time()
        seeds = args.seeds or range(args.first_seed, args.first_seed + args.n)
        args.n = len(seeds)
        res, dec = run(test, seeds, args.workers, args.chunk, **kw)
        c = Counter(r["outcome"] for r in res)
        print(f"{test} {kw}: landed {c['landed']}/{args.n}  {dict(c)}  "
              f"({time.time() - t0:.0f} s, {dec} decisions)", flush=True)  # fmt: skip
        for r in res:
            if r["outcome"] != "landed":
                print("   ", r, flush=True)
        saved[test] = {"flights": res, "seconds": round(time.time() - t0), "kw": kw}
    if args.out:
        (HERE / "runs").mkdir(exist_ok=True)
        (HERE / "runs" / f"{args.out}.json").write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
