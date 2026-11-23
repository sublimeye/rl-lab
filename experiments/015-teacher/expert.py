"""The teacher (post 15): a look-ahead planner that copies the simulator.

Every 0.2 s, each drone is copied once per candidate plan and every copy is flown ahead in the
real simulator (Fleet.take); the drone follows the plan that went best. It sees what no network
sees: where every bird in sight will steer and when every frog will strike. It is a privileged
expert, the teacher for the network in dagger.py, not a pilot a network could match.

A plan is up to three moves in a row, each a simple controller held for a while:
- cruise (height h, speed v along the route; v = 0 holds still, v < 0 backs off);
- jink: the sticks held at full deflection (full throttle or none, tilted or not), the sharp
  dodge the 014 planner lacked;
- land: fly to the pad low and fast, brake, and set down;
- wait: hold beside the pad, out of its frog's reach.

Candidates are every first move followed by every second move from a smaller set (a jink
there is followed by a recovery cruise), plus what is left of the plan chosen last time (so the
choice never gets worse: the world is deterministic). A copy scores 1000 for landing (less the
time it took), -1000 for failing (plus a little for lasting longer), and otherwise its progress
toward the pad, less a cost for passing close to a threat and a cost for every bird or missile
still after it when the look-ahead ends (without it, the planner puts off every waypoint whose
threat would arrive just past its horizon, and stalls; the 014 planner did). Every waypoint
still ahead is charged too, so putting one off never pays. Copies that end with a threat still
after them, or near the pad, get a search of their own from there (deepening), and so do
crossings of a waypoint, from the moment its threat appears.

A teacher also has to be fair: by default (fair=True) the copies know nothing of waypoints the
drone hasn't reached, so no new bird or missile appears in the look-ahead. The planner then
reacts only to threats a network could see too. (With foreknowledge it also lands 200/200, but
its best missile escapes start before the missile exists, which no network can learn.)

Last, a teacher has to be copyable: among the plans scoring within `band` (16) of the best, it
takes the one whose sticks right now are closest to a plain cruise's. Where the choice doesn't
matter it always does the same thing, and where it must swerve it swerves least.

uv run experiments/015-teacher/expert.py [--n 200] [--tests gauntlet] [--workers 10]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))

from gauntlet_env import (  # noqa: E402
    FAILED,
    FLYING,
    FROG_REACH,
    LANDED,
    LEG,
    MAX_ENC,
    OUTCOMES,
    ROUGH,
    Fleet,
    G,
    make_course,
    thrust_scale,
)

CRUISE, JINK, LAND, WAIT, LEVEL = range(5)


HEIGHTS = (1.5, 3.0, 5.0, 7.0, 9.0, 11.0)


def _table():
    moves = []  # (kind, a, b)
    for h in HEIGHTS:
        for v in (-3.0, 0.0, 4.0, 7.0, 10.0):
            moves.append((CRUISE, h, v))
    for thr in (1.0, -1.0):
        for tilt in (-1.0, 0.0, 1.0):
            moves.append((JINK, thr, tilt))
    for s in (2.5, 3.5, 5.0):
        moves.append((LAND, s, 0.0))
    for off in (3.2, 5.0):
        for b in (0.1, 0.45):
            moves.append((WAIT, off, b))
    for v in (4.0, 7.0, 10.0):
        moves.append((LEVEL, 0.0, v))
    return np.array(moves)


MOVES = _table()
MK, MA, MB = MOVES[:, 0].astype(int), MOVES[:, 1], MOVES[:, 2]


def move_id(kind, a, b=0.0) -> int:
    hit = np.nonzero((MK == kind) & np.isclose(MA, a) & np.isclose(MB, b))[0]
    assert len(hit) == 1, (kind, a, b)
    return int(hit[0])


CRUISES = np.nonzero(MK == CRUISE)[0]
JINKS = np.nonzero(MK == JINK)[0]
LEVELS = np.nonzero(MK == LEVEL)[0]
PAD_MOVES = np.nonzero((MK == LAND) | (MK == WAIT))[0]
DEFAULT = move_id(CRUISE, 7.0, 7.0)
CANON = move_id(CRUISE, 7.0, 10.0)  # the plain cruise the band prefers

PAD_NEAR = 25.0  # land and wait are candidates within this distance of the pad
PAD_DEEP = 8.0  # copies ending this close to the pad, not landed, get a landing search
# jinks: (throttle, tilt along the route), full deflection
JINK_STICKS = ((1.0, -1.0), (1.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0))
HOSTILE_PER_KIND = np.array([0, 1, 2, 1, 2, 0])  # aimed birds + missiles, in ENCOUNTERS order


def control(f: Fleet, m: np.ndarray) -> np.ndarray:
    """Sticks for every drone of f flying move m (an index into MOVES), (N, 2)."""
    kind, a, b = MK[m], MA[m], MB[m]
    gx, gy = f.goal()
    dx = gx - f.x
    d = f.dir
    half = (f.t_x1 - f.t_x0) / 2
    # cruise
    want_vx = d * b
    want_y = a
    vy_lim = np.full(f.n, 3.0)
    tilt_lim = np.ones(f.n)
    # land: approach just above the pad top, brake, descend once the feet are over it
    over = (np.abs(dx) < half - 0.15) & (np.abs(f.vx) < 1.2)
    near = np.abs(dx) < half + 1.0
    # below the pad's top and close beside it: climb before closing in (or hit its side)
    low = (f.y - LEG < f.t_top + 0.1) & (np.abs(dx) > half + 0.2) & (np.abs(dx) < half + 4.0)
    land_vx = np.where(low, 0.0, np.clip(2.0 * dx, -a, a))
    land_y = np.where(over, gy - 0.6, np.where(near, gy + 0.35, gy + 0.5))
    is_land = kind == LAND
    want_vx = np.where(is_land, land_vx, want_vx)
    want_y = np.where(is_land, land_y, want_y)
    vy_lim = np.where(is_land & over, 1.1, vy_lim)
    tilt_lim = np.where(is_land & near, 0.6, tilt_lim)
    # wait: hold `a` m short of the pad's centre, `b` m above its landing height
    is_wait = kind == WAIT
    want_vx = np.where(is_wait, np.clip(1.2 * (gx - d * a - f.x), -4.0, 4.0), want_vx)
    want_y = np.where(is_wait, gy + b, want_y)
    vy = np.clip(1.5 * (want_y - f.y), -vy_lim, vy_lim)
    # level: hold whatever height it is at (climbing off a pillar it sits on)
    vy = np.where(kind == LEVEL, np.where(f.grounded == 1, 1.0, 0.0), vy)
    # the wind, felt as a sideways push (the planner knows it; a network has to infer it)
    t = f.ticks * f.ph.dt
    air = f.wind + f.gust * np.sin(2 * np.pi * t / f.period + f.phase)
    lean = -np.arctan2(air / f.mass, G) / f.ph.tilt_max
    tilt = np.clip(lean + 0.35 * (want_vx - f.vx), -tilt_lim, tilt_lim)
    # and the throttle that holds a hover for this mass and charge
    hover = f.mass / (f.ph.thrust_max * thrust_scale(f.charge) * np.cos(f.th))
    thr = np.clip(2.0 * hover - 1.0 + 0.6 * (vy - f.vy), -1.0, 1.0)
    jink = kind == JINK
    thr = np.where(jink, a, thr)
    tilt = np.where(jink, d * b, tilt)
    return np.stack([thr, tilt], 1)


class Planner:
    """Plans for the drones of one Fleet. plan[i] = (m1, t1, m2, t2, m3): move m1 until update
    t1 (counted from now), m2 until t2, then m3."""

    def __init__(
        self,
        n: int,
        horizon: float = 4.5,
        every: int = 4,
        first: float = 0.6,
        jinks=(0.3, 0.6, 0.9),
        delays=(0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.6, 3.2, 3.8),
        safe: float = 1.5,
        near_cost: float = 10.0,
        open_cost: float = 15.0,
        loyal: float = 1.0,
        jink_cost: float = 1.0,
        deep_k: int = 4,
        deep_rounds: int = 3,
        event_k: int = 3,
        event_h: float = 4.5,
        band: float = 16.0,
        fair: bool = True,
        deep: float = 3.0,
        deep_delays=(0.0, 0.25, 0.5, 0.8, 1.2, 1.6),
    ):
        upd = 3 / 60
        self.H = int(round(horizon / upd))
        self.every = every
        self.first_u = int(round(first / upd))
        self.jink_u = [int(round(t / upd)) for t in jinks]
        self.delays_u = [int(round(t / upd)) for t in delays]
        self.safe, self.near_cost, self.open_cost, self.loyal = safe, near_cost, open_cost, loyal
        self.jink_cost = jink_cost
        self.deep_k, self.deep_rounds, self.H2 = deep_k, deep_rounds, int(round(deep / upd))
        self.event_k, self.H3 = event_k, int(round(event_h / upd))
        self.band, self.fair = band, fair
        self.deep_delays_u = [int(round(t / upd)) for t in deep_delays]
        self.plan = np.zeros((n, 5), np.int64)
        self.plan[:] = (DEFAULT, 10**6, DEFAULT, 10**6, DEFAULT)
        self.age = np.zeros(n, np.int64)  # updates since the plan was chosen
        self.fresh = np.ones(n, bool)  # no plan yet (a new course)
        self.decisions = 0
        self._cand: dict = {}

    def reset(self, idx) -> None:
        self.plan[idx] = (DEFAULT, 10**6, DEFAULT, 10**6, DEFAULT)
        self.age[idx] = 0
        self.fresh[idx] = True

    # --- candidates ------------------------------------------------------------------------
    def candidates(self, near_pad: bool, k_h: int) -> np.ndarray:
        """(C, 5) plans from now, for a drone nearest cruise height HEIGHTS[k_h]."""
        key = (near_pad, k_h)
        if key not in self._cand:
            self._cand[key] = self._candidates(near_pad, k_h)
        return self._cand[key]

    def _candidates(self, near_pad: bool, k_h: int) -> np.ndarray:
        pads = list(PAD_MOVES) if near_pad else []
        firsts = list(CRUISES) + list(LEVELS) + pads
        seconds = [move_id(CRUISE, h, v) for h in (3.0, 7.0, 11.0) for v in (0.0, 7.0)]
        seconds += [move_id(CRUISE, 1.5, 7.0), move_id(CRUISE, 5.0, 10.0)]
        seconds += [move_id(CRUISE, 9.0, 10.0)] + pads
        never = 10**6
        out = []
        t1 = self.first_u
        for m1 in firsts:  # two moves: the first for 0.6 s, the second to the end
            for m2 in seconds:
                out.append((m1, t1, m2, never, m2))
        # a jink after a delay: keep this height (slow, cruising or fast) for 0-2.6 s,
        # jink 0.3-0.9 s, then cruise a height step up or down
        up, down = HEIGHTS[min(k_h + 1, len(HEIGHTS) - 1)], HEIGHTS[max(k_h - 1, 0)]
        for v in (4.0, 7.0, 10.0):
            m1 = move_id(LEVEL, 0.0, v)
            for thr, tilt in JINK_STICKS:
                rec = move_id(CRUISE, up if thr > 0 else down, 7.0)
                for dur in self.jink_u:
                    for delay in self.delays_u:
                        out.append((m1, delay, move_id(JINK, thr, tilt), delay + dur, rec))
        if near_pad:  # wait beside the pad for 0-2.6 s, then land
            for w in np.nonzero(MK == WAIT)[0]:
                for m2 in np.nonzero(MK == LAND)[0]:
                    for delay in self.delays_u:
                        out.append((w, delay, m2, never, m2))
        return np.array(out, np.int64)

    def current(self) -> np.ndarray:
        """The move each drone's plan says now."""
        p, a = self.plan, self.age
        return np.where(a < p[:, 1], p[:, 0], np.where(a < p[:, 3], p[:, 2], p[:, 4]))

    # --- look-ahead ------------------------------------------------------------------------
    def rollout(self, sim: Fleet, plans: np.ndarray, H: int):
        """Flies every copy its plan for H updates: (updates flown, closest pass, progress
        toward the pad summed over the updates)."""
        alive = np.zeros(sim.n)
        dmin = np.full(sim.n, 1e9)
        d0 = sim.distance()
        gain = np.zeros(sim.n)
        self.spawn_u = np.full(sim.n, -1)  # the update in which it first met a threat
        rows = np.arange(sim.n)
        m1, t1, m2, t2, m3 = plans.T
        for u in range(H):
            flying = sim.outcome == FLYING
            if not flying.any():
                gain += np.where(sim.outcome == LANDED, d0, 0.0) * (H - u)
                break
            alive += flying
            m = np.where(u < t1, m1, np.where(u < t2, m2, m3))
            act = control(sim, m)
            k = sim.enc_next.copy()
            sim.step(act)
            met = (sim.enc_next > k) & (sim.enc_kind[rows, np.minimum(k, MAX_ENC - 1)] != 0)
            self.spawn_u = np.where(met & (self.spawn_u < 0), u, self.spawn_u)
            dmin = np.minimum(dmin, clearance(sim))
            gain += np.where(sim.outcome == LANDED, d0, d0 - sim.distance())
        return alive, dmin, gain

    def progress_score(self, sim: Fleet, plans, d0, gain) -> np.ndarray:
        """Half where it ends, half how soon it got there (mean progress over the horizon),
        less a little per jink."""
        s = 0.5 * (d0 - sim.distance()) + 0.5 * gain / self.H
        jinks = (MK[plans[:, 0]] == JINK) & (plans[:, 1] > 0)
        jinks = jinks.astype(float) + ((MK[plans[:, 2]] == JINK) & (plans[:, 3] > 0))
        return s - self.jink_cost * jinks

    @staticmethod
    def open_threats(sim: Fleet) -> np.ndarray:
        """Birds and missiles still after the drone, each weighted by closeness (1 within
        25 m)."""
        hostile = sim.p_on & sim.p_hostile & ~sim.p_done
        r = np.hypot(sim.p_x - sim.x[:, None], sim.p_y - sim.y[:, None])
        return (hostile * np.clip(25.0 / np.maximum(r, 1e-6), 0, 1)).sum(1)

    def landings(self) -> np.ndarray:
        """Plans for the deepening at the pad: wait beside it for 0-2.6 s, then land."""
        if "land" not in self._cand:
            out = [(w, d, m, 10**6, m) for w in np.nonzero(MK == WAIT)[0]
                   for m in np.nonzero(MK == LAND)[0] for d in self.delays_u]  # fmt: skip
            self._cand["land"] = np.array(out, np.int64)
        return self._cand["land"]

    def dodges(self, k_h: int) -> np.ndarray:
        """Plans for the deepening: cruise on at this height, a jink after 0-1.6 s, or a
        height step up or down."""
        key = ("dodge", k_h)
        if key not in self._cand:
            h = HEIGHTS[k_h]
            up, down = HEIGHTS[min(k_h + 1, len(HEIGHTS) - 1)], HEIGHTS[max(k_h - 1, 0)]
            out = [(m, 10**6, m, 10**6, m) for m in (move_id(CRUISE, h, 7.0),
                   move_id(CRUISE, up, 7.0), move_id(CRUISE, down, 7.0))]  # fmt: skip
            for v in (4.0, 7.0):
                m1 = move_id(LEVEL, 0.0, v)
                for thr, tilt in JINK_STICKS:
                    rec = move_id(CRUISE, up if thr > 0 else down, 7.0)
                    for dur in self.jink_u:
                        for delay in self.deep_delays_u:
                            out.append((m1, delay, move_id(JINK, thr, tilt), delay + dur, rec))
            self._cand[key] = np.array(out, np.int64)
        return self._cand[key]

    def event_deepen(self, f, src, groups, plans, spawn_u, d0, core, threat, failed, resolved,
                     dmin) -> None:  # fmt: skip
        """Deepening from the moment a threat appears: for each drone, its earliest few
        crossings of a waypoint that end badly (failed, or a threat still after it) are flown
        again up to the crossing, and a dodge search starts there. A crossing that one branch
        survives counts as resolved, scored at the same time (the horizon) as every other
        plan. Without this, a crossing late in the look-ahead got a deeper search than an
        early one, and the planner kept putting the crossing off."""
        bad = (spawn_u >= 0) & (failed | (threat > 0))
        sel, k = [], 0
        for g in groups:
            rows_g = k + np.nonzero(bad[k : k + len(g)])[0]
            order = rows_g[np.lexsort((-core[rows_g], spawn_u[rows_g]))]
            seen, pick = set(), []
            for r in order:  # earliest first, one per 0.25 s of crossing time
                b = int(spawn_u[r]) // 5
                if b not in seen:
                    seen.add(b)
                    pick.append(r)
                if len(pick) == self.event_k:
                    break
            sel.extend(pick)
            k += len(g)
        if not sel:
            return
        sel = np.array(sel)
        pre = f.take(src[sel])
        U = spawn_u[sel] + 1  # updates up to and including the crossing
        m1, t1, m2, t2, m3 = plans[sel].T
        gain_pre = np.zeros(len(sel))
        dmin_pre = np.full(len(sel), 1e9)
        d0s = d0[sel]
        snaps = [None] * len(sel)
        for u in range(int(U.max()) + 1):
            for j in np.nonzero(U == u)[0]:
                snaps[j] = pre.take([j])
            if u == U.max():
                break
            m = np.where(u < t1, m1, np.where(u < t2, m2, m3))
            pre.step(control(pre, m))
            on = u < U
            dmin_pre = np.where(on, np.minimum(dmin_pre, clearance(pre)), dmin_pre)
            gain_pre += np.where(on, d0s - pre.distance(), 0.0)
        snap = concat(snaps)
        ok_snap = snap.outcome == FLYING
        kh = np.argmin(np.abs(np.array(HEIGHTS)[None] - snap.y[:, None]), 1)
        dp = [self.dodges(int(h)) for h in kh]
        nb = np.array([len(p) for p in dp])
        owner = np.repeat(np.arange(len(sel)), nb)
        br = snap.take(owner)
        bp = np.concatenate(dp)
        bm1, bt1, bm2, bt2, bm3 = bp.T
        mark = np.maximum(self.H - U[owner], 0)  # branch updates until the horizon
        d_at = br.distance()
        gain_b = np.zeros(br.n)
        dmin_b = np.full(br.n, 1e9)
        d0b = d0s[owner]
        for u in range(self.H3):
            if not (br.outcome == FLYING).any():
                break
            m = np.where(u < bt1, bm1, np.where(u < bt2, bm2, bm3))
            br.step(control(br, m))
            dmin_b = np.minimum(dmin_b, clearance(br))
            dist = np.where(br.outcome == LANDED, 0.0, br.distance())
            before = u < mark
            gain_b += np.where(before, d0b - dist, 0.0)
            d_at = np.where(u + 1 == mark, dist, d_at)
        ok = ~np.isin(br.outcome, FAILED)
        score_b = 0.5 * (d0b - d_at) + 0.5 * (gain_pre[owner] + gain_b) / self.H
        dm = np.minimum(dmin_pre[owner], dmin_b)
        score_b = score_b - self.near_cost * np.clip(self.safe - dm, 0.0, None)
        score_b = np.where(ok, score_b, -np.inf)
        start = np.concatenate([[0], np.cumsum(nb)[:-1]])
        for j, r in enumerate(sel):
            if not ok_snap[j]:
                continue
            sb = score_b[start[j] : start[j] + nb[j]]
            b = int(np.argmax(sb))
            if np.isfinite(sb[b]):
                resolved[r] = True
                failed[r] = False
                core[r] = (
                    0.5 * (d0s[j] - d_at[start[j] + b])
                    + 0.5 * (gain_pre[j] + gain_b[start[j] + b]) / self.H
                )
                dmin[r] = dm[start[j] + b]

    def decide(self, f: Fleet, idx: np.ndarray) -> None:
        """New plans for drones idx of f (all flying)."""
        if len(idx) == 0:
            return
        groups, sc, _ = self.evaluate(f, idx)
        act0 = self.act0
        canon = control(f.take(idx), np.full(len(idx), CANON))
        idx_pos = {int(i): q for q, i in enumerate(idx)}
        k = 0
        for g, i in zip(groups, idx, strict=True):
            s = sc[k : k + len(g)].copy()
            if not self.fresh[i]:
                s[0] += self.loyal
            j = int(np.argmax(s))
            if self.band and s[j] > -500:
                # among plans scoring within `band` of the best, the one whose sticks now are
                # closest to a plain cruise's (7 m up, 10 m/s): where the choice doesn't
                # matter, the teacher always does the same thing, and where it must swerve, it
                # swerves least. A network can copy that; it can't copy a coin toss
                near = np.nonzero(s >= s[j] - self.band)[0]
                d = ((act0[k + near] - canon[idx_pos[i]]) ** 2).sum(1)
                j = int(near[np.lexsort((-s[near], d))[0]])
            self.plan[i] = g[j]
            k += len(g)
        self.age[idx] = 0
        self.fresh[idx] = False
        self.decisions += len(idx)

    def evaluate(self, f: Fleet, idx: np.ndarray):
        """Every candidate plan of drones idx, flown ahead: (plans per drone, scores, the
        copies at the horizon)."""
        gx, _ = f.goal()
        groups, rows = [], []
        for i in idx:
            k_h = int(np.argmin(np.abs(np.array(HEIGHTS) - f.y[i])))
            c = self.candidates(abs(gx[i] - f.x[i]) < PAD_NEAR, k_h)
            if not self.fresh[i]:  # what is left of the last plan, first in line
                p = self.plan[i].copy()
                p[1] -= self.age[i]
                p[3] -= self.age[i]
                c = np.concatenate([p[None], c])
            groups.append(c)
            rows.append(np.full(len(c), i))
        plans = np.concatenate(groups)
        sim = f.take(np.concatenate(rows))
        if self.fair:  # the copies know nothing of waypoints not yet reached
            sim.enc_at[:] = 1e9
        d0 = sim.distance()
        m1, t1, m2, t2, m3 = plans.T  # each plan's sticks right now
        self.act0 = control(sim, np.where(0 < t1, m1, np.where(0 < t2, m2, m3)))
        alive, dmin, gain = self.rollout(sim, plans, self.H)
        spawn_u = self.spawn_u
        core = self.progress_score(sim, plans, d0, gain)
        threat = self.open_threats(sim)
        failed = np.isin(sim.outcome, FAILED)
        landed = sim.outcome == LANDED
        lasted = alive / self.H
        resolved = np.zeros(sim.n, bool)
        if self.event_k:
            self.event_deepen(f, np.concatenate(rows), groups, plans, spawn_u, d0, core, threat,
                              failed, resolved, dmin)  # fmt: skip
        # deepening: the best few copies per drone that end with a threat still after them
        # get a dodge search of their own; they score by its best branch, the rest pay. If
        # none of a drone's first few gets away, the next few are tried (up to deep_rounds)
        deep = (sim.outcome == FLYING) & (threat > 0)
        ranked, k = [], 0
        for g in groups:
            rows_g = k + np.nonzero(deep[k : k + len(g)])[0]
            ranked.append(rows_g[np.argsort(-core[rows_g], kind="stable")])
            k += len(g)
        for rnd in range(self.deep_rounds):
            lo, hi = rnd * self.deep_k, (rnd + 1) * self.deep_k
            sel = [r for rows_g in ranked if not resolved[rows_g].any() for r in rows_g[lo:hi]
                   if not resolved[r]]  # fmt: skip
            if not sel:
                break
            sel = np.array(sel)
            kh = np.argmin(np.abs(np.array(HEIGHTS)[None] - sim.y[sel][:, None]), 1)
            dp = [self.dodges(int(h)) for h in kh]
            sub = sim.take(np.repeat(sel, [len(p) for p in dp]))
            a2, dm2, _ = self.rollout(sub, np.concatenate(dp), self.H2)
            ok = ~np.isin(sub.outcome, FAILED)
            j = 0
            for r, p in zip(sel, dp, strict=True):
                okr, dmr = ok[j : j + len(p)], dm2[j : j + len(p)]
                if okr.any():
                    resolved[r] = True
                    dmin[r] = min(dmin[r], dmr[okr].max())
                else:
                    failed[r] = True
                    lasted[r] = (alive[r] + a2[j : j + len(p)].max()) / (self.H + self.H2)
                j += len(p)
        # deepening at the pad: the best few copies per drone that end near it, flying and with
        # no threat after them, get a wait-then-land search; landing in it counts as landing
        land_at = np.full(sim.n, -1.0)
        gx, _ = sim.goal()
        near = (sim.outcome == FLYING) & (threat == 0) & (np.abs(gx - sim.x) < PAD_DEEP)
        sel, k = [], 0
        for g in groups:
            rows_g = k + np.nonzero(near[k : k + len(g)])[0]
            sel.extend(rows_g[np.argsort(-core[rows_g], kind="stable")][: self.deep_k])
            k += len(g)
        if sel:
            sel = np.array(sel)
            lp = self.landings()
            sub = sim.take(np.repeat(sel, len(lp)))
            a2, dm2, _ = self.rollout(sub, np.tile(lp, (len(sel), 1)), self.H2)
            down = (sub.outcome == LANDED).reshape(len(sel), len(lp))
            a2 = a2.reshape(len(sel), len(lp))
            dm2 = dm2.reshape(len(sel), len(lp))
            for q, r in enumerate(sel):
                if down[q].any():
                    b = np.nonzero(down[q])[0][np.argmin(a2[q][down[q]])]
                    land_at[r] = alive[r] + a2[q, b]
                    dmin[r] = min(dmin[r], dm2[q, b])
        # every waypoint not yet reached is coming anyway: putting one off saves nothing
        j = np.arange(MAX_ENC)[None]
        ahead = (j >= sim.enc_next[:, None]) & (j < sim.n_enc[:, None])
        pending = self.open_cost * (HOSTILE_PER_KIND[sim.enc_kind] * ahead).sum(1)
        # and each waypoint, and the mid-route frog, costs a close pass at worst
        worst = self.near_cost * self.safe
        pending = pending + worst * (ahead.sum(1) + (sim.frog_on[:, 0] & ~sim.frog_past[:, 0]))
        # ending the look-ahead inside a frog's reach, not landed or past it: a strike to come
        inside = np.abs(sim.frog_p[..., 0] - sim.x[:, None]) < FROG_REACH + 1.0
        in_reach = (sim.frog_on & ~sim.frog_past & inside).sum(1)
        sc = core - self.open_cost * (np.where(resolved, 0.0, threat) + in_reach) - pending
        sc = np.where(landed, 1000.0 - alive, sc)
        sc = np.where(land_at >= 0, 1000.0 - land_at, sc)
        sc = sc - self.near_cost * np.clip(self.safe - dmin, 0.0, None)
        sc = np.where(failed, -1000.0 + 10.0 * lasted, sc)
        self.last = {"core": core, "threat": threat, "resolved": resolved, "pending": pending,
                     "in_reach": in_reach, "dmin": dmin}  # fmt: skip
        return groups, sc, sim

    def act(self, f: Fleet) -> np.ndarray:
        """Sticks for every drone (re-planning those due), then advance the clocks."""
        flying = np.nonzero(f.outcome == FLYING)[0]
        due = flying[self.fresh[flying] | (self.age[flying] >= self.every)]
        self.decide(f, due)
        a = control(f, self.current())
        self.age += 1
        return a


def concat(parts: list[Fleet]) -> Fleet:
    """One Fleet holding the drones of several (copies from Fleet.take)."""
    base = parts[0]
    new = Fleet.__new__(Fleet)
    new.n, new.ph, new.feat = sum(p.n for p in parts), base.ph, base.feat
    for k, v in vars(base).items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (base.n,):
            setattr(new, k, np.concatenate([getattr(p, k) for p in parts]))
    new.courses = sum((p.courses for p in parts), [])
    return new


def clearance(f: Fleet) -> np.ndarray:
    """(N,): the distance to the nearest bird, missile or frog, less its hit radius's excess
    over a bird's (so 0.55 m is a touch for every kind)."""
    x, y = f.x[:, None], f.y[:, None]
    d = np.where(f.p_on, np.hypot(f.p_x - x, f.p_y - y), 1e9).min(1)
    if f.frog_on.any():
        fx, fy, _, _ = f.frog_pos()
        dist = np.where(f.frog_on, np.hypot(fx - x, fy - y) - 0.05, 1e9).min(1)
        d = np.minimum(d, dist)
    return d


def fly_expert(feat, seeds, weather=ROUGH, record=False, **kw):
    """The planner flies these courses; returns the Fleet (and, with record, tracks)."""
    f = Fleet(len(seeds), feat=feat)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(int(s), weather, feat, f.ph))
    pl = Planner(f.n, **kw)
    t = 0
    tracks = [[] for _ in seeds] if record else None
    while (f.outcome == FLYING).any() and t < 1300:
        a = pl.act(f)
        f.step(a)
        if record:
            for i in np.nonzero(f.outcome == FLYING)[0]:
                tracks[i].append((f.x[i], f.y[i]))
        t += 1
    return (f, pl, tracks) if record else (f, pl)


def _job(args):
    test, seeds, kw = args
    os.environ["OMP_NUM_THREADS"] = "1"
    from evaluate import TESTS

    feat = TESTS[test][0]
    f, pl = fly_expert(feat, seeds, **kw)
    out = []
    for i, s in enumerate(seeds):
        out.append(
            {
                "seed": int(s),
                "outcome": OUTCOMES[f.outcome[i]],
                "time": round(float(f.ticks[i] * f.ph.dt), 2),
                "x": float(f.x[i]),
                "togo": float(abs(f.goal()[0][i] - f.x[i])),
                "hit_by": int(f.hit_enc[i]),
                "met": f.enc_count[i].tolist(),
            }
        )
    return out, pl.decisions


def run_parallel(test, seeds, workers=10, chunk=10, **kw):
    from multiprocessing import Pool

    seeds = list(seeds)
    jobs = [(test, seeds[i : i + chunk], kw) for i in range(0, len(seeds), chunk)]
    with Pool(workers) as p:
        parts = p.map(_job, jobs)
    res = sum((r for r, _ in parts), [])
    return res, sum(d for _, d in parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=20_000)
    ap.add_argument("--tests", nargs="+", default=["gauntlet"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--horizon", type=float, default=4.5)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--unfair", action="store_true", help="let it foresee the waypoints")
    ap.add_argument("--show", action="store_true", help="print every failed flight")
    ap.add_argument("--out", default=None, help="save per-course results to runs/<out>.json")
    args = ap.parse_args()
    saved = {}
    seeds = range(args.first_seed, args.first_seed + args.n)
    kw = {"horizon": args.horizon, "every": args.every, "fair": not args.unfair}
    for test in args.tests:
        t0 = time.time()
        res, dec = run_parallel(test, seeds, args.workers, args.chunk, **kw)
        c = Counter(r["outcome"] for r in res)
        print(
            f"{test}: landed {c['landed']}/{args.n}  {dict(c)}  ({time.time() - t0:.0f} s, "
            f"{dec} decisions)",
            flush=True,
        )
        if args.show:
            for r in res:
                if r["outcome"] != "landed":
                    print("   ", r)
        saved[test] = {"flights": res, "seconds": round(time.time() - t0), "decisions": dec}
    if args.out:
        import json

        (HERE / "runs").mkdir(exist_ok=True)
        (HERE / "runs" / f"{args.out}.json").write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
