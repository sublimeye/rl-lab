"""Scripted pilots for tuning the gauntlet before any training (and the escape test).

cruise_pilot flies to the pad at a fixed height and lands; it ignores every threat.
"""

from __future__ import annotations

import numpy as np

from gauntlet_env import FAILED, FLYING, Fleet

MANEUVERS = np.array([(t, s) for t in (-1.0, 0.0, 1.0) for s in (-1.0, 0.0, 1.0)])


def cruise_pilot(f: Fleet, height: float = 7.0, speed: float = 7.0) -> np.ndarray:
    gx, gy = f.goal()
    dx = gx - f.x
    near = np.abs(dx) < 6.0
    want_vx = np.where(near, np.clip(0.8 * dx, -2.5, 2.5), f.dir * speed)
    want_y = np.where(near & (np.abs(dx) < 1.0), gy - 0.5, np.maximum(height, gy + 1.5))
    want_vy = np.clip(1.0 * (want_y - f.y), -1.0, 2.0)
    tilt = np.clip(0.35 * (want_vx - f.vx), -1, 1)
    thr = np.clip(0.05 + 0.6 * (want_vy - f.vy), -1, 1)
    return np.stack([thr, tilt], 1)


def fly_snapshots(fleet: Fleet, pilot, max_updates: int = 1300):
    snaps = []
    rows = np.arange(fleet.n)
    for _ in range(max_updates):
        if not (fleet.outcome == FLYING).any():
            break
        snaps.append(fleet.take(rows))
        fleet.step(pilot(fleet))
    return snaps


def escapable(snaps, fleet: Fleet, hits: np.ndarray, lead: float, after: float = 0.6):
    """For each drone in hits: can one of 9 full-stick maneuvers started `lead` s before the
    hit get it to `after` s past the hit without failing?"""
    upd = fleet.ph.dt * fleet.ph.substeps
    out = np.zeros(len(hits), bool)
    for j, i in enumerate(hits):
        end_u = int(np.ceil(fleet.ticks[i] / fleet.ph.substeps))
        u0 = max(0, end_u - int(round(lead / upd)))
        f = snaps[u0].take(np.full(len(MANEUVERS), i))
        for _ in range(int(round((lead + after) / upd))):
            if not (f.outcome == FLYING).any():
                break
            f.step(MANEUVERS)
        out[j] = (~np.isin(f.outcome, FAILED)).any()
    return out
