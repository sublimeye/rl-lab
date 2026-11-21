"""The escape test: was a hit avoidable? Rewind to `lead` seconds before it and try 9 full-stick
escapes (throttle down/hold/up x tilt left/none/right), each held until well past the moment of
the hit; if any of them survives, a pilot with that much notice could have got away.

Also a scripted hover pilot (no dodging) for tuning threats before any training.
"""

from __future__ import annotations

import numpy as np

from dodge_env import FAILED, FLYING, Fleet

MANEUVERS = np.array([(t, s) for t in (-1.0, 0.0, 1.0) for s in (-1.0, 0.0, 1.0)])


def hover_pilot(f: Fleet) -> np.ndarray:
    """Flies to the goal and holds it; ignores threats."""
    gx, gy = f.goal()
    ex = np.clip(gx - f.x, -3, 3)
    tilt = np.clip(0.25 * ex - 0.35 * f.vx, -1, 1)
    thr = np.clip(0.0 + 0.5 * (gy - f.y) - 0.6 * f.vy, -1, 1)
    return np.stack([thr, tilt], 1)


def fly_snapshots(fleet: Fleet, pilot, max_updates: int = 500):
    """Flies everyone with pilot(fleet) -> sticks, copying the fleet before each update."""
    snaps = []
    rows = np.arange(fleet.n)
    for _ in range(max_updates):
        if not (fleet.outcome == FLYING).any():
            break
        snaps.append(fleet.take(rows))
        fleet.step(pilot(fleet))
    return snaps


def escapable(snaps, fleet: Fleet, hits: np.ndarray, lead: float, after: float = 0.6):
    """For each drone index in hits (hit at its final tick): can one of the maneuvers started
    `lead` s before the hit get it to `after` s past the hit without failing?"""
    upd = fleet.ph.dt * fleet.ph.substeps
    out = np.zeros(len(hits), bool)
    for j, i in enumerate(hits):
        end_u = int(np.ceil(fleet.ticks[i] / fleet.ph.substeps))  # updates flown
        u0 = max(0, end_u - int(round(lead / upd)))
        f = snaps[u0].take(np.full(len(MANEUVERS), i))
        n_up = int(round((lead + after) / upd))
        for _ in range(n_up):
            if not (f.outcome == FLYING).any():
                break
            f.step(MANEUVERS)
        out[j] = (~np.isin(f.outcome, FAILED)).any()
    return out
