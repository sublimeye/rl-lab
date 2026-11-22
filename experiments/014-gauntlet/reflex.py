"""A hand-written dodge reflex for a trained pilot (post 14's hybrid).

The network flies; the reflex takes the throttle only while a bird or missile in sight is
predicted (from the pilot's own closest-approach inputs) to pass within MISS m in the next
T_CA s, and climbs or dives away from that pass. It reads nothing the pilot can't see.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gauntlet_env import FULL, K_SEE, N_BASE, N_PREDICT, N_SLOT, frame_width

W = frame_width(replace(FULL, predict=True))
SLOT = N_SLOT + N_PREDICT


class Reflex:
    def __init__(self, pilot, miss: float = 1.5, t_ca: float = 1.5, push: float = 1.0):
        self.pilot, self.miss, self.t_ca, self.push = pilot, miss, t_ca, push
        self.predict = True

    def danger(self, obs: np.ndarray):
        o = obs[:, :W].astype(float)
        sl = o[:, N_BASE : N_BASE + K_SEE * SLOT].reshape(len(o), K_SEE, SLOT)
        t, mx, my = sl[..., 7] * 3, sl[..., 8] * 5, sl[..., 9] * 5
        bad = (sl[..., 4] > 0.5) & (t < self.t_ca) & (np.hypot(mx, my) < self.miss)
        k = np.argmin(np.where(bad, t, np.inf), 1)
        return bad.any(1), my[np.arange(len(o)), k]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        act = self.pilot(obs).copy()
        on, my = self.danger(obs)
        y = obs[:, 10].astype(float) * 10  # height (input 10)
        up = np.where(y < 3.0, True, np.where(y > 10.0, False, my <= 0))  # not into the water
        thr = np.where(up, self.push, -0.3 * self.push)  # or the ceiling; a dive is gentle
        act[:, 0] = np.where(on, thr, act[:, 0])
        return act
