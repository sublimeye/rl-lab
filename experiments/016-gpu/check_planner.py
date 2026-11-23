"""Does the GPU planner decide as the numpy fair planner does?

Drones fly gauntlet courses (45 deg, the test features). At every decision both planners
decide from the same state: numpy's FairPlanner (its decide, per hypothesis batch) and the
torch one (gpu_planner, all groups at once). Compared: every candidate's score (the worst over
the hypotheses) and the plan chosen. The flight follows numpy's choice, so a difference can't
carry over. One drone starts on its pillar, one just before its first waypoint (a bird or
missile appears: the dodge searches), one 12 m from the pad (the landing searches, the pad's
frog).

uv run experiments/016-gpu/check_planner.py [--updates 60] [--seeds 20000 20001 20002]
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gpu_planner as gp  # noqa: E402

fp = gp.fp
from gauntlet_env import FLYING, FULL, OUTCOMES, ROUGH, Fleet, Physics, make_course  # noqa: E402

PH45 = replace(Physics(), tilt_max=float(np.radians(45.0)))


class Checker:
    def __init__(self, pl, tol):
        self.pl, self.tol = pl, tol
        self.decisions = self.same = 0
        self.worst = 0.0
        self.diffs = []

    def __call__(self, f, due, near, chosen):
        pl = self.pl
        snap = (pl.plan.copy(), pl.age.copy(), pl.fresh.copy(), pl.decisions)
        captured = []
        orig = fp.FairPlanner.evaluate

        def ev(self_, ff, idx):
            out = orig(self_, ff, idx)
            captured.append((np.asarray(idx), out[0], out[1]))
            return out

        pl.evaluate = types.MethodType(ev, pl)
        try:
            fp.FairPlanner.decide(pl, f, due[near])
            fp.FairPlanner.decide(pl, f, due[~near])
        finally:
            del pl.evaluate
        ref = pl.plan[due].copy()
        pl.plan[:], pl.age[:], pl.fresh[:] = snap[0], snap[1], snap[2]
        pl.decisions = snap[3]
        g = pl.last_gpu
        best = g["best"].cpu().numpy()
        keep = g["keep"].cpu().numpy()
        drone_of = g["drone_of"].cpu().numpy()
        pos = {int(i): q for q, i in enumerate(due)}
        for idx, groups, sc in captured:
            k = 0
            for grp, i in zip(groups, idx, strict=True):
                q = pos[int(i)]
                mine = best[keep & (drone_of == q)]
                theirs = sc[k : k + len(grp)]
                d = float(np.max(np.abs(mine - theirs)))
                self.worst = max(self.worst, d)
                if d > self.tol:
                    j = int(np.argmax(np.abs(mine - theirs)))
                    self.diffs.append((int(f.ticks[i]), int(i), j, theirs[j], mine[j], grp[j]))
                k += len(grp)
        self.decisions += len(due)
        same = (ref == chosen).all(1)
        self.same += int(same.sum())
        for q in np.nonzero(~same)[0]:
            self.diffs.append(("plan", int(f.ticks[due[q]]), int(due[q]), ref[q], chosen[q]))
        return ref


def setup(seeds):
    f = Fleet(len(seeds), PH45, feat=FULL)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(int(s), ROUGH, FULL, f.ph))
    sx = (f.s_x0 + f.s_x1) / 2
    if f.n > 1:  # just before the first waypoint
        f.x[1] = sx[1] + f.dir[1] * (f.enc_at[1, 0] - 1.0)
        f.y[1], f.grounded[1] = 5.0, 0
    if f.n > 2:  # 12 m short of the pad
        gx, gy = f.goal()
        f.x[2], f.y[2], f.grounded[2] = gx[2] - f.dir[2] * 12.0, gy[2] + 3.0, 0
        f.enc_next[2] = f.n_enc[2]
        f.frog_past[2, 0] = f.frog_on[2, 0]
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20_000, 20_001, 20_002])
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.float32 else torch.float64
    f = setup(args.seeds)
    pl = gp.GPUPlanner(f.n, device=args.device, dtype=dtype)
    chk = Checker(pl, 1e-3 if args.float32 else 1e-6)
    pl.check = chk
    t0 = time.time()
    for u in range(args.updates):
        if not (f.outcome == FLYING).any():
            break
        f.step(pl.act(f))
        if u % 20 == 19:
            print(f"update {u + 1}: {chk.same}/{chk.decisions} plans the same, largest score "
                  f"difference {chk.worst:.2e} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    print("outcomes:", [OUTCOMES[o] for o in f.outcome], "progress:", np.round(f.progress(), 1))
    print(f"{chk.same}/{chk.decisions} plans the same; largest score difference {chk.worst:.2e}")
    print("searches:", dict(pl.stats))
    for d in chk.diffs[:10]:
        print("  ", d)


if __name__ == "__main__":
    main()
