"""Does the GPU shield decide as pilot.Teacher's numpy shield does?

As check_planner.py, with the shield on: at each decision every drone is allowed 5 random
moves (the network's top 5 in a real flight; random here, so that sometimes every allowed plan
fails and the full search runs). Numpy's side is pilot.Teacher._decide (its filtered candidates
and full-search fallback) on the same planner state.

uv run experiments/016-gpu/check_shield.py [--updates 40]
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import check_planner as cp  # noqa: E402
import dagger_gpu as dg  # noqa: E402

pilot = dg.pilot
FLYING = dg.FLYING


def _believe(self, f, idx, h):  # pilot.Teacher.believe
    self._order = iter(list(idx))
    return dg.gp.fp.FairPlanner.believe(self, f, idx, h)


def _candidates(self, near_pad, k_h):  # pilot.Teacher.candidates
    c = dg.expert.Planner.candidates(self, near_pad, k_h)
    i = next(self._order, None)
    if i is None or self.allowed[i].all():
        return c
    keep = self.allowed[i][pilot.move_now(c)]
    return c[keep] if keep.any() else c


class ShieldChecker(cp.Checker):
    def __call__(self, f, due, near, chosen):
        pl = self.pl
        snap = (pl.plan.copy(), pl.age.copy(), pl.fresh.copy(), pl.decisions, pl.full_searches,
                pl.mscore.copy(), pl.decided.copy())  # fmt: skip
        # numpy: pilot.Teacher's decide, with its filtered candidates
        pl.candidates = types.MethodType(_candidates, pl)
        pl.believe = types.MethodType(_believe, pl)
        pl._order = iter(())
        try:
            for idx in (due[near], due[~near]):
                pilot.Teacher._decide(pl, pl.assumed(f), idx)
        finally:
            del pl.candidates, pl.believe
        ref, ref_ms, ref_full = pl.plan[due].copy(), pl.mscore[due].copy(), pl.full_searches
        pl.plan[:], pl.age[:], pl.fresh[:] = snap[0], snap[1], snap[2]
        pl.decisions = snap[3]
        self.np_full += ref_full - snap[4]
        # the GPU's best score per move (as recorded for the danger head) vs numpy's
        d = np.abs(np.where(ref_ms > pilot.ABSENT + 1, ref_ms, 0) - np.where(
            self.gpu_ms[due] > pilot.ABSENT + 1, self.gpu_ms[due], 0))  # fmt: skip
        self.worst = max(self.worst, float(d.max()))
        self.decisions += len(due)
        same = (ref == chosen).all(1)
        self.same += int(same.sum())
        for q in np.nonzero(~same)[0]:
            self.diffs.append(("plan", int(f.ticks[due[q]]), int(due[q]), ref[q], chosen[q]))
        return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20_001, 20_003, 20_008])
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dg.DEVICE = "cpu"
    f = cp.setup(args.seeds)
    pl = dg.GPUTeacher(f.n, dtype=torch.float64)
    chk = ShieldChecker(pl, 1e-6)
    chk.np_full = 0
    gpu_record = pl._record

    def record(due, best, drone_of, plans):  # keep the GPU's per-move scores to compare
        gpu_record(due, best, drone_of, plans)
        chk.gpu_ms = pl.mscore.copy()

    pl._record = record
    pl.check = chk
    rng = np.random.default_rng(0)
    t0 = time.time()
    for u in range(args.updates):
        if not (f.outcome == FLYING).any():
            break
        due = pl.fresh | (pl.age >= pl.every)
        if due.any():
            allowed = np.zeros((f.n, pilot.NM), bool)
            for i in range(f.n):
                allowed[i, rng.choice(pilot.NM, 5, replace=False)] = True
            pl.allowed = allowed
        m = pl.plan_now(f)
        a = dg.expert.control(pl.assumed(f), m)
        pl.remember(f, a)
        pl.age += 1
        f.step(a)
        if u % 10 == 9:
            print(f"update {u + 1}: {chk.same}/{chk.decisions} plans the same, largest per-move "
                  f"score difference {chk.worst:.2e}, full searches GPU {pl.full_searches} / "
                  f"numpy {chk.np_full} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    print(f"{chk.same}/{chk.decisions} plans the same; full searches GPU {pl.full_searches}, "
          f"numpy {chk.np_full}; largest per-move score difference {chk.worst:.2e}")  # fmt: skip
    for d in chk.diffs[:10]:
        print("  ", d)


if __name__ == "__main__":
    main()
