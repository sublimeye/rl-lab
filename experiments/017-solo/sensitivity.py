"""Does the fair planner agree with itself? (post 17, diagnosis)

A network copying the planner sees the drone and the threats through noisy sensors, one update
late. If a nudge the size of that noise changes the planner's move, no network can copy it there:
near-identical inputs carry different labels.

The planner (016's Teacher) flies training-like courses. At every third decision, a copy of the
planner with the same memory decides again from a nudged copy of the world: the drone's position
moved by 5 cm and its speed by 0.1 m/s (one standard deviation, the sensors' noise), each bird
and missile by 5 cm. The two choices are compared.

uv run experiments/017-solo/sensitivity.py --flights 40
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "016-flawless"))

import pilot  # noqa: E402  (016's Teacher; patches expert's moves)

expert = pilot.expert
from gauntlet_env import FLYING, ROUGH, Config, FleetEnv  # noqa: E402

FIRST_SEED = 2_100_000  # courses no DAgger run flew


def context(f, i: int) -> str:
    """What is near drone i: an urgent bird or missile (closest approach within 2 s and 2 m,
    flying straight), a frog within 8 m, a bird or missile in sight, or nothing."""
    x, y, vx, vy, on = f.pursuers()
    dx, dy = x[i] - f.x[i], y[i] - f.y[i]
    rvx, rvy = vx[i] - f.vx[i], vy[i] - f.vy[i]
    t = np.clip(-(dx * rvx + dy * rvy) / np.maximum(rvx**2 + rvy**2, 1e-6), 0, 6)
    miss = np.hypot(dx + rvx * t, dy + rvy * t)
    seen = on[i] & (np.hypot(dx, dy) < 40)
    if (seen & (t < 2) & (miss < 2)).any():
        return "urgent"
    frog = f.frog_on[i] & ~f.frog_past[i] & (np.abs(f.frog_p[i, :, 0] - f.x[i]) < 8)
    if frog.any():
        return "frog"
    return "in sight" if seen.any() else "calm"


def _fly(job):
    seeds, every, sig_p, sig_v, seed = job
    os.environ["OMP_NUM_THREADS"] = "1"
    env = FleetEnv(len(seeds), Config(pilot.PH45, ROUGH, pilot.FEAT), seed=seed)
    env.start(seeds)
    f = env.fleet
    pl = pilot.Teacher(f.n)
    rng = np.random.default_rng(seed)
    rows, t, n_dec = [], 0, 0
    while (f.outcome == FLYING).any() and t < 1300:
        pl.ph_real = f.ph
        pl.observe(f)
        pl.estimate(f)
        flying = f.outcome == FLYING
        due = flying & (pl.fresh | (pl.age >= pl.every)) & (f.grounded == 0)
        twin = None
        if due.any():
            n_dec += 1
            if n_dec % every == 0:
                pl2 = copy.deepcopy(pl)
                g = f.take(np.arange(f.n))
                for k, s in (("x", sig_p), ("y", sig_p), ("vx", sig_v), ("vy", sig_v)):
                    getattr(g, k)[:] += s * rng.standard_normal(f.n)
                g.p_x += sig_p * rng.standard_normal(g.p_x.shape)
                g.p_y += sig_p * rng.standard_normal(g.p_y.shape)
                twin = (pl2.plan_now(g), pl2.mscore.copy())
        m = pl.plan_now(f)
        if twin is not None:
            for i in np.nonzero(due)[0]:
                s = np.sort(pl.mscore[i])[::-1]
                rows.append({
                    "ctx": context(f, i), "a": int(m[i]), "b": int(twin[0][i]),
                    "safe": int((pl.mscore[i] >= pilot.FAIL).sum()),
                    "tried": int((pl.mscore[i] > pilot.ABSENT + 1).sum()),
                    "gap": float(s[0] - s[1]),
                    # the nudged planner's move, as the real planner scores it
                    "b_score": float(pl.mscore[i, twin[0][i]]), "best": float(s[0]),
                })  # fmt: skip
        a = expert.control(pl.assumed(f), m)
        pl.remember(f, a)
        pl.age += 1
        env.step(a)
        t += 1
    return rows, int((f.outcome == 1).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=40)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--pos", type=float, default=0.05)
    ap.add_argument("--vel", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="sensitivity")
    args = ap.parse_args()
    from multiprocessing import Pool

    seeds = list(range(FIRST_SEED, FIRST_SEED + args.flights))
    jobs = [(seeds[i : i + 4], args.every, args.pos, args.vel, 500 + i)
            for i in range(0, len(seeds), 4)]  # fmt: skip
    with Pool(args.workers) as p:
        parts = p.map(_fly, jobs)
    rows = sum((r for r, _ in parts), [])
    landed = sum(n for _, n in parts)
    MK = expert.MK
    out = {"flights": args.flights, "landed": landed, "pos": args.pos, "vel": args.vel, "by": {}}
    print(f"planner landed {landed}/{args.flights}; {len(rows)} decisions made twice")
    for ctx in ("all", "calm", "in sight", "frog", "urgent"):
        r = [x for x in rows if ctx == "all" or x["ctx"] == ctx]
        if not r:
            continue
        a, b = np.array([x["a"] for x in r]), np.array([x["b"] for x in r])
        flip = a != b
        kind = MK[a] != MK[b]
        fatal = np.array([x["b_score"] < pilot.FAIL for x in r]) & np.array(
            [x["best"] >= pilot.FAIL for x in r]
        )
        untried = np.array([x["b_score"] <= pilot.ABSENT + 1 for x in r])
        out["by"][ctx] = {"n": len(r), "flip": int(flip.sum()), "kind": int(kind.sum()),
                          "fatal": int((fatal & ~untried).sum())}  # fmt: skip
        print(f"  {ctx:9s}: {len(r):5d} decisions; a different move {flip.mean():.3f}; a different "
              f"kind of move {kind.mean():.3f}; the nudged choice fails in the real planner's "
              f"copies {np.mean(fatal & ~untried):.3f}")  # fmt: skip
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / f"{args.out}.json").write_text(json.dumps({"summary": out, "rows": rows}))


if __name__ == "__main__":
    main()
