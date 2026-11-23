"""The fair planner on a GPU flies the gauntlet's test courses, all at once.

Same planner and settings as fair_planner.py --tilt 45 (frogs ghost, air estimated, 3
hypotheses, deepenings 3 s, band 16): its reference is runs/fair-t45-v6.json in 016-flawless,
200/200 on courses 20000-20199 in 2370 s on 10 CPU cores.

uv run experiments/016-gpu/fly_gpu.py --n 200 --out gpu-t45
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gpu_planner as gp  # noqa: E402
import torch_rollout as tr  # noqa: E402

fp = gp.fp
from gauntlet_env import FLYING, FULL, OUTCOMES, ROUGH, Physics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--tilt", type=float, default=45.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--eager", action="store_true", help="don't compile the update")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = torch.float64 if args.float64 else torch.float32
    seeds = args.seeds or list(range(args.first_seed, args.first_seed + args.n))
    ph = replace(Physics(), tilt_max=float(np.radians(args.tilt)))
    body = tr.rollout_body
    if not args.eager:
        body = torch.compile(tr.rollout_body, dynamic=True, fullgraph=True)
    f = fp.Fleet(len(seeds), ph, feat=FULL)
    for i, s in enumerate(seeds):
        f.set_course(i, fp.make_course(int(s), ROUGH, FULL, f.ph))
    pl = gp.GPUPlanner(f.n, device=args.device, dtype=dtype, body=body, frogs="ghost",
                       air="estimated", hyps=3, deep=3.0, band=16.0)  # fmt: skip
    gpu_s = [0.0]
    decide = pl.gpu_decide

    def timed(*a):
        t = time.time()
        decide(*a)
        if args.device != "cpu":
            torch.cuda.synchronize()
        gpu_s[0] += time.time() - t

    pl.gpu_decide = timed
    t0 = time.time()
    u = 0
    while (f.outcome == FLYING).any() and u < 1300:
        f.step(pl.act(f))
        u += 1
        if u % 50 == 0:
            c = Counter(OUTCOMES[o] for o in f.outcome)
            print(f"update {u}: {dict(c)}  {time.time() - t0:.0f} s (decide {gpu_s[0]:.0f} s, "
                  f"{pl.decisions} decisions)", flush=True)  # fmt: skip
    wall = time.time() - t0
    c = Counter(OUTCOMES[o] for o in f.outcome)
    print(f"landed {c['landed']}/{len(seeds)}  {dict(c)}  ({wall:.0f} s, decide {gpu_s[0]:.0f} s, "
          f"{pl.decisions} decisions, {u} updates)")  # fmt: skip
    res = []
    for i, s in enumerate(seeds):
        r = {"seed": int(s), "outcome": OUTCOMES[f.outcome[i]],
             "time": round(float(f.ticks[i] * f.ph.dt), 2), "x": float(f.x[i]),
             "togo": float(abs(f.goal()[0][i] - f.x[i])), "hit_by": int(f.hit_enc[i]),
             "met": f.enc_count[i].tolist()}  # fmt: skip
        res.append(r)
        if r["outcome"] != "landed":
            print("   ", r)
    print("searches:", dict(pl.stats))
    if args.out:
        (HERE / "runs").mkdir(exist_ok=True)
        dev = torch.cuda.get_device_name() if args.device != "cpu" else "cpu"
        run = {
            "flights": res,
            "seconds": round(wall),
            "decide_s": round(gpu_s[0]),
            "decisions": pl.decisions,
            "updates": u,
            "dtype": str(dtype),
            "device": dev,
        }
        out = {"gauntlet": run}
        (HERE / "runs" / f"{args.out}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
