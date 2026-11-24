"""Test a move-picking network alone (post 17): no planner, no simulator copies.

uv run experiments/017-solo/solo_eval.py --net runs/a1/last.npz --name a1 --kinds
uv run experiments/017-solo/solo_eval.py --net scripted --name scripted --kinds

Tests, 200 fixed courses each (45 degree drone, rough weather):
- test: the gauntlet's test courses 20000-20199 (post 14's);
- fresh: 30000-30199 (post 16's validation courses);
- one kind of course at a time, on the test seeds (post 14's table): no threats, one encounter
  kind at every waypoint with no frogs, the two frogs alone.
Each is flown twice: the network's top move every decision, and a move drawn from its
distribution (seeded, so repeatable).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))

import solo_env as se  # noqa: E402
from solo_net import load  # noqa: E402

from gauntlet_env import CLEAR, ENCOUNTERS  # noqa: E402

TEST, FRESH = 20_000, 30_000


def only(k: int):
    mix = [0.0] * len(ENCOUNTERS)
    mix[k] = 1.0
    return replace(se.FEAT, mix=tuple(mix), route_frog=0.0, guard=0.0)


KINDS = {
    "clear": replace(CLEAR, predict=True),
    **{ENCOUNTERS[k]: only(k) for k in range(1, len(ENCOUNTERS))},
    "frogs": replace(CLEAR, predict=True, route_frog=1.0, guard=1.0),
}


def _job(job):
    seeds, net, feat, hold, sampled = job
    os.environ["OMP_NUM_THREADS"] = "1"
    if net == "scripted":
        policy = se.scripted
    elif sampled:
        rng = np.random.default_rng(seeds[0])

        def policy(x, mask):
            return net.sample(x, mask, rng)
    else:
        policy = net.act
    out, f = se.fly(policy, seeds, feat, hold=hold)
    # encounters met, by kind, and which kind's bird or missile ended the flight
    hits = np.bincount(f.hit_enc[f.hit_enc >= 0], minlength=len(ENCOUNTERS))
    # flights that left the course: out through the ceiling, or past the pad and off the far end
    lost = np.array(out) == "lost"
    high = lost & (f.y > f.ph.ceiling)
    past = lost & ~high & (f.dir * (f.x - f.goal()[0]) > 0)
    left = [int(high.sum()), int(past.sum())]
    return out, f.enc_count.sum(0), hits, (f.ticks * f.ph.dt).tolist(), left


def run(net, first: int, feat=se.FEAT, hold: int = 1, sampled: bool = False, pool=None,
        n: int = 200) -> dict:  # fmt: skip
    jobs = [(list(range(first + i, first + i + 20)), net, feat, hold, sampled)
            for i in range(0, n, 20)]  # fmt: skip
    parts = pool.map(_job, jobs)
    out = sum((p[0] for p in parts), [])
    times = sum((p[3] for p in parts), [])
    c = Counter(out)
    landed_t = [t for o, t in zip(out, times, strict=True) if o == "landed"]
    return {
        "landed": c["landed"],
        "n": n,
        "outcomes": dict(c),
        "met": np.sum([p[1] for p in parts], 0).tolist(),
        "hit_by": np.sum([p[2] for p in parts], 0).tolist(),
        "lost_ceiling": sum(p[4][0] for p in parts),
        "lost_past_pad": sum(p[4][1] for p in parts),
        "seconds_landed": round(float(np.mean(landed_t)), 2) if landed_t else None,
        "flights": out,
    }


def main() -> None:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="a network (.npz under 017-solo), or 'scripted'")
    ap.add_argument("--name", required=True)
    ap.add_argument("--hold", type=int, default=1)
    ap.add_argument("--kinds", action="store_true", help="also one kind of course at a time")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    net = "scripted" if args.net == "scripted" else load(HERE / args.net)
    res = {"net": args.net, "hold": args.hold}
    with Pool(args.workers) as pool:
        for mode in ("top",) if net == "scripted" else ("top", "sampled"):
            smp = mode == "sampled"
            r = res[mode] = {}
            for tag, first in (("test", TEST), ("fresh", FRESH)):
                r[tag] = run(net, first, hold=args.hold, sampled=smp, pool=pool)
                print(f"{args.name} {mode:7s} {tag:5s}: landed {r[tag]['landed']}/200 "
                      f"{r[tag]['outcomes']}", flush=True)  # fmt: skip
            if args.kinds:
                for kind, feat in KINDS.items():
                    r[kind] = run(net, TEST, feat, args.hold, smp, pool)
                    k = ENCOUNTERS.index(kind) if kind in ENCOUNTERS else 0
                    met, hit = r[kind]["met"][k], sum(r[kind]["hit_by"])
                    print(f"{args.name} {mode:7s} {kind:15s}: landed {r[kind]['landed']:3d}/200 "
                          f"{r[kind]['outcomes']}" + (f"; hit by {hit} of {met}" if k else ""),
                          flush=True)  # fmt: skip
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / f"eval-{args.name}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
