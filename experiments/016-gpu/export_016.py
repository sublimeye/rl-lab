"""Post 16's site data: recorded flights and the numbers the post quotes.

--record N: the shield, the network alone and the fair planner fly the first N test courses
(numpy, 45 deg, pilot.py's flight loop), their sticks rounded to float32 before each update so
the browser port replays exactly what they flew (site/test/flawless-replay.test.mjs).
--export: site/public/data/016-flawless/results.json, every number the post quotes, read from
the runs of 016-flawless (the CPU thread) and 016-gpu (this one).

uv run experiments/016-gpu/export_016.py --record 24 --export
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
FLAWLESS = HERE.parent / "016-flawless"
sys.path.insert(0, str(FLAWLESS))
import pilot  # noqa: E402

SITE = HERE.parent.parent / "site"
DATA = SITE / "public" / "data" / "016-flawless"
NET = HERE / "runs" / "g1" / "net.npz"
MODES = {"shield": "shield-flights.json", "net": "network-flights.json",
         "planner": "planner-flights.json"}  # fmt: skip


def _record(job):
    """pilot._fly for `seeds` in `mode`, recording the sticks each drone flew."""
    seeds, mode = job
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np

    from gauntlet_env import FLYING, OUTCOMES, ROUGH, Config, FleetEnv

    expert = pilot.expert
    net = pilot.load_net(NET.parent, NET.stem) if mode != "planner" else None
    env = FleetEnv(len(seeds), Config(pilot.PH45, ROUGH, pilot.FEAT), seed=7)
    obs = env.start(seeds)
    f = env.fleet
    pl = pilot.Teacher(f.n)
    prev = np.full(f.n, expert.DEFAULT)
    sticks = [[] for _ in seeds]
    t = 0
    while (f.outcome == FLYING).any() and t < 1300:
        flying = f.outcome == FLYING
        pl.ph_real = f.ph
        pl.observe(f)
        pl.estimate(f)
        x = np.concatenate([obs, pilot.beliefs(pl, f, prev)], 1).astype(np.float32)
        if net is not None:
            logits, _ = net(x)
        if mode == "shield":
            due = pl.fresh | (pl.age >= pl.every)
            if due.any():
                top = np.argsort(-logits, 1)[:, :5]
                allowed = np.zeros((f.n, pilot.NM), bool)
                allowed[np.arange(f.n)[:, None], top] = True
                pl.allowed = allowed
            m = pl.plan_now(f)
        elif mode == "planner":
            m = pl.plan_now(f)
        else:
            m = np.argmax(logits, 1)
        a = expert.control(pl.assumed(f), m).astype(np.float32).astype(np.float64)
        for i in np.nonzero(flying)[0]:
            sticks[i].append(a[i])
        pl.remember(f, a)
        pl.age += 1
        prev = np.where(flying, m, prev)
        obs, _, _, _ = env.step(a)
        t += 1
    out = []
    for i, s in enumerate(seeds):
        st = np.asarray(sticks[i], "<f4")
        out.append({"seed": int(s), "outcome": OUTCOMES[f.outcome[i]], "updates": len(st),
                    "end": [float(f.x[i]), float(f.y[i])],
                    "sticks": base64.b64encode(st.tobytes()).decode()})  # fmt: skip
    return mode, out


def record(n: int, workers: int = 10) -> None:
    from multiprocessing import Pool

    seeds = list(range(20_000, 20_000 + n))
    jobs = [(seeds[i : i + 3], m) for m in MODES for i in range(0, n, 3)]
    with Pool(workers) as p:
        parts = p.map(_record, jobs)
    DATA.mkdir(parents=True, exist_ok=True)
    for mode, name in MODES.items():
        flights = sum((fl for m, fl in parts if m == mode), [])
        path = DATA / name
        path.write_text(json.dumps({"flights": flights}))
        c = Counter(f["outcome"] for f in flights)
        print(f"{mode}: {dict(c)}  {path.stat().st_size / 1e3:.0f} KB")


def _tally(flights) -> dict:
    c = Counter(r["outcome"] for r in flights)
    return {"landed": c["landed"], "n": len(flights), "outcomes": dict(c)}


def export() -> None:
    runs16, runs_gpu = FLAWLESS / "runs", HERE / "runs"

    def run(path, key="gauntlet"):
        d = json.loads(path.read_text())
        return d[key] if key in d else d

    # the fair planner's versions on the test courses (fair_planner.py, 016-flawless/runs)
    steps = [
        ("30 deg: frogs from strikes seen, air estimated", "fair-v1"),
        ("45 deg", "fair-t45"),
        (
            "+ the frog's column owned during a possible strike; waits past the pad",
            "fair-t45-ghost",
        ),
        ("+ margins round every threat; higher waits", "fair-t45-v4"),
        ("band off (no preference for a plain cruise)", "fair-t45-v5"),
        ("+ the band's cruise heads to the pad", "fair-t45-v6"),
    ]
    planner = []
    for label, name in steps:
        r = run(runs16 / f"{name}.json")
        planner.append({"label": label, "run": name, **_tally(r["flights"]),
                        "seconds": r["seconds"]})  # fmt: skip
    fresh = run(runs16 / "fair-t45-v6-fresh.json")
    t30 = run(runs16 / "fair-t30-v6.json")
    gpu_t = run(runs_gpu / "gpu-t45.json")
    gpu_f = run(runs_gpu / "gpu-t45-fresh.json")
    d1 = json.loads((runs16 / "d1" / "rounds.json").read_text())
    g1 = json.loads((runs_gpu / "g1" / "rounds.json").read_text())
    shield = {}
    for k, first in ((5, 20_000), (5, 30_000), (3, 20_000), (2, 20_000)):
        p = runs_gpu / "g1" / f"test-net-shield-k{k}-{first}.json"
        if p.exists():
            r = json.loads(p.read_text())
            shield[f"k{k}-{first}"] = {**_tally(r["flights"]), "seconds": r["seconds"],
                                       "decisions": r["stats"]["decisions"],
                                       "full": r["stats"]["full"]}  # fmt: skip
    out = {
        "planner": planner,
        "planner_fresh": {**_tally(fresh["flights"]), "seconds": fresh["seconds"]},
        "planner_30deg": {**_tally(t30["flights"]), "seconds": t30["seconds"]},
        "gpu_planner": {
            "test": {
                **_tally(gpu_t["flights"]),
                "seconds": gpu_t["seconds"],
                "decisions": gpu_t["decisions"],
            },
            "fresh": {**_tally(gpu_f["flights"]), "seconds": gpu_f["seconds"]},
            "device": gpu_t["device"],
        },  # fmt: skip
        "dagger": {
            "cpu": [
                {
                    "round": r["round"],
                    "train_flights": r["train_flights"],
                    "train_landed": r["train_landed"],
                    "test_landed": r["test_landed"],
                    "accuracy": r["accuracy"],
                    "minutes": r["minutes"],
                }
                for r in d1
            ],
            "gpu": [
                {
                    "round": r["round"],
                    "train_flights": r["train_flights"],
                    "train_landed": r["train_landed"],
                    "test_landed": r["test_landed"],
                    "accuracy": r["accuracy"],
                    "minutes": r["minutes"],
                }
                for r in g1
            ],
            "cpu_paused_minutes": 45.25,  # d1 paused twice (14:38-15:01, 15:21-15:44)
        },  # fmt: skip
        "shield": shield,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.record:
        record(args.record, args.workers)
    if args.export:
        export()


if __name__ == "__main__":
    main()
