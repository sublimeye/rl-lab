"""Post 17's site data: recorded flights and every number the post quotes.

--record N: the pilots fly the first N test courses in solo_env; their sticks (float32, as
flown) are saved so the browser port replays each flight exactly
(site/test/solo-replay.test.mjs).
--export: site/public/data/017-solo/results.json and eval.json, read from this folder's runs/
(diagnose.py, refit.py, sensitivity.py, train.py, solo_eval.py).

uv run experiments/017-solo/publish.py --record 24 --export
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import solo_env as se  # noqa: E402
from solo_net import load  # noqa: E402

RUNS = HERE / "runs"
DATA = HERE.parent.parent / "site" / "public" / "data" / "017-solo"
# name on the site -> (network, sampled moves?)
PILOTS = {
    "solo": ("b1/last.npz", True),
    "copy": ("solo-sr.npz", True),
}
REFITS = ("base", "noprev", "reg", "short", "safe", "safe-noprev", "safe-reg", "soft-noprev",
          "eighth", "quarter", "half", "sr")  # fmt: skip


def _record(job):
    """solo_eval's flight of one chunk of 20 test courses, keeping the sticks."""
    name, first = job
    os.environ["OMP_NUM_THREADS"] = "1"
    path, sampled = PILOTS[name]
    net = load(RUNS / path)
    seeds = list(range(first, first + 20))
    if sampled:
        rng = np.random.default_rng(seeds[0])

        def policy(x, mask):
            return net.sample(x, mask, rng)
    else:
        policy = net.act
    out, f, tape = se.fly(policy, seeds, hold=1, record=True)
    flights = []
    for i, s in enumerate(seeds):
        st = np.asarray(tape[i], "<f4")
        flights.append({"seed": int(s), "outcome": out[i], "updates": len(st),
                        "end": [float(f.x[i]), float(f.y[i])],
                        "sticks": base64.b64encode(st.tobytes()).decode()})  # fmt: skip
    return name, flights


def record(n: int, workers: int) -> None:
    from multiprocessing import Pool

    jobs = [(name, first) for name in PILOTS for first in range(20_000, 20_000 + n, 20)]
    with Pool(workers) as p:
        parts = p.map(_record, jobs)
    DATA.mkdir(parents=True, exist_ok=True)
    for name in PILOTS:
        flights = sum((fl for nm, fl in parts if nm == name), [])[:n]
        path = DATA / f"{name}-flights.json"
        path.write_text(json.dumps({"flights": flights}))
        c = Counter(f["outcome"] for f in flights)
        print(f"{name}: {dict(c)}  {path.stat().st_size / 1e3:.0f} KB")


def _read(name: str):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else None


def _strip(ev: dict) -> dict:
    """An evaluation without the per-flight lists."""
    out = {}
    for mode in ("top", "sampled"):
        if mode in ev:
            out[mode] = {k: {q: v[q] for q in v if q != "flights"} for k, v in ev[mode].items()}
    return out


def _training(run: str) -> dict | None:
    hist, cfg = _read(f"{run}/train.json"), _read(f"{run}/config.json")
    if not hist:
        return None
    checks = [h for h in hist if "dev_top" in h]
    final = {}
    for line in (RUNS / run / "log.txt").read_text().splitlines():
        m = re.match(
            r"last network alone, (top|sampled) move, 1000 dev courses: landed (\d+)", line
        )
        if m:
            final[m.group(1)] = int(m.group(2))
    return {
        "run": run,
        "config": cfg,
        "minutes": hist[-1]["minutes"],
        "decisions": hist[-1]["decisions"],
        "dev_n": cfg.get("dev_n"),
        "final_dev": final,
        "checks": [
            {
                "round": h["round"],
                "minutes": h["minutes"],
                "decisions": h["decisions"],
                "top": h["dev_top"].get("landed", 0),
                "sampled": h["dev_sampled"].get("landed", 0),
                "sampled_outcomes": h["dev_sampled"],
            }
            for h in checks
        ],  # fmt: skip
    }


def export(runs: list[str], lineage: list[str], scratch: str) -> None:
    out = {
        "diagnosis": _read("diagnosis.json"),
        "sensitivity": (_read("sensitivity.json") or {}).get("summary"),
        "refits": {},
        "evals": {},
    }
    for name in REFITS:
        r = _read(f"refit-{name}.json")
        if r:
            keys = ("no_prev", "no_air", "loss", "dropout", "noise", "wd", "steps", "share")
            out["refits"][name] = {k: r[k] for k in r if k != "args"}
            out["refits"][name]["args"] = {k: r["args"].get(k) for k in keys}
    for p in sorted(RUNS.glob("eval-*.json")):
        out["evals"][p.stem[5:]] = _strip(json.loads(p.read_text()))
    out["trainings"] = [t for t in map(_training, runs) if t]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "results.json").write_text(json.dumps(out, indent=1))
    # DroneChart's shape: dev courses landed at each check (the network drawing its moves).
    # "copy" is the practice from the safe-move copy: the first run, then each run that carried
    # on from it, its minutes and decisions counted on; "scratch" is the run with no copy.
    by = {t["run"]: t for t in out["trainings"]}
    line, t0, d0 = [], 0.0, 0
    for run in [r for r in lineage if r in by]:
        t = by[run]
        for h in t["checks"][1 if line else 0 :]:
            line.append({"steps": d0 + h["decisions"], "seconds": (t0 + h["minutes"]) * 60,
                         "landed": h["sampled"], "n": t["dev_n"]})  # fmt: skip
        t0, d0 = t0 + t["minutes"], d0 + t["decisions"]
    curves = {"copy": {"practice": line}}
    if scratch in by:
        t = by[scratch]
        curves["scratch"] = {scratch: [{"steps": h["decisions"], "seconds": h["minutes"] * 60,
                                        "landed": h["sampled"], "n": t["dev_n"]}
                                       for h in t["checks"]]}  # fmt: skip
    (DATA / "eval.json").write_text(json.dumps({"curves": curves, "table": {}}))
    for f in sorted(DATA.glob("*.json")):
        print(f"{f.name}: {f.stat().st_size / 1e3:.0f} KB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--runs", nargs="+", default=["a1", "b1", "s1", "n1", "a2"],
                    help="training runs to report (those found)")  # fmt: skip
    ap.add_argument("--line", nargs="+", default=["b1"],
                    help="the practice from the copy, each run carrying on the last")  # fmt: skip
    ap.add_argument("--scratch", default="s1", help="the run that started from nothing")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.record:
        record(args.record, args.workers)
    if args.export:
        export(args.runs, args.line, args.scratch)


if __name__ == "__main__":
    main()
