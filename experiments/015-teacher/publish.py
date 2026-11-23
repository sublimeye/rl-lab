"""Post 15's numbers and site data.

uv run experiments/015-teacher/publish.py --table ft-s1 ft-dodge dagger-s1/round_03   # every test
uv run experiments/015-teacher/publish.py --export        # site/public/data/015-teacher/*.json
uv run experiments/015-teacher/publish.py --summary       # prints the numbers quoted in the post

A run name means its last checkpoint (fine-tunes) or a named zip (run/stem).
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))

from dagger import RUNS, NumpyNet, actor_layers  # noqa: E402

SITE = HERE.parent.parent / "site"
DATA = SITE / "public" / "data" / "015-teacher"
DATA14 = SITE / "public" / "data" / "014-gauntlet"
# fine-tunes: (on full test courses | post 14's drills and dials) x (kept close to the teacher
# | not) x (dodge bonus | not)
FINETUNES = ("ft-s1", "ft-dodge", "ft-free", "ft-school", "ft-school-dodge", "ft-school-free",
             "ft-school-free-dodge")  # fmt: skip
MODELS = {  # site name: (checkpoint, label)
    "tuned": ("ft-school-free", "Fine-tuned (PPO)"),
    "taught": ("dagger-s1/round_03", "Taught (DAgger)"),
}
# The planner's versions on the 200 test courses (logs/expert-v*.log, expert-band16.log,
# expert-fair.log): what each change bought
PLANNER_STEPS = [
    ("Post 14's planner", "logs: 014 planner rerun (runs/planner-014.json)"),
    ("+ jinks, open-threat charge, deepening at the threat and the pad", "logs/expert-v1.log"),
    ("+ hover throttle for the mass and charge", "logs/expert-v2.log"),
    ("+ a level move, more jink delays, deepening in rounds", "logs/expert-v3.log"),
    ("+ deepening from the moment a threat appears", "logs/expert-v4.log"),
    ("+ a band that prefers a plain cruise (easier to copy)", "logs/expert-band16.log"),
    ("+ no foreknowledge of waypoints (fair)", "logs/expert-fair.log"),
]


def ckpt_path(name: str) -> Path:
    if "/" in name:
        return RUNS / f"{name}.zip"
    return sorted((RUNS / name).glob("ckpt_*.zip"))[-1]


def load_net(name: str) -> NumpyNet:
    from stable_baselines3 import PPO

    return NumpyNet(actor_layers(PPO.load(ckpt_path(name), device="cpu")))


def table(names) -> None:
    from evaluate import run_tests, show

    for name in names:
        print(f"{name} ({ckpt_path(name).stem})")
        res = run_tests(load_net(name))
        for n, s in res.items():
            show(n, s)
        out = RUNS / f"eval-{name.replace('/', '-')}.json"
        out.write_text(json.dumps({"ckpt": ckpt_path(name).stem, **res}))


def planner_versions() -> list:
    """(label, landed, outcomes) per planner version, read from the logs."""
    out = []
    for label, src in PLANNER_STEPS:
        if src.startswith("logs: 014"):
            res = json.loads((RUNS / "planner-014.json").read_text())
            c = Counter(r["outcome"] for r in res)
        else:
            line = (HERE / src).read_text().splitlines()[0]
            c = Counter(ast.literal_eval(line.split("  ")[1]))
        out.append({"label": label, "landed": c["landed"], "outcomes": dict(c)})
    return out


def packed(net: NumpyNet) -> dict:
    from export import b64  # 014's: float16, base64

    return {
        "layers": [{"shape": list(w.T.shape), "w": b64(w.T), "b": b64(b)} for w, b in net.layers]
    }


def curve_points(run: str) -> list:
    """Landed per test against minutes of training. A gap between tests over twice the usual
    one is the laptop asleep (once, 27 min, in the resumed school runs): it counts as a usual
    gap, so the minutes are minutes of compute."""
    import numpy as np

    pts = json.loads((RUNS / run / "points.json").read_text())
    gaps = np.diff([p["minutes"] for p in pts])
    usual = float(np.median(gaps)) if len(gaps) else 0.0
    out, mins = [], pts[0]["minutes"] if pts else 0.0
    for i, p in enumerate(pts):
        if i:
            g = gaps[i - 1]
            mins += usual if g > 2 * usual else g
        out.append({"steps": p["steps"], "seconds": round(mins * 60, 1),
                    "landed": p["test"]["landed"], "n": 200})  # fmt: skip
    return out


def export() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (ckpt, label) in MODELS.items():
        net = load_net(ckpt)
        (DATA / f"model-{name}.json").write_text(
            json.dumps({"label": label, "act": "tanh", "predict": True, **packed(net)})
        )
    ev = {"planner": planner_versions(), "dagger": {}, "finetune": {}, "table": {}, "curves": {}}
    fair = json.loads((RUNS / "planner-fair.json").read_text())["gauntlet"]
    ev["planner_fair"] = {
        "seconds": fair["seconds"],
        "decisions": fair["decisions"],
        "time": round(sum(f["time"] for f in fair["flights"]) / len(fair["flights"]), 1),
    }
    for run in ("dagger-band0", "dagger-unfair", "dagger-s1"):
        ev["dagger"][run] = json.loads((RUNS / run / "rounds.json").read_text())
    ev["side"] = json.loads((RUNS / "side-tests.json").read_text())
    for run in FINETUNES:
        ev["finetune"][run] = json.loads((RUNS / run / "points.json").read_text())
    for p in sorted(RUNS.glob("eval-*.json")):
        ev["table"][p.stem[5:]] = json.loads(p.read_text())
    ev14 = json.loads((DATA14 / "eval.json").read_text())
    ev["table"]["post 14"] = ev14["table"]["final-s1"]
    # DroneChart: landed per test against minutes; the fine-tune against post 14's PPO
    # (one run per group: DroneChart averages a group's runs as seeds)
    for key, run in (("tuned", "ft-school-free"), ("tuned-dodge", "ft-school-free-dodge"),
                     ("anchored", "ft-school"), ("anchored-dodge", "ft-school-dodge")):  # fmt: skip
        ev["curves"][key] = {run: curve_points(run)}
    ev["curves"]["post14"] = ev14["curves"]["final"]  # both seeds
    ev["table"]["autopilot"] = None
    (DATA / "eval.json").write_text(json.dumps(ev))
    for p in sorted(DATA.glob("*.json")):
        print(f"{p.name}: {p.stat().st_size / 1e3:.0f} KB")


def summary() -> None:
    ev = json.loads((DATA / "eval.json").read_text())
    print("planner versions (200 test courses)")
    for v in ev["planner"]:
        print(f"  {v['label']:<70} landed {v['landed']}  {v['outcomes']}")
    print("  fair planner:", ev["planner_fair"])
    for run, rounds in ev["dagger"].items():
        print(f"DAgger {run}")
        for r in rounds:
            t = r["test"]
            print(
                f"  round {r['round']} ({r['flown_by']}): training landed {r['train_landed']}/"
                f"{r['train_flights']}, {r['states']:,} states, held out "
                f"{r.get('held_out_loss')}, test landed {t['landed']} {t['outcomes']}"
            )
    for run, pts in ev["finetune"].items():
        print(f"fine-tune {run}")
        for p in pts:
            t = p["test"]
            when = f"{p['steps'] / 1e6:7.1f}M {p['minutes']:6.1f} min"
            print(f"  {when}: landed {t['landed']} {t['outcomes']}")
    print("tables")
    for run, res in ev["table"].items():
        if not res:
            continue
        print(f"  {run} ({res.get('ckpt')})")
        for test, s in res.items():
            if test == "ckpt":
                continue
            enc = "  ".join(f"{k} {v['hit']}/{v['met']}" for k, v in s["encounters"].items())
            print(f"    {test:<14} landed {s['landed']:>3} {s['outcomes']}  {enc}")


def _record(seeds):
    """The planner (fair, as taught) on test courses, its sticks rounded to float32 before each
    update so the site replays exactly what it flew."""
    import os

    import numpy as np

    os.environ["OMP_NUM_THREADS"] = "1"
    from expert import Planner
    from gauntlet_env import FLYING, FULL, OUTCOMES, ROUGH, Fleet, make_course

    f = Fleet(len(seeds), feat=FULL)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(s, ROUGH, FULL, f.ph))
    pl = Planner(f.n)
    sticks = [[] for _ in seeds]
    while (f.outcome == FLYING).any():
        flying = f.outcome == FLYING
        a = pl.act(f).astype(np.float32).astype(np.float64)
        for i in np.nonzero(flying)[0]:
            sticks[i].append(a[i])
        f.step(a)
    out = []
    for i, s in enumerate(seeds):
        st = np.asarray(sticks[i], "<f4")
        x, y = float(f.x[i]), float(f.y[i])
        out.append({"seed": s, "outcome": OUTCOMES[f.outcome[i]], "updates": len(st),
                    "end": [x, y], "sticks": base64.b64encode(st.tobytes()).decode()})  # fmt: skip
    return out


def record(n: int, workers: int = 2) -> None:
    from multiprocessing import Pool

    seeds = list(range(20_000, 20_000 + n))
    with Pool(workers) as p:
        parts = p.map(_record, [seeds[i : i + 2] for i in range(0, n, 2)])
    flights = sum(parts, [])
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "planner-flights.json"
    path.write_text(json.dumps({"flights": flights}))
    print(Counter(f["outcome"] for f in flights), f"{path.stat().st_size / 1e3:.0f} KB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", nargs="+", default=[])
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--record", type=int, default=0, help="record the planner on n test courses")
    args = ap.parse_args()
    if args.record:
        record(args.record)
    if args.table:
        table(args.table)
    if args.export:
        export()
    if args.summary:
        summary()


if __name__ == "__main__":
    main()
