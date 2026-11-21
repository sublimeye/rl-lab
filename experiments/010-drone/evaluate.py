"""Scores pilots on 200 held-out courses and exports the site data.

uv run experiments/010-drone/evaluate.py --runs ppo-s1 ppo-s2 --curves     # per-checkpoint, calm
uv run experiments/010-drone/evaluate.py --table ppo=ppo-s1 sac=sac-s1  # final, all weathers
uv run experiments/010-drone/evaluate.py --export spec.json   # site/public/data/010-drone
uv run experiments/010-drone/evaluate.py --parity ppo-s1                   # site/test fixture

Every pilot is scored with its weights rounded the way the site gets them (6 significant digits),
so the numbers are the browser network's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drone_env import (
    CALM,
    HEAVY,
    ROUGH,
    WINDY,
    Drone,
    Net,
    Physics,
    autopilot,
    make_course,
    observe,
)

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SITE = HERE.parent.parent / "site"
DATA = SITE / "public" / "data" / "010-drone"
TEST_SEEDS = [1_000_000_000 + i for i in range(200)]
WEATHER = {"calm": CALM, "windy": WINDY, "heavy": HEAVY, "rough": ROUGH}
DIGITS = 6


def rounded(a):
    a = np.asarray(a, float)
    return np.array([float(f"{v:.{DIGITS}g}") for v in a.ravel()]).reshape(a.shape)


def load_spec(ckpt: str) -> dict:
    """A checkpoint's actor with site-precision weights."""
    from train import actor, load_model

    spec = actor(load_model(ckpt))
    cfg = json.loads((RUNS / ckpt.split("/")[0] / "config.json").read_text())
    spec["physics"] = {"stick_rate": cfg.get("stick_rate", 0.0)}
    spec["layers"] = [{"w": rounded(L["w"]), "b": rounded(L["b"])} for L in spec["layers"]]
    return spec


def last_ckpt(run: str) -> str:
    return f"{run}/{sorted((RUNS / run).glob('ckpt_*.zip'))[-1].stem}"


def flight(seed: int, pilot, weather=CALM, record=False, ph: Physics | None = None) -> dict:
    ph = ph or Physics()
    d = Drone(make_course(seed, weather, ph), ph)
    sticks = []
    while not d.done:
        a = pilot(d)
        d.step(*a)
        sticks.append(d.stick)
    s = np.array(sticks)
    change = np.abs(np.diff(s, axis=0)).sum(1).mean() if len(s) > 1 else 0.0
    out = {
        "outcome": d.outcome,
        "time": d.time,
        "touch": abs(d.touch[1]),
        "crash": d.crash,
        "change": float(change),  # mean stick movement per update (both sticks, units of 2=full)
        "pinned": float((np.abs(s) > 0.95).mean()),  # share of stick readings at an end
    }
    if record:
        out["sticks"] = sticks
    return out


def score(pilot, weather=CALM, seeds=TEST_SEEDS) -> dict:
    """pilot: a function of the drone, or a Net (flown with the physics it was trained with)."""
    ph = Physics(**pilot.physics) if isinstance(pilot, Net) else None
    fn = pilot.pilot if isinstance(pilot, Net) else pilot
    fl = [flight(s, fn, weather, ph=ph) for s in seeds]
    landed = [f for f in fl if f["outcome"] == "landed"]
    hard = [f for f in fl if f["crash"] == "hard"]
    counts = {
        k: sum(f["outcome"] == k for f in fl)
        for k in ("landed", "crashed", "water", "lost", "timeout")
    }
    return {
        **counts,
        "n": len(fl),
        "time": round(float(np.mean([f["time"] for f in landed])), 2) if landed else None,
        "touch": round(float(np.mean([f["touch"] for f in landed])), 3) if landed else None,
        "hard": len(hard),  # crashes that were landings: too fast down or sideways, or tilted
        "hard_vy": round(float(np.mean([f["touch"] for f in hard])), 3) if hard else None,
        "change": round(float(np.mean([f["change"] for f in fl])), 4),
        "pinned": round(float(np.mean([f["pinned"] for f in fl])), 4),
    }


def curve(run: str, weather="calm") -> list[dict]:
    """Calm-air scores at every checkpoint, cached next to the run, with wall-clock seconds."""
    cache = RUNS / run / f"eval_{weather}.json"
    marks = dict(json.loads((RUNS / run / "train_log.json").read_text())["marks"])
    have = json.loads(cache.read_text()) if cache.exists() else {}
    out = []
    for ck in sorted((RUNS / run).glob("ckpt_*.zip")):
        steps = int(ck.stem.split("_")[1])
        key = str(steps)
        if key not in have:
            have[key] = score(Net(load_spec(f"{run}/{ck.stem}")), WEATHER[weather])
            cache.write_text(json.dumps(have))
        out.append({"steps": steps, "seconds": marks.get(steps), **have[key]})
    return out


def table(pilots: dict[str, str]) -> dict:
    """Final checkpoints (and the autopilot) in every weather."""
    rows = {"autopilot": {w: score(autopilot, wx) for w, wx in WEATHER.items()}}
    for label, ckpt in pilots.items():
        net = Net(load_spec(ckpt if "/" in ckpt else last_ckpt(ckpt)))
        rows[label] = {w: score(net, wx) for w, wx in WEATHER.items()}
    return rows


def print_score(label, s):
    print(
        f"  {label:<14} landed {s['landed']:3d}/{s['n']}  crash {s['crashed']:3d}"
        f" water {s['water']:3d}"
        f" lost {s['lost']:3d} timeout {s['timeout']:3d}  time {s['time']}  touch {s['touch']}"
        f"  stick-change {s['change']:.3f}  pinned {s['pinned']:.1%}"
    )


def model_json(ckpt: str, label: str) -> dict:
    spec = load_spec(ckpt if "/" in ckpt else last_ckpt(ckpt))
    return {
        "label": label,
        "physics": spec["physics"],
        "act": spec["act"],
        "squash": spec["squash"],
        "layers": [{"w": L["w"].tolist(), "b": L["b"].tolist()} for L in spec["layers"]],
    }


def export_parity(ckpts: list[str]) -> None:
    """Courses, whole flights (sticks, states, observations) and network outputs for the JS port:
    every network flies two courses with the physics it was trained with, plus the autopilot."""

    def r(v):
        return float(f"{v:.13g}")

    fx = {"courses": [], "nets": [], "flights": []}
    for s in [0, 1, 2, 7, 42, 1_000_000_123, 3_000_000_000]:
        for w in ("calm", "rough"):
            fx["courses"].append({"seed": s, "weather": w, **_course(make_course(s, WEATHER[w]))})
    pilots = [("autopilot", None)]
    for ckpt in ckpts:
        spec = load_spec(ckpt if "/" in ckpt else last_ckpt(ckpt))
        fx["nets"].append({k: _plain(v) for k, v in spec.items()})
        pilots.append((len(fx["nets"]) - 1, spec))
    plan = [(1_000_000_000, "calm"), (1_000_000_001, "rough"), (5, "windy"), (9, "heavy")]
    for i, (net_i, spec) in enumerate(pilots):
        net = Net(spec) if spec else None
        ph = Physics(**spec["physics"]) if spec else Physics()
        for seed, w in plan[(i % 2) * 2 :][:2] if spec else plan:
            d = Drone(make_course(seed, WEATHER[w], ph), ph)
            sticks, states, outs = [], [], []
            while not d.done and len(sticks) < 150:  # long enough; keeps the fixture small
                if net:
                    o = observe(d)
                    a = net(o)
                    outs.append([r(v) for v in a])
                    a = (float(a[0]), float(a[1]))
                else:
                    a = autopilot(d)
                    outs.append([r(v) for v in a])
                d.step(*a)
                sticks.append([r(v) for v in d.stick])
                st = [d.x, d.y, d.vx, d.vy, d.th, d.om, d.motor, d.ticks] + observe(d)
                states.append([r(v) for v in st])
            fx["flights"].append(
                {
                    "seed": seed,
                    "weather": w,
                    "pilot": "autopilot" if spec is None else net_i,
                    "physics": spec["physics"] if spec else {"stick_rate": 0.0},
                    "wants": outs,
                    "sticks": sticks,
                    "states": states,
                    "outcome": d.outcome,  # None when cut at 150 updates
                }
            )
    path = SITE / "test" / "drone-parity.json"
    path.write_text(json.dumps(fx, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} KB)")


def _plain(v):
    if isinstance(v, list):
        return [{"w": np.asarray(L["w"]).tolist(), "b": np.asarray(L["b"]).tolist()} for L in v]
    return v


def _course(c) -> dict:
    return {
        "start": [c.start.x0, c.start.x1, c.start.top],
        "target": [c.target.x0, c.target.x1, c.target.top],
        "wind": c.wind,
        "gust": c.gust,
        "period": c.gust_period,
        "phase": c.gust_phase,
        "mass": c.mass,
    }


def write(name: str, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    path.write_text(json.dumps(obj, separators=(",", ":")))
    size = path.stat().st_size
    print(f"wrote {path.relative_to(SITE)} ({size / 1e3:.1f} KB)")
    assert size < 1_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--table", nargs="*", default=None, help="label=run-or-ckpt ...")
    ap.add_argument(
        "--parity", nargs="*", default=None, help="checkpoints (or runs) for the JS fixture"
    )
    ap.add_argument("--export", default=None, help="JSON spec of the site export, see summarize")
    args = ap.parse_args()

    if args.curves:
        for run in args.runs:
            print(run)
            for c in curve(run):
                print_score(f"{c['steps']:>9,}", c)
    if args.table is not None:
        pilots = dict(p.split("=", 1) for p in args.table)
        for label, rows in table(pilots).items():
            print(label)
            for w, s in rows.items():
                print_score(w, s)
    if args.parity is not None:
        export_parity(args.parity)
    if args.export:
        export(json.loads(Path(args.export).read_text()))


def export(spec: dict) -> None:
    """spec: {"models": {file: [ckpt, label]}, "groups": {name: [runs]}, "table": {label: ckpt}}"""
    for fname, (ckpt, label) in spec["models"].items():
        write(fname, model_json(ckpt, label))
    curves = {g: {r: curve(r) for r in runs} for g, runs in spec["groups"].items()}
    rows = table(spec["table"])
    write("eval.json", {"seeds": len(TEST_SEEDS), "curves": curves, "table": rows})
    write("traces.json", traces(spec["traces"]["seed"], spec["traces"]["pilots"]))


def pilot_of(ckpt: str):
    if ckpt == "autopilot":
        return autopilot, None
    net = Net(load_spec(ckpt if "/" in ckpt else last_ckpt(ckpt)))
    return net.pilot, Physics(**net.physics)


def traces(seed: int, pilots: dict[str, str]) -> dict:
    """Every stick update of each pilot on one calm course (sticks rounded to 3 decimals)."""
    c = make_course(seed)
    out = []
    for label, ckpt in pilots.items():
        fn, ph = pilot_of(ckpt)
        f = flight(seed, fn, record=True, ph=ph)
        out.append(
            {
                "label": label,
                "outcome": f["outcome"],
                "time": round(f["time"], 2),
                "change": round(f["change"], 4),
                "pinned": round(f["pinned"], 4),
                "sticks": [[round(a, 3), round(b, 3)] for a, b in f["sticks"]],
            }
        )
    facing = 1 if c.target.cx > c.start.cx else -1
    return {"seed": seed, "dt": 0.05, "facing": facing, "course": _course(c), "pilots": out}


if __name__ == "__main__":
    main()
