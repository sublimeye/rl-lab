"""Site data and the JS parity fixture for post 12 (called from evaluate.py)."""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import numpy as np

from dodge_env import (
    FLYING,
    FULL,
    FULL_VISION,
    ONLY,
    ROUGH,
    SCHOOL,
    TIMEOUT,
    Config,
    FleetEnv,
    make_course,
)
from evaluate import DATA, RUNS, SITE, TEST_SEEDS, TESTS, Net, curve, load_spec, report, run_config


def r13(v) -> float:
    return float(f"{v:.13g}")


def b64(a) -> str:
    """float16, little-endian, base64."""
    return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode()


def packed(spec: dict) -> list[dict]:
    return [
        {"shape": list(L["w"].shape), "w": b64(L["w"]), "b": b64(L["b"])} for L in spec["layers"]
    ]


def model_json(ckpt: str, label: str) -> dict:
    spec = load_spec(ckpt)
    return {"label": label, "task": spec["task"], "obs": spec["obs"], "act": "tanh",
            "layers": packed(spec)}  # fmt: skip


def school(obs: str):
    return replace(FULL_VISION if obs == "vision" else FULL, threats=SCHOOL, noise=0.0)


def parity_seeds() -> list[int]:
    """Three school courses that between them have every threat: a course with a missile and
    rocks, an arena with a missile, a course with birds and rocks."""
    ph = Config().physics
    want = [
        lambda c: not c["arena"] and c["missile"] and len(c["rocks"]) == 2,
        lambda c: c["arena"] and c["missile"] and c["rocks"],
        lambda c: not c["arena"] and len(c["birds"]) >= 3 and c["rocks"] and not c["missile"],
    ]
    out, s = [], 1_000_000_000
    for ok in want:
        while not ok(make_course(s, ROUGH, school("state"), ph)):
            s += 1
        out.append(s)
        s += 1
    return out


STATE_KEYS = ("x y vx vy th om motor charge pad_x0 m_x m_y m_psi").split()


def export_parity(ckpts: list[str]) -> None:
    """Courses from many seeds, and every network flying three school courses with sensor noise
    off: the drone's and the threats' state after every update (up to 150), what the pilot saw
    every 25th update, and the network's choice."""
    fx = {"courses": [], "nets": [], "flights": []}
    ph = Config().physics
    for obs in ("state", "vision"):
        for s in [0, 3, 7, 42, 1_000_000_005, 3_000_000_000, 12345, 777]:
            fx["courses"].append({"seed": s, "obs": obs, **make_course(s, ROUGH, school(obs), ph)})
    aimed = replace(FULL, threats=ONLY["aimed"], noise=0.0)
    for s in [5, 11, 1_000_000_007]:
        fx["courses"].append(
            {"seed": s, "obs": "state", "threats": "aimed", **make_course(s, ROUGH, aimed, ph)}
        )
    seeds = parity_seeds()
    for ckpt in ckpts:
        spec = load_spec(ckpt)
        net = Net(spec)
        fx["nets"].append({"task": spec["task"], "obs": spec["obs"], "layers": packed(spec)})
        env = FleetEnv(len(seeds), Config(weather=ROUGH, features=school(spec["obs"])))
        env.start(seeds)
        f = env.fleet
        flights = [
            {
                "seed": s,
                "net": len(fx["nets"]) - 1,
                "obs": spec["obs"],
                "states": [],
                "sticks": [],
                "wants": [],
                "looks": [],
            }  # fmt: skip
            for s in seeds
        ]
        t = 0
        while (f.outcome == FLYING).any() and t < 150:
            seen = env.frames.reshape(len(seeds), -1).clip(-5, 5)  # float64, before the cast
            want = net(seen)
            flying = f.outcome == FLYING
            env.step(want)
            for i, fl in enumerate(flights):
                if not flying[i]:
                    continue
                fl["wants"].append([r13(v) for v in want[i]])
                fl["sticks"].append([r13(f.stick0[i]), r13(f.stick1[i])])
                fl["states"].append(
                    [r13(getattr(f, k)[i]) for k in STATE_KEYS]
                    + [r13(v) for v in f.rock_x[i]]
                    + [int(v) for v in f.rock_n[i]]
                    + [int(f.m_state[i]), int(f.ticks[i]), int(f.outcome[i])]
                )
                if t % 25 == 0:
                    fl["looks"].append([t, [r13(v) for v in seen[i]]])
            t += 1
        fx["flights"] += flights
    fx["state_keys"] = list(STATE_KEYS)
    path = SITE / "test" / "dodge-parity.json"
    path.write_text(json.dumps(fx, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} KB)")


def write(name: str, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    path.write_text(json.dumps(obj, separators=(",", ":")))
    size = path.stat().st_size
    print(f"wrote {path.relative_to(SITE)} ({size / 1e3:.1f} KB)")
    assert size < 1_000_000


def hover_scores() -> dict:
    """The scripted pilot that flies to the point and holds it, ignoring every threat (escape.py),
    on each arena test: what holding still is worth."""
    from dodge_env import Fleet, Physics
    from escape import hover_pilot

    out = {}
    for test in ("arena", *ONLY):
        feat = replace(FULL, threats=TESTS[test], noise=0.0)
        f = Fleet(len(TEST_SEEDS), Physics(), feat)
        for i, s in enumerate(TEST_SEEDS):
            f.set_course(i, make_course(s, ROUGH, feat, f.ph))
        while (f.outcome == FLYING).any():
            f.step(hover_pilot(f))
        out[test] = int((f.outcome == TIMEOUT).sum())
    return out


def curves_of(spec: dict) -> dict:
    curves = {}
    for group, g in spec["curves"].items():
        t = g["test"]
        key = "landed" if t in ("mix", "none") else "held"
        curves[group] = {
            run: [
                {"steps": c["steps"], "seconds": c["seconds"], "landed": c[t][key], "n": c[t]["n"]}
                for c in curve(run, tests=(t,))
            ]
            for run in g["runs"]
        }
    return curves


def export_curves(spec: dict) -> None:
    """Only refreshes the curves in an exported eval.json."""
    path = DATA / "eval.json"
    ev = json.loads(path.read_text())
    ev["curves"] = curves_of(spec)
    write("eval.json", ev)


def export(spec: dict) -> None:
    """spec: {"models": {file: [ckpt, label]}, "curves": {group: {"runs": [...], "test": t}},
    "table": [runs], "speed": [runs]}. Curves count landings on courses and holds in arenas."""
    for fname, (ckpt, label) in spec["models"].items():
        write(fname, model_json(ckpt, label))
    curves = {}
    for group, g in spec["curves"].items():
        t = g["test"]
        key = "landed" if t in ("mix", "none") else "held"
        curves[group] = {
            run: [
                {"steps": c["steps"], "seconds": c["seconds"], "landed": c[t][key], "n": c[t]["n"]}
                for c in curve(run, tests=(t,))
            ]
            for run in g["runs"]
        }
    table = {run: report(load_spec(run)) for run in spec["table"]}
    speed = {}
    for run in spec["speed"]:
        log = json.loads((RUNS / run / "train_log.json").read_text())
        cfg = run_config(run)
        speed[run] = {k: cfg.get(k) for k in ("task", "envs", "epochs", "threads", "warm")}
        speed[run].update(seconds=log["seconds"], steps=log["marks"][-1][0])
    levels = {
        run: json.loads((RUNS / run / "train_log.json").read_text()).get("levels", [])
        for run in spec.get("levels", [])
    }
    write(
        "eval.json",
        {"seeds": 200, "curves": curves, "table": table, "speed": speed, "levels": levels,
         "hover": hover_scores()},
    )  # fmt: skip
