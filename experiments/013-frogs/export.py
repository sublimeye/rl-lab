"""Site data and the JS parity fixture for post 13 (called from evaluate.py)."""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import numpy as np

from evaluate import DATA, RUNS, SITE, SWITCH, Net, condition, curve, load_spec, report, run_config
from frog_env import FLYING, ROUGH, Config, FleetEnv, make_course


def r13(v) -> float:
    return float(f"{v:.13g}")


def b64(a) -> str:
    """float16, little-endian, base64."""
    return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode()


def packed(spec: dict) -> dict:
    out = {
        "layers": [
            {"shape": list(L["w"].shape), "w": b64(L["w"]), "b": b64(L["b"])}
            for L in spec["layers"]
        ]
    }
    if spec.get("aux"):
        a = spec["aux"]
        out["aux"] = {"shape": list(a["w"].shape), "w": b64(a["w"]), "b": b64(a["b"])}
    return out


def model_json(ckpt: str, label: str) -> dict:
    spec = load_spec(ckpt)
    cfg = condition(spec["task"]).features
    return {
        "label": label,
        "task": spec["task"],
        "memory": cfg.memory,
        "tell": cfg.tell,
        "act": "tanh",
        **packed(spec),
    }


STATE_KEYS = "x y vx vy th om motor charge".split()


def export_parity(ckpts: list[str]) -> None:
    """Courses from many seeds (with and without a rhythm switch), and every network flying
    three frog courses with sensor noise off: the drone's and frogs' state after every update
    (up to 200), what the pilot saw every 25th update, and the network's choice (and its
    countdown, if it has one)."""
    fx = {"courses": [], "nets": [], "flights": [], "state_keys": STATE_KEYS}
    ph = Config().physics
    for task in ("hist", "guard"):
        for sw in (None, list(SWITCH)):
            feat = replace(condition(task).features, noise=0.0, switch=sw)
            for s in [0, 3, 7, 42, 1_000_000_005, 3_000_000_000, 12345, 777]:
                c = make_course(s, ROUGH, feat, ph)
                fx["courses"].append({"seed": s, "task": task, "switch": sw, **c})
    seeds = [1_000_000_002, 1_000_000_009, 1_000_000_014]
    for ckpt in ckpts:
        spec = load_spec(ckpt)
        net = Net(spec)
        task = spec["task"]
        fx["nets"].append({"task": task, **packed(spec)})
        feat = replace(condition(task, SWITCH).features, noise=0.0)
        env = FleetEnv(len(seeds), Config(weather=ROUGH, features=feat))
        obs = env.start(seeds)
        f = env.fleet
        flights = [
            {
                "seed": s,
                "net": len(fx["nets"]) - 1,
                "task": task,
                "switch": list(SWITCH),
                "states": [],
                "sticks": [],
                "wants": [],
                "counts": [],
                "looks": [],
            }  # fmt: skip
            for s in seeds
        ]
        t = 0
        while (f.outcome == FLYING).any() and t < 200:
            seen = np.concatenate(
                [env.frames.reshape(len(seeds), -1)]
                + ([env.hist[:, ::4].reshape(len(seeds), -1)] if env.memory else []),
                1,
            ).clip(-5, 5)  # float64, before the cast; the labels left off
            want = net(seen)
            count = net.countdown() if net.aux is not None else None
            flying = f.outcome == FLYING
            obs, _, _, _ = env.step(want)
            for i, fl in enumerate(flights):
                if not flying[i]:
                    continue
                fl["wants"].append([r13(v) for v in want[i]])
                if count is not None:
                    fl["counts"].append([r13(v) for v in count[i]])
                fl["sticks"].append([r13(f.stick0[i]), r13(f.stick1[i])])
                fx_, fy_, _, _ = f.frog_pos(np.array([i]))
                fl["states"].append(
                    [r13(getattr(f, k)[i]) for k in STATE_KEYS]
                    + [r13(v) for v in fx_[0]]
                    + [r13(v) for v in fy_[0]]
                    + [int(v) for v in f.frog_n[i]]
                    + [int(f.ticks[i]), int(f.outcome[i])]
                )
                if t % 25 == 0:
                    fl["looks"].append([t, [r13(v) for v in seen[i]]])
            t += 1
        fx["flights"] += flights
    # no network: the drone sits on its pad for 10 s, past the rhythm switch, frogs crouching
    feat = replace(condition("tell", SWITCH).features, noise=0.0, frogs=(2, 2))
    sit = {"seeds": seeds, "frames": []}
    f = FleetEnv(len(seeds), Config(weather=ROUGH, features=feat))
    f.start(seeds)
    g = f.fleet
    for _ in range(200):
        f.step(np.tile([-1.0, 0.0], (len(seeds), 1)))
        x, y, _, _ = g.frog_pos()
        sit["frames"].append(
            [
                [r13(v) for v in (*x[i], *y[i])]
                + g.frog_n[i].tolist()
                + g.crouching()[i].astype(int).tolist()
                + [int(g.outcome[i])]
                for i in range(len(seeds))
            ]  # fmt: skip
        )
    fx["sit"] = sit
    path = SITE / "test" / "frog-parity.json"
    path.write_text(json.dumps(fx, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} KB)")


def write(name: str, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    path.write_text(json.dumps(obj, separators=(",", ":")))
    size = path.stat().st_size
    print(f"wrote {path.relative_to(SITE)} ({size / 1e3:.1f} KB)")
    assert size < 1_000_000


def export(spec: dict) -> None:
    """spec: {"models": {file: [ckpt, label]}, "groups": {name: [runs]}, "table": [runs]}"""
    for fname, (ckpt, label) in spec["models"].items():
        write(fname, model_json(ckpt, label))
    curves = {g: {r: curve(r) for r in runs} for g, runs in spec["groups"].items()}
    table = {}
    for run in spec["table"]:
        log = json.loads((RUNS / run / "train_log.json").read_text())
        secs = log.get("seconds") or log["marks"][-1][1]
        steps = log["marks"][-1][0]
        table[run] = {**report(run), "seconds": secs, "steps": steps, "config": run_config(run)}
    levels = {
        run: json.loads((RUNS / run / "train_log.json").read_text()).get("levels", [])
        for run in spec["table"]
    }
    write(
        "eval.json",
        {"seeds": 200, "switch": list(SWITCH), "curves": curves, "table": table, "levels": levels,
         "phase_check": phase_check()},
    )  # fmt: skip


def phase_check() -> dict:
    """A scripted pilot that flies straight at the pad, past one frog (the first design: a 2-3 s
    beat, frog in the gap): how often the frog gets it, by how long before the frog's next strike
    it first came within reach. Shows when in the rhythm it is safe to go."""
    from frog_env import FROG, FROG_HIT, FROG_REACH, Fleet, Physics

    feat = replace(condition("none").features, frogs=(1, 1), noise=0.0)
    seeds = [1_000_000_000 + i for i in range(1500)]
    f = Fleet(len(seeds), Physics(), feat)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(s, ROUGH, feat, f.ph))
    entry = np.full(len(seeds), np.nan)
    while (f.outcome == FLYING).any():
        gx, gy = f.goal()
        tilt = np.clip(0.5 * np.clip(gx - f.x, -3, 3) - 0.12 * f.vx, -1, 1)
        ty = np.where(np.abs(gx - f.x) > 1.0, np.maximum(gy, f.s_top + 1.5), gy)
        thr = np.clip(0.05 + 0.5 * (ty - f.y) - 0.6 * f.vy, -1, 1)
        f.step(np.stack([thr, tilt], 1))
        near = np.abs(f.x - f.frog_p[:, 0, 0]) < FROG_REACH + FROG_HIT
        new = np.isnan(entry) & near & (f.outcome == FLYING)
        entry[new] = f.next_strike()[new, 0]
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
    rows = []
    for a, b in bins:
        m = (entry >= a) & (entry < b)
        rows.append([int((f.outcome[m] == FROG).sum()), int(m.sum())])
    return {"bins": bins, "rows": rows}


def add_phase_check() -> None:
    path = DATA / "eval.json"
    ev = json.loads(path.read_text())
    ev["phase_check"] = phase_check()
    write("eval.json", ev)


def add_table(runs: list[str]) -> None:
    """Adds (or refreshes) runs in an exported eval.json's table and levels."""
    path = DATA / "eval.json"
    ev = json.loads(path.read_text())
    for run in runs:
        log = json.loads((RUNS / run / "train_log.json").read_text())
        secs = log.get("seconds") or log["marks"][-1][1]
        steps = log["marks"][-1][0]
        ev["table"][run] = {
            **report(run),
            "seconds": secs,
            "steps": steps,
            "config": run_config(run),
        }
        ev["levels"][run] = log.get("levels", [])
    write("eval.json", ev)
