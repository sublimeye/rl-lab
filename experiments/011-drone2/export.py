"""Site data and the JS parity fixture for post 11 (called from evaluate.py)."""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import numpy as np

from evaluate import (
    DATA,
    RUNS,
    SITE,
    TASKS,
    TEST_SEEDS,
    Net,
    bird_warnings,
    blindfold,
    condition,
    curve,
    fly,
    last_ckpt,
    load_spec,
    run_config,
    summary,
)
from fleet_env import FLYING, ROUGH, Config, FleetEnv


def r13(v) -> float:
    return float(f"{v:.13g}")


def model_json(ckpt: str, label: str) -> dict:
    spec = load_spec(ckpt)
    feat = TASKS[spec["task"]]
    return {
        "label": label,
        "task": spec["task"],
        "features": {"delay": feat.delay, "stack": feat.stack, "obs": feat.obs},
        "act": "tanh",
        "layers": [
            {"shape": list(L["w"].shape), "w": b64(L["w"]), "b": b64(L["b"])}
            for L in spec["layers"]
        ],
    }


def b64(a) -> str:
    """float16, little-endian, base64."""
    return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode()


def export_parity(ckpts: list[str]) -> None:
    """Every network flies three rough-weather courses (birds, moving pads, low batteries
    among them) with sensor noise off; the fixture keeps the drone's state after every update
    and what the pilot saw every 10th update, plus the network's own choice."""
    fx = {"courses": [], "nets": [], "flights": []}
    for task in ("state", "vision"):
        feat = replace(TASKS[task], noise=0.0)
        for s in [0, 3, 7, 42, 1_000_000_005, 3_000_000_000]:
            from fleet_env import make_course

            c = make_course(s, ROUGH, feat, Config().physics)
            fx["courses"].append({"seed": s, "task": task, **c})
    seeds = [1_000_000_003, 1_000_000_011, 1_000_000_017]
    for ckpt in ckpts:
        spec = load_spec(ckpt)
        net = Net(spec)
        fx["nets"].append(
            {
                "task": spec["task"],
                "layers": [
                    {"shape": list(L["w"].shape), "w": b64(L["w"]), "b": b64(L["b"])}
                    for L in spec["layers"]
                ],
            }
        )
        feat = replace(TASKS[spec["task"]], noise=0.0)
        env = FleetEnv(len(seeds), Config(weather=ROUGH, features=feat))
        env.start(seeds)
        f = env.fleet
        flights = [
            {
                "seed": s,
                "net": len(fx["nets"]) - 1,
                "task": spec["task"],
                "states": [],
                "sticks": [],
                "wants": [],
                "looks": [],
            }
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
                    [
                        r13(v)
                        for v in (
                            f.x[i],
                            f.y[i],
                            f.vx[i],
                            f.vy[i],
                            f.th[i],
                            f.om[i],
                            f.motor[i],
                            f.charge[i],
                            f.pad_x0[i],
                        )
                    ]
                    + [int(f.ticks[i]), int(f.outcome[i])]
                )
                if t % 25 == 0:
                    fl["looks"].append([t, [r13(v) for v in seen[i]]])
            t += 1
        fx["flights"] += flights
    path = SITE / "test" / "drone2-parity.json"
    path.write_text(json.dumps(fx, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} KB)")


def write(name: str, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    path.write_text(json.dumps(obj, separators=(",", ":")))
    size = path.stat().st_size
    print(f"wrote {path.relative_to(SITE)} ({size / 1e3:.1f} KB)")
    assert size < 1_000_000


def traces(seed: int, pilots: dict[str, str]) -> dict:
    """Every stick update of each pilot on one course (rough weather, its own task)."""
    out = []
    for label, ckpt in pilots.items():
        spec = load_spec(ckpt)
        res = fly(Net(spec), condition(spec["task"], "rough"), seeds=[seed], record=True)
        n = int(res["alive"][:, 0].sum())
        out.append(
            {
                "label": label,
                "outcome": res["outcome"][0],
                "time": round(res["time"][0], 2),
                "change": round(res["change"][0], 4),
                "pinned": round(res["pinned"][0], 4),
                "sticks": [[round(a, 3), round(b, 3)] for a, b in res["sticks"][:n, 0]],
            }
        )
    c = res["courses"][0]
    facing = 1 if c["target"][0] > c["start"][0] else -1
    return {"seed": seed, "dt": 0.05, "facing": facing, "pilots": out}


def export(spec: dict) -> None:
    """spec: {"models": {file: [ckpt, label]}, "groups": {name: [runs]}, "table": {label: ckpt},
    "speed": [runs], "traces": {"seed": n, "pilots": {label: ckpt}}}"""
    for fname, (ckpt, label) in spec["models"].items():
        write(fname, model_json(ckpt, label))
    curves = {g: {r: curve(r) for r in runs} for g, runs in spec["groups"].items()}
    table = {}
    for label, ckpt in spec["table"].items():
        s = load_spec(ckpt)
        table[label] = {
            "task": s["task"],
            **{w: summary(fly(Net(s), condition(s["task"], w))) for w in ("calm", "rough")},
        }
    speed = {}
    for run in spec["speed"]:
        log = json.loads((RUNS / run / "train_log.json").read_text())
        cfg = run_config(run)
        speed[run] = {
            "sim": cfg["sim"],
            "envs": cfg["envs"],
            "threads": cfg["threads"],
            "epochs": cfg.get("epochs", 10),
            "batch": cfg.get("batch"),
            "seconds": log["seconds"],
            "steps": log["marks"][-1][0],
        }
    bench = json.loads((RUNS / "bench.json").read_text())
    caps = {
        r: json.loads((RUNS / r / "train_log.json").read_text()).get("caps", [])[-1:]
        for r in spec["groups"].get("caps", [])
    }
    blind = {}
    for run in spec.get("blindfold", []):
        s = load_spec(run)
        net = Net(s)
        blind[run] = {
            "seeing": summary(fly(net, condition("state", "rough"))),
            "blindfolded": summary(fly(blindfold(net), condition("state", "rough"))),
            "warnings": bird_warnings(net),
        }
    write(
        "eval.json",
        {
            "seeds": len(TEST_SEEDS),
            "curves": curves,
            "table": table,
            "speed": speed,
            "bench": bench,
            "caps": caps,
            "blindfold": blind,
        },
    )
    write("traces.json", traces(spec["traces"]["seed"], spec["traces"]["pilots"]))


__all__ = ["export", "export_parity", "last_ckpt"]
_ = np
