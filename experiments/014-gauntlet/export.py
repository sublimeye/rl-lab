"""Site data and the JS parity fixture for post 14 (called from evaluate.py --export)."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate import RUNS, TESTS, Net, checkpoints, load
from gauntlet_env import (
    CLEAR,
    FLYING,
    FULL,
    P_KIND,
    ROUGH,
    Config,
    FleetEnv,
    Physics,
    make_course,
)

HERE = Path(__file__).parent
SITE = HERE.parent.parent / "site"
DATA = SITE / "public" / "data" / "014-gauntlet"


def r13(v) -> float:
    return float(f"{v:.13g}")


def b64(a) -> str:
    """float16, little-endian, base64."""
    return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode()


def ckpt_path(ckpt: str) -> Path:
    run, _, stem = ckpt.partition("/")
    return RUNS / run / f"{stem}.zip" if stem else checkpoints(run)[-1]


def packed(net: Net) -> dict:
    return {
        "layers": [{"shape": list(w.T.shape), "w": b64(w.T), "b": b64(b)} for w, b in net.layers]
    }


def model_json(ckpt: str, label: str) -> dict:
    net = Net(load(ckpt_path(ckpt)))
    return {"label": label, "act": "tanh", "predict": net.predict, **packed(net)}


STATE_KEYS = "x y vx vy th om motor charge".split()


def _seen(env, n):
    return np.concatenate([env.frames.reshape(n, -1), env.hist[:, ::4].reshape(n, -1)], 1).clip(
        -5, 5
    )


def parity_seeds(net, feat, candidates=range(20_000, 20_040), cap=600) -> list[int]:
    """Three test courses this pilot finishes within `cap` updates (so the fixture holds whole
    flights), between them meeting a bird and the missile."""
    seeds = list(candidates)
    env = FleetEnv(len(seeds), Config(weather=ROUGH, features=feat))
    env.start(seeds)
    f = env.fleet
    kinds = np.zeros((len(seeds), 2), bool)
    for _ in range(cap):
        if not (f.outcome == FLYING).any():
            break
        flying = f.outcome == FLYING
        env.step(net(_seen(env, len(seeds))))
        for k in (0, 1):
            kinds[flying, k] |= (f.p_on & (P_KIND == k))[flying].any(1)
    done = [i for i in range(len(seeds)) if f.outcome[i] != FLYING]
    both = [i for i in done if kinds[i].all()]
    pick = (both + [i for i in done if i not in both])[:3]
    assert len(pick) == 3 and kinds[pick].any(0).all(), "no three short flights meet both"
    return [seeds[i] for i in pick]


def export_parity(ckpt: str) -> dict:
    """Courses from many seeds, and the network flying three gauntlet courses with sensor
    noise off (ones it finishes within 600 updates): the drone, every pursuer and frog after
    every update, what the pilot saw every 25th update, and the network's choice."""
    ph = Physics()
    fx = {"courses": [], "flights": [], "state_keys": STATE_KEYS}
    for name, feat in (
        ("gauntlet", FULL),
        ("clear", CLEAR),
        ("bird + missile", TESTS["bird + missile"][0]),
    ):
        for s in [0, 3, 7, 42, 1_000_000_005, 3_000_000_000, 12345, 777]:
            fx["courses"].append({"seed": s, "test": name, **make_course(s, ROUGH, feat, ph)})
    net = Net(load(ckpt_path(ckpt)), f16=True)  # the weights as the site gets them
    fx["net"] = {"predict": net.predict, **packed(net)}
    fx["predict"] = net.predict
    feat = replace(FULL, noise=0.0, predict=net.predict)
    seeds = parity_seeds(net, feat)
    env = FleetEnv(len(seeds), Config(weather=ROUGH, features=feat))
    obs = env.start(seeds)
    f = env.fleet
    flights = [{"seed": s, "states": [], "wants": [], "looks": []} for s in seeds]
    t = 0
    while (f.outcome == FLYING).any() and t < 600:
        seen = _seen(env, len(seeds))
        want = net(seen)
        flying = f.outcome == FLYING
        obs, _, _, _ = env.step(want)
        fxs, fys, _, _ = f.frog_pos()
        for i, fl in enumerate(flights):
            if not flying[i]:
                continue
            fl["wants"].append([r13(v) for v in want[i]])
            fl["states"].append(
                [r13(getattr(f, k)[i]) for k in STATE_KEYS]
                + [r13(v) for v in f.p_x[i]]
                + [r13(v) for v in f.p_y[i]]
                + [int(v) for v in f.p_on[i]]
                + [r13(v) for v in fxs[i]]
                + [r13(v) for v in fys[i]]
                + [int(f.enc_next[i]), int(f.ticks[i]), int(f.outcome[i])]
            )
            if t % 25 == 0:
                fl["looks"].append([t, [r13(v) for v in seen[i]]])
        t += 1
    fx["flights"] = flights
    return fx


def write_parity(ckpt: str) -> None:
    path = SITE / "test" / "gauntlet-parity.json"
    path.write_text(json.dumps(export_parity(ckpt)))
    print(f"{path.name}: {path.stat().st_size / 1e3:.0f} KB")


def export(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (ckpt, label) in spec["models"].items():
        (DATA / f"model-{name}.json").write_text(json.dumps(model_json(ckpt, label)))
    ev = {"table": {}, "curves": {}, "levels": {}, "seconds": {}}
    for run in spec["table"]:
        ev["table"][run] = json.loads((RUNS / run / "eval.json").read_text())
    for group, runs in spec["curves"].items():  # {group: [runs]}: DroneChart's layout
        ev["curves"][group] = {r: json.loads((RUNS / r / "curve.json").read_text()) for r in runs}
    for run in spec["levels"]:
        log = json.loads((RUNS / run / "train_log.json").read_text())
        ev["levels"][run] = log["levels"]
        secs = log.get("seconds") or log["marks"][-1][1]
        ev["seconds"][run] = [secs, log["marks"][-1][0]]
    ev["diagnostics"] = json.loads((RUNS / "diagnostics.json").read_text())
    (DATA / "eval.json").write_text(json.dumps(ev))
    write_parity(spec["parity"])
    for p in sorted(DATA.glob("*.json")):
        print(f"{p.name}: {p.stat().st_size / 1e3:.0f} KB")
