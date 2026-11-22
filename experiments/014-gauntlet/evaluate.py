"""Test gauntlet pilots (post 14).

uv run experiments/014-gauntlet/evaluate.py --table main-s1 main-s2   # every test, final pilots
uv run experiments/014-gauntlet/evaluate.py --curve main-s1           # the gauntlet per checkpoint
uv run experiments/014-gauntlet/evaluate.py --export export.json      # site data

Tests (200 fixed courses each, rough weather):
- gauntlet: the full test (60-120 m, every encounter kind, a frog on the pad, half the courses
  a frog mid-route);
- clear: no threats, no frogs (just the long flight);
- one encounter kind at every waypoint, no frogs: bird, two birds, missile, bird + missile,
  flock;
- frogs: calm waypoints, both frogs;
- blindfolds: the gauntlet with the pursuers or the frogs hidden from the pilot's inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gauntlet_env import (
    CLEAR,
    ENCOUNTERS,
    FLYING,
    FULL,
    OUTCOMES,
    ROUGH,
    Config,
    FleetEnv,
    Physics,
)

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SEEDS = [20_000 + i for i in range(200)]


def only(k: int):
    mix = [0.0] * len(ENCOUNTERS)
    mix[k] = 1.0
    return replace(FULL, mix=tuple(mix), route_frog=0.0, guard=0.0)


TESTS = {
    "gauntlet": (FULL, ()),
    "clear": (CLEAR, ()),
    **{ENCOUNTERS[k]: (only(k), ()) for k in range(1, len(ENCOUNTERS))},
    "frogs": (replace(CLEAR, route_frog=1.0, guard=1.0), ()),
    "blind threats": (FULL, ("threats",)),
    "blind frogs": (FULL, ("frogs",)),
}


class Net:
    """The actor as numpy (deterministic: the mean action)."""

    def __init__(self, model, f16: bool = False):
        lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
        lin.append(model.policy.action_net)
        self.layers = [(m.weight.detach().numpy().T, m.bias.detach().numpy()) for m in lin]
        if f16:  # rounded as the site's float16 copy is
            h = np.float16
            self.layers = [(w.astype(h).astype(float), b.astype(h).astype(float))
                           for w, b in self.layers]  # fmt: skip
        width = self.layers[0][0].shape[0]  # with closest-approach inputs or without
        dims = {
            p: FleetEnv(1, Config(features=replace(FULL, predict=p))).observation_space.shape[0]
            for p in (False, True)
        }
        self.predict = width == dims[True]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = obs.astype(np.float64)
        for k, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if k < len(self.layers) - 1:
                h = np.tanh(h)
        return np.clip(h, -1.0, 1.0)


def fly(pilot, feat, seeds=SEEDS, hide=(), weather=ROUGH, record=False):
    """Every course once; returns per-flight results (and, with record, each drone's track)."""
    feat = replace(feat, predict=getattr(pilot, "predict", False))
    env = FleetEnv(len(seeds), Config(Physics(), weather, feat), seed=7, hide=hide)
    obs = env.start(seeds)
    f = env.fleet
    tracks = [[] for _ in seeds] if record else None
    while (f.outcome == FLYING).any():
        act = pilot(obs)
        obs, _, _, _ = env.step(act)
        if record:
            for i in np.nonzero(f.outcome == FLYING)[0]:
                tracks[i].append((f.x[i], f.y[i]))
    frog_fail = f.frog_failed()
    out = []
    for i in range(len(seeds)):
        out.append(
            {
                "outcome": OUTCOMES[f.outcome[i]],
                "time": round(f.ticks[i] * f.ph.dt, 2),
                "length": round(f.courses[i]["length"], 1),
                "met": f.enc_count[i].tolist(),
                "hit_by": int(f.hit_enc[i]),
                "frog_fail": bool(frog_fail[i]),
                "frogs": int(f.frog_on[i].sum()),
            }
        )
    return (out, tracks) if record else out


def summary(res: list[dict]) -> dict:
    oc = {k: sum(r["outcome"] == k for r in res) for k in OUTCOMES[1:]}
    met = np.sum([r["met"] for r in res], 0)
    hits = np.bincount([r["hit_by"] for r in res if r["hit_by"] >= 0], minlength=len(ENCOUNTERS))
    enc = {
        ENCOUNTERS[k]: {"met": int(met[k]), "hit": int(hits[k])}
        for k in range(1, len(ENCOUNTERS))
        if met[k]
    }
    landed = [r["time"] for r in res if r["outcome"] == "landed"]
    return {
        "landed": oc["landed"],
        "outcomes": {k: v for k, v in oc.items() if v and k != "landed"},
        "encounters": enc,
        "frog_fail": sum(r["frog_fail"] for r in res),
        "time": round(float(np.mean(landed)), 1) if landed else None,
    }


def run_tests(pilot, names=tuple(TESTS)) -> dict:
    return {n: summary(fly(pilot, TESTS[n][0], hide=TESTS[n][1])) for n in names}


def checkpoints(run: str) -> list[Path]:
    return sorted((RUNS / run).glob("ckpt_*.zip"))


def load(path: Path):
    from stable_baselines3 import PPO

    return PPO.load(path, device="cpu")


def curve(run: str, test: str = "gauntlet") -> list[dict]:
    log = json.loads((RUNS / run / "train_log.json").read_text())
    secs = dict(log["marks"])
    pts = []
    for p in checkpoints(run):
        steps = int(p.stem.split("_")[1])
        s = summary(fly(Net(load(p)), TESTS[test][0], hide=TESTS[test][1]))
        pts.append({"steps": steps, "seconds": secs.get(steps), "landed": s["landed"], "n": 200})
        print(f"  {run} {steps / 1e6:7.1f}M: landed {s['landed']}", flush=True)
    return pts


def show(name: str, s: dict) -> None:
    enc = "  ".join(f"{k} {v['hit']}/{v['met']}" for k, v in s["encounters"].items())
    print(
        f"  {name:<14} landed {s['landed']:>3}  {s['outcomes']}  frog fails {s['frog_fail']}"
        f"  time {s['time']}  hit/met: {enc}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", nargs="+", default=[])
    ap.add_argument("--curve", nargs="+", default=[])
    ap.add_argument("--ckpt", default=None, help="a checkpoint stem instead of the last")
    ap.add_argument("--tests", nargs="+", default=list(TESTS))
    ap.add_argument("--export", default=None)
    ap.add_argument("--parity", default=None, help="run[/ckpt_stem]: the JS parity fixture only")
    args = ap.parse_args()
    for run in args.table:
        path = RUNS / run / f"{args.ckpt}.zip" if args.ckpt else checkpoints(run)[-1]
        print(f"{run} ({path.stem})")
        res = run_tests(Net(load(path)), args.tests)
        for n, s in res.items():
            show(n, s)
        (RUNS / run / "eval.json").write_text(json.dumps({"ckpt": path.stem, **res}))
    for run in args.curve:
        pts = curve(run)
        (RUNS / run / "curve.json").write_text(json.dumps(pts))
    if args.parity:
        from export import write_parity

        write_parity(args.parity)
    if args.export:
        from export import export

        export(HERE / args.export)


if __name__ == "__main__":
    main()
