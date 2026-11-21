"""Scores frog pilots on held-out courses (all flown at once) and exports the site data.

uv run experiments/013-frogs/evaluate.py --curves none-s1 hist-s1          # per checkpoint
uv run experiments/013-frogs/evaluate.py --table none-s1 hist-s1 aux-s1    # final pilots
uv run experiments/013-frogs/evaluate.py --export export.json  # site/public/data/013-frogs
uv run experiments/013-frogs/evaluate.py --parity hist-s1 aux-s1           # site/test fixture

Tests (200 courses each, rough weather): "frogs" (the pilot's own task), "switch" (the same
courses, but right after its first strike the frog's beat speeds up to 0.8 of its period: a
pilot that predicts the old beat expects a window that is no longer there; one that keeps
watching can still find the shorter gap). Weights are rounded to float16, as on the site.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from frog_env import FLYING, FROG, FROG_HIT, FROG_REACH, K_FROGS, OUTCOMES, ROUGH, Config, FleetEnv

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SITE = HERE.parent.parent / "site"
DATA = SITE / "public" / "data" / "013-frogs"
TEST_SEEDS = [1_000_000_000 + i for i in range(200)]
SWITCH = (0, 0.8)


def rounded(a):
    return np.asarray(a, dtype=np.float16).astype(np.float64)


class Net:
    """The exported actor (tanh MLP, output clipped to the stick range), batched; it reads the
    first columns of the observation it was trained on (the labels at the end are dropped).
    hidden() gives the last hidden layer, countdown() the head's guess (s to each strike)."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.layers = [(np.asarray(L["w"]), np.asarray(L["b"])) for L in spec["layers"]]
        self.aux = (
            (np.asarray(spec["aux"]["w"]), np.asarray(spec["aux"]["b"]))
            if spec.get("aux")
            else None
        )
        self.width = self.layers[0][0].shape[1]
        self.last = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = np.clip(np.asarray(obs, float)[:, : self.width], -5.0, 5.0)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
                self.last = h
        return np.clip(h, -1.0, 1.0)

    def countdown(self) -> np.ndarray:
        w, b = self.aux
        return 3.0 * (self.last @ w.T + b)


class LstmPilot:
    """A RecurrentPPO checkpoint flying a batch (its memory reset at each course start)."""

    def __init__(self, ckpt: str):
        from sb3_contrib import RecurrentPPO

        self.model = RecurrentPPO.load(RUNS / f"{ckpt}.zip", device="cpu")
        self.state = None
        self.start = None
        self.last = None

    def __call__(self, obs):
        if self.start is None:
            self.start = np.ones(len(obs), bool)
        act, self.state = self.model.predict(
            obs, state=self.state, episode_start=self.start, deterministic=True
        )
        self.start = np.zeros(len(obs), bool)
        self.last = self.state[0][0]  # the LSTM's hidden state (memory)
        return np.clip(act, -1, 1)


def run_config(run: str) -> dict:
    return json.loads((RUNS / run / "config.json").read_text())


def last_ckpt(run: str) -> str:
    return f"{run}/{sorted((RUNS / run).glob('ckpt_*.zip'))[-1].stem}"


def load_spec(ckpt: str) -> dict:
    from train import actor, load_model

    run = ckpt.split("/")[0]
    if "/" not in ckpt:
        ckpt = last_ckpt(run)
    spec = actor(load_model(ckpt))
    spec["layers"] = [{"w": rounded(L["w"]), "b": rounded(L["b"])} for L in spec["layers"]]
    if spec.get("aux") and run_config(run)["aux"]:
        spec["aux"] = {"w": rounded(spec["aux"]["w"]), "b": rounded(spec["aux"]["b"])}
    else:
        spec.pop("aux", None)
    spec["task"] = run_config(run)["task"]
    return spec


def pilot_for(run_or_ckpt: str):
    run = run_or_ckpt.split("/")[0]
    if run_config(run).get("lstm"):
        ck = run_or_ckpt if "/" in run_or_ckpt else last_ckpt(run)
        return LstmPilot(ck), run_config(run)["task"]
    spec = load_spec(run_or_ckpt)
    return Net(spec), spec["task"]


def condition(task: str, switch=None) -> Config:
    from train import TASKS

    return Config(weather=ROUGH, features=replace(TASKS[task], switch=switch))


def fly(pilot, cfg: Config, seeds=TEST_SEEDS, probe=False, noise_seed=0):
    """Flies every course at once. Records, per course: outcome, time, when it first came
    within reach of each frog (and the time to that frog's next strike then), how long it
    hovered (speed under 0.5 m/s off the pads), and with probe=True the pilot's last hidden
    layer and the true time to each strike at every update."""
    env = FleetEnv(len(seeds), cfg, seed=noise_seed)
    obs = env.start(seeds)
    f = env.fleet
    n = len(seeds)
    entry = np.full((n, K_FROGS), np.nan)
    hover = np.zeros(n)
    feats, labels, guess = [], [], []
    while (f.outcome == FLYING).any():
        flying = f.outcome == FLYING
        act = pilot(obs)
        if probe:
            feats.append(pilot.last[flying].copy())
            labels.append(f.next_strike()[flying].copy())
            if getattr(pilot, "aux", None) is not None:
                guess.append(pilot.countdown()[flying].copy())
        obs, _, _, _ = env.step(act)
        near = np.abs(f.x[:, None] - f.frog_p[..., 0]) < FROG_REACH + FROG_HIT
        new = np.isnan(entry) & near & f.frog_on & flying[:, None]
        entry[new] = f.next_strike()[new]
        slow = np.hypot(f.vx, f.vy) < 0.5
        hover += (flying & slow & (f.grounded == 0)) * f.ph.dt * f.ph.substeps
    out = {
        "outcome": [OUTCOMES[o] for o in f.outcome],
        "time": (f.ticks * f.ph.dt).tolist(),
        "entry": entry,
        "hover": hover,
        "frogs": f.frog_on.sum(1),
        "courses": [f.courses[i] for i in range(n)],
    }
    if probe:
        out["feats"] = np.concatenate(feats)
        out["labels"] = np.concatenate(labels)
        out["guess"] = np.concatenate(guess) if guess else None
    return out


def summary(res: dict) -> dict:
    oc = res["outcome"]
    landed = [i for i, o in enumerate(oc) if o == "landed"]
    e = res["entry"][~np.isnan(res["entry"])]
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
    return {
        **{k: oc.count(k) for k in OUTCOMES[1:]},
        "n": len(oc),
        "time": round(float(np.mean([res["time"][i] for i in landed])), 2) if landed else None,
        "hover": round(float(np.mean(res["hover"])), 2),
        "by_frogs": [
            [
                sum(oc[i] == "landed" for i in range(len(oc)) if res["frogs"][i] == k),
                int((res["frogs"] == k).sum()),
            ]
            for k in (1, 2)
        ],  # fmt: skip
        "entry_hist": np.histogram(np.clip(e, 0, 3.49), bins)[0].tolist(),
        "entry_bins": bins,
    }


def probe(res: dict) -> dict:
    """How well a straight line through the pilot's last hidden layer (or its memory, for the
    LSTM) reads off the time to each frog's next strike: fit on even updates, scored on odd
    ones; mean error in seconds, against always guessing the average. With a countdown head,
    the head's own error too."""
    x, y = res["feats"], np.clip(res["labels"], 0, 3)
    have = y[:, 0] < 3  # frog 0 exists in every course
    x, y = x[have], y[have, 0]
    x1 = np.concatenate([x, np.ones((len(x), 1))], 1)
    tr, te = slice(0, None, 2), slice(1, None, 2)
    w = np.linalg.solve(x1[tr].T @ x1[tr] + 1e-3 * np.eye(x1.shape[1]), x1[tr].T @ y[tr])
    err = float(np.mean(np.abs(x1[te] @ w - y[te])))
    base = float(np.mean(np.abs(y[te] - y[tr].mean())))
    out = {"error": round(err, 3), "guess_mean": round(base, 3)}
    if res.get("guess") is not None:
        g = np.clip(res["guess"][have, 0], 0, 3)
        out["head_error"] = round(float(np.mean(np.abs(g - y))), 3)
    return out


def report(run: str) -> dict:
    pilot, task = pilot_for(run)
    out = {"task": task}
    res = fly(pilot, condition(task), probe=True)
    out["frogs"] = summary(res)
    out["probe"] = probe(res)
    pilot, _ = pilot_for(run)
    out["switch"] = summary(fly(pilot, condition(task, SWITCH)))
    return out


def curve(run: str) -> list[dict]:
    cache = RUNS / run / "eval.json"
    marks = dict(json.loads((RUNS / run / "train_log.json").read_text())["marks"])
    have = json.loads(cache.read_text()) if cache.exists() else {}
    cks = sorted((RUNS / run).glob("ckpt_*.zip"))
    steps_of = [int(c.stem.split("_")[1]) for c in cks]
    if len(cks) > 1 and steps_of[-1] - steps_of[-2] < 0.01 * steps_of[-1]:
        cks = cks[:-2] + cks[-1:]
    out = []
    for ck in cks:
        steps = int(ck.stem.split("_")[1])
        if str(steps) not in have:
            pilot, task = pilot_for(f"{run}/{ck.stem}")
            have[str(steps)] = summary(fly(pilot, condition(task)))
            cache.write_text(json.dumps(have))
        out.append({"steps": steps, "seconds": marks.get(steps), **have[str(steps)]})
    return out


def print_report(run: str, r: dict) -> None:
    for test in ("frogs", "switch"):
        s = r[test]
        print(
            f"  {run:<10} {test:<7} landed {s['landed']:3d}/200 frog {s['frog']:3d}"
            f" crashed {s['crashed']} timeout {s['timeout']} lost {s['lost']} water {s['water']}"
            f"  time {s['time']} hover {s['hover']}  by frogs {s['by_frogs']}"
            f"  entry {s['entry_hist']}"
        )
    print(f"  {'':<10} probe {r['probe']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", nargs="*", default=[])
    ap.add_argument("--table", nargs="*", default=None)
    ap.add_argument("--parity", nargs="*", default=None)
    ap.add_argument("--export", default=None)
    args = ap.parse_args()
    for run in args.curves:
        print(run)
        for c in curve(run):
            secs = c["seconds"] or 0
            print(
                f"  {c['steps']:>11,} {secs / 60:5.1f} min  landed {c['landed']:3d}"
                f" frog {c['frog']:3d}  time {c['time']}"
            )
    if args.table is not None:
        for run in args.table:
            print_report(run, report(run))
    if args.parity is not None:
        from export import export_parity

        export_parity(args.parity)
    if args.export:
        from export import export

        export(json.loads(Path(args.export).read_text()))


if __name__ == "__main__":
    main()


_ = FROG
