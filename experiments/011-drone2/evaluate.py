"""Scores pilots on 200 held-out courses (all flown at once) and exports the site data.

uv run experiments/011-drone2/evaluate.py --curves basic-s1 fleet-s1        # per checkpoint
uv run experiments/011-drone2/evaluate.py --table state=state-s1 vision=vision-s1
uv run experiments/011-drone2/evaluate.py --export export.json   # site/public/data/011-drone2
uv run experiments/011-drone2/evaluate.py --parity state-s1 vision-s1      # site/test fixture

Weights are rounded to float16, as on the site.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from fleet_env import (
    CALM,
    FLYING,
    FULL,
    FULL_VISION,
    OUTCOMES,
    ROUGH,
    V010,
    Config,
    FleetEnv,
)

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SITE = HERE.parent.parent / "site"
DATA = SITE / "public" / "data" / "011-drone2"
TEST_SEEDS = [1_000_000_000 + i for i in range(200)]
TASKS = {
    "v010": V010,
    "state": FULL,
    "vision": FULL_VISION,
    "vision1": replace(FULL_VISION, stack=1),  # no memory
}


def rounded(a):
    """float16, the precision the site gets."""
    return np.asarray(a, dtype=np.float16).astype(np.float64)


class Net:
    """The exported actor (tanh MLP, output clipped to the stick range), batched."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.layers = [(np.asarray(L["w"]), np.asarray(L["b"])) for L in spec["layers"]]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = np.clip(np.asarray(obs, float), -5.0, 5.0)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return np.clip(h, -1.0, 1.0)


def run_config(run: str) -> dict:
    return json.loads((RUNS / run / "config.json").read_text())


def load_spec(ckpt: str) -> dict:
    from train import actor, load_model

    run = ckpt.split("/")[0]
    if "/" not in ckpt:
        ckpt = last_ckpt(run)
    spec = actor(load_model(ckpt))
    spec["layers"] = [{"w": rounded(L["w"]), "b": rounded(L["b"])} for L in spec["layers"]]
    cfg = run_config(run)
    spec["task"] = cfg["task"]
    return spec


def last_ckpt(run: str) -> str:
    return f"{run}/{sorted((RUNS / run).glob('ckpt_*.zip'))[-1].stem}"


def condition(task: str, weather: str) -> Config:
    return Config(weather=CALM if weather == "calm" else ROUGH, features=TASKS[task])


def fly(pilot, cfg: Config, seeds=TEST_SEEDS, noise_seed: int = 0, record: bool = False):
    """Flies every course at once. pilot(obs) -> sticks. Returns per-course results."""
    env = FleetEnv(len(seeds), cfg, seed=noise_seed)
    obs = env.start(seeds)
    f = env.fleet
    sticks, alive = [], []
    while (f.outcome == FLYING).any():
        flying = f.outcome == FLYING
        obs, _, _, _ = env.step(pilot(obs))
        sticks.append(np.stack([f.stick0, f.stick1], 1).copy())
        alive.append(flying)
    s = np.array(sticks)  # (T, N, 2)
    a = np.array(alive)  # (T, N): stick reading counts while the drone was flying
    diff = np.abs(np.diff(s, axis=0)).sum(2)
    both = a[1:] & a[:-1]
    change = (diff * both).sum(0) / np.maximum(both.sum(0), 1)
    pinned = ((np.abs(s) > 0.95).mean(2) * a).sum(0) / np.maximum(a.sum(0), 1)
    out = {
        "outcome": [OUTCOMES[o] for o in f.outcome],
        "time": (f.ticks * f.ph.dt).tolist(),
        "touch": np.abs(f.touch[:, 1]).tolist(),
        "crash": f.crash.tolist(),
        "change": change.tolist(),
        "pinned": pinned.tolist(),
        "courses": [env.fleet.courses[i] for i in range(len(seeds))],
    }
    if record:
        out["sticks"] = s
        out["alive"] = a
    return out


def summary(res: dict) -> dict:
    oc = res["outcome"]
    n = len(oc)
    landed = [i for i in range(n) if oc[i] == "landed"]
    counts = {k: sum(o == k for o in oc) for k in OUTCOMES[1:]}

    def mean(key, idx):
        return round(float(np.mean([res[key][i] for i in idx])), 3) if idx else None

    birds = [len(c["birds"]) for c in res["courses"]]
    moving = [c["amp"] > 0 for c in res["courses"]]
    low = [c["charge"] < 0.15 for c in res["courses"]]

    def rate(mask):
        idx = [i for i in range(n) if mask[i]]
        return [sum(oc[i] == "landed" for i in idx), len(idx)]

    return {
        **counts,
        "n": n,
        "time": mean("time", landed),
        "touch": mean("touch", landed),
        "change": mean("change", list(range(n))),
        "pinned": mean("pinned", list(range(n))),
        "by_birds": [rate([b == k for b in birds]) for k in range(4)],
        "pad_moving": rate(moving),
        "pad_still": rate([not m for m in moving]),
        "low_battery": rate(low),
    }


def score(spec_or_pilot, task: str, weather: str) -> dict:
    pilot = Net(spec_or_pilot) if isinstance(spec_or_pilot, dict) else spec_or_pilot
    return summary(fly(pilot, condition(task, weather)))


def curve(run: str) -> list[dict]:
    """Test scores at every checkpoint (in the run's own task and weather), cached."""
    cfg = run_config(run)
    cache = RUNS / run / "eval.json"
    marks = dict(json.loads((RUNS / run / "train_log.json").read_text())["marks"])
    have = json.loads(cache.read_text()) if cache.exists() else {}
    out = []
    cks = sorted((RUNS / run).glob("ckpt_*.zip"))
    steps_of = [int(c.stem.split("_")[1]) for c in cks]
    # the run's last save lands a few thousand steps after its last planned one: keep one
    if len(cks) > 1 and steps_of[-1] - steps_of[-2] < 0.01 * steps_of[-1]:
        cks = cks[:-2] + cks[-1:]
    for ck in cks:
        steps = int(ck.stem.split("_")[1])
        if str(steps) not in have:
            spec = load_spec(f"{run}/{ck.stem}")
            have[str(steps)] = score(spec, cfg["task"], cfg["weather"])
            cache.write_text(json.dumps(have))
        out.append({"steps": steps, "seconds": marks.get(steps), **have[str(steps)]})
    return out


def print_score(label: str, s: dict) -> None:
    fails = " ".join(f"{k} {s[k]}" for k in ("crashed", "water", "lost", "timeout", "bird"))
    print(
        f"  {label:<12} landed {s['landed']:3d}/{s['n']}  {fails}  time {s['time']}"
        f"  touch {s['touch']}  move {s['change']}  pinned {s['pinned']}"
    )
    if "by_birds" in s:
        print(
            f"  {'':<12} birds 0-3 {s['by_birds']}  moving pad {s['pad_moving']}"
            f"  still {s['pad_still']}  low battery {s['low_battery']}"
        )


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
            print_score(f"{c['steps']:,}", c)
    if args.table is not None:
        for item in args.table:
            label, ckpt = item.split("=", 1)
            spec = load_spec(ckpt)
            print(label, spec["task"])
            for w in ("calm", "rough"):
                print_score(w, score(spec, spec["task"], w))
    if args.parity is not None:
        from export import export_parity

        export_parity(args.parity)
    if args.export:
        from export import export

        export(json.loads(Path(args.export).read_text()))


if __name__ == "__main__":
    main()


def blindfold(net: Net, width: int = 26, stack: int = 3):
    """The exact-position pilot with its bird inputs replaced by "no bird" in every snapshot:
    how many bird hits does seeing them save?"""
    none = np.array([2.0, 0.0, 0.0, 0.0, 0.0] * 2)

    def pilot(obs):
        o = np.array(obs, float).reshape(len(obs), stack, width)
        o[:, :, 16:26] = none
        return net(o.reshape(len(obs), -1))

    return pilot


def bird_warnings(net: Net, radius: float = 3.0) -> list[float]:
    """For each bird hit on the test courses: seconds from a bird first coming within `radius`
    of the drone to the collision."""
    from fleet_env import BIRD

    env = FleetEnv(len(TEST_SEEDS), condition("state", "rough"), seed=0)
    obs = env.start(TEST_SEEDS)
    f = env.fleet
    first = np.full(len(TEST_SEEDS), -1.0)
    while (f.outcome == FLYING).any():
        obs, _, _, _ = env.step(net(obs))
        bx, by, _, _ = f.bird_pos()
        d = np.sqrt((bx - f.x[:, None]) ** 2 + (by - f.y[:, None]) ** 2)
        d = np.where(f.bird_on, d, np.inf).min(1)
        newly = (first < 0) & (d < radius)
        first[newly] = (f.ticks * f.ph.dt)[newly]
    hits = np.nonzero(f.outcome == BIRD)[0]
    return sorted(round(float(f.ticks[i] * f.ph.dt - first[i]), 2) for i in hits)
