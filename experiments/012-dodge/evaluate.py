"""Scores pilots on held-out courses and arenas (all flown at once) and exports the site data.

uv run experiments/012-dodge/evaluate.py --curves school-s1 mix-s1   # per checkpoint
uv run experiments/012-dodge/evaluate.py --table school-s1 mix-s1    # final pilots, every test
uv run experiments/012-dodge/evaluate.py --export export.json        # site/public/data/012-dodge
uv run experiments/012-dodge/evaluate.py --parity school-s1 vschool-s1  # site/test fixture

Tests: "mix" = 200 normal courses with the test mix of threats (rough weather); "arena" = 200
dodge-school arenas; "bird"/"rock"/"missile" = 200 arenas with only that threat; "none" = the
200 courses with no threats (does dodging cost landing skill?). Weights are rounded to float16,
as on the site.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from dodge_env import (
    ARENA,
    BIRD,
    FAILED,
    FLYING,
    FULL,
    FULL_VISION,
    MISSILE,
    MIX,
    NONE,
    ONLY,
    OUTCOMES,
    ROCK,
    ROUGH,
    Config,
    FleetEnv,
)
from escape import MANEUVERS

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SITE = HERE.parent.parent / "site"
DATA = SITE / "public" / "data" / "012-dodge"
TEST_SEEDS = [1_000_000_000 + i for i in range(200)]
TESTS = {"mix": MIX, "arena": ARENA, **ONLY, "none": NONE}
VISION_TASKS = ("vmix", "vschool", "vschool2")


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


def last_ckpt(run: str) -> str:
    return f"{run}/{sorted((RUNS / run).glob('ckpt_*.zip'))[-1].stem}"


def load_spec(ckpt: str) -> dict:
    from train import actor, load_model

    run = ckpt.split("/")[0]
    if "/" not in ckpt:
        ckpt = last_ckpt(run)
    spec = actor(load_model(ckpt))
    spec["layers"] = [{"w": rounded(L["w"]), "b": rounded(L["b"])} for L in spec["layers"]]
    spec["task"] = run_config(run)["task"]
    spec["obs"] = "vision" if spec["task"] in VISION_TASKS else "state"
    return spec


def condition(obs: str, test: str) -> Config:
    base = FULL_VISION if obs == "vision" else FULL
    return Config(weather=ROUGH, features=replace(base, threats=TESTS[test]))


def fly(pilot, cfg: Config, seeds=TEST_SEEDS, hide=(), snapshots=False, noise_seed=0):
    """Flies every course at once. pilot(obs) -> sticks."""
    env = FleetEnv(len(seeds), cfg, seed=noise_seed, hide=hide)
    obs = env.start(seeds)
    f = env.fleet
    snaps, sticks, alive = [], [], []
    rows = np.arange(f.n)
    while (f.outcome == FLYING).any():
        if snapshots:
            snaps.append(f.take(rows))
        flying = f.outcome == FLYING
        obs, _, _, _ = env.step(pilot(obs))
        sticks.append(np.stack([f.stick0, f.stick1], 1).copy())
        alive.append(flying)
    s, a = np.array(sticks), np.array(alive)
    diff = np.abs(np.diff(s, axis=0)).sum(2)
    both = a[1:] & a[:-1]
    change = (diff * both).sum(0) / np.maximum(both.sum(0), 1)
    return {
        "outcome": [OUTCOMES[o] for o in f.outcome],
        "time": (f.ticks * f.ph.dt).tolist(),
        "change": change.tolist(),
        "courses": [f.courses[i] for i in range(len(seeds))],
        "fleet": f,
        "snaps": snaps,
        "sticks": s,
        "alive": a,
    }


def summary(res: dict) -> dict:
    oc = res["outcome"]
    n = len(oc)
    counts = {k: sum(o == k for o in oc) for k in OUTCOMES[1:]}
    cs = res["courses"]

    def rate(mask, what="landed"):
        idx = [i for i in range(n) if mask[i]]
        return [sum(oc[i] == what for i in idx), len(idx)]

    arena = [c["arena"] for c in cs]
    out = {
        **counts,
        "n": n,
        "hits": counts["bird"] + counts["rock"] + counts["missile"],
        "change": round(float(np.mean(res["change"])), 3),
    }
    if any(arena):
        out["held"] = counts["timeout"]
    else:
        out["with_missile"] = rate([c["missile"] is not None for c in cs])
        out["no_missile"] = rate([c["missile"] is None for c in cs])
        out["by_birds"] = [rate([len(c["birds"]) == k for c in cs]) for k in range(4)]
        out["by_rocks"] = [rate([len(c["rocks"]) == k for c in cs]) for k in range(3)]
    return out


def pilot_of(spec_or_pilot):
    return Net(spec_or_pilot) if isinstance(spec_or_pilot, dict) else spec_or_pilot


def score(spec: dict, test: str, hide=()) -> dict:
    return summary(fly(Net(spec), condition(spec["obs"], test), hide=hide))


def curve(run: str, tests=("mix", "arena")) -> list[dict]:
    """Test scores at every checkpoint, cached in runs/<run>/eval.json."""
    cache = RUNS / run / "eval.json"
    marks = dict(json.loads((RUNS / run / "train_log.json").read_text())["marks"])
    have = json.loads(cache.read_text()) if cache.exists() else {}
    cks = sorted((RUNS / run).glob("ckpt_*.zip"))
    steps_of = [int(c.stem.split("_")[1]) for c in cks]
    if len(cks) > 1 and steps_of[-1] - steps_of[-2] < 0.01 * steps_of[-1]:
        cks = cks[:-2] + cks[-1:]  # the final save right after a planned one: keep one
    out = []
    for ck in cks:
        steps = int(ck.stem.split("_")[1])
        key = str(steps)
        if key not in have or any(t not in have[key] for t in tests):
            spec = load_spec(f"{run}/{ck.stem}")
            have[key] = {t: score(spec, t) for t in tests}
            cache.write_text(json.dumps(have))
        out.append({"steps": steps, "seconds": marks.get(steps), **have[key]})
    return out


HIT_OF = {"bird": BIRD, "rock": ROCK, "missile": MISSILE}
KIND_OF = {"bird": "bird", "aimed": "bird", "rock": "rock", "missile": "missile"}


def escapes(res: dict, kind: str, leads=(0.3, 0.5, 0.8, 1.2), after=0.6) -> dict:
    """For each hit by `kind`: could one of 9 full-stick escapes, started `lead` s before the
    hit, have got away? (see escape.py)"""
    f = res["fleet"]
    hits = np.nonzero(f.outcome == HIT_OF[KIND_OF[kind]])[0]
    upd = f.ph.dt * f.ph.substeps
    out = {"hits": len(hits)}
    for lead in leads:
        ok = 0
        for i in hits:
            end_u = int(np.ceil(f.ticks[i] / f.ph.substeps))
            u0 = max(0, end_u - int(round(lead / upd)))
            g = res["snaps"][u0].take(np.full(len(MANEUVERS), i))
            for _ in range(int(round((lead + after) / upd))):
                if not (g.outcome == FLYING).any():
                    break
                g.step(MANEUVERS)
            ok += bool((~np.isin(g.outcome, FAILED)).any())
        out[str(lead)] = ok
    return out


def report(spec: dict) -> dict:
    """Everything the post quotes about one final pilot."""
    out = {"task": spec["task"], "obs": spec["obs"]}
    for test in TESTS:
        res = fly(Net(spec), condition(spec["obs"], test), snapshots=test in ONLY)
        out[test] = summary(res)
        if test in ONLY:
            out[test]["escape"] = escapes(res, test)
            blind = fly(Net(spec), condition(spec["obs"], test), hide=(KIND_OF[test],))
            out[test]["blind"] = summary(blind)
    blind_mix = fly(Net(spec), condition(spec["obs"], "mix"), hide=("bird", "rock", "missile"))
    out["mix"]["blind"] = summary(blind_mix)
    return out


def hold_share(spec: dict, test: str) -> float:
    """In an arena: the share of flying time spent within HOLD_R of the point (the rest is
    mostly the trip there from the pillar)."""
    from dodge_env import HOLD_R

    env = FleetEnv(len(TEST_SEEDS), condition(spec["obs"], test), seed=0)
    obs = env.start(TEST_SEEDS)
    f, net = env.fleet, Net(spec)
    near = live = 0
    while (f.outcome == FLYING).any():
        fl = f.outcome == FLYING
        obs, _, _, _ = env.step(net(obs))
        live += fl.sum()
        near += (fl & (f.distance() < HOLD_R)).sum()
    return round(float(near / live), 3)


def add_hold(runs: list[str]) -> None:
    """Adds hold shares (every arena test) for runs to the exported eval.json."""
    path = DATA / "eval.json"
    ev = json.loads(path.read_text())
    ev["hold"] = {r: {t: hold_share(load_spec(r), t) for t in ("arena", *ONLY)} for r in runs}
    path.write_text(json.dumps(ev, separators=(",", ":")))
    print(json.dumps(ev["hold"]))


def print_report(label: str, r: dict) -> None:
    print(f"{label} ({r['task']})")
    m = r["mix"]
    print(
        f"  mix    landed {m['landed']:3d}/200  hits {m['hits']} (bird {m['bird']} rock {m['rock']}"
        f" missile {m['missile']})  crashed {m['crashed']} water {m['water']} lost {m['lost']}"
        f" timeout {m['timeout']}  move {m['change']}"
    )
    print(
        f"         with missile {m['with_missile']} without {m['no_missile']}"
        f"  blindfolded: landed {m['blind']['landed']} hits {m['blind']['hits']}"
    )
    print(f"  none   landed {r['none']['landed']}/200")
    a = r["arena"]
    print(f"  arena  held {a['held']}/200  hits {a['hits']}")
    for k in ONLY:
        o, kk = r[k], KIND_OF[k]
        e = o["escape"]
        esc = " ".join(f"{L}s:{e[L]}" for L in e if L != "hits")
        print(
            f"  {k:<7} held {o['held']}/200  hits {o[kk]}  blind: held {o['blind']['held']}"
            f" hits {o['blind'][kk]}  escapable of {e['hits']}: {esc}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", nargs="*", default=[])
    ap.add_argument("--table", nargs="*", default=None)
    ap.add_argument("--parity", nargs="*", default=None)
    ap.add_argument("--export", default=None)
    ap.add_argument("--export-curves", default=None, help="refresh only the curves")
    ap.add_argument("--hold", nargs="*", default=None, help="add hold shares to eval.json")
    args = ap.parse_args()
    for run in args.curves:
        print(run)
        for c in curve(run):
            m, a = c["mix"], c["arena"]
            secs = c["seconds"] or 0
            print(
                f"  {c['steps']:>11,} {secs / 60:5.1f} min  mix landed {m['landed']:3d}"
                f" hits {m['hits']:3d}   arena held {a['held']:3d} hits {a['hits']:3d}"
            )
    if args.table is not None:
        for run in args.table:
            print_report(run, report(load_spec(run)))
    if args.parity is not None:
        from export import export_parity

        export_parity(args.parity)
    if args.hold:
        add_hold(args.hold)
    if args.export_curves:
        from export import export_curves

        export_curves(json.loads(Path(args.export_curves).read_text()))
    if args.export:
        from export import export

        export(json.loads(Path(args.export).read_text()))


if __name__ == "__main__":
    main()
