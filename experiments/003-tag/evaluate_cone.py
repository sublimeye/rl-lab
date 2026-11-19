"""Score the field-of-view seekers for post 007 and export the site data.

uv run experiments/003-tag/evaluate_cone.py            # scores (cached) -> eval.json
uv run experiments/003-tag/evaluate_cone.py --export   # + models, parity fixture, frames

Same test episodes as post 6 (evaluate_seek): post 5's 200 test maps against post 5's best
runner and the scripted hider, and 200 starts with a still runner out of sight. The seeker
that handles any settings is scored at HUMAN and with one setting changed at a time.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

import maps
from cone import ALLROUND, HUMAN, ConeArena, Senses, wrap
from evaluate import RUNS, SITE
from evaluate_maps import cases, net
from evaluate_seek import REFIND, RUNNERS, SP40, _map_json, _physics, cached, prey_of, still_cases
from sight import SeekBot, Still
from tag_env import ACTIONS, Config, NetEvader

OUT = SITE / "public" / "data" / "007-tag-cone"
ANY = ["cone-any-s1", "cone-any-s2"]
WIDE = ["cone-any128-s1"]
SPECIALISTS = ["cone-human-s1", "cone-human-s2"]
POST6 = "seek-map-s2/ckpt_10000000"  # post 6's seeker (the one on its page)
FINAL = 20_000_000
CURVE = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000]


def _h(**kw) -> Senses:
    return Senses(**{**asdict(HUMAN), **kw})


# HUMAN, then one setting changed at a time, then post 6's all-round sight
SETTINGS: dict[str, Senses] = {
    "human": HUMAN,
    "fov60": _h(fov=60.0),
    "fov140": _h(fov=140.0),
    "fov200": _h(fov=200.0),
    "fov360": _h(fov=360.0),
    "turn90": _h(turn=90.0),
    "turn180": _h(turn=180.0),
    "turn600": _h(turn=600.0),
    "turn1080": _h(turn=1080.0),
    "react0": _h(reaction=0.0),
    "react10": _h(reaction=0.1),
    "react40": _h(reaction=0.4),
    "back40": _h(backpedal=0.4),
    "back75": _h(backpedal=0.75),
    "back100": _h(backpedal=1.0),
    "allround": ALLROUND,
    # why a fast head hurts: the same 1080 deg/s head without the reaction delay, and
    # without the slower walking backwards
    "turn1080react0": _h(turn=1080.0, reaction=0.0),
    "turn1080back100": _h(turn=1080.0, backpedal=1.0),
}


def play(chaser, runner: str, senses: Senses, ratio: float = 0.9, eps=None) -> dict:
    """chaser: a SeekBot, a cone checkpoint path, or POST6 (post 6's seeker; its head looks
    where it goes, or at the runner while it sees it)."""
    eps = eps or (still_cases() if runner == "still" else cases("maps"))
    bot = chaser if isinstance(chaser, SeekBot) else None
    m = None if bot else net(chaser)
    old = m is not None and m.layers[0][0].shape[1] == 49
    cfg = Config(evaders=lambda r: None, maps="random", rays=16)
    envs = [ConeArena(cfg, senses) for _ in eps]
    head = "auto" if old else "net"
    obs = np.stack(
        [
            e.reset(
                seed=i,
                options={**o, "prey": prey_of(runner), "bot": bot, "ratio": ratio, "head": head},
            )[0]
            for i, (e, o) in enumerate(zip(envs, eps, strict=True))
        ]
    ).astype(np.float64)
    n = len(envs)
    alive = np.ones(n, dtype=bool)
    length = np.zeros(n, dtype=int)
    caught = np.zeros(n, dtype=bool)
    first_seen = np.full(n, -1)
    seen = np.zeros(n)
    turned = np.zeros(n)  # total head turning, radians
    moving = np.zeros(n)  # steps moving faster than 1 unit/s
    aside = np.zeros(n)  # ... while facing more than 45 degrees off where it goes
    was = np.array([e.tracker.seen for e in envs])
    lost_at = np.full(n, -1)
    spells: list[int] = []
    while alive.any():
        idx = np.flatnonzero(alive)
        if bot:
            acts = np.zeros((len(idx), 2), dtype=int)
        else:
            X = obs[idx][:, :49] if old else obs[idx]
            for li, (w, b) in enumerate(m.layers):
                X = X @ w.T + b
                if li < len(m.layers) - 1:
                    X = np.tanh(X)
            k = len(ACTIONS)
            acts = np.stack(
                [X[:, :k].argmax(axis=1), X[:, k:].argmax(axis=1) if not old else 0 * idx], axis=1
            )
        for i, a in zip(idx, acts, strict=True):
            e = envs[i]
            f0 = e.tracker.facing
            o, _, term, trunc, info = e.step(a)
            obs[i] = o
            length[i] += 1
            turned[i] += abs(wrap(e.tracker.facing - f0))
            c = e.chaser
            if math.hypot(c.vx, c.vy) > 1.0:
                moving[i] += 1
                aside[i] += abs(wrap(math.atan2(c.vy, c.vx) - e.tracker.facing)) > math.pi / 4
            now = e.tracker.seen
            if now and first_seen[i] < 0:
                first_seen[i] = length[i]
            if was[i] and not now:
                lost_at[i] = length[i]
            elif now and not was[i] and lost_at[i] >= 0:
                spells.append(int(length[i] - lost_at[i]))
                lost_at[i] = -1
            was[i] = now
            if term or trunc:
                alive[i] = False
                caught[i] = info["outcome"] == "caught"
                seen[i] = e.seen_steps / length[i]
                if lost_at[i] >= 0 and lost_at[i] <= length[i] - REFIND:
                    spells.append(-1)
    dt = 0.05
    fs = np.where(first_seen < 0, 400, first_seen) * dt
    fam = np.array([o["map"][1] for o in eps])
    sp = np.array(spells)
    return {
        "caught": float(caught.mean()),
        "seconds": float(length.mean() * dt),
        "seenShare": float(seen.mean()),
        "firstSeenMean": float(fs.mean()),
        "found": float((first_seen >= 0).mean()),
        "refound": float(((sp >= 0) & (sp <= REFIND)).mean()) if len(sp) else None,
        "turnPerSecond": float(np.degrees(turned.sum() / (length.sum() * dt))),
        "lookAside": float(aside.sum() / max(moving.sum(), 1)),
        "byFamily": {f: float(caught[fam == f].mean()) for f in maps.FAMILIES if (fam == f).any()},
    }


def ckpt(run: str, steps: int = FINAL) -> str:
    return f"{run}/ckpt_{steps:08d}"


def job(args: tuple[str, str, str]) -> dict:
    """(who, runner, setting). who: a cone run's checkpoint, "post6" or "bot:<kind>"."""
    who, runner, setting = args
    tag = who.replace("/", "_").replace(":", "_")
    out = RUNS / "evalcone" / f"{tag}_{runner}_{setting}.json"
    out.parent.mkdir(exist_ok=True)
    if who.startswith("bot:"):
        chaser = SeekBot(who[4:])
    else:
        chaser = POST6 if who == "post6" else who
    return cached(out, lambda: play(chaser, runner, SETTINGS[setting]))


def jobs() -> list[tuple[str, str, str]]:
    out = []
    for r in ANY + WIDE:
        out += [(ckpt(r), rn, s) for s in SETTINGS for rn in RUNNERS]
    for r in ANY + SPECIALISTS:
        out += [(ckpt(r, st), rn, "human") for st in CURVE[:-1] for rn in RUNNERS]
    for r in SPECIALISTS:
        out += [(ckpt(r), rn, "human") for rn in RUNNERS]
    for who in ("post6", "bot:search"):
        out += [(who, rn, s) for s in ("human", "allround") for rn in RUNNERS]
    return out


def log_of(run: str) -> dict:
    return json.loads((RUNS / run / "cone_log.json").read_text())


# ---------------------------------------------------------------- site exports


def best(runs: list[str]) -> str:
    """The seed that catches best on average (all three runners, HUMAN, 90% speed)."""
    return max(
        runs, key=lambda r: np.mean([job((ckpt(r), rn, "human"))["caught"] for rn in RUNNERS])
    )


def export_all() -> None:
    any_run, human_run = best(ANY), best(SPECIALISTS)
    seekers = [
        {"id": "any", "label": "Trained seeker (any settings)", "path": ckpt(any_run)},
        {"id": "human", "label": "Trained on the default settings only", "path": ckpt(human_run)},
        {"id": "post6", "label": "Post 6's seeker (looks where it walks)", "path": POST6},
    ]
    seekers[-1]["auto"] = True  # its head is scripted (cone.auto_look)
    for s in seekers:
        s["net"] = net(s["path"]).to_json()
    runners = [{"id": "sp40", "label": "Post 5's runner", "rays": 16, "net": net(SP40).to_json()}]
    world = _physics()
    data = {"world": world, "human": asdict(HUMAN), "seekers": seekers, "runners": runners}
    (OUT / "models.json").write_text(json.dumps(data))
    print("models.json", round(len(json.dumps(data)) / 1024), "KB")
    export_parity(ckpt(any_run))
    export_frames(ckpt(any_run))


PARITY_STEPS = 160


def record_episode(
    case: dict, chaser: str, runner: str, ratio: float, senses: Senses, seeker=None
) -> dict:
    """One episode with every state, for the parity test. chaser: a SeekBot kind or "net"."""
    from cone import act2
    from tag_env import observe

    env = ConeArena(Config(evaders=lambda r: None, maps="random", rays=16), senses)
    prey = NetEvader(net(SP40), sample=False) if runner == "net" else prey_of(runner)
    bot = None if chaser == "net" else SeekBot(chaser)
    o, _ = env.reset(
        seed=0, options={**case, "prey": prey, "bot": bot, "ratio": ratio, "facing": 1.0}
    )
    p = env.cfg.physics
    out = {"map": {"seed": case["map"][0], "family": case["map"][1]}, "ratio": ratio}
    out |= {"chaser": chaser, "runner": runner, "senses": asdict(senses), "facing": 1.0}
    out["start"] = {"evader": case["evader"], "chaser": case["chaser_pos"]}
    out["runnerActions"], out["chaserActions"] = [], []

    def snap(obs):
        tr = env.tracker
        return [
            env.evader.state(),
            env.chaser.state(),
            tr.facing,
            int(tr.seen),
            int(tr.possible.sum()),
            float(tr.weight.sum()),
            [float(v) for v in obs],
        ]

    out["states"] = [snap(o)]
    term = False
    for _ in range(PARITY_STEPS):
        if runner == "net":
            out["runnerActions"].append(
                prey.net.act(observe(env.evader, env.chaser, p, env.arena, 16))
            )
        a = (0, 0)
        if chaser == "net":
            a = act2(seeker, np.asarray(o, dtype=np.float64))
            out["chaserActions"].append(list(a))
        o, _, term, trunc, _ = env.step(a)
        out["states"].append(snap(o))
        if term or trunc:
            break
    out["caught"] = bool(term)
    return out


def export_parity(seeker_path: str) -> None:
    eps, still = cases("maps"), still_cases()
    seeker = net(seeker_path)
    specs = [
        (eps[3], "search", "hider", 0.9, HUMAN),
        (still[90], "search", "still", 0.9, SETTINGS["fov60"]),
        (eps[130], "lastseen", "hider", 1.0, SETTINGS["turn90"]),
        (eps[170], "net", "net", 0.9, HUMAN),
        (still[150], "net", "still", 1.0, Senses(140.0, 500.0, 0.1, 0.7)),
        (still[40], "net", "still", 0.9, ALLROUND),
    ]
    episodes = [record_episode(c, ch, rn, ratio, s, seeker) for c, ch, rn, ratio, s in specs]
    rng = np.random.default_rng(0)
    samples = []
    for ep in episodes[-3:]:
        for k in rng.choice(len(ep["states"]), 8, replace=False):
            ob = ep["states"][int(k)][6]
            samples.append({"obs": ob, "logits": seeker.logits(ob).tolist()})
    fixture = {
        "world": _physics(),
        "episodes": episodes,
        "maps": {
            f"{c['map'][0]}-{c['map'][1]}": _map_json(maps.generate(*c["map"])) for c, *_ in specs
        },
        "policy": {"net": seeker.to_json(digits=None), "samples": samples},
    }
    (SITE / "test" / "tag-cone-parity.json").write_text(json.dumps(fixture))
    print("parity:", [len(e["states"]) for e in episodes])


FRAMES = [90, 122]  # still-runner test episodes in the post (walls #11, maze #3)


def export_frames(seeker_path: str) -> None:
    still = still_cases()
    panels = [_frames_of(seeker_path, still[i], i) for i in FRAMES]
    (OUT / "frames.json").write_text(json.dumps({"world": _physics(), "episodes": panels}))


def _frames_of(seeker_path: str, case: dict, index: int, senses: Senses = HUMAN) -> dict:
    from cone import act2

    env = ConeArena(Config(evaders=lambda r: None, maps="random", rays=16), senses)
    o, _ = env.reset(seed=0, options={**case, "prey": Still(), "ratio": 0.9})
    m = net(seeker_path)
    path = [[round(env.chaser.x, 2), round(env.chaser.y, 2)]]
    snaps = []

    def snap():
        tr = env.tracker
        w = {int(k): round(float(tr.weight[k]), 2) for k in np.flatnonzero(tr.weight > 0.02)}
        return {
            "step": env.steps,
            "chaser": path[-1],
            "facing": round(tr.facing, 4),
            "path": list(path),
            "weight": w,
        }

    snaps.append(snap())
    found = None
    while True:
        o, _, term, trunc, _ = env.step(act2(m, np.asarray(o, dtype=np.float64)))
        path.append([round(env.chaser.x, 2), round(env.chaser.y, 2)])
        if env.steps % 5 == 0:
            snaps.append(snap())
        if env.tracker.seen and found is None:
            found = env.steps
            snaps.append(snap())
        if term or trunc:
            break
    # keep only the four snapshots SeekFrames draws (start, a third, two thirds, first sight)
    end = found if found is not None else snaps[-1]["step"]

    def pick(step: float) -> dict:
        return min(snaps, key=lambda s: abs(s["step"] - step))

    snaps = [pick(0), pick(end / 3), pick(2 * end / 3), pick(end)]
    return {
        "index": index,
        "map": _map_json(env.arena),
        "mapLabel": f"{case['map'][1].capitalize()} map #{case['map'][0]}",
        "runner": [round(case["evader"][0], 2), round(case["evader"][1], 2)],
        "cell": env.grid.nav.cell,
        "nx": env.grid.nav.nx,
        "fov": senses.fov,
        "found": found,
        "caught": env.steps if term else None,
        "snaps": snaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    js = jobs()
    with ProcessPoolExecutor(args.procs) as ex:
        res = dict(zip(js, ex.map(job, js), strict=True))
    runs = [r for r in ANY + WIDE + SPECIALISTS if (RUNS / r / "cone_log.json").exists()]
    data = {
        "settings": {k: asdict(v) for k, v in SETTINGS.items()},
        "results": [{"who": w, "runner": rn, "setting": s, **res[(w, rn, s)]} for w, rn, s in js],
        "runs": [
            {
                "name": r,
                "senses": log_of(r)["senses"],
                "arch": log_of(r)["arch"],
                "minutes": round(log_of(r).get("seconds", 0) / 60, 1),
                "training": log_of(r)["phases"],
            }
            for r in runs
        ],
    }
    (OUT / "eval.json").write_text(json.dumps(data))
    print("wrote", OUT / "eval.json")
    if args.export:
        export_all()


if __name__ == "__main__":
    main()
