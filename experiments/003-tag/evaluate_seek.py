"""Score the hide-and-seek chasers for post 006 and export the site data.

uv run experiments/003-tag/evaluate_seek.py            # scores (cached) -> eval.json
uv run experiments/003-tag/evaluate_seek.py --export   # + models, traces, parity fixture

Test maps are post 5's: seeds 1-40 of every family (200 maps), one start each. Runners:
post 5's best runner (sees everything, picks moves with some randomness), the scripted
hider, and a runner standing still somewhere out of sight (a pure search test).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import numpy as np

import maps
from evaluate import RUNS, SITE
from evaluate_maps import batch_act, cases, net
from sight import Hider, SeekArena, SeekBot, Still, obs_size, sees
from tag_env import Config, NetEvader

OUT = SITE / "public" / "data" / "006-tag-seek"
SP40 = "msp2-s1/evader_r20.zip"  # post 5's best runner
RUNNERS = ("sp40", "hider", "still")


def prey_of(name: str):
    if name == "sp40":
        return NetEvader(net(SP40), sample=True)
    return Hider() if name == "hider" else Still()


@cache
def still_cases() -> list[dict]:
    """Test maps again, but the runner starts somewhere the chaser can't see."""
    env = SeekArena(Config(evaders=lambda r: Still(), maps="random", rays=16), "none")
    out = []
    for fam in maps.FAMILIES:
        for seed in range(1, 41):
            env.reset(seed=30_000 + len(out), options={"map": (seed, fam), "prey": Still()})
            assert not sees(env.chaser.x, env.chaser.y, env.evader.x, env.evader.y, env.arena)
            out.append(
                {
                    "map": (seed, fam),
                    "evader": (env.evader.x, env.evader.y),
                    "chaser_pos": (env.chaser.x, env.chaser.y),
                }
            )
    return out


def sight_of(path: str) -> str:
    n = net(path).layers[0][0].shape[1]
    return {obs_size(s): s for s in ("none", "last", "map")}[n]  # xray shares "none"'s size


def play(chaser, runner: str, ratio: float = 0.9, sight: str | None = None, record=0, eps=None):
    """chaser: a SeekBot or a checkpoint path (sight inferred unless given). eps: test
    episodes (default: all 200 for this runner)."""
    eps = eps or (still_cases() if runner == "still" else cases("maps"))
    bot = chaser if isinstance(chaser, SeekBot) else None
    m = None if bot else net(chaser)
    sight = sight or ("map" if bot else sight_of(chaser))
    cfg = Config(evaders=lambda r: None, maps="random", rays=16)
    envs = [SeekArena(cfg, sight) for _ in eps]
    obs = np.stack(
        [
            e.reset(seed=i, options={**o, "prey": prey_of(runner), "bot": bot, "ratio": ratio})[0]
            for i, (e, o) in enumerate(zip(envs, eps, strict=True))
        ]
    ).astype(np.float64)
    n = len(envs)
    alive = np.ones(n, dtype=bool)
    length = np.zeros(n, dtype=int)
    caught = np.zeros(n, dtype=bool)
    first_seen = np.full(n, -1)
    seen = np.zeros(n)
    was = np.array([e.tracker.seen for e in envs])
    lost_at = np.full(n, -1)
    spells: list[int] = []  # steps from losing sight to seeing again (-1: never, censored)
    tracks = [[_frame(e)] for e in envs[:record]]
    while alive.any():
        idx = np.flatnonzero(alive)
        acts = np.zeros(len(idx), dtype=int) if bot else batch_act(m, obs[idx], None)
        for i, a in zip(idx, acts, strict=True):
            e = envs[i]
            o, _, term, trunc, info = e.step(int(a))
            obs[i] = o
            length[i] += 1
            now = e.tracker.seen
            if now and first_seen[i] < 0:
                first_seen[i] = length[i]
            if was[i] and not now:
                lost_at[i] = length[i]
            elif now and not was[i] and lost_at[i] >= 0:
                spells.append(int(length[i] - lost_at[i]))
                lost_at[i] = -1
            was[i] = now
            if i < record:
                tracks[i].append(_frame(e))
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
    res = {
        "caught": float(caught.mean()),
        "seconds": float(length.mean() * dt),
        "seenShare": float(seen.mean()),
        "firstSeen": float(np.median(fs)),
        "firstSeenMean": float(fs.mean()),
        "found": float((first_seen >= 0).mean()),
        "lost": int(len(sp)),
        "refound": float(((sp >= 0) & (sp <= REFIND)).mean()) if len(sp) else None,
        "byFamily": {f: float(caught[fam == f].mean()) for f in maps.FAMILIES if (fam == f).any()},
    }
    if record:
        res["tracks"] = tracks
    return res


REFIND = 40  # steps: "found again within 2 seconds"


def _frame(e: SeekArena) -> list:
    return [
        round(e.evader.x, 3),
        round(e.evader.y, 3),
        round(e.chaser.x, 3),
        round(e.chaser.y, 3),
        int(e.tracker.seen),
    ]


def cached(path, fn):
    if path.exists():
        return json.loads(path.read_text())
    res = fn()
    path.write_text(json.dumps(res))
    return res


SIGHT_RUNS = {s: [f"seek-{s}-s1", f"seek-{s}-s2"] for s in ("xray", "none", "last", "map")}
CURVE = [0, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000]
BOTS = ("xray", "lastseen", "search")


def run_sight(run: str) -> str:
    return json.loads((RUNS / run / "seek_log.json").read_text())["sight"]


def ckpt_job(args: tuple[str, int, str, float]) -> dict:
    run, steps, runner, ratio = args
    path = f"{run}/ckpt_{steps:07d}.zip"
    out = RUNS / run / f"evalseek_{steps:07d}_{runner}_{int(ratio * 100)}.json"
    return cached(out, lambda: play(path, runner, ratio, run_sight(run)))


def bot_job(args: tuple[str, str, float]) -> dict:
    k, r, ratio = args
    out = RUNS / f"seekbot_{k}_{r}_{int(ratio * 100)}.json"
    return cached(out, lambda: play(SeekBot(k), r, ratio))


# ---------------------------------------------------------------- site exports


def best_run(sight: str) -> str:
    """The seed that catches best on average (all three runners, 90% speed)."""

    def score(r):
        return np.mean([ckpt_job((r, CURVE[-1], rn, 0.9))["caught"] for rn in RUNNERS])

    return max(SIGHT_RUNS[sight], key=score)


LABELS = {
    "map": "Trained seeker: memory + map",
    "last": "Trained seeker: last sighting only",
    "none": "Trained seeker: sight only",
    "xray": "Trained chaser that sees through walls",
}


def export_all() -> None:
    seekers = []
    for s in ("map", "last", "none", "xray"):
        r = best_run(s)
        p = f"{r}/ckpt_{CURVE[-1]:07d}.zip"
        seekers.append(
            {"id": s, "label": LABELS[s], "sight": s, "path": p, "net": net(p).to_json()}
        )
    runners = [{"id": "sp40", "label": "Post 5's runner", "rays": 16, "net": net(SP40).to_json()}]
    data = {"world": _physics(), "seekers": seekers, "runners": runners}
    (OUT / "models.json").write_text(json.dumps(data))
    print("models.json", round(len(json.dumps(data)) / 1024), "KB")
    export_parity(seekers[0]["path"])
    export_frames(seekers[0]["path"])


def _physics() -> dict:
    from tag_env import TagArena

    w = TagArena().describe()
    w["pillars"] = []
    return w


def _map_json(a) -> dict:
    return {"pillars": [list(p) for p in a.pillars], "blocks": [list(b) for b in a.blocks]}


def record_episode(case: dict, chaser, runner: str, ratio: float, seeker=None) -> dict:
    """One episode with every state, for the parity test. chaser: a SeekBot kind or "net"."""
    from tag_env import observe

    env = SeekArena(Config(evaders=lambda r: None, maps="random", rays=16), "map")
    prey = NetEvader(net(SP40), sample=False) if runner == "net" else prey_of(runner)
    bot = None if chaser == "net" else SeekBot(chaser)
    o, _ = env.reset(seed=0, options={**case, "prey": prey, "bot": bot, "ratio": ratio})
    p = env.cfg.physics
    out = {"map": {"seed": case["map"][0], "family": case["map"][1]}, "ratio": ratio}
    out |= {"chaser": chaser, "runner": runner, "start": {"evader": case["evader"]}}
    out["start"]["chaser"] = case["chaser_pos"]
    out["runnerActions"], out["chaserActions"] = [], []

    def snap(obs):
        tr = env.tracker
        return [
            env.evader.state(),
            env.chaser.state(),
            int(tr.seen),
            int(tr.possible.sum()),
            float(tr.weight.sum()),
            [float(v) for v in obs],
        ]

    out["states"] = [snap(o)]
    for _ in range(PARITY_STEPS):
        if runner == "net":
            out["runnerActions"].append(
                prey.net.act(observe(env.evader, env.chaser, p, env.arena, 16))
            )
        a = 0
        if chaser == "net":
            a = seeker.act(np.asarray(o, dtype=np.float64))
            out["chaserActions"].append(a)
        o, _, term, trunc, _ = env.step(a)
        out["states"].append(snap(o))
        if term or trunc:
            break
    out["caught"] = bool(term)
    return out


PARITY_STEPS = 160  # per recorded episode (keeps the fixture small)


def export_parity(seeker_path: str) -> None:
    eps = cases("maps")
    still = still_cases()
    seeker = net(seeker_path)
    specs = [
        (eps[3], "search", "hider", 0.9),
        (eps[45], "lastseen", "hider", 1.0),
        (still[90], "search", "still", 0.9),
        (eps[130], "xray", "hider", 0.9),
        (eps[170], "net", "net", 0.9),
        (still[150], "net", "still", 1.0),
    ]
    episodes = [record_episode(c, ch, rn, ratio, seeker) for c, ch, rn, ratio in specs]
    rng = np.random.default_rng(0)
    samples = []
    for ep in episodes[-2:]:
        for k in rng.choice(len(ep["states"]), 10, replace=False):
            ob = ep["states"][int(k)][5]
            samples.append({"obs": ob, "logits": seeker.logits(ob).tolist()})
    fixture = {
        "world": _physics(),
        "episodes": episodes,
        "maps": {
            f"{c['map'][0]}-{c['map'][1]}": _map_json(maps.generate(*c["map"])) for c, *_ in specs
        },
        "policy": {"net": seeker.to_json(digits=None), "samples": samples},
    }
    (SITE / "test" / "tag-seek-parity.json").write_text(json.dumps(fixture))
    print("parity:", [len(e["states"]) for e in episodes])


FRAMES = [99, 120]  # still-runner test episodes shown in the post (walls #20, maze #1)


def export_frames(seeker_path: str) -> None:
    """Snapshots of the map seeker searching for a still runner, for the post's figure.
    The episodes were picked from those where it first sees the runner after 3-6 seconds."""
    still = still_cases()
    panels = [_frames_of(seeker_path, still[i], i) for i in FRAMES]
    (OUT / "frames.json").write_text(json.dumps({"world": _physics(), "episodes": panels}))


def _frames_of(seeker_path: str, case: dict, index: int) -> dict:
    env = SeekArena(Config(evaders=lambda r: None, maps="random", rays=16), "map")
    o, _ = env.reset(seed=0, options={**case, "prey": Still(), "ratio": 0.9})
    m = net(seeker_path)
    path = [[round(env.chaser.x, 2), round(env.chaser.y, 2)]]
    snaps = []

    def snap():
        tr = env.tracker
        w = {int(k): round(float(tr.weight[k]), 2) for k in np.flatnonzero(tr.weight > 0.02)}
        return {"step": env.steps, "chaser": path[-1], "path": list(path), "weight": w}

    snaps.append(snap())
    found = None
    while True:
        o, _, term, trunc, _ = env.step(m.act(np.asarray(o, dtype=np.float64)))
        path.append([round(env.chaser.x, 2), round(env.chaser.y, 2)])
        if env.steps % 10 == 0:
            snaps.append(snap())
        if env.tracker.seen and found is None:
            found = env.steps
            snaps.append(snap())
        if term or trunc:
            break
    return {
        "index": index,
        "map": _map_json(env.arena),
        "mapLabel": f"{case['map'][1].capitalize()} map #{case['map'][0]}",
        "runner": [round(case["evader"][0], 2), round(case["evader"][1], 2)],
        "cell": env.grid.nav.cell,
        "nx": env.grid.nav.nx,
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
    runs = [r for rs in SIGHT_RUNS.values() for r in rs if (RUNS / r / "seek_log.json").exists()]
    finals = [(r, json.loads((RUNS / r / "seek_log.json").read_text())) for r in runs]
    finals = [(r, log) for r, log in finals if "seconds" in log]
    jobs = [(k, r, ratio) for k in BOTS for r in RUNNERS for ratio in (0.9, 1.0)]
    cjobs = [(r, st, rn, 0.9) for r, _ in finals for st in CURVE for rn in RUNNERS]
    cjobs += [(r, CURVE[-1], rn, 1.0) for r, _ in finals for rn in RUNNERS]
    with ProcessPoolExecutor(args.procs) as ex:
        bots = dict(zip(jobs, ex.map(bot_job, jobs), strict=True))
        curves = dict(zip(cjobs, ex.map(ckpt_job, cjobs), strict=True))
    data = {
        "bots": [
            {"kind": k, "runner": r, "ratio": ratio, **bots[(k, r, ratio)]} for k, r, ratio in jobs
        ],
        "runs": [
            {
                "name": r,
                "sight": log["sight"],
                "minutes": round(log["seconds"] / 60, 1),
                "training": log["phases"],
                "evals": [
                    {"steps": st, "runner": rn, "ratio": ratio, **curves[(r, st, rn, ratio)]}
                    for rr, st, rn, ratio in cjobs
                    if rr == r
                ],
            }
            for r, log in finals
        ],
    }
    (OUT / "eval.json").write_text(json.dumps(data))
    print("wrote", OUT / "eval.json")
    if args.export:
        export_all()


if __name__ == "__main__":
    main()
