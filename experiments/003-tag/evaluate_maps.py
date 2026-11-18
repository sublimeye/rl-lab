"""Score the random-map runs for post 005 and export the site data.

uv run experiments/003-tag/evaluate_maps.py            # scores (cached) -> eval.json
uv run experiments/003-tag/evaluate_maps.py --export   # + models, traces, parity fixture

Test maps are seeds 1-40 of every family (200 maps), one start each, never used in
training (training draws seeds from 1,000,000 up). Post 3's room is scored separately with
its own 200 starts. Scripted bots plan paths on a grid; the chaser runs at 90% speed.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import numpy as np

import maps
from evaluate import RUNS, SITE, starts
from tag_env import ACTIONS, MLP, Arena, Chaser, Config, NetChaser, TagArena

OUT = SITE / "public" / "data" / "005-tag-maps"
RATIO = 0.9
PER_FAMILY = 40
TESTS = {
    "pursuit-90": Chaser("pursuit", RATIO),
    "intercept-90": Chaser("intercept", RATIO),
    "keyboard-90": Chaser("intercept", RATIO, reaction=4, dirs8=True),
}


@cache
def cases(where: str = "maps") -> list[dict]:
    """Reset options for every test episode: a map and the two start positions."""
    if where == "classic":
        return [
            {"arena": Arena(), "evader": e, "chaser_pos": c} for e, c in starts()
        ]  # post 3's room and starts
    env = TagArena(Config(maps="random"))
    out = []
    for fam in maps.FAMILIES:
        for seed in range(1, PER_FAMILY + 1):
            env.reset(seed=20_000 + len(out), options={"map": (seed, fam)})
            out.append(
                {
                    "map": (seed, fam),
                    "evader": (env.evader.x, env.evader.y),
                    "chaser_pos": (env.chaser.x, env.chaser.y),
                }
            )
    return out


@cache
def net(path: str) -> MLP:
    import torch
    from stable_baselines3 import PPO

    from selfplay import net_of

    torch.set_num_threads(1)
    return net_of(PPO.load(RUNS / path, device="cpu"))


def rays_of(path: str) -> int:
    """Input size tells how many ray sensors a network was trained with."""
    return net(path).layers[0][0].shape[1] - 9


def batch_act(m: MLP, X: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    for i, (w, b) in enumerate(m.layers):
        X = X @ w.T + b
        if i < len(m.layers) - 1:
            X = np.tanh(X)
    if rng is None:
        return X.argmax(axis=1)
    p = np.exp(X - X.max(axis=1, keepdims=True))
    c = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
    return np.minimum((c < rng.random((len(X), 1))).sum(axis=1), len(ACTIONS) - 1)


def play(evader: str, chaser, where="maps", sample=False, ratio=RATIO, record=0) -> dict:
    """Escape rate of `evader` (checkpoint path) against a scripted Chaser or a chaser
    checkpoint path, over the test episodes."""
    m = net(evader)
    rays = rays_of(evader)
    if isinstance(chaser, str):
        chaser = NetChaser(net(chaser), ratio=ratio)
        # the evader's and the chaser's sensors must match: the env builds both observations
        assert rays_of_net(chaser.net) == rays
    eps = where if isinstance(where, list) else cases(where)
    envs = [TagArena(Config(maps="random", rays=rays)) for _ in eps]
    obs = np.stack(
        [
            e.reset(seed=i, options={**o, "chaser": chaser})[0]
            for i, (e, o) in enumerate(zip(envs, eps, strict=True))
        ]
    ).astype(np.float64)
    rng = np.random.default_rng(0) if sample else None
    alive = np.ones(len(envs), dtype=bool)
    length = np.zeros(len(envs), dtype=int)
    escaped = np.zeros(len(envs), dtype=bool)
    tracks = [[[e.evader.x, e.evader.y, e.chaser.x, e.chaser.y]] for e in envs[:record]]
    while alive.any():
        idx = np.flatnonzero(alive)
        acts = batch_act(m, obs[idx], rng)
        for i, a in zip(idx, acts, strict=True):
            o, _, term, trunc, info = envs[i].step(int(a))
            obs[i] = o
            length[i] += 1
            if i < record:
                e = envs[i]
                tracks[i].append([e.evader.x, e.evader.y, e.chaser.x, e.chaser.y])
            if term or trunc:
                alive[i] = False
                escaped[i] = info["outcome"] == "escaped"
    res = {"escaped": float(escaped.mean()), "seconds": float(length.mean() * 0.05)}
    if where == "maps":
        fam = np.array([o["map"][1] for o in eps])
        res["byFamily"] = {f: float(escaped[fam == f].mean()) for f in maps.FAMILIES}
    if record:
        res["tracks"] = tracks
    return res


def rays_of_net(m: MLP) -> int:
    return m.layers[0][0].shape[1] - 9


def cached(path, fn):
    if path.exists():
        return json.loads(path.read_text())
    res = fn()
    path.write_text(json.dumps(res))
    return res


def steps_of(p) -> int:
    return int(re.findall(r"\d+", p.stem)[-1])


# ---------------------------------------------------------------- stage 1: scripted bots

STAGE1 = ["maps-s1", "maps-s2", "maps-s3", "maps128-s1", "maps-norays-s1"]
CURVE = [0, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000]
SP3 = "sp-s3/evader_r20"  # post 4's best runner (no ray sensors)


def ckpt_job(args: tuple[str, int]) -> dict:
    run, steps = args
    path = f"{run}/ckpt_{steps:07d}.zip"

    def compute():
        res = {"steps": steps}
        for name, ch in TESTS.items():
            res[name] = play(path, ch)
        return res

    return cached(RUNS / run / f"evalmaps_{steps:07d}.json", compute)


def final_job(path: str) -> dict:
    """A runner against every bot, on the test maps and in post 3's room, both pick modes."""
    tag = path.replace("/", "_").removesuffix(".zip")

    def compute():
        res = {"path": path}
        for sample in (False, True):
            key = "sample" if sample else "best"
            res[key] = {
                where: {n: play(path, ch, where, sample) for n, ch in TESTS.items()}
                for where in ("maps", "classic")
            }
        return res

    return cached(RUNS / f"final_{tag}.json", compute)


# ---------------------------------------------------------------- exploiters


def exploit_job(xrun: str) -> dict:
    meta = json.loads((RUNS / xrun / "exploit.json").read_text())
    target = meta["evader"] + ".zip"

    def one(p):
        st = steps_of(p)
        res = cached(
            RUNS / xrun / f"evalmaps_{st:07d}.json",
            lambda: {
                k: v
                for k, v in play(
                    target, f"{xrun}/{p.name}", "maps", meta["sample"], meta["ratio"]
                ).items()
                if k != "tracks"
            },
        )
        return {"steps": st, **res}

    ck = sorted((RUNS / xrun).glob("ckpt_*.zip"), key=steps_of)
    return {
        "name": xrun,
        "target": meta["evader"],
        "ratio": meta["ratio"],
        "sample": meta["sample"],
        "checkpoints": [one(p) for p in ck],
    }


SP_ROUNDS = [0, 2, 5, 10, 15, 20]
SP_SEEDS = (1, 2, 3)


def sp_round_job(args: tuple[str, int]) -> dict:
    """A self-play runner (picking with randomness) against the scripted bots and every
    seed's final self-play chaser."""
    run, r = args
    path = f"{run}/evader_r{r:02d}.zip"

    def compute():
        res = {"round": r}
        for name, ch in TESTS.items():
            res[name] = play(path, ch, sample=True)
        for k in SP_SEEDS:
            res[f"chaser-s{k}"] = play(path, f"msp-s{k}/chaser_r20.zip", sample=True)
        return res

    return cached(RUNS / run / f"evalmaps_round_{r:02d}.json", compute)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--finals", nargs="*", default=[], help="extra runner checkpoints to score")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        (run, st) for run in STAGE1 for st in CURVE if (RUNS / run / f"ckpt_{st:07d}.zip").exists()
    ]
    finals = [f"{run}/ckpt_{10_000_000:07d}.zip" for run in STAGE1] + [f"{SP3}.zip"]
    finals += [f"{p}{k}/evader_r20.zip" for p in ("msp-s", "msp2-s") for k in SP_SEEDS]
    finals += [f if f.endswith(".zip") else f + ".zip" for f in args.finals]
    finals = [f for f in finals if (RUNS / f).exists()]
    xruns = sorted(p.parent.name for p in RUNS.glob("m*-*/exploit.json"))
    with ProcessPoolExecutor(max_workers=8) as ex:
        curve = list(ex.map(ckpt_job, jobs))
        final = list(ex.map(final_job, finals))
        exploits = list(ex.map(exploit_job, xruns))
        sp_jobs = [
            (f"msp-s{s}", r)
            for s in SP_SEEDS
            for r in SP_ROUNDS
            if all((RUNS / f"msp-s{k}/chaser_r20.zip").exists() for k in SP_SEEDS)
        ]
        sp_scores = list(ex.map(sp_round_job, sp_jobs))
    runs = []
    for run in STAGE1:
        log = RUNS / run / "train_log.json"
        meta = json.loads(log.read_text()) if log.exists() else {}
        runs.append(
            {
                "name": run,
                "arch": meta.get("arch"),
                "rays": meta.get("rays"),
                "trainSeconds": meta.get("seconds"),
                "checkpoints": [c for (r, _), c in zip(jobs, curve, strict=True) if r == run],
            }
        )
    sp = []
    for p in sorted(RUNS.glob("msp*-s*/selfplay_log.json")):
        name = p.parent.name
        rounds = [sc for (run, _), sc in zip(sp_jobs, sp_scores, strict=True) if run == name]
        sp.append({"name": name, "log": json.loads(p.read_text()), "rounds": rounds})
    data = {
        "testMaps": len(cases()),
        "ratio": RATIO,
        "runs": runs,
        "finals": final,
        "selfplay": sp,
        "exploits": exploits,
    }
    (OUT / "eval.json").write_text(json.dumps(data))
    print("wrote", OUT / "eval.json")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- site exports


def export_parity(evader: str) -> None:
    """Fixture for site/test/tag-maps.test.mjs: maps, rays, bot paths and network outputs."""
    import torch
    from stable_baselines3 import PPO

    from tag_env import flee_action

    model = PPO.load(RUNS / evader, device="cpu")
    rays = rays_of(evader)
    rng = np.random.default_rng(11)
    prng = {}
    for seed in (1, 42, 123456, 2**31 - 1):
        r = maps.mulberry32(seed)
        prng[str(seed)] = [r() for _ in range(5)]
    gen = [
        {"seed": s, "family": f, **_map_json(maps.generate(s, f))}
        for f in maps.FAMILIES
        for s in range(1, 7)
    ]
    probe = maps.generate(3, "maze")
    pts = rng.uniform([0.4, 0.4], [11.6, 7.6], size=(40, 2))
    pts = [p for p in pts if not maps.square_hits(p[0], p[1], 0.3, probe)][:20]
    ray_cases = [{"x": x, "y": y, "rays": maps.rays(x, y, probe, rays)} for x, y in pts]
    nav = maps.NavGrid(probe)
    nav_cases = []
    for (fx, fy), (tx, ty) in zip(pts[:10], pts[10:20], strict=False):
        nav_cases.append(
            {"from": [fx, fy], "to": [tx, ty], "dir": list(nav.direction(fx, fy, tx, ty, 0.36))}
        )
    cases = [
        ("maze, pursuit bot, network runner", (3, "maze"), Chaser("pursuit", 0.9), "policy"),
        (
            "walls, 8-dir slow bot, random moves",
            (7, "walls"),
            Chaser("intercept", 0.85, 3, True, 1.2),
            "random",
        ),
        ("mixed, intercept bot, flee", (5, "mixed"), Chaser("intercept", 1.0), "flee"),
        ("blocks, pursuit bot, network runner", (2, "blocks"), Chaser("pursuit", 0.8), "policy"),
    ]
    episodes = []
    for i, (name, mp, chaser, mode) in enumerate(cases):
        env = TagArena(Config(maps="random", rays=rays))
        obs, _ = env.reset(seed=300 + i, options={"chaser": chaser, "map": mp})
        start = {"evader": env.evader.state(), "chaser": env.chaser.state()}
        actions, states, done = [], [], False
        for _ in range(240):
            if mode == "random":
                a = int(rng.integers(len(ACTIONS)))
            elif mode == "flee":
                a = flee_action(env.evader, env.chaser)
            else:
                a = int(model.predict(obs, deterministic=True)[0])
            obs, _, term, trunc, info = env.step(a)
            actions.append(a)
            states.append([env.evader.state(), env.chaser.state(), obs.astype(float).tolist()])
            if term or trunc:
                done = info["outcome"] == "caught"
                break
        episodes.append(
            {
                "name": name,
                "map": {"seed": mp[0], "family": mp[1]},
                "chaser": chaser.__dict__,
                "start": start,
                "actions": actions,
                "states": states,
                "caught": done,
            }
        )
    samples = []
    for obs in rng.uniform(-1, 1, size=(12, 9 + rays)):
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32)[None]
            z = model.policy.action_net(model.policy.mlp_extractor.policy_net(t))[0]
        samples.append({"obs": obs.tolist(), "logits": z.numpy().astype(float).tolist()})
    fixture = {
        "world": _physics(),
        "rays": rays,
        "prng": prng,
        "maps": gen,
        "rayProbe": {"map": {"seed": 3, "family": "maze"}, "cases": ray_cases},
        "nav": nav_cases,
        "episodes": episodes,
        "policy": {"net": net(evader).to_json(digits=None), "samples": samples},
    }
    (SITE / "test" / "tag-maps-parity.json").write_text(json.dumps(fixture))


def _map_json(a: Arena) -> dict:
    return {"pillars": [list(p) for p in a.pillars], "blocks": [list(b) for b in a.blocks]}


def _physics() -> dict:
    """World settings for the browser without a map (the game adds one)."""
    w = TagArena().describe()
    w["pillars"] = []
    return w


def export_models(entries: list[tuple[str, str, str, str]]) -> None:
    """The game's menus: (id, label, checkpoint, role)."""
    models = [
        {
            "id": i,
            "label": label,
            "path": p,
            "role": role,
            "rays": rays_of(p),
            "net": net(p).to_json(),
        }
        for i, label, p, role in entries
    ]
    (OUT / "models.json").write_text(json.dumps({"world": _physics(), "models": models}))


def export_traces(cases_: list[dict]) -> None:
    """Paths for TagTraces, each on its own test map: {model, chaser, evader, chaserPath,
    sample, index (test episode), mapLabel}."""
    panels = []
    eps = cases()
    for c in cases_:
        i = c["index"]
        r = play(c["evader"], c["chaserPath"], [eps[i]], c["sample"], RATIO, record=1)
        track = [[round(v, 2) for v in s] for s in r["tracks"][0]]
        panels.append(
            {
                "model": c["model"],
                "chaser": c["chaser"],
                "mapLabel": c.get("mapLabel"),
                "map": _map_json(maps.generate(*eps[i]["map"])),
                "track": track,
                "escaped": r["escaped"] == 1.0,
                "seconds": round((len(track) - 1) * 0.05, 2),
            }
        )
    (OUT / "traces.json").write_text(json.dumps({"world": _physics(), "panels": panels}))
