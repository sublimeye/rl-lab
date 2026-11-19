"""Score every checkpoint on 200 held-out maps and export the data for the post.

uv run experiments/008-flood/evaluate.py s1 s2 s3        # score runs (cached per checkpoint)
uv run experiments/008-flood/evaluate.py s1 s2 s3 --export round1 --live s2
uv run experiments/008-flood/evaluate.py wall-s1 wall-s2 wall-s3 --export round2 --parity

Each checkpoint plays the same 200 map seeds (never used in training: training draws seeds
below 2**31 from its own RNG, these start at TEST_SEED). The network samples its actions
from its probabilities, as in training and in the browser game (and, separately, always takes
its top action), using a numpy copy of the network. Writes, under
site/public/data/008-flood/: eval-<tag>.json (scores, baselines, training curves, tide tests),
replays-<tag>.json (actions per checkpoint on a few maps; the browser re-simulates them) and
model-<tag>.json (the network that plays live). With --parity: site/test/flood-parity.json for
the JS port test.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from flood_env import (
    ACTIONS,
    WAIT,
    Config,
    FloodWorld,
    World,
    builder_action,
    make_map,
    observe,
    wall_site,
)

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SITE = HERE.parent.parent / "site"
OUT = SITE / "public" / "data" / HERE.name
EPISODES = 200
TEST_SEED = 3_000_000_000  # above anything training draws (< 2**31)
REPLAY_MAPS = 4


class MLP:
    """The actor network as numpy arrays (tanh hidden layers, linear logits), like the JS."""

    def __init__(self, layers: list[dict]):
        self.layers = [(np.asarray(L["w"], dtype=np.float64), np.asarray(L["b"])) for L in layers]

    def logits(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def act(self, obs) -> int:
        return int(np.argmax(self.logits(obs)))

    def sample(self, obs, rng: np.random.Generator) -> int:
        z = self.logits(obs)
        p = np.exp(z - z.max())
        return int(rng.choice(len(p), p=p / p.sum()))


def policy_net(model, digits: int | None = 5) -> dict:
    """The actor MLP as plain lists: [{w: out x in, b: out}], tanh between layers."""
    lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    lin.append(model.policy.action_net)

    def r(a):
        a = a.detach().cpu().numpy().astype(float)
        return np.round(a, digits).tolist() if digits else a.tolist()

    return {"layers": [{"w": r(m.weight), "b": r(m.bias)} for m in lin]}


def ckpt_steps(p: Path) -> int:
    return int(re.search(r"(\d+)", p.stem).group(1))


def checkpoints(run: str) -> list[Path]:
    return sorted((RUNS / run).glob("ckpt_*.zip"), key=ckpt_steps)


def run_config(run: str) -> Config:
    """The world a run trained in (train.py writes runs/<run>/config.json)."""
    c = json.loads((RUNS / run / "config.json").read_text())
    return Config(world=World(house_wall=c["house_wall"]))


def load_net(path: Path, digits: int | None = 5) -> dict:
    from stable_baselines3 import PPO

    return policy_net(PPO.load(path, device="cpu"), digits)


# -- one episode -----------------------------------------------------------------------------
def play(
    policy: str, seed: int, net: MLP | None = None, rng_seed: int = 0, cfg: Config | None = None
) -> dict:
    """policy: 'net' (samples from the network, as in training), 'net-best' (always its top
    action), 'nothing', 'random' or 'builder'. Returns the episode summary."""
    env = FloodWorld(cfg or Config())
    env.reset(seed=0, options={"map_seed": seed})
    start = env.sand.copy()
    rng = np.random.default_rng(rng_seed)
    c = wall_site(env)
    actions = []
    while True:
        if policy == "net":
            a = net.sample(observe(env), rng)
        elif policy == "net-best":
            a = net.act(observe(env))
        elif policy == "builder":
            a = builder_action(env, c)
        elif policy == "random":
            a = int(rng.integers(len(ACTIONS)))
        else:
            a = WAIT
        actions.append(a)
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    n = env.n
    added = env.sand[:n] - start[:n]
    built = np.flatnonzero(added > 0)
    # The dam: the highest built-up column (nearest the house on ties), if anything was built.
    order = built[::-1]
    dam = int(order[np.argmax(env.sand[order])]) if len(built) else -1
    return {
        "seed": seed,
        "outcome": info["outcome"],
        "t": env.t,
        "dug": env.dug,
        "dropped": env.dropped,
        "dam": dam,  # column of the dam's crest, -1 if nothing built
        "damFromHouse": n - dam if dam >= 0 else None,  # columns between crest and house
        "damHeight": int(env.sand[dam]) if dam >= 0 else None,
        "damAdded": int(added[dam]) if dam >= 0 else 0,  # grains on top of the old ground there
        "peak": round(env.map.peak, 3),
        "floor": env.map.house_floor,
        "actions": actions,
    }


def _eval_ckpt(args) -> dict:
    path, cache, cfg = args
    if cache.exists():
        return json.loads(cache.read_text())
    net = MLP(load_net(path, digits=None)["layers"])
    eps = [play("net", TEST_SEED + i, net, rng_seed=i, cfg=cfg) for i in range(EPISODES)]
    best = [play("net-best", TEST_SEED + i, net, cfg=cfg) for i in range(EPISODES)]
    for e in eps + best:
        e.pop("actions")
    res = {"steps": ckpt_steps(path), "episodes": eps, "best": best}
    cache.write_text(json.dumps(res))
    return res


def eval_run(run: str) -> list[dict]:
    cfg = run_config(run)
    jobs = [(p, RUNS / run / f"eval_{ckpt_steps(p):08d}.json", cfg) for p in checkpoints(run)]
    with ProcessPoolExecutor(8) as ex:
        return list(ex.map(_eval_ckpt, jobs))


def _eval_baseline(args) -> list[dict]:
    policy, cfg = args
    eps = [play(policy, TEST_SEED + i, rng_seed=i, cfg=cfg) for i in range(EPISODES)]
    for e in eps:
        e.pop("actions")
    return eps


def summary(eps: list[dict]) -> dict:
    dry = [e for e in eps if e["outcome"] == "dry"]
    flooded = [e for e in eps if e["outcome"] == "flooded"]
    return {
        "dry": len(dry),
        "n": len(eps),
        "floodT": round(float(np.mean([e["t"] for e in flooded])), 1) if flooded else None,
        "dropped": round(float(np.mean([e["dropped"] for e in eps])), 1),
        "dug": round(float(np.mean([e["dug"] for e in eps])), 1),
    }


def training_curve(run: str, bins: int = 40) -> list[dict]:
    log = json.loads((RUNS / run / "train_log.json").read_text())
    eps = log["episodes"]
    total = log.get("total_steps") or eps[-1][0]
    width = total / bins
    out = [
        {"from": int(i * width), "to": int((i + 1) * width), "dry": 0, "n": 0, "drops": 0}
        for i in range(bins)
    ]
    for steps, _ret, _len, outcome, dropped in eps:
        b = out[min(int(steps // width), bins - 1)]
        b["n"] += 1
        b["dry"] += outcome == "dry"
        b["drops"] += dropped
    for b in out:
        b["drops"] = round(b["drops"] / b["n"], 1) if b["n"] else None
    return out


# Tides lower and higher than training saw (training: 1-3 cells above the house floor).
TIDES = {
    "lower": (0.3, 1.0),
    "trained": (1.0, 3.0),
    "higher": (3.0, 4.5),
    "much higher": (4.5, 6.0),
}


def _tide_job(args) -> dict:
    policy, path, lo, hi, base = args
    from dataclasses import replace

    cfg = Config(world=replace(base.world, tide_over=(lo, hi)))
    net = MLP(load_net(path, digits=None)["layers"]) if path else None
    eps = [play(policy, TEST_SEED + i, net, rng_seed=i, cfg=cfg) for i in range(EPISODES)]
    for e in eps:
        e.pop("actions")
    s = summary(eps)
    dams = [e for e in eps if e["dam"] >= 0]
    s["damOverPeak"] = (
        round(float(np.mean([e["damHeight"] - e["peak"] for e in dams])), 2) if dams else None
    )
    # How high the crest stands above the house floor, and the tide's peak above the floor.
    s["damOverFloor"] = (
        round(float(np.mean([e["damHeight"] - e["floor"] for e in dams])), 2) if dams else None
    )
    s["peakOverFloor"] = round(float(np.mean([e["peak"] - e["floor"] for e in eps])), 2)
    return s


def tide_test(runs: list[str], cfg: Config) -> dict:
    jobs, keys = [], []
    for name, (lo, hi) in TIDES.items():
        for r in runs:
            jobs.append(("net", checkpoints(r)[-1], lo, hi, cfg))
            keys.append((name, r))
        for b in ("builder", "nothing"):
            jobs.append((b, None, lo, hi, cfg))
            keys.append((name, b))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_tide_job, jobs))
    out: dict = {name: {"range": list(TIDES[name])} for name in TIDES}
    for (name, who), s in zip(keys, res, strict=True):
        out[name][who] = s
    return out


# -- exports ---------------------------------------------------------------------------------
def world_json(cfg: Config) -> dict:
    w = cfg.world
    return {
        "width": w.width,
        "height": w.height,
        "houseWidth": w.house_width,
        "houseFloor": list(w.house_floor),
        "tideOver": list(w.tide_over),
        "tideStart": w.tide_start,
        "rise": w.rise,
        "beach": list(w.beach),
        "flow": w.flow,
        "damping": w.damping,
        "substeps": w.substeps,
        "settlePasses": w.settle_passes,
        "floodDepth": w.flood_depth,
        "carry": w.carry,
        "maxStep": w.max_step,
        "houseWall": w.house_wall,
        "maxSteps": cfg.max_steps,
        "rewardDry": cfg.rewards.dry,
        "rewardFlood": cfg.rewards.flood,
    }


def _transfer_job(args) -> dict:
    path, cfg = args
    net = MLP(load_net(path, digits=None)["layers"])
    eps = [play("net", TEST_SEED + i, net, rng_seed=i, cfg=cfg) for i in range(EPISODES)]
    for e in eps:
        e.pop("actions")
    return summary(eps)


def export(
    tag: str, runs: list[str], live: str, replay_run: str, transfer: list[str] | None = None
) -> None:
    """Writes eval-<tag>.json, replays-<tag>.json and model-<tag>.json. All runs must share
    one world."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = run_config(runs[0])
    assert all(run_config(r) == cfg for r in runs + [live, replay_run]), "mixed worlds"
    results = {r: eval_run(r) for r in runs}
    names = ("nothing", "random", "builder")
    with ProcessPoolExecutor(3) as ex:
        base = dict(zip(names, ex.map(_eval_baseline, [(b, cfg) for b in names]), strict=True))

    def per_ckpt(res: dict) -> dict:
        s = summary(res["episodes"])
        s["steps"] = res["steps"]
        s["best"] = summary(res["best"])
        s["best"]["outcomes"] = "".join("d" if e["outcome"] == "dry" else "f" for e in res["best"])
        s["outcomes"] = "".join("d" if e["outcome"] == "dry" else "f" for e in res["episodes"])
        dams = [e for e in res["episodes"] if e["outcome"] == "dry" and e["dam"] >= 0]
        s["damFromHouse"] = [e["damFromHouse"] for e in dams]
        s["damAdded"] = [e["damAdded"] for e in dams]
        s["damOverPeak"] = [round(e["damHeight"] - e["peak"], 2) for e in dams]
        s["peak"] = [e["peak"] for e in dams]
        return s

    eval_json = {
        "episodes": EPISODES,
        "world": world_json(cfg),
        "baselines": {
            k: {
                **summary(v),
                "outcomes": "".join("d" if e["outcome"] == "dry" else "f" for e in v),
            }
            for k, v in base.items()
        },
        "runs": {
            r: {
                "checkpoints": [per_ckpt(x) for x in res],
                "training": training_curve(r),
                "seconds": json.loads((RUNS / r / "train_log.json").read_text()).get("seconds"),
            }
            for r, res in results.items()
        },
    }
    eval_json["tides"] = tide_test(runs, cfg)
    if transfer:  # final networks trained in another world, scored in this one
        with ProcessPoolExecutor(4) as ex:
            res = ex.map(_transfer_job, [(checkpoints(r)[-1], cfg) for r in transfer])
            eval_json["transfer"] = dict(zip(transfer, res, strict=True))
    (OUT / f"eval-{tag}.json").write_text(json.dumps(eval_json, separators=(",", ":")))

    # Replays: the same few maps for every checkpoint of one run; the browser re-simulates.
    seeds = [TEST_SEED + i for i in pick_replay_maps(results[replay_run][-1], base["builder"])]
    reps = []
    for p in checkpoints(replay_run):
        net = MLP(load_net(p, digits=None)["layers"])
        runs_ = []
        for s in seeds:
            e = play("net", s, net, rng_seed=s - TEST_SEED, cfg=cfg)
            runs_.append({"seed": s, "outcome": e["outcome"], "t": e["t"], "actions": e["actions"]})
        reps.append({"steps": ckpt_steps(p), "runs": runs_})
    builder = [play("builder", s, cfg=cfg) for s in seeds]
    (OUT / f"replays-{tag}.json").write_text(
        json.dumps(
            {
                "world": world_json(cfg),
                "run": replay_run,
                "seeds": seeds,
                "checkpoints": reps,
                "builder": [{"outcome": e["outcome"], "t": e["t"]} for e in builder],
            },
            separators=(",", ":"),
        )
    )

    final = checkpoints(live)[-1]
    (OUT / f"model-{tag}.json").write_text(
        json.dumps(
            {
                "world": world_json(cfg),
                "run": live,
                "steps": ckpt_steps(final),
                "net": load_net(final),
            },
            separators=(",", ":"),
        )
    )
    for f in sorted(OUT.glob("*.json")):
        print(f"{f.name}: {f.stat().st_size / 1024:.0f} KB")


def pick_replay_maps(final: dict, builder: list[dict]) -> list[int]:
    """A few test maps where the final network keeps the house dry: the first one where the
    scripted builder floods (if there is one), the rest spread over where the dam stands."""
    eps = final["episodes"]
    dry = [i for i, e in enumerate(eps) if e["outcome"] == "dry" and e["dam"] >= 0]
    beats = [i for i in dry if builder[i]["outcome"] == "flooded"][:1]
    rest = sorted((i for i in dry if i not in beats), key=lambda i: eps[i]["damFromHouse"])
    k = REPLAY_MAPS - len(beats)
    if len(rest) <= k:
        return beats + rest
    idx = np.linspace(0, len(rest) - 1, k).round().astype(int)
    return beats + [rest[i] for i in idx]


def export_parity(run: str) -> None:
    """Fixture for site/test/flood-sim.test.mjs: maps, Python states and observations for a
    few episodes, and PyTorch logits for some observations. Always in the round-2 world, so
    the spill rule is covered (a round-1 network piles sand against the house and spills)."""
    import torch
    from stable_baselines3 import PPO

    cfg = Config(world=World(house_wall=2))
    model = PPO.load(checkpoints(run)[-1], device="cpu")
    net = MLP(policy_net(model, digits=None)["layers"])
    w = cfg.world
    maps = []
    for seed in [0, 1, 2, 3, 12345, 99999, 2**31 - 1, TEST_SEED, TEST_SEED + 7]:
        m = make_map(seed, w)
        maps.append(
            {
                "seed": seed,
                "sand": m.sand.tolist(),
                "houseFloor": m.house_floor,
                "peak": m.peak,
                "startX": m.start_x,
            }
        )
    rng = np.random.default_rng(5)
    episodes = []
    for name, seed, mode in [
        ("random actions", 11, "random"),
        ("scripted builder", 12, "builder"),
        ("trained network", TEST_SEED + 3, "net"),
        ("doing nothing", 13, "nothing"),
        ("trained network, another map", TEST_SEED + 11, "net"),
    ]:
        env = FloodWorld(cfg)
        env.reset(seed=0, options={"map_seed": seed})
        c = wall_site(env)
        actions, states = [], []
        while True:
            if mode == "random":
                a = int(rng.integers(len(ACTIONS)))
            elif mode == "builder":
                a = builder_action(env, c)
            elif mode == "net":
                a = net.act(observe(env))
            else:
                a = WAIT
            _, r, term, trunc, info = env.step(a)
            actions.append(a)
            if env.t % 10 and not (term or trunc):
                continue  # keep every 10th state (and the last) to keep the fixture small
            states.append(
                {
                    "t": env.t,
                    "sand": env.sand.tolist(),
                    "water": env.water.tolist(),
                    "x": env.x,
                    "carrying": env.carrying,
                    "spilled": env.spilled,
                    "reward": r,
                    "obs": observe(env),
                }
            )
            if term or trunc:
                break
        episodes.append(
            {
                "name": name,
                "seed": seed,
                "actions": actions,
                "states": states,
                "outcome": info["outcome"],
            }
        )
    samples = []
    for e in episodes:
        for st in e["states"][:: max(1, len(e["states"]) // 4)][:4]:
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(st["obs"], dtype=np.float32))[None]
                z = model.policy.action_net(model.policy.mlp_extractor.policy_net(t))[0]
            samples.append({"obs": st["obs"], "logits": z.numpy().astype(float).tolist()})
    fixture = {
        "world": world_json(cfg),
        "maps": maps,
        "episodes": episodes,
        "policy": {"net": policy_net(model, digits=None), "samples": samples},
    }
    (SITE / "test" / "flood-parity.json").write_text(json.dumps(fixture, separators=(",", ":")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--export", metavar="TAG", help="write eval/replays/model-<TAG>.json")
    ap.add_argument("--parity", action="store_true", help="write the JS parity fixture (--live)")
    ap.add_argument("--transfer", nargs="*", help="runs from another world to score in this one")
    ap.add_argument("--model", nargs="*", help="also write model-<run>.json for these runs")
    ap.add_argument("--live", help="run whose final network plays live (default: first run)")
    ap.add_argument("--replay", help="run whose checkpoints are replayed (default: --live)")
    args = ap.parse_args()
    live = args.live or args.runs[0]
    if args.export:
        export(args.export, args.runs, live, args.replay or live, args.transfer)
    if args.parity:
        export_parity(live)
    for r in args.model or []:
        final = checkpoints(r)[-1]
        (OUT / f"model-{r}.json").write_text(
            json.dumps(
                {
                    "world": world_json(run_config(r)),
                    "run": r,
                    "steps": ckpt_steps(final),
                    "net": load_net(final),
                },
                separators=(",", ":"),
            )
        )
    if args.export or args.parity or args.model:
        return
    for r in args.runs:
        for res in eval_run(r):
            s, b = summary(res["episodes"]), summary(res["best"])
            print(
                f"{r} {res['steps']:>10,}  sampled: dry {s['dry']:3d}/{s['n']} "
                f"drops {s['dropped']:5.1f}   top action: dry {b['dry']:3d}/{b['n']}"
            )


if __name__ == "__main__":
    main()
