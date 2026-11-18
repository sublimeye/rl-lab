"""Evaluate every checkpoint against fixed test chasers and export data for the post.

uv run experiments/003-tag/evaluate.py simple-s1 simple-s2 simple-s3 mixed-s1 mixed-s2 mixed-s3

Each test plays the same 200 start positions with the evader taking its best action.
Results are cached per checkpoint in runs/<name>/eval_<steps>.json.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tag_env import ACTIONS, Chaser, Config, TagArena, flee_action

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SITE = HERE.parent.parent / "site"
OUT = SITE / "public" / "data" / "003-tag"
EPISODES = 200

# Test chasers. "keyboard" stands in for a person: aims ahead, 8 directions, 0.2 s reactions.
TESTS: dict[str, Chaser] = {
    "pursuit-80": Chaser("pursuit", 0.8),
    "pursuit-90": Chaser("pursuit", 0.9),
    "pursuit-100": Chaser("pursuit", 1.0),
    "intercept-80": Chaser("intercept", 0.8),
    "intercept-90": Chaser("intercept", 0.9),
    "intercept-100": Chaser("intercept", 1.0),
    "keyboard-90": Chaser("intercept", 0.9, reaction=4, dirs8=True),
}


def ckpt_steps(p: Path) -> int:
    return int(re.search(r"(\d+)", p.stem).group(1))


def starts(n: int = EPISODES, seed: int = 2026) -> list[tuple[tuple, tuple]]:
    env = TagArena()
    out = []
    for i in range(n):
        env.reset(seed=seed + i)
        out.append(((env.evader.x, env.evader.y), (env.chaser.x, env.chaser.y)))
    return out


def play(policy, chaser: Chaser, pairs, record: int = 0) -> dict:
    """Run all start pairs at once. policy(obs_batch, envs) -> actions."""
    envs = [TagArena(Config()) for _ in pairs]
    obs = np.stack(
        [
            e.reset(seed=0, options={"evader": ev, "chaser_pos": ch, "chaser": chaser})[0]
            for e, (ev, ch) in zip(envs, pairs, strict=True)
        ]
    )
    alive = np.ones(len(envs), dtype=bool)
    length = np.zeros(len(envs), dtype=int)
    escaped = np.zeros(len(envs), dtype=bool)
    tracks = [[[e.evader.x, e.evader.y, e.chaser.x, e.chaser.y]] for e in envs[:record]]
    while alive.any():
        idx = np.flatnonzero(alive)
        actions = policy(obs[idx], [envs[i] for i in idx])
        for i, a in zip(idx, actions, strict=True):
            o, _, term, trunc, info = envs[i].step(int(a))
            obs[i] = o
            length[i] += 1
            if i < record:
                e = envs[i]
                tracks[i].append([e.evader.x, e.evader.y, e.chaser.x, e.chaser.y])
            if term or trunc:
                alive[i] = False
                escaped[i] = info["outcome"] == "escaped"
    dt = envs[0].cfg.physics.dt
    res = {
        "escaped": float(escaped.mean()),
        "seconds": float(length.mean() * dt),  # mean time before being caught (or 20 s)
        "secondsCaught": float(length[~escaped].mean() * dt) if (~escaped).any() else None,
    }
    if record:
        res["tracks"] = [[[round(v, 3) for v in s] for s in t] for t in tracks]
    return res


def model_policy(model):
    def policy(obs, _envs):
        actions, _ = model.predict(obs, deterministic=True)
        return actions

    return policy


def baseline_policy(kind: str):
    rng = np.random.default_rng(0)

    def policy(obs, envs):
        if kind == "still":
            return [0] * len(envs)
        if kind == "random":
            return rng.integers(len(ACTIONS), size=len(envs))
        return [flee_action(e.evader, e.chaser) for e in envs]

    return policy


def eval_ckpt(path: Path) -> dict:
    cache = path.with_name(f"eval_{ckpt_steps(path):07d}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    model = PPO.load(path, device="cpu")
    pairs = starts()
    res = {"steps": ckpt_steps(path)}
    for name, chaser in TESTS.items():
        r = play(model_policy(model), chaser, pairs)
        res[name] = {k: v for k, v in r.items() if k != "tracks"}
    cache.write_text(json.dumps(res))
    return res


def eval_run(name: str) -> dict:
    ckpts = sorted((RUNS / name).glob("ckpt_*.zip"), key=ckpt_steps)
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(eval_ckpt, ckpts))
    log = json.loads((RUNS / name / "train_log.json").read_text())
    return {
        "name": name,
        "chasers": log["chasers"],
        "rewards": log["rewards"],
        "seed": log["seed"],
        "trainSeconds": log["seconds"],
        "checkpoints": results,
    }


def baselines() -> dict:
    pairs = starts()
    out = {}
    for kind in ("still", "random", "flee"):
        out[kind] = {
            name: {k: v for k, v in play(baseline_policy(kind), ch, pairs).items() if k != "tracks"}
            for name, ch in TESTS.items()
        }
    return out


# -- exports for the site ------------------------------------------------------


def policy_net(model, digits: int | None = 5) -> dict:
    """The actor MLP as plain lists: [{w: out x in, b: out}], tanh between layers."""
    lin = [m for m in model.policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    lin.append(model.policy.action_net)

    def r(a):
        a = a.detach().cpu().numpy().astype(float)
        return np.round(a, digits).tolist() if digits else a.tolist()

    return {"layers": [{"w": r(m.weight), "b": r(m.bias)} for m in lin]}


def export_models(entries: list[tuple[str, str, str, int]]) -> None:
    """entries: (id, label, run/steps) -> models.json for the game."""
    from stable_baselines3 import PPO

    world = TagArena().describe()
    models = []
    for mid, label, run, steps in entries:
        model = PPO.load(RUNS / run / f"ckpt_{steps:07d}.zip", device="cpu")
        models.append(
            {"id": mid, "label": label, "run": run, "steps": steps, "net": policy_net(model)}
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models.json").write_text(json.dumps({"world": world, "models": models}))


def export_parity(run: str, steps: int) -> None:
    """Fixture for site/test/tag-sim.test.mjs: Python states + PyTorch logits."""
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(RUNS / run / f"ckpt_{steps:07d}.zip", device="cpu")
    rng = np.random.default_rng(7)
    cases = [
        ("pursuit, random actions", Chaser("pursuit", 0.9), "random"),
        ("intercept 8-dir, slow reactions", Chaser("intercept", 0.85, 3, True, 1.2), "random"),
        ("policy vs intercept", Chaser("intercept", 1.0), "policy"),
        ("flee into a corner", Chaser("pursuit", 0.8), "flee"),
    ]
    episodes = []
    for i, (name, chaser, mode) in enumerate(cases):
        env = TagArena()
        obs, _ = env.reset(seed=100 + i, options={"chaser": chaser})
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
                "chaser": chaser.__dict__,
                "start": start,
                "actions": actions,
                "states": states,
                "caught": done,
            }
        )
    samples = []
    for obs in rng.uniform(-1, 1, size=(12, 9)):
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32)[None]
            feats = model.policy.mlp_extractor.policy_net(t)
            z = model.policy.action_net(feats)[0].numpy().astype(float).tolist()
        samples.append({"obs": obs.tolist(), "logits": z})
    fixture = {
        "world": TagArena().describe(),
        "episodes": episodes,
        "policy": {"net": policy_net(model, digits=None), "samples": samples},
    }
    (SITE / "test" / "tag-parity.json").write_text(json.dumps(fixture))


def main() -> None:
    names = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [eval_run(n) for n in names]
    for r in runs:
        print(r["name"], "done")
    data = {
        "episodes": EPISODES,
        "tests": {k: v.__dict__ for k, v in TESTS.items()},
        "baselines": baselines(),
        "runs": runs,
    }
    (OUT / "eval.json").write_text(json.dumps(data))
    print("wrote", OUT / "eval.json")


if __name__ == "__main__":
    main()


# -- report exports --------------------------------------------------------------

PLAYABLE = [
    ("untrained", "Untrained", "mixed-s1", 0),
    ("early", "250k steps, mixed chasers", "mixed-s1", 250_000),
    ("mixed", "5M steps, mixed chasers", "mixed-s1", 5_000_000),
    ("simple", "5M steps, one simple chaser", "simple-s1", 5_000_000),
]


def export_curves() -> None:
    """Escape rate per checkpoint, simple vs mixed, one 'run' per seed (EvalChart's shape)."""
    data = json.loads((OUT / "eval.json").read_text())
    by = {(r["chasers"], r["rewards"], r["seed"]): r for r in data["runs"]}
    runs = []
    for seed in (1, 2, 3):
        s, m = by[("simple", "default", seed)], by[("mixed", "default", seed)]
        cks = []
        for cs, cm in zip(s["checkpoints"], m["checkpoints"], strict=True):
            assert cs["steps"] == cm["steps"]
            row = {"steps": cs["steps"]}
            for t in TESTS:
                key = t.replace("-", "")
                row[f"{key}_simple"] = {"success": cs[t]["escaped"]}
                row[f"{key}_mixed"] = {"success": cm[t]["escaped"]}
            cks.append(row)
        runs.append({"name": f"seed {seed}", "checkpoints": cks})
    (OUT / "curves.json").write_text(json.dumps({"runs": runs}))


def export_traces(index: int) -> None:
    """One start position played by both final models against two test chasers."""
    from stable_baselines3 import PPO

    pair = starts()[index : index + 1]
    panels = []
    for run, label in (
        ("simple-s1", "Trained on one simple chaser"),
        ("mixed-s1", "Trained on mixed chasers"),
    ):
        model = PPO.load(RUNS / run / "ckpt_5000000.zip", device="cpu")
        for test, tlabel in (("pursuit-90", "pursuit bot"), ("intercept-90", "intercept bot")):
            r = play(model_policy(model), TESTS[test], pair, record=1)
            track = [[round(v, 2) for v in s] for s in r["tracks"][0]]
            panels.append(
                {
                    "model": label,
                    "chaser": tlabel,
                    "track": track,
                    "escaped": r["escaped"] == 1.0,
                    "seconds": round((len(track) - 1) * 0.05, 2),
                }
            )
    (OUT / "traces.json").write_text(json.dumps({"world": TagArena().describe(), "panels": panels}))


def export_report(trace_index: int) -> None:
    export_curves()
    export_traces(trace_index)
    export_models(PLAYABLE)
    export_parity("mixed-s1", 5_000_000)
