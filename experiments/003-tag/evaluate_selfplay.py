"""Score the self-play runs and exploiters for post 004, and export the site data.

uv run experiments/003-tag/evaluate_selfplay.py

Every number is from the same 200 start positions as post 003, both sides taking their best
action (as in the browser game), the chaser at 90% speed unless stated.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from evaluate import OUT as OUT003
from evaluate import RUNS, TESTS, play, starts
from tag_env import MLP, NetChaser, TagArena

OUT = OUT003.parent / "004-tag-selfplay"
SEEDS = (1, 2, 3)
RATIO = 0.9


@cache
def net(path: str) -> MLP:
    import torch
    from stable_baselines3 import PPO

    from selfplay import net_of

    torch.set_num_threads(1)
    return net_of(PPO.load(RUNS / path, device="cpu"))


def evader_policy(m: MLP, sample: bool = False):
    rng = np.random.default_rng(0) if sample else None
    return lambda obs, _envs: [m.act(o, rng) for o in obs]


def escape(evader: str, chaser, ratio: float = RATIO, sample: bool = False) -> dict:
    """chaser: a scripted Chaser or a checkpoint path (network chaser at `ratio`).

    sample: the evader samples its actions instead of taking the best one.
    """
    if isinstance(chaser, str):
        chaser = NetChaser(net(chaser), ratio=ratio)
    r = play(evader_policy(net(evader), sample), chaser, starts())
    return {"escaped": r["escaped"], "seconds": r["seconds"]}


def cached(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text())
    res = fn()
    path.write_text(json.dumps(res))
    return res


def ckpts(run: str, prefix: str = "ckpt_") -> list[str]:
    files = sorted(
        (RUNS / run).glob(f"{prefix}*.zip"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])
    )
    return [f"{run}/{p.name}" for p in files]


def exploit_job(xrun: str) -> dict:
    """Escape rate of the exploiter's target at every exploiter checkpoint."""
    meta = json.loads((RUNS / xrun / "exploit.json").read_text())
    target = meta["evader"] + ".zip"

    def one(ck: str):
        steps = int(re.findall(r"\d+", ck)[-1])
        res = cached(
            RUNS / xrun / f"eval_{steps:07d}.json",
            lambda: escape(target, ck, meta["ratio"], meta.get("sample", False)),
        )
        return {"steps": steps, **res}

    return {
        "name": xrun,
        "target": meta["evader"],
        "ratio": meta["ratio"],
        "sample": meta.get("sample", False),
        "checkpoints": [one(c) for c in ckpts(xrun)],
    }


# Self-play stages: (run prefix, number of rounds, chasers each round is scored against).
STAGES = {
    "sp": (20, "r", "selfplay_log.json", [f"sp-s{s}/chaser_r20.zip" for s in SEEDS]),
    "league": (16, "r", "selfplay_log.json", [f"league-s{s}/chaser_r16.zip" for s in SEEDS]),
    "psro": (8, "i", "psro_log.json", [f"psro-s{s}/chaser_i08.zip" for s in SEEDS]),
}


def round_job(args: tuple[str, str, int]) -> dict:
    """One evader round: the scripted tests + every seed's final chaser of the same stage."""
    stage, run, r = args
    evader = f"{run}/evader_{STAGES[stage][1]}{r:02d}.zip"

    def compute():
        res = {"round": r}
        for name, ch in TESTS.items():
            res[name] = escape(evader, ch)
        for k, ch in enumerate(STAGES[stage][3], start=1):
            res[f"chaser-s{k}"] = escape(evader, ch)
        return res

    return cached(RUNS / run / f"eval_round_{r:02d}.json", compute)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xruns = sorted(p.parent.name for p in RUNS.glob("*/exploit.json"))
    jobs = [
        (stage, f"{stage}-s{s}", r)
        for stage, (rounds, *_) in STAGES.items()
        for s in SEEDS
        for r in range(rounds + 1)
    ]
    with ProcessPoolExecutor(max_workers=8) as ex:
        exploits = list(ex.map(exploit_job, xruns))
        scores = list(ex.map(round_job, jobs))
    runs = []
    for stage in STAGES:
        for s in SEEDS:
            name = f"{stage}-s{s}"
            log = json.loads((RUNS / name / STAGES[stage][2]).read_text())
            rounds = [sc for (_, run, _), sc in zip(jobs, scores, strict=True) if run == name]
            runs.append({"name": name, "stage": stage, "seed": s, "log": log, "rounds": rounds})
    data = {"episodes": 200, "ratio": RATIO, "exploits": exploits, "runs": runs}
    (OUT / "eval.json").write_text(json.dumps(data))
    print("wrote", OUT / "eval.json")


def export_models(entries: list[tuple[str, str, str, str]]) -> None:
    """The game's menus: (id, label, checkpoint, role). Evaders drive blue; chasers orange."""
    models = [
        {"id": i, "label": label, "path": p, "role": role, "net": net(p).to_json()}
        for i, label, p, role in entries
    ]
    (OUT / "models.json").write_text(json.dumps({"world": TagArena().describe(), "models": models}))


def export_traces(cases: list[dict], index: int = 0) -> None:
    """Paths for TagTraces: each case = {model, chaser, evader, chaserPath, ratio, sample}."""
    pair = starts()[index : index + 1]
    panels = []
    for c in cases:
        ch = NetChaser(net(c["chaserPath"]), ratio=c["ratio"])
        r = play(evader_policy(net(c["evader"]), c["sample"]), ch, pair, record=1)
        track = [[round(v, 2) for v in s] for s in r["tracks"][0]]
        panels.append(
            {
                "model": c["model"],
                "chaser": c["chaser"],
                "track": track,
                "escaped": r["escaped"] == 1.0,
                "seconds": round((len(track) - 1) * 0.05, 2),
            }
        )
    (OUT / "traces.json").write_text(json.dumps({"world": TagArena().describe(), "panels": panels}))


if __name__ == "__main__":
    main()
