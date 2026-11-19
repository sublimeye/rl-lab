"""Score beavers and rivers on 200 held-out boards, export post 009's data and the parity fixture.

uv run experiments/009-beaver/evaluate.py --beavers b-s1 b-s2 b-s3            # curves + table
uv run experiments/009-beaver/evaluate.py --export round1 --beavers b-s1 ...  # site JSON
uv run experiments/009-beaver/evaluate.py --parity b-s1                       # JS fixture

Every beaver meets the same test rivers on the same boards (TEST_SEED + i): the scripted
river in three styles (straight to the den, dodging the beaver, wandering), plus any trained
rivers given with --rivers. Networks sample their moves (as in training) unless noted 'best'.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from beaver_env import (
    B_ACTIONS,
    Board,
    Game,
    Net,
    NetBeaver,
    NetRiver,
    RiverBot,
    Rules,
    make_board,
    observe,
    plan_route,
    river_legal,
    warden_action,
)

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SITE = HERE.parent.parent / "site"
OUT = SITE / "public" / "data" / HERE.name
BOARDS = 200
TEST_SEED = 3_000_000_000  # above anything training draws (< 2**31)
STYLES = ("route", "dodge", "wander")


# -- networks ---------------------------------------------------------------------------------
def ckpt_steps(p: Path) -> int:
    return int(re.search(r"(\d+)", p.stem).group(1))


def checkpoints(run: str) -> list[Path]:
    return sorted((RUNS / run).glob("ckpt_*.zip"), key=ckpt_steps)


def final(run: str) -> str:
    return f"{run}/{checkpoints(run)[-1].stem}"


def half(a: np.ndarray) -> np.ndarray:
    """Rounds weights to float16, the precision the browser gets."""
    return np.asarray(a, dtype=np.float16).astype(np.float64)


def load(ckpt: str) -> dict:
    """A checkpoint's actor, float16-rounded (what the site runs): {"convs"?, "layers"}."""
    from train import load_spec

    spec = load_spec(ckpt)
    return {k: [{"w": half(L["w"]), "b": half(L["b"])} for L in v] for k, v in spec.items()}


def pack(spec: dict) -> dict:
    """float16, little-endian, base64: about a third the size of JSON numbers."""

    def b64(a):
        return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode()

    return {
        k: [
            {"shape": list(np.asarray(L["w"]).shape), "w": b64(L["w"]), "b": b64(L["b"])} for L in v
        ]
        for k, v in spec.items()
    }


# -- players ----------------------------------------------------------------------------------
def test_river(style: str, rng: np.random.Generator):
    if style == "route":
        return RiverBot(rng)
    if style == "dodge":
        return RiverBot(rng, dodge=3.0)
    if style == "wander":
        return RiverBot(rng, noise=2.0, eps=0.3)
    raise ValueError(style)


class LookaheadRiver:
    """A river that plans: for every dig it could make now, it plays the rest of the game out
    `rollouts` times (the same beaver it's facing, then the scripted river) and picks the dig
    that floods the den most often (sooner on ties). A stand-in for a careful human."""

    def __init__(self, beaver, rng: np.random.Generator, rollouts: int = 4):
        self.beaver = beaver
        self.rng = rng
        self.rollouts = rollouts

    def __call__(self, g: Game) -> int:
        best, best_v = -1, -1.0
        for o in g.diggable() + [-1]:
            v = 0.0
            for _ in range(self.rollouts):
                h = g.copy()
                h.dig(o)
                river = RiverBot(self.rng, dodge=3.0)
                first = True
                while not h.done:
                    if not first:
                        h.dig(river(h))
                    first = False
                    h.beaver(self.beaver(h))
                    h.flow()
                v += (1.0 + (h.r.rounds - h.round) / h.r.rounds) if h.flooded else 0.0
            if v > best_v:
                best, best_v = o, v
        return best


def make_beaver(spec, rng: np.random.Generator):
    """spec: 'warden', 'nothing', 'random', or (layers, sample)."""
    if spec == "warden":
        return warden_action
    if spec == "nothing":
        return lambda g: 0
    if spec == "random":

        def rand(g: Game) -> int:
            legal = np.flatnonzero(g.beaver_legal())
            return int(rng.choice(legal))

        return rand
    layers, sample = spec
    return NetBeaver(Net(layers, sample=sample, rng=rng))


def play_game(board: Board, river, beaver, rules: Rules | None = None) -> dict:
    g = Game(board, rules)
    if hasattr(river, "reset"):
        river.reset(g)
    digs, moves, logs = [], [], []
    den = board.den
    n = g.n
    while not g.done:
        g.dig(river(g))
        digs.append(g.last_dig)
        a = beaver(g)
        moves.append(int(a))
        wet_before = list(g.wet)
        g.beaver(a)
        if g.last_log >= 0:
            i = g.last_log
            logs.append(
                {
                    "round": g.round + 1,
                    "wet": wet_before[i],  # dammed water already there (vs dry ground ahead)
                    "fresh": i == g.last_dig,  # plugged the tile the river just dug
                    "fromDen": abs(i % n - den % n) + abs(i // n - den // n),
                    "fromSpring": g.spring_dist[i],
                    # how many water tiles the dam cut off (drained next flow)
                    "cut": 0,
                }
            )
        wet_mid = list(g.wet)
        g.flow()
        if g.last_log >= 0:
            logs[-1]["cut"] = sum(w and not v for w, v in zip(wet_mid, g.wet, strict=True))
    return {
        "seed": board.seed,
        "flooded": g.flooded,
        "round": g.round,
        "dams": g.dams,
        "dug": g.dug,
        "digs": digs,
        "moves": moves,
        "logs": logs,
    }


def _games(args) -> list[dict]:
    """Beaver spec vs one river style (or a river net) on all test boards."""
    beaver_spec, river_spec, keep = args
    out = []
    for i in range(BOARDS):
        rng = np.random.default_rng(i)
        board = make_board(TEST_SEED + i)
        beaver = make_beaver(beaver_spec, rng)
        if river_spec == "lookahead":
            river = LookaheadRiver(beaver, np.random.default_rng(10_000 + i))
        elif isinstance(river_spec, str):
            river = test_river(river_spec, rng)
        else:
            river = NetRiver(Net(river_spec, sample=True, rng=rng))
        res = play_game(board, river, beaver)
        if not keep:
            res.pop("digs")
            res.pop("moves")
        out.append(res)
    return out


def score(games: list[dict]) -> dict:
    flooded = [g for g in games if g["flooded"]]
    return {
        "dry": len(games) - len(flooded),
        "n": len(games),
        "floodRound": round(float(np.mean([g["round"] for g in flooded])), 1) if flooded else None,
        "dams": round(float(np.mean([g["dams"] for g in games])), 2),
        "dug": round(float(np.mean([g["dug"] for g in games])), 1),
        **log_stats([L for g in games for L in g.get("logs", [])]),
    }


def log_stats(logs: list[dict]) -> dict:
    """Where and when the logs went, over all games."""
    if not logs:
        return {"logs": 0}
    m = lambda k: round(float(np.mean([L[k] for L in logs])), 3)  # noqa: E731
    return {
        "logs": len(logs),
        "logWet": m("wet"),
        "logFresh": m("fresh"),
        "logFromDen": m("fromDen"),
        "logRound": m("round"),
        "logFromSpring": m("fromSpring"),
        "logCut": m("cut"),
    }


def evaluate(beavers: dict, rivers: dict, keep: bool = False) -> dict:
    """{beaver: {river: games}} for every pair, in parallel."""
    jobs = [(b, r) for b in beavers for r in rivers]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_games, [(beavers[b], rivers[r], keep) for b, r in jobs]))
    out: dict = {b: {} for b in beavers}
    for (b, r), games in zip(jobs, res, strict=True):
        out[b][r] = games
    return out


def bot_rivers() -> dict:
    return {s: s for s in STYLES}


# -- checkpoint curves --------------------------------------------------------------------------
def curve(run: str, rivers: dict) -> list[dict]:
    """Dry rate per checkpoint (cached per checkpoint in runs/<run>/eval_*.json)."""
    out = []
    for p in checkpoints(run):
        cache = RUNS / run / f"eval_{ckpt_steps(p):08d}.json"
        if cache.exists():
            row = json.loads(cache.read_text())
        else:
            res = evaluate({"net": (load(f"{run}/{p.stem}"), True)}, rivers)["net"]
            row = {"steps": ckpt_steps(p), "vs": {r: score(g) for r, g in res.items()}}
            cache.write_text(json.dumps(row))
        out.append(row)
    return out


def table(beavers: dict, rivers: dict) -> dict:
    res = evaluate(beavers, rivers)
    return {b: {r: score(g) for r, g in v.items()} for b, v in res.items()}


def print_table(t: dict) -> None:
    rivers = list(next(iter(t.values())))
    print(f"{'':24s}" + "".join(f"{r:>12s}" for r in rivers) + f"{'all':>8s}")
    for b, row in t.items():
        dry = sum(row[r]["dry"] for r in rivers)
        n = sum(row[r]["n"] for r in rivers)
        print(
            f"{b:24s}"
            + "".join(f"{row[r]['dry']:>9d}/{row[r]['n']}" for r in rivers)
            + f"{dry / n:8.1%}"
        )


# -- parity fixture -------------------------------------------------------------------------------
def export_parity(run: str, river_ckpt: str | None = None) -> None:
    """Boards, whole games (every state), bot moves and network outputs for the JS test;
    with river_ckpt, also games where that river network digs."""
    river_games: list[dict] = []
    layers = load(final(run))
    net = Net(layers, sample=False)
    boards = []
    for s in list(range(20)) + [TEST_SEED + i for i in range(5)]:
        b = make_board(s)
        boards.append(
            {"seed": s, "tiles": b.tiles, "spring": b.spring, "den": b.den, "start": b.start}
        )
    games = []
    for k in range(6):
        rng = np.random.default_rng(100 + k)
        board = make_board(TEST_SEED + k)
        g = Game(board)
        river = RiverBot(rng, dodge=3.0 if k % 2 else 0.0)
        river.reset(g)
        steps = []
        while not g.done:
            obs = observe(g)
            dig = river(g)
            g.dig(dig)
            obs = observe(g)
            warden = warden_action(g)
            z = net.logits(obs)
            a = net.choose(obs, g.beaver_legal()) if k < 3 else warden
            g.beaver(a)
            g.flow()
            steps.append(
                {
                    "dig": dig,
                    "route": plan_route(g),
                    "warden": warden,
                    "move": a,
                    "logits": [round(float(v), 9) for v in z] if len(steps) % 3 == 0 else None,
                    "obsSum": round(float(np.dot(obs, np.arange(len(obs)) % 7 + 1)), 6),
                    "tiles": "".join(map(str, g.tiles)),
                    "wet": "".join("1" if w else "0" for w in g.wet),
                    "beaver": g.beaver_at,
                    "logs": g.logs,
                    "done": g.done,
                    "flooded": g.flooded,
                }
            )
        games.append({"seed": board.seed, "dodge": 3.0 if k % 2 else 0.0, "steps": steps})
    if river_ckpt:
        rnet = Net(load(river_ckpt), sample=False)
        for k in range(2):
            board = make_board(TEST_SEED + 10 + k)
            g = Game(board)
            steps = []
            while not g.done:
                obs = observe(g)
                z = rnet.logits(obs)
                d = rnet.choose(obs, river_legal(g))
                d = -1 if d == g.n * g.n else d
                g.dig(d)
                a = net.choose(observe(g), g.beaver_legal())
                g.beaver(a)
                g.flow()
                steps.append(
                    {
                        "dig": d,
                        "move": a,
                        "logits": [round(float(v), 9) for v in z] if len(steps) % 3 == 0 else None,
                        "tiles": "".join(map(str, g.tiles)),
                        "wet": "".join("1" if w else "0" for w in g.wet),
                        "beaver": g.beaver_at,
                        "done": g.done,
                    }
                )
            river_games.append({"seed": board.seed, "steps": steps})
    legal = river_legal(Game(make_board(TEST_SEED)))
    fixture = {
        "rules": asdict(Rules()),
        "net": pack(layers),
        "boards": boards,
        "games": games,
        "riverLegal": legal,
        "river": pack(load(river_ckpt)) if river_ckpt else None,
        "riverGames": river_games,
    }
    p = SITE / "test" / "beaver-parity.json"
    p.write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote {p} ({p.stat().st_size / 1e3:.0f} KB)")


# -- site export --------------------------------------------------------------------------------
def model_json(ckpt: str) -> dict:
    return {"rules": asdict(Rules()), "net": pack(load(ckpt)), "ckpt": ckpt}


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"wrote {p.relative_to(SITE)} ({p.stat().st_size / 1e3:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beavers", nargs="*", default=[], help="beaver runs (final checkpoints)")
    ap.add_argument("--rivers", nargs="*", default=[], help="river nets: [label=]run or run/ckpt")
    ap.add_argument("--curves", action="store_true", help="per-checkpoint eval vs the bots")
    ap.add_argument("--best", action="store_true", help="also score each net's top choices")
    ap.add_argument("--lookahead", action="store_true", help="also test vs the planning river")
    ap.add_argument("--export", help="write site/public/data/009-beaver/eval-<tag>.json")
    ap.add_argument("--model", nargs=2, metavar=("CKPT", "NAME"), action="append", default=[])
    ap.add_argument("--parity", help="beaver run for the JS parity fixture")
    ap.add_argument("--parity-river", help="river checkpoint for the parity fixture")
    args = ap.parse_args()

    for ckpt, name in args.model:
        write(f"model-{name}.json", model_json(ckpt if "/" in ckpt else final(ckpt)))
    if args.parity:
        export_parity(args.parity, args.parity_river)
    if not args.beavers:
        return

    rivers = bot_rivers()
    if args.lookahead:
        rivers["lookahead"] = "lookahead"
    for r in args.rivers:
        label, _, ckpt = r.rpartition("=")
        rivers[label or ckpt] = load(ckpt if "/" in ckpt else final(ckpt))
    beavers: dict = {"nothing": "nothing", "random": "random", "warden": "warden"}
    for run in args.beavers:
        layers = load(final(run))
        beavers[run] = (layers, True)
        if args.best:
            beavers[f"{run} best"] = (layers, False)
    t = table(beavers, rivers)
    print_table(t)
    out = {"rules": asdict(Rules()), "rivers": list(rivers), "table": t, "actions": B_ACTIONS}
    if args.curves:
        out["curves"] = {run: curve(run, bot_rivers()) for run in args.beavers}
        for run, rows in out["curves"].items():
            print(run, [(r["steps"], sum(v["dry"] for v in r["vs"].values())) for r in rows])
    if args.export:
        write(f"eval-{args.export}.json", out)


if __name__ == "__main__":
    main()
