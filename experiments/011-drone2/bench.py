"""Raw simulator speed, post 10's one-drone-at-a-time env vs the vectorised fleet (same task).

uv run experiments/011-drone2/bench.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "010-drone"))


def basic(n_envs: int, subproc: bool, seconds: float = 4.0) -> float:
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from drone_env import DroneFlight

    env = make_vec_env(
        DroneFlight, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv if subproc else DummyVecEnv
    )
    env.reset()
    rng = np.random.default_rng(0)
    steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        env.step(np.clip(rng.normal([0.3, 0], 0.6, (n_envs, 2)), -1, 1))
        steps += n_envs
    rate = steps / (time.time() - t0)
    env.close()
    return rate


def fleet(n: int, features, seconds: float = 4.0) -> float:
    from fleet_env import Config, FleetEnv

    env = FleetEnv(n, Config(features=features), seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        env.step(np.clip(rng.normal([0.3, 0], 0.6, (n, 2)), -1, 1))
        steps += n
    return steps / (time.time() - t0)


def main() -> None:
    from fleet_env import FULL, FULL_VISION, V010

    out = {
        "basic, 1 process, 16 drones": basic(16, False),
        "basic, 16 processes (post 10)": basic(16, True),
    }
    for n in (16, 256, 1024, 4096):
        out[f"vectorised, {n} drones"] = fleet(n, V010)
    for n in (256, 1024):
        out[f"vectorised + post 11 features (state), {n}"] = fleet(n, FULL)
        out[f"vectorised + post 11 features (vision), {n}"] = fleet(n, FULL_VISION)
    for k, v in out.items():
        print(f"{k:<48} {v:>10,.0f} steps/s")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "bench.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
