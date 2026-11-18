import numpy as np
from gymnasium.utils.env_checker import check_env

from cube_env import Config, CubeArena, Layout


def test_gym_api_contract():
    check_env(CubeArena(), skip_render_check=True)


def test_walls_block_movement():
    env = CubeArena()
    env.reset(seed=0)
    env.pos = np.array([4.0, 1.0])
    for _ in range(60):
        env.step(4)  # push right into the wall at x=4.8
    assert env.pos[0] <= 4.8 - env.cfg.physics.half


def test_lava_terminates_with_penalty():
    env = CubeArena()
    env.reset(seed=0)
    env.pos = np.array([2.0, 4.0])
    for _ in range(40):
        _, reward, terminated, _, info = env.step(1)  # straight up into the band
        if terminated:
            break
    assert terminated and info["outcome"] == "lava" and reward < 0


def test_coin_collected_once():
    env = CubeArena(Config(layout=Layout(coins=((1.0, 1.0),))))
    env.reset(seed=0)
    env.pos = np.array([1.0, 1.0])
    _, r1, *_ = env.step(0)
    _, r2, *_ = env.step(0)
    assert r1 > 0.9 and r2 < 0


def test_flag_ends_episode():
    env = CubeArena()
    env.reset(seed=0)
    env.pos = np.array(env.cfg.layout.flag)
    _, reward, terminated, _, info = env.step(0)
    assert terminated and info["outcome"] == "flag" and reward > 5


def test_timeout_truncates():
    env = CubeArena(Config(max_steps=5))
    env.reset(seed=0)
    for _ in range(5):
        *_, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and info["outcome"] == "timeout"


def test_start_region_spreads_starts_and_avoids_walls():
    env = CubeArena(Config(start_region=(0.3, 0.3, 9.7, 3.8)))
    xs = []
    for seed in range(200):
        env.reset(seed=seed)
        x, y = env.pos
        assert 0.3 <= x <= 9.7 and 0.3 <= y <= 3.8
        assert not (4.55 < x < 5.45 and y < 2.65)  # wall at x 4.8-5.2, y < 2.4, plus cube size
        xs.append(x)
    assert max(xs) - min(xs) > 8
