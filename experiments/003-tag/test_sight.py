import numpy as np
from gymnasium.utils.env_checker import check_env

import maps
from sight import SIGHTS, Grid, Hider, SeekArena, SeekBot, Still, obs_size, sees
from tag_env import Arena, Config


def _env(sight="map", prey=None):
    return SeekArena(Config(evaders=lambda r: prey or Hider(), maps="random", rays=16), sight)


def test_line_of_sight():
    a = Arena(pillars=((6.0, 4.0, 1.0),), blocks=((2.0, 1.0, 2.3, 7.0),))
    assert not sees(4, 4, 8, 4, a)  # through the pillar
    assert sees(4, 6, 8, 6, a)  # above it
    assert not sees(1, 4, 3, 4, a)  # through the wall
    assert sees(1, 7.5, 3, 7.5, a)  # past its end
    assert sees(3, 3, 3, 3, a)


def test_vectorised_sight_matches_scalar():
    for fam in maps.FAMILIES:
        for seed in range(1, 6):
            a = maps.generate(seed, fam)
            g = Grid(maps.nav_for(a))
            rng = np.random.default_rng(seed)
            for _ in range(4):
                x, y = rng.uniform(0.3, 11.7), rng.uniform(0.3, 7.7)
                got = g._sees_points(x, y)
                want = [sees(x, y, px, py, a) for px, py in zip(g.px, g.py, strict=True)]
                assert got.tolist() == want


def test_envs_follow_the_gym_api():
    for s in SIGHTS:
        env = _env(s)
        check_env(env, skip_render_check=True)
        assert env.observation_space.shape == (obs_size(s),)
    assert [obs_size(s) for s in SIGHTS] == [26, 26, 32, 49]


def test_xray_always_sees_the_runner():
    env = _env("xray", Still())
    o, _ = env.reset(seed=3)
    assert o[20] == 1.0  # the "can see it" input
    env = _env("none", Still())
    o, _ = env.reset(seed=3)
    assert o[20] == 0.0  # a still runner starts out of sight


def test_the_map_keeps_track_of_a_hidden_runner():
    """While the chaser can't see it, the runner's cell stays on the possibility map."""
    hidden = missed = 0
    env = _env("map", Still())
    for seed in range(40):
        env.reset(seed=seed, options={"bot": SeekBot("search"), "ratio": 0.9})
        for _ in range(200):
            _, _, term, trunc, _ = env.step(0)
            if not env.tracker.seen:
                hidden += 1
                missed += not env.tracker.possible[env.grid.nav.cell_of(env.evader.x, env.evader.y)]
            if term or trunc:
                break
    assert hidden > 500
    assert missed / hidden < 0.01


def test_search_bot_finds_a_still_runner():
    env = _env("map", Still())
    found = 0
    for seed in range(30):
        env.reset(seed=100 + seed, options={"bot": SeekBot("search"), "ratio": 0.9})
        for _ in range(400):
            _, _, term, trunc, _ = env.step(0)
            if env.tracker.seen:
                found += 1
                break
    assert found >= 29
