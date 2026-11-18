import math

import numpy as np
from gymnasium.utils.env_checker import check_env

import maps
from tag_env import Arena, Body, Chaser, Config, Physics, TagArena, flee_action, move


def test_mulberry32_known_values():
    # Same numbers as the JavaScript version (see site/test/tag-sim.test.mjs).
    rand = maps.mulberry32(1)
    assert [round(rand(), 10) for _ in range(3)] == [0.6270739406, 0.0027357212, 0.52744704]


def test_every_family_makes_valid_maps():
    for fam in maps.FAMILIES:
        for seed in range(150):
            a = maps.generate(seed, fam)
            assert a.pillars or a.blocks
            for cx, cy, r in a.pillars:
                assert maps.GAP - 1e-9 <= cx - r and cx + r <= a.width - maps.GAP + 1e-9
                assert maps.GAP - 1e-9 <= cy - r and cy + r <= a.height - maps.GAP + 1e-9
            for x0, y0, x1, y1 in a.blocks:
                assert maps.GAP - 1e-9 <= x0 < x1 <= a.width - maps.GAP + 1e-9
                assert maps.GAP - 1e-9 <= y0 < y1 <= a.height - maps.GAP + 1e-9
            assert maps.navigable(a)


def test_obstacles_keep_their_distance():
    """Separate obstacles never come closer than GAP, so a square can pass between any two."""
    for seed in range(100):
        a = maps.generate(seed, "mixed")
        circles = [("c", *p) for p in a.pillars]
        for i, p in enumerate(circles):
            for q in circles[i + 1 :]:
                assert maps._prim_gap(p, q) >= maps.GAP - 1e-9


def test_navigable_rejects_a_closed_room():
    # Four walls around the middle: the inside can't be reached from the outside.
    box = Arena(
        pillars=(),
        blocks=((4, 2, 8, 2.3), (4, 5.7, 8, 6), (4, 2, 4.3, 6), (7.7, 2, 8, 6)),
    )
    assert not maps.navigable(box)
    assert maps.navigable(Arena(pillars=(), blocks=((4, 2, 8, 2.3),)))


def test_block_stops_a_square():
    p = Physics()
    a = Arena(pillars=(), blocks=((5.0, 3.0, 6.0, 5.0),))
    b = Body(3.0, 4.0)
    for _ in range(80):
        move(b, 1, 0, p.max_speed, p, a)
    assert abs(b.x + p.half - 5.0) < 1e-9 and b.vx == 0


def test_rays_measure_distances():
    a = Arena(pillars=((9.0, 4.0, 1.0),), blocks=((5.0, 6.0, 7.0, 7.0),))
    r = maps.rays(6.0, 4.0, a, n=16, reach=6.0)
    assert math.isclose(r[0] * 6, 2.0)  # east: pillar surface at x = 8
    assert math.isclose(r[4] * 6, 2.0)  # north: block bottom at y = 6
    assert math.isclose(r[8] * 6, 6.0)  # west: wall at x = 0 is 6 away, the cap
    assert math.isclose(r[12] * 6, 4.0)  # south: wall at y = 0


def test_nav_reaches_every_open_cell():
    for fam in maps.FAMILIES:
        for seed in range(20):
            g = maps.NavGrid(maps.generate(seed, fam))
            free = g.free.reshape(-1)
            assert (g.dist[np.ix_(free, free)] < maps.UNREACHED).all()


def test_bots_catch_on_every_kind_of_map():
    for fam in maps.FAMILIES:
        env = TagArena(Config(maps=fam, rays=16))
        for seed in range(15):
            env.reset(seed=seed, options={"chaser": Chaser("pursuit", 0.9), "map": (seed, fam)})
            for _ in range(400):
                _, _, term, trunc, info = env.step(flee_action(env.evader, env.chaser))
                if term or trunc:
                    break
            assert info["outcome"] == "caught", (fam, seed)


def test_map_env_api():
    env = TagArena(Config(maps="random", rays=16))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (25,) and env.map_id is not None
    first = env.map_id
    for _ in range(env.cfg.map_repeat - 1):
        env.reset()
        assert env.map_id == first
    env.reset()
    assert env.map_id != first
