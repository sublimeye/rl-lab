import numpy as np
from gymnasium.utils.env_checker import check_env

from flood_env import (
    DIG_L,
    DIG_R,
    DROP_L,
    DROP_R,
    LEFT,
    RIGHT,
    WAIT,
    Config,
    FloodWorld,
    Map,
    World,
    builder_action,
    flow_water,
    make_map,
    settle,
    wall_site,
)


def flat_map(height: int = 4, floor: int = 6, peak: float = 8.0, x: int = 20) -> Map:
    w = World()
    sand = np.full(w.width, height, dtype=np.int64)
    sand[w.width - w.house_width :] = floor
    return Map(sand, floor, peak, x)


def play(env: FloodWorld, policy) -> dict:
    while True:
        _, _, term, trunc, info = env.step(policy(env))
        if term or trunc:
            return info


def test_gym_api_contract():
    check_env(FloodWorld(), skip_render_check=True)


def test_doing_nothing_always_floods():
    for seed in range(200):
        env = FloodWorld()
        env.reset(seed=seed)
        assert play(env, lambda e: WAIT)["outcome"] == "flooded", seed


def test_terrain_stays_below_the_tide():
    w = World()
    for seed in range(500):
        m = make_map(seed, w)
        n = w.width - w.house_width
        assert m.sand[:n].max() <= m.peak - 0.5
        assert (m.sand[n:] == m.house_floor).all()
        assert m.peak > m.house_floor
        assert np.abs(np.diff(m.sand[:n])).max() <= 1  # generated terrain is gentle


def test_scripted_builder_usually_keeps_the_house_dry():
    dry = 0
    for seed in range(100):
        env = FloodWorld()
        env.reset(seed=seed)
        c = wall_site(env)
        dry += play(env, lambda e, c=c: builder_action(e, c))["outcome"] == "dry"
    assert dry >= 70


def test_settle_conserves_sand_and_respects_the_angle_of_repose():
    sand = np.array([1, 9, 1, 1, 6, 6, 1, 2, 9, 9], dtype=np.int64)
    total = sand.sum()
    for _ in range(50):
        settle(sand, len(sand), 2)
    assert sand.sum() == total
    assert np.abs(np.diff(sand)).max() <= 2


def test_settle_leaves_columns_past_n_alone():
    sand = np.array([1, 1, 1, 9, 9], dtype=np.int64)
    for _ in range(10):
        settle(sand, 3, 2)
    assert sand.tolist() == [1, 1, 1, 9, 9]


def test_water_is_conserved_away_from_the_sea():
    w = World()
    rng = np.random.default_rng(1)
    sand = rng.integers(1, 8, size=w.width).astype(np.int64)
    sand[0] = 20  # a high wall at the sea: nothing in or out
    water = rng.uniform(0, 3, size=w.width)
    water[0] = 0.0
    flux = np.zeros(w.width)
    total = water.sum()
    for _ in range(500):
        water, flux = flow_water(sand, water, flux, 0.0, w)
        assert (water >= 0).all()
    assert abs(water.sum() - total) < 1e-9


def test_water_settles_level():
    w = World()
    sand = np.full(w.width, 2, dtype=np.int64)
    sand[0] = 30
    water = np.zeros(w.width)
    water[1:6] = 4.0
    flux = np.zeros(w.width)
    for _ in range(4000):
        water, flux = flow_water(sand, water, flux, 0.0, w)
    surf = sand[1:] + water[1:]
    assert surf.max() - surf.min() < 0.01


def test_a_wall_above_the_sea_holds_with_no_seepage():
    w = World()
    sand = np.full(w.width, 2, dtype=np.int64)
    sand[10] = 9
    water = np.zeros(w.width)
    flux = np.zeros(w.width)
    for i in range(3000):  # a tide rising to 8, a cell below the wall's top
        water, flux = flow_water(sand, water, flux, min(8.0, 2.0 + i / 300), w)
    assert water[:10].min() > 5.0  # the sea side is full
    assert water[10:].max() == 0.0  # not a drop behind the wall


def test_digger_climbs_at_most_max_step():
    env = FloodWorld()
    m = flat_map(x=20)
    m.sand[21] = 7  # a 3-cell step
    m.sand[19] = 6  # a 2-cell step
    env.reset(options={"map": m})
    env.step(RIGHT)
    assert env.x == 20
    env.step(LEFT)
    assert env.x == 19


def test_dig_and_drop_move_one_grain_and_respect_carry():
    env = FloodWorld()
    env.reset(options={"map": flat_map(x=20)})
    before = env.sand.sum()
    for _ in range(5):
        env.step(DIG_R)
    assert env.carrying == env.cfg.world.carry
    assert env.sand.sum() + env.carrying == before
    env.step(DROP_L)
    assert env.carrying == env.cfg.world.carry - 1
    assert env.sand.sum() + env.carrying == before


def test_house_columns_cannot_be_dug_walked_into_or_built_on():
    env = FloodWorld()
    w = env.cfg.world
    n = w.width - w.house_width
    m = flat_map(height=6, floor=6, x=n - 1)
    env.reset(options={"map": m})
    env.step(DIG_R)
    assert env.carrying == 0
    env.step(RIGHT)
    assert env.x == n - 1
    env.step(DIG_L)
    env.step(DROP_R)
    assert env.carrying == 1
    assert (env.sand[n:] == 6).all()


def test_round_one_house_holds_any_pile_against_it():
    env = FloodWorld()
    n = env.n
    m = flat_map(height=6, floor=6, x=n - 2)
    env.reset(options={"map": m})
    for _ in range(3):
        env.sand[n - 1] = 14
        env.step(WAIT)
    assert env.sand[n - 1] >= 12 and env.spilled == 0


def test_round_two_sand_above_the_house_wall_spills():
    env = FloodWorld(Config(world=World(house_wall=2)))
    n = env.n
    env.reset(options={"map": flat_map(height=6, floor=6, x=n - 3)})
    env.sand[n - 1] = 11
    env.sand[n - 2] = 9
    before = env.sand.sum()
    for _ in range(20):
        env.step(WAIT)
    assert env.sand[n - 1] <= 6 + 2
    assert env.sand.sum() + env.spilled == before and env.spilled > 0


def test_bedrock_cannot_be_dug():
    env = FloodWorld()
    m = flat_map(height=1, x=20)
    env.reset(options={"map": m})
    env.step(DIG_L)
    assert env.carrying == 0 and env.sand[19] == 1


def test_reward_is_one_for_a_dry_episode_and_flood_ends_it():
    env = FloodWorld(Config(max_steps=50))
    env.reset(options={"map": flat_map(height=4, floor=6, peak=5.0)})
    total, info = 0.0, {}
    for _ in range(50):
        _, r, term, trunc, info = env.step(WAIT)
        total += r
        if term or trunc:
            break
    assert trunc and info["outcome"] == "dry" and abs(total - 1.0) < 1e-9

    env = FloodWorld()
    env.reset(options={"map": flat_map(height=4, floor=6, peak=8.0)})
    _, r, term, _, info = env.step(WAIT)
    while not term:
        _, r, term, _, info = env.step(WAIT)
    assert info["outcome"] == "flooded" and r == env.cfg.rewards.flood


def test_actions_out_of_reach_do_nothing():
    env = FloodWorld()
    env.reset(options={"map": flat_map(x=0)})
    env.step(LEFT)
    env.step(DIG_L)
    assert env.x == 0 and env.carrying == 0
    env.step(DROP_R)  # nothing to drop
    assert env.sand[1] == 4
