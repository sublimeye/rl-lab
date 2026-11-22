from dataclasses import replace

import numpy as np

from gauntlet_env import (
    BIRD,
    BIRD1,
    CLEAR,
    ENCOUNTERS,
    FAR,
    FLYING,
    FULL,
    HIST,
    K_FROGS,
    LANDED,
    MISSILE,
    MISSILE1,
    N_BASE,
    N_FRAME,
    N_HIST,
    N_SLOT,
    ROUGH,
    SIGHT,
    Config,
    Fleet,
    FleetEnv,
    base_inputs,
    make_course,
)
from scripted import cruise_pilot, escapable, fly_snapshots


def only(k):
    mix = [0.0] * len(ENCOUNTERS)
    mix[k] = 1.0
    return replace(CLEAR, mix=tuple(mix))


def fleet(feat, seeds):
    f = Fleet(len(seeds), feat=feat)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(s, ROUGH, feat, f.ph))
    return f


def fly(f, pilot=cruise_pilot, updates=1300):
    for _ in range(updates):
        if not (f.outcome == FLYING).any():
            break
        f.step(pilot(f))
    return f


def test_courses_are_long_and_repeatable():
    for s in range(30):
        a = make_course(s, ROUGH, FULL, Fleet(1).ph)
        assert a == make_course(s, ROUGH, FULL, Fleet(1).ph)
        assert 60 <= a["length"] <= 120
        gap = abs((a["target"][0] + a["target"][1]) / 2 - (a["start"][0] + a["start"][1]) / 2)
        assert abs(gap - a["length"]) < 1e-9
        ats = [e[0] for e in a["encounters"]]
        assert ats == sorted(ats) and all(at < a["length"] - 12 for at in ats)


def test_a_scripted_pilot_can_fly_the_clear_course():
    f = fly(fleet(CLEAR, range(100, 200)))
    assert (f.outcome == LANDED).sum() >= 90


def test_the_pad_is_only_a_direction_until_in_sight():
    f = fleet(CLEAR, [3])
    b = base_inputs(f)[0]
    gap = abs(f.goal()[0][0] - f.x[0])
    assert gap > SIGHT
    assert b[0] == np.sign(f.goal()[0][0] - f.x[0]) * SIGHT / 10
    assert b[1] == 0.0 and b[14] == 0.0 and b[17] == f.dir[0]


def test_aimed_birds_hit_a_pilot_that_ignores_them_and_can_be_dodged():
    f = fleet(only(BIRD1), range(40))
    snaps = fly_snapshots(f, cruise_pilot)
    hit = np.nonzero(f.outcome == BIRD)[0]
    assert len(hit) >= 36
    assert escapable(snaps, f, hit[:10], lead=0.5).all()


def test_the_missile_hits_a_pilot_that_ignores_it():
    f = fly(fleet(only(MISSILE1), range(40)))
    assert (f.outcome == MISSILE).sum() >= 36


def test_take_copies_mid_flight():
    f = fleet(FULL, range(8))
    for _ in range(200):
        f.step(cruise_pilot(f))
    g = f.take(np.arange(8))
    for _ in range(100):
        a = cruise_pilot(f)
        f.step(a)
        g.step(a)
    assert np.array_equal(f.x, g.x) and np.array_equal(f.p_x, g.p_x)
    assert np.array_equal(f.outcome, g.outcome)


def test_observation_layout_and_blindfold():
    env = FleetEnv(16, Config(features=FULL), seed=3)
    obs = env.start(range(16))
    assert obs.shape == (16, N_FRAME * 3 + HIST * K_FROGS * N_HIST)
    blind = FleetEnv(16, Config(features=FULL), seed=3, hide=("threats", "frogs"))
    ob = blind.start(range(16))
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = rng.uniform(-0.2, 0.6, (16, 2))
        obs, *_ = env.step(a)
        ob, *_ = blind.step(a)
    seen = obs[:, N_BASE : N_BASE + 4 * N_SLOT].reshape(16, 4, N_SLOT)
    hidden = ob[:, N_BASE : N_BASE + 4 * N_SLOT].reshape(16, 4, N_SLOT)
    assert seen[..., 4].any()  # something was in sight
    assert (hidden[..., 4] == 0).all() and np.allclose(hidden[..., 0], FAR, atol=0.1)
    assert (ob[:, N_FRAME * 3 :] == 0).all()  # no frog history either
