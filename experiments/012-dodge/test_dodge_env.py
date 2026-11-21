import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from dodge_env import (
    ARENA_T,
    BIRD,
    BIRD_MARGIN,
    FLYING,
    FULL,
    FULL_VISION,
    K_SEE,
    LEG,
    MISSILE,
    N_RAYS,
    N_STATE,
    N_VISION,
    NONE,
    ROCK,
    ROCK_TOP,
    ROUGH,
    SCHOOL,
    TIMEOUT,
    Config,
    Fleet,
    FleetEnv,
    Physics,
    Reward,
    make_course,
    obs_state,
    rays,
    rewards,
    threat_slots,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "011-drone2"))
import fleet_env as f11  # noqa: E402


def test_same_as_post_11_without_threats():
    """No threats: post 11's fleet and this one fly the same courses number for number."""
    feat = replace(FULL, threats=NONE)
    feat11 = replace(f11.FULL, birds=False)
    seeds = [7000 + i for i in range(40)]
    a, b = Fleet(len(seeds), feat=feat), f11.Fleet(len(seeds), feat=feat11)
    for i, s in enumerate(seeds):
        c = make_course(s, ROUGH, feat, a.ph)
        a.set_course(i, c)
        b.set_course(i, {**c, "birds": []})
    rng = np.random.default_rng(3)
    for _ in range(400):
        act = np.clip(np.array([0.1, 0.0]) + rng.normal(0, 0.4, (len(seeds), 2)), -1, 1)
        a.step(act)
        b.step(act)
        for k in ("x", "y", "vx", "vy", "th", "om", "motor", "charge", "outcome", "ticks"):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k
        assert np.array_equal(obs_state(a, None)[:, :16], f11.obs_state(b, None)[:, :16])


def one(course: dict, feat=FULL) -> Fleet:
    f = Fleet(1, feat=feat)
    f.set_course(0, course)
    return f


def plain(**kw) -> dict:
    c = {
        "start": (2.0, 4.0, 1.0),
        "target": (16.0, 18.0, 1.0),
        "wind": 0.0,
        "gust": 0.0,
        "period": 3.0,
        "phase": 0.0,
        "mass": 1.0,
        "amp": 0.0,
        "pad_period": 6.0,
        "pad_phase": 0.0,
        "arena": False,
        "goal": (12.0, 6.0),
        "birds": [],
        "rocks": [],
        "missile": None,
        "charge": 1.0,
    }
    return {**c, **kw}


def sit(f: Fleet, n: int, stop=True):
    for _ in range(n):
        f.step(np.array([[-1.0, 0.0]]))
        if stop and f.outcome[0] != FLYING:
            break


def test_courses_keep_earlier_draws():
    for s in range(50):
        a = f11.make_course(s, f11.ROUGH, f11.FULL, f11.Physics())
        b = make_course(s, ROUGH, FULL, Physics())
        if not b["arena"]:
            for k in ("start", "target", "wind", "mass", "amp", "pad_phase"):
                assert a[k] == b[k], k
        assert len(b["birds"]) <= 3 and len(b["rocks"]) <= 2


def test_bird_comes_back_from_beyond_the_edge():
    # flying left from x = 0.5: after 1 s it's 2.5 m off-screen, then re-enters 6 m right of it
    f = one(plain(birds=[(10.0, -3.0, BIRD_MARGIN + 0.5, 0.0, 2.0, 0.0)]))
    sit(f, 20)
    x = f.threats()[0][0, 0]
    assert abs(x - (0.5 - 3.0)) < 1e-9
    sit(f, 40)  # 3 s: 9 m travelled, so it went out at -6 and came back in at 30
    x = f.threats()[0][0, 0]
    assert abs(x - (f.ph.width + BIRD_MARGIN - 2.5)) < 1e-9


def test_bird_hit():
    f = one(plain(birds=[(1.0 + LEG, -3.0, BIRD_MARGIN + 7.0, 0.0, 2.0, 0.0)]))
    sit(f, 60)
    assert f.outcome[0] == BIRD


def test_rock_drops_on_the_beat_and_falls():
    # c = 0.5: the first rock is aimed right at the drone sitting on its pad
    f = one(plain(rocks=[(3.0, 0.5, 0.0, 0.5)]))
    sit(f, 9)  # 0.45 s: not yet
    assert not f.threats()[4][0, 4]
    sit(f, 2)
    x, y, _, vy, on = f.threats()
    assert on[0, 4] and abs(x[0, 4] - 3.0) < 1e-9 and y[0, 4] < ROCK_TOP and vy[0, 4] < 0
    sit(f, 60)
    assert f.outcome[0] == ROCK
    t = f.ticks[0] * f.ph.dt
    assert abs(0.5 + math.sqrt(2 * (ROCK_TOP - 1.0 - LEG - 0.5) / 9.81) - t) < 0.1


def test_missile_homes_in_and_runs_out_of_fuel():
    f = one(plain(missile=(0.5, 1.0, 5.0, 11.0)))  # from the right, 5 m up, at 0.5 s
    sit(f, 80)
    assert f.outcome[0] == MISSILE
    # the same missile against a drone far away runs out of fuel
    g = one(plain(missile=(0.5, 0.0, 5.0, 11.0), start=(60.0, 62.0, 1.0)))
    g.ph = replace(g.ph, width=80.0)
    for _ in range(120):
        g.step(np.array([[-1.0, 0.0]]))
    assert g.m_state[0] == 2 and g.outcome[0] == FLYING


def test_arena_has_no_landing_and_ends_held():
    f = one(plain(arena=True, target=(-100.0, -100.0, 0.0)))
    assert f.end_tick[0] == round(ARENA_T / f.ph.dt)
    gx, gy = f.goal()
    assert (gx[0], gy[0]) == (12.0, 6.0)
    sit(f, 400)
    assert f.outcome[0] == TIMEOUT
    # holding the point pays; far away doesn't
    f.outcome[0] = FLYING
    f.x[0], f.y[0] = 12.5, 6.0
    r = rewards(f, f.distance(), Reward(), np.array([True]))
    assert r[0] == Reward().hold


def test_most_urgent_threat_comes_first():
    # a slow bird 3 m away drifting off, and a missile 8 m away coming fast
    f = one(plain(birds=[(1.0 + LEG, 1.0, BIRD_MARGIN + 6.0, 0.0, 2.0, 0.0)]))
    f.m_state[0] = 1
    f.m_x[0], f.m_y[0], f.m_psi[0] = 11.0, 1.0 + LEG, math.pi
    s = threat_slots(f)
    assert s.shape == (1, K_SEE, 8)
    assert s[0, 0, 7] == 1 and s[0, 1, 5] == 1  # missile first, then the bird
    assert s[0, 2, 4] == 0 and s[0, 2, 0] == 2.0  # empty slots read as far away
    hidden = threat_slots(f, hide=("missile",))
    assert hidden[0, 0, 5] == 1 and hidden[0, 1, 4] == 0


def test_camera_sees_pad_threat_and_shadow():
    f = one(plain(), FULL_VISION)
    f.x[0], f.y[0], f.grounded[0] = 17.0, 6.0, 0
    f._pad(np.array([0]))
    depth, pad, threat, shadow = rays(f)
    down = N_RAYS // 2
    assert pad[0, down] == 1 and abs(depth[0, down] * 15.0 - 5.0) < 0.2
    assert threat.sum() == 0 and shadow.sum() == 0
    # a rock overhead, falling: the camera can't see it, but its shadow on the pad below
    f.rock_on[0, 0], f.rock_n[0, 0] = True, 1
    f.rock_t[0, 0], f.rock_x[0, 0] = f.ticks[0] * f.ph.dt, 17.0
    depth, pad, threat, shadow = rays(f)
    assert threat.sum() == 0 and shadow[0, down] == 1 and shadow[0, down - 1] == 1


def test_take_copies():
    f = one(plain(rocks=[(3.0, 0.5, 0.0, 0.5)]))
    sit(f, 15)
    g = f.take([0, 0])
    g.step(np.array([[1.0, 0.0], [1.0, 1.0]]))
    assert f.ticks[0] == 45 and g.ticks.tolist() == [48, 48] and g.x[0] != g.x[1]


def test_delay_and_memory():
    feat = replace(FULL, noise=0.0, threats=NONE, pad_move=0.0)
    env = FleetEnv(2, Config(features=feat), seed=0)
    first = env.start([11, 12])
    assert first.shape == (2, 3 * N_STATE)
    o1, _, _, _ = env.step(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(o1[:, :N_STATE], first[:, :N_STATE])  # 50 ms late
    o2, _, _, _ = env.step(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(o2[:, N_STATE : 2 * N_STATE], o1[:, :N_STATE])
    assert not np.allclose(o2[:, :N_STATE], o1[:, :N_STATE])


def test_env_shapes_and_restarts():
    for base, width in ((FULL, N_STATE), (FULL_VISION, N_VISION)):
        feat = replace(base, threats=SCHOOL)
        env = FleetEnv(16, Config(features=feat), seed=1)
        obs = env.reset()
        assert obs.shape == (16, width * feat.stack)
        finished = 0
        rng = np.random.default_rng(0)
        for _ in range(300):
            obs, r, done, info = env.step(rng.uniform(-1, 1, (16, 2)))
            assert np.isfinite(obs).all() and np.isfinite(r).all()
            finished += int(done.sum())
            for i in np.nonzero(done)[0]:
                assert "outcome" in info[i] and "terminal_observation" in info[i]
        assert finished > 0
        assert (env.fleet.outcome == FLYING).all()
