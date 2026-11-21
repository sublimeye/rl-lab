import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from fleet_env import (
    BIRD,
    CALM,
    FLYING,
    FULL,
    FULL_VISION,
    LANDED,
    LEG,
    N_RAYS,
    N_STATE,
    N_VISION,
    ROUGH,
    V010,
    Config,
    Features,
    Fleet,
    FleetEnv,
    Physics,
    make_course,
    obs_v010,
    rays,
    thrust_scale,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "010-drone"))
import drone_env as d10  # noqa: E402


def test_same_as_post_10_with_features_off():
    """Vectorised and one-at-a-time drones fly the same, number for number."""
    for w10, w11 in ((d10.CALM, CALM), (d10.ROUGH, ROUGH)):
        seeds = [7000 + i for i in range(40)]
        fleet = Fleet(len(seeds))
        for i, s in enumerate(seeds):
            fleet.set_course(i, make_course(s, w11, V010, fleet.ph))
        drones = [d10.Drone(d10.make_course(s, w10)) for s in seeds]
        rng = np.random.default_rng(3)
        for _ in range(400):
            a = np.zeros((len(seeds), 2))
            for i, d in enumerate(drones):
                if not d.done:
                    a[i] = d10.autopilot(d)
            a = np.clip(a + rng.normal(0, 0.3, a.shape), -1, 1)
            for i, d in enumerate(drones):
                if not d.done:
                    d.step(*a[i])
            fleet.step(a)
            obs = obs_v010(fleet)
            for i, d in enumerate(drones):
                assert (d.x, d.y, d.vx, d.vy, d.th, d.motor) == (
                    fleet.x[i],
                    fleet.y[i],
                    fleet.vx[i],
                    fleet.vy[i],
                    fleet.th[i],
                    fleet.motor[i],
                )
                assert np.array_equal(np.array(d10.observe(d)), obs[i])
        names = ["flying", "landed", "crashed", "water", "lost", "timeout", "bird"]
        assert [d.outcome for d in drones] == [names[o] for o in fleet.outcome]


def one(course: dict, feat=FULL) -> Fleet:
    f = Fleet(1, feat=feat)
    f.set_course(0, course)
    return f


def plain(**kw) -> dict:
    c = {
        "start": (2.0, 4.0, 1.0),
        "target": (11.0, 13.0, 1.0),
        "wind": 0.0,
        "gust": 0.0,
        "period": 3.0,
        "phase": 0.0,
        "mass": 1.0,
        "amp": 0.0,
        "pad_period": 6.0,
        "pad_phase": 0.0,
        "birds": [],
        "charge": 1.0,
    }
    return {**c, **kw}


def test_courses_keep_post_10_pillars():
    for s in range(50):
        a = make_course(s, ROUGH, V010, Physics())
        b = make_course(s, ROUGH, FULL, Physics())
        assert (a["start"], a["target"], a["wind"], a["mass"]) == (
            b["start"],
            b["target"],
            b["wind"],
            b["mass"],
        )
        assert len(b["birds"]) <= 3 and 0.06 <= b["charge"] <= 1.0
        assert b["amp"] == 0.0 or b["amp"] >= 0.0


def test_pad_moves_and_landing_is_relative_to_it():
    f = one(plain(amp=2.0, pad_period=6.0))
    x0 = f.pad_x0[0]
    f.step(np.array([[-1.0, 0.0]]))
    assert f.pad_x0[0] != x0 and abs(f.pad_vx[0]) > 0
    # set down gently on the deck while moving with it
    f.grounded[0] = 0
    f.x[0] = (f.pad_x0[0] + f.pad_x1[0]) / 2
    f.y[0] = 1.0 + LEG + 0.05
    f.vx[0] = f.pad_vx[0]
    f.vy[0] = -0.5
    f.motor[0] = 0.5
    for _ in range(10):
        f.step(np.array([[0.0, 0.0]]))
    assert f.outcome[0] == LANDED


def test_bird_hit_ends_the_flight():
    # a bird at the height of the drone sitting on its pad, flying straight at it
    f = one(plain(birds=[(1.0 + LEG, -3.0, 7.0, 0.0, 2.0, 0.0)]))
    for _ in range(40):
        f.step(np.array([[-1.0, 0.0]]))
        if f.outcome[0] != FLYING:
            break
    assert f.outcome[0] == BIRD


def test_battery_sags_only_near_empty():
    assert thrust_scale(1.0) == 1.0 - 0.2 * math.exp(-1 / 0.06)
    assert thrust_scale(0.5) > 0.98 and thrust_scale(0.2) > 0.96
    assert thrust_scale(0.05) < 0.9
    f = one(plain(charge=0.5))
    for _ in range(100):
        f.step(np.array([[0.2, 0.0]]))
    assert 0.45 < f.charge[0] < 0.5


def test_flat_battery_cuts_the_motors():
    f = one(plain(charge=0.0))
    for _ in range(20):
        f.step(np.array([[1.0, 0.0]]))
    assert f.grounded[0] == 1 and f.motor[0] == 0.0


def test_camera_sees_the_pad_below():
    f = one(plain(), FULL_VISION)
    f.x[0] = 12.0
    f.y[0] = 6.0
    f.grounded[0] = 0
    depth, pad, bird = rays(f)
    down = N_RAYS // 2  # the two middle rays straddle straight down
    assert pad[0, down] == 1 and pad[0, down - 1] == 1
    assert abs(depth[0, down] * 15.0 - 5.0) < 0.2
    assert bird.sum() == 0


def test_delay_and_memory():
    feat = replace(FULL, noise=0.0, birds=False, pad_move=0.0)
    env = FleetEnv(2, Config(features=feat), seed=0)
    first = env.start([11, 12])
    assert first.shape == (2, 3 * N_STATE)
    assert np.allclose(first[:, :N_STATE], first[:, N_STATE : 2 * N_STATE])
    o1, _, _, _ = env.step(np.array([[1.0, 0.0], [1.0, 0.0]]))
    # 50 ms late: the first update still shows the starting reading
    assert np.allclose(o1[:, :N_STATE], first[:, :N_STATE])
    o2, _, _, _ = env.step(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(o2[:, N_STATE : 2 * N_STATE], o1[:, :N_STATE])
    assert not np.allclose(o2[:, :N_STATE], o1[:, :N_STATE])


def test_env_shapes_and_restarts():
    for feat, width in ((V010, 14), (FULL, N_STATE), (FULL_VISION, N_VISION)):
        env = FleetEnv(8, Config(weather=ROUGH, features=feat), seed=1)
        obs = env.reset()
        assert obs.shape == (8, width * feat.stack)
        finished = 0
        for _ in range(300):
            obs, r, done, info = env.step(np.random.default_rng(0).uniform(-1, 1, (8, 2)))
            assert np.isfinite(obs).all() and np.isfinite(r).all()
            finished += int(done.sum())
            for i in np.nonzero(done)[0]:
                assert "outcome" in info[i] and "terminal_observation" in info[i]
        assert finished > 0
        assert (env.fleet.outcome == FLYING).all()  # finished slots restarted


def test_features_default_off():
    assert Features() == V010
