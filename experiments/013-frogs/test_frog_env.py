import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from frog_env import (
    CROUCH,
    FLYING,
    FROG,
    FULL,
    HIST,
    HOLD,
    K_FROGS,
    LEAP,
    N_FROG_OBS,
    N_HIST,
    NONE,
    ROUGH,
    Config,
    Fleet,
    FleetEnv,
    make_course,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "012-dodge"))
import dodge_env as d12  # noqa: E402

FROGS = replace(FULL, obs="frog", threats=NONE, frogs=(1, 2), pad_move=0.0)


def test_same_as_post_12_without_frogs():
    feat12 = d12.FULL
    feat = replace(FULL, frogs=(0, 0))
    seeds = [5000 + i for i in range(40)]
    a, b = Fleet(len(seeds), feat=feat), d12.Fleet(len(seeds), feat=feat12)
    for i, s in enumerate(seeds):
        c = make_course(s, ROUGH, feat, a.ph)
        assert c == {**d12.make_course(s, d12.ROUGH, feat12, b.ph), "frogs": [], "switch": None}
        a.set_course(i, c)
        b.set_course(i, c)
    rng = np.random.default_rng(1)
    for _ in range(300):
        act = np.clip(np.array([0.1, 0.0]) + rng.normal(0, 0.4, (len(seeds), 2)), -1, 1)
        a.step(act)
        b.step(act)
        for k in ("x", "y", "vx", "vy", "th", "outcome", "ticks", "m_x", "rock_x"):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k


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
        "frogs": [],
        "switch": None,
    }
    return {**c, **kw}


def one(course, feat=FROGS) -> Fleet:
    f = Fleet(1, feat=feat)
    f.set_course(0, course)
    return f


def hold_at(f: Fleet, x: float, y: float) -> None:
    f.grounded[0], f.x[0], f.y[0], f.vx[0], f.vy[0] = 0, x, y, 0.0, 0.0


def test_strikes_on_the_beat_and_hits_inside_its_reach():
    # c = 0.5: no jitter on strike 0; period 2.5 s, first strike at 1.0 s
    f = one(plain(frogs=[(10.0, 6.0, 2.5, 1.0, 0.5, 0.0)]))
    assert abs(f.next_strike()[0, 0] - 1.0) < 1e-9
    for _ in range(20):  # 1 s sitting on the pad, out of reach: it strikes the edge of its reach
        f.step(np.array([[-1.0, 0.0]]))
    assert f.frog_n[0, 0] == 1 and f.outcome[0] == FLYING
    for _ in range(int((LEAP + HOLD) / 0.05)):
        f.step(np.array([[-1.0, 0.0]]))
    assert abs(f.frog_tx[0, 0] - 8.0) < 1e-9  # clamped to the column - reach
    # inside its reach at the next strike: hit
    g = one(plain(frogs=[(10.0, 6.0, 2.5, 1.0, 0.5, 0.0)]))
    while g.ticks[0] < 120 and g.outcome[0] == FLYING:
        hold_at(g, 9.0, 4.0)
        g.step(np.array([[0.0, 0.0]]))
    assert g.outcome[0] == FROG


def test_rhythm_switch_and_jitter():
    sw = one(plain(frogs=[(10.0, 6.0, 2.5, 1.0, 0.3, 0.0)], switch=(2, 0.5)))
    times = np.array(
        [sw.strike_time(np.array([0]), np.full((1, K_FROGS), k))[0, 0] for k in range(6)]
    )
    gaps = np.diff(times)
    assert np.all(np.abs(gaps[:2] - 2.5) <= 0.2 + 1e-9)
    assert np.all(np.abs(gaps[2:] - 1.25) <= 0.2 + 1e-9)


def test_tell_crouches_before_a_strike():
    tell = replace(FROGS, tell=True)
    f = one(plain(frogs=[(10.0, 6.0, 2.5, 1.0, 0.5, 0.0)]), tell)
    seen = []
    for _ in range(20):
        f.step(np.array([[-1.0, 0.0]]))
        seen.append((f.ticks[0] * f.ph.dt, bool(f.crouching()[0, 0])))
    crouched = [t for t, c in seen if c]
    assert crouched and min(crouched) >= 1.0 - CROUCH - 1e-9 and max(crouched) < 1.0


def test_memory_and_labels():
    for memory, extra in (("none", 0), ("hist", HIST * K_FROGS * N_HIST)):
        env = FleetEnv(8, Config(features=replace(FROGS, memory=memory)), seed=0)
        obs = env.reset()
        assert obs.shape == (8, 3 * N_FROG_OBS + extra + K_FROGS)
        rng = np.random.default_rng(0)
        for _ in range(100):
            obs, r, done, _ = env.step(rng.uniform(-1, 1, (8, 2)))
            assert np.isfinite(obs).all()
        lab = obs[:, -K_FROGS:]
        assert (lab >= 0).all() and (lab <= 1).all()
        on = env.fleet.frog_on
        want = np.clip(env.fleet.next_strike() / 3, 0, 1)
        assert np.allclose(lab[on], want[on], atol=1e-6) and np.all(lab[~on] == 1.0)
