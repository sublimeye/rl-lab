"""The move-flown gauntlet (post 17). Numpy only: CI runs it without torch.

solo_env is imported inside the tests: importing it adds post 16's moves to expert's table,
which post 15's tests (collected in the same run) must not see.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def test_frog_memory_matches_the_fair_planner():
    """The vectorised strike memory gives the numbers post 16's network was fed, which
    fair_planner works out one drone at a time."""
    import solo_env as se

    sys.path.insert(0, str(Path(__file__).parent.parent / "016-flawless"))
    import pilot

    env = se.SoloEnv(6, hold=1, seed=3)
    x = env.start(list(range(50_000, 50_006)))
    f = env.fleet
    pl = se.fp.FairPlanner(f.n)
    seen = 0
    for _ in range(260):
        pl.observe(f)
        ref = pilot.beliefs(pl, f, env.prev)
        assert np.abs(ref[:, 9:25] - x[:, 318:334]).max() < 1e-5
        assert np.abs(ref[:, 25:] - x[:, 334:]).max() < 1e-6
        seen = max(seen, int(x[:, 318].sum() + x[:, 326].sum()))
        x, _, _, _ = env.step(se.scripted(x, env.mask()))
    assert seen > 0  # some frog came into sight
    assert x.shape == (6, se.DIM)


def test_a_pilot_that_ignores_threats_lands_clear_courses():
    """Still air assumed, 45 degrees: cruise, then land. No threats, so nothing to dodge."""
    from dataclasses import replace

    import solo_env as se

    from gauntlet_env import CLEAR

    out, _ = se.fly(se.scripted, list(range(20_000, 20_012)), replace(CLEAR, predict=True))
    assert out.count("landed") >= 11


def test_pad_moves_only_near_the_pad_and_restarts_are_clean():
    import solo_env as se

    env = se.SoloEnv(8, hold=2, seed=5)
    x = env.reset()
    mask = env.mask()
    f = env.fleet
    far = np.abs(f.goal()[0] - f.x) >= se.PAD_NEAR
    assert far.all() and not mask[:, se.PAD_MOVES].any() and mask[:, ~se.PAD_MOVES].all()
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(400):
        m = np.where(env.mask(), rng.random((8, se.NM)), -1).argmax(1)
        x, r, done, ended = env.step(m)
        finished += len(ended)
        assert len(ended) == done.sum()
        # a slot that restarted reads as a fresh course: the default move was made last
        assert (env.prev[done] == se.expert.DEFAULT).all()
        assert np.isfinite(x).all() and np.isfinite(r).all()
    assert finished > 0


def test_fork_copies_one_drone_mid_flight():
    import solo_env as se

    env = se.SoloEnv(3, hold=1, seed=7)
    x = env.start([50_000, 50_001, 50_002])
    for _ in range(40):
        x = env.step(se.scripted(x, env.mask()))[0]
    fork = env.fork(1, 4, seed=3)
    f, g = env.fleet, fork.fleet
    assert g.n == 4 and np.allclose(g.x, f.x[1]) and np.allclose(g.y, f.y[1])
    # the same moves from here: the copies fly the original's flight on (the physics has no
    # dice; only the sensors' noise is each copy's own)
    xf = fork.inputs()
    for _ in range(30):
        m = se.scripted(x, env.mask())
        x = env.step(m)[0]
        xf = fork.step(np.full(4, m[1]))[0]
    assert np.allclose(g.x, f.x[1]) and np.allclose(g.y, f.y[1])
    assert (g.outcome == f.outcome[1]).all()
    assert not np.allclose(xf[0], xf[1])


def test_practice_brings_failed_courses_back():
    import solo_env as se

    def repeats(replay: float) -> float:
        env = se.SoloEnv(32, se.FEAT, None, 3, 1, replay=replay, timeout=-50.0)
        x = env.reset()
        failed, again, total = set(), 0, 0
        for _ in range(500):
            x, _, d, _ = env.step(se.scripted(x, env.mask()))
            for i in np.nonzero(d)[0]:
                again += (int(env.env.seeds[i]), int(env.env.drill[i])) in failed
                total += 1
            failed |= set(getattr(env.env, "hard", ()))
        hard = getattr(env.env, "hard", [])
        assert all(hard[j] == k for k, j in getattr(env.env, "place", {}).items())
        assert len(set(hard)) == len(hard)
        return again / max(total, 1)

    assert repeats(0.0) == 0.0
    assert 0.3 < repeats(0.5) < 0.7  # the scripted pilot fails nearly every course


def test_practice_restarts_from_rewound_failures():
    import solo_env as se

    # a drone put into another fleet's slot flies on exactly as its copy does
    a = se.SoloEnv(3, hold=1, seed=7)
    x = a.start([50_000, 50_001, 50_002])
    for _ in range(60):
        x = a.step(se.scripted(x, a.mask()))[0]
    rec = a.fork(1, 1)
    b = se.SoloEnv(4, se.FEAT, None, 9, 1)
    xb = b.reset()
    for _ in range(25):
        xb = b.step(se.scripted(xb, b.mask()))[0]
    b._adopt(2, rec)
    assert np.allclose(b.fleet.x[2], rec.fleet.x[0])
    for _ in range(20):
        m = se.scripted(b.inputs(), b.mask())
        if b.fleet.outcome[2] != 0 or rec.fleet.outcome[0] != 0:
            break
        b.step(m)
        rec.step(m[2:3])
        assert np.allclose(b.fleet.x[2], rec.fleet.x[0])
        assert np.allclose(b.fleet.y[2], rec.fleet.y[0])
    # with again=1 every flight after the first failures begins mid-course
    env = se.SoloEnv(32, se.FEAT, None, 3, 1, again=1.0)
    x = env.reset()
    mid = begun = 0
    for _ in range(700):
        x, _, d, _ = env.step(se.scripted(x, env.mask()))
        begun += int(d.sum())
        mid += int((env.fleet.ticks[d] > 0).sum())
    assert len(env._pool) > 0 and begun > 50
    assert mid > 0.8 * (begun - 32)  # all but the flights begun before any failure was kept
