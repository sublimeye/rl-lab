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
