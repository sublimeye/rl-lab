"""The planner's premise and its moves (post 15). Numpy only: CI runs it without torch."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "014-gauntlet"))
sys.path.insert(0, str(Path(__file__).parent))

from expert import CRUISE, LAND, MOVES, Planner, control, move_id  # noqa: E402
from gauntlet_env import CLEAR, FLYING, FULL, LANDED, ROUGH, Fleet, make_course  # noqa: E402


def _fleet(seeds, feat=FULL):
    f = Fleet(len(seeds), feat=feat)
    for i, s in enumerate(seeds):
        f.set_course(i, make_course(s, ROUGH, feat, f.ph))
    return f


def test_copies_fly_exactly_like_the_original():
    """The look-ahead is only as good as Fleet.take: a copy given the same sticks must end
    exactly where the original does, threats, frogs and all."""
    f = _fleet([20_000, 20_001, 20_002])
    rng = np.random.default_rng(0)
    for _ in range(60):
        f.step(np.c_[rng.uniform(0.0, 0.6, 3), rng.uniform(0.2, 0.8, 3) * f.dir])
    g = f.take([0, 1, 2])
    acts = [np.c_[rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)] for _ in range(80)]
    for a in acts:
        f.step(a)
        g.step(a)
    for k in ("x", "y", "vx", "vy", "p_x", "p_y", "frog_s", "outcome", "enc_next"):
        assert np.array_equal(getattr(f, k), getattr(g, k)), k


def test_moves_are_sticks_in_range():
    f = _fleet([20_000] * len(MOVES))
    a = control(f, np.arange(len(MOVES)))
    assert a.shape == (len(MOVES), 2)
    assert np.all(np.abs(a) <= 1.0)


def test_landing_move_lands_on_a_clear_course():
    """Cruise most of the way, then the land move alone sets the drone down."""
    f = _fleet([20_001], CLEAR)
    cruise, land = move_id(CRUISE, 7.0, 7.0), move_id(LAND, 3.5)
    for _ in range(1200):
        if f.outcome[0] != FLYING:
            break
        gx, _ = f.goal()
        m = land if abs(gx[0] - f.x[0]) < 20 else cruise
        f.step(control(f, np.array([m])))
    assert f.outcome[0] == LANDED


def test_planner_plans_every_few_updates():
    f = _fleet([20_002], CLEAR)
    pl = Planner(1, every=8, horizon=2.0)
    for _ in range(9):
        f.step(pl.act(f))
    assert pl.decisions == 2
