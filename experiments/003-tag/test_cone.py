import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from cone import (
    ALLROUND,
    HUMAN,
    ConeArena,
    Senses,
    obs_size,
    speed_factor,
    turn_toward,
    wrap,
)
from sight import Hider, SeekBot, Still
from tag_env import Arena, Config


def _env(senses=None, prey=None):
    return ConeArena(Config(evaders=lambda r: prey or Hider(), maps="random", rays=16), senses)


def test_env_follows_the_gym_api():
    env = _env()
    check_env(env, skip_render_check=True)
    assert env.observation_space.shape == (obs_size(),) == (63,)
    assert list(env.action_space.nvec) == [9, 9]


def test_angles():
    assert wrap(3 * math.pi) == math.pi
    assert abs(wrap(-math.pi / 2 - 2 * math.pi) + math.pi / 2) < 1e-12
    assert abs(turn_toward(0.0, math.pi / 2, 0.1) - 0.1) < 1e-12
    assert abs(turn_toward(0.0, -math.pi / 2, 0.1) + 0.1) < 1e-12
    assert abs(turn_toward(3.0, -3.0, 0.5) + 3.0) < 1e-12  # the short way round, across +-pi
    assert speed_factor(1, 0, 0.0, 0.5) == 1.0
    assert speed_factor(-1, 0, 0.0, 0.5) == 0.5
    assert abs(speed_factor(0, 1, 0.0, 0.5) - 0.75) < 1e-12


EMPTY = Arena(pillars=(), blocks=())


def _place(senses, runner, chaser, facing=0.0):
    env = _env(senses, Still())
    env.reset(
        seed=1,
        options={"arena": EMPTY, "evader": (*runner, 0, 0), "chaser_pos": chaser, "facing": facing},
    )
    return env


def test_it_sees_only_in_front():
    env = _place(HUMAN, (1.0, 4.0), (5.0, 4.0))  # runner behind it, to the west
    tr = env.tracker
    assert tr.in_cone(1.0, 0.5) and not tr.in_cone(-1.0, 0.0) and tr.in_cone(-0.3, 0.2)
    for _ in range(20):
        env.step([0, 0])  # stand still, keep facing east
    nav = env.grid.nav
    assert not tr.seen
    assert not tr.possible[nav.cell_of(9.0, 4.0)]  # in front: checked
    assert tr.possible[nav.cell_of(1.0, 4.0)]  # behind: could be there
    env.step([0, 5])  # turn west (the look actions follow ACTIONS: 5 = west)
    assert tr.facing > 0  # turning the short way, 15 degrees a step at 300 degrees/s


def test_reaction_time_delays_noticing():
    for reaction, k in ((0.0, 1), (0.25, 6), (0.4, 9)):
        env = _place(Senses(100, 300, reaction, 1.0), (8.0, 4.0), (2.0, 4.0))
        assert env.tracker.k == k
        steps = 0
        while not env.tracker.seen:
            env.step([0, 0])
            steps += 1
        assert steps == k - 1  # in view from the first look, noticed after k looks


def test_the_map_keeps_track_of_a_hidden_runner():
    hidden = missed = 0
    for prey in (Still(), Hider()):
        env = _env(HUMAN, prey)
        for seed in range(20):
            env.reset(seed=seed, options={"bot": SeekBot("search"), "ratio": 0.9})
            for _ in range(200):
                _, _, term, trunc, _ = env.step([0, 0])
                if not env.tracker.seen:
                    hidden += 1
                    missed += not env.tracker.possible[
                        env.grid.nav.cell_of(env.evader.x, env.evader.y)
                    ]
                if term or trunc:
                    break
    assert hidden > 2000
    assert missed / hidden < 0.01


def test_a_cone_makes_searching_slower():
    def first_sight(senses):
        env = _env(senses, Still())
        t = []
        for seed in range(20):
            env.reset(seed=200 + seed, options={"bot": SeekBot("search"), "ratio": 0.9})
            k = 0
            while not env.tracker.seen and k < 400:
                env.step([0, 0])
                k += 1
            t.append(k)
        return np.mean(t)

    assert first_sight(ALLROUND) < first_sight(HUMAN)
