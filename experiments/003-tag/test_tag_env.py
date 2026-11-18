import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from tag_env import (
    ACTIONS,
    MLP,
    Arena,
    Body,
    ChaserArena,
    Config,
    NetChaser,
    NetEvader,
    Physics,
    TagArena,
    caught,
    flee_action,
    move,
    steer,
)


def test_env_api():
    check_env(TagArena(), skip_render_check=True)


def test_resets_are_valid():
    env = TagArena()
    for seed in range(50):
        env.reset(seed=seed)
        a, h = env.cfg.arena, env.cfg.physics.half
        d = math.hypot(env.evader.x - env.chaser.x, env.evader.y - env.chaser.y)
        assert d >= env.cfg.min_start_distance
        for b in (env.evader, env.chaser):
            assert h <= b.x <= a.width - h and h <= b.y <= a.height - h
            assert all(math.hypot(b.x - cx, b.y - cy) > r for cx, cy, r in a.pillars)


def test_chaser_is_slower():
    p, a = Physics(), Arena(pillars=())
    fast, slow = Body(1, 1), Body(1, 2)
    for _ in range(100):
        move(fast, 1, 0, p.max_speed, p, a)
        move(slow, 1, 0, p.max_speed * 0.8, p, a)
        if fast.x > 8:
            break
    assert fast.x > slow.x
    assert math.hypot(slow.vx, slow.vy) <= p.max_speed * 0.8 + 1e-9


def test_pillar_blocks():
    p, a = Physics(), Arena()
    b = Body(2.0, 4.0)
    for _ in range(80):
        move(b, 1, 0, p.max_speed, p, a)
    cx, cy, r = a.pillars[0]
    # The square stops against the left side of the first pillar.
    assert b.x + p.half <= cx - r + 1e-6


def test_steer_goes_around_pillar():
    a = Arena()
    dx, dy = steer(2.0, 4.0, 6.0, 4.0, a, clearance=0.45)
    assert abs(dy) > 0.2  # not straight into the pillar
    dx, dy = steer(2.0, 1.0, 6.0, 1.0, a, clearance=0.45)
    assert (dx, dy) == (1.0, 0.0)  # clear path: straight


def test_pursuit_catches_a_still_target():
    env = TagArena(Config())
    env.reset(seed=0, options={"evader": (1.0, 1.0), "chaser_pos": (11.0, 7.0)})
    for _ in range(400):
        _, r, term, trunc, info = env.step(0)
        if term or trunc:
            break
    assert term and info["outcome"] == "caught" and r < 0


def test_flee_outlasts_standing_still_but_gets_pinned():
    def run(policy):
        env = TagArena(Config())
        env.reset(seed=0, options={"evader": (6.0, 2.0), "chaser_pos": (6.0, 7.0)})
        for t in range(1, 401):
            _, _, term, trunc, info = env.step(policy(env))
            if term or trunc:
                return t, info["outcome"]

    still = run(lambda env: 0)
    flee = run(lambda env: flee_action(env.evader, env.chaser))
    assert flee[0] > still[0]
    assert flee[1] == "caught"  # running straight away ends against a wall


def test_catch_is_square_overlap():
    assert caught(Body(1, 1), Body(1.5, 1.5), 0.3)
    assert not caught(Body(1, 1), Body(1.7, 1.0), 0.3)


def test_observation_in_range():
    env = TagArena()
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, *_rest, term, trunc, _ = env.step(int(rng.integers(9)))
        assert np.all(np.abs(obs) <= 2.0)
        if term or trunc:
            obs, _ = env.reset()


def _tiny_net(seed: int = 0) -> MLP:
    rng = np.random.default_rng(seed)
    shapes = [(64, 9), (64, 64), (9, 64)]
    return MLP([{"w": rng.normal(0, 0.3, s), "b": rng.normal(0, 0.1, s[0])} for s in shapes])


def test_mlp_act_and_sample():
    net = _tiny_net()
    obs = np.linspace(-1, 1, 9)
    assert net.act(obs) == int(np.argmax(net.logits(obs)))
    rng = np.random.default_rng(0)
    assert all(0 <= net.act(obs, rng) < 9 for _ in range(20))
    assert MLP(net.to_json(digits=None)["layers"]).act(obs) == net.act(obs)


def test_net_chaser_drives_the_chaser():
    cfg = Config(opponents=lambda rng: NetChaser(_tiny_net(1), ratio=0.9))
    env = TagArena(cfg)
    env.reset(seed=0)
    start = (env.chaser.x, env.chaser.y)
    for _ in range(20):
        env.step(0)
    assert (env.chaser.x, env.chaser.y) != start


def test_chaser_arena_is_zero_sum_and_catches():
    cfg = Config(evaders=lambda rng: NetEvader(_tiny_net(2)))
    check_env(ChaserArena(cfg), skip_render_check=True)
    env = ChaserArena(cfg)
    obs, _ = env.reset(seed=0)
    total, term = 0.0, False
    for _ in range(400):
        e, c = env.evader, env.chaser
        # greedy chase: pick the action pointing most toward the evader
        dx, dy = e.x - c.x, e.y - c.y
        a = max(range(1, 9), key=lambda i: ACTIONS[i][0] * dx + ACTIONS[i][1] * dy)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    assert term and info["outcome"] == "caught" and r == 5.0
