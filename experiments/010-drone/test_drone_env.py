import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from drone_env import (
    CALM,
    LEG,
    N_OBS,
    ROUGH,
    Course,
    Drone,
    DroneFlight,
    Pad,
    Reward,
    autopilot,
    fly,
    make_course,
    observe,
    reward,
)


def course(start_top=1.0, target_x=12.0, target_top=1.0):
    return Course(
        Pad(2.0, 4.0, start_top), Pad(target_x - 1, target_x + 1, target_top), 0, 0, 3, 0, 1
    )


def hold(d, throttle, tilt, seconds):
    for _ in range(round(seconds * 20)):
        if d.done:
            break
        d.step(throttle, tilt)


def test_env_api():
    check_env(DroneFlight(), skip_render_check=True)


def test_courses_are_deterministic_mirrored_and_weather_free():
    sides = set()
    for s in range(200):
        a, b = make_course(s), make_course(s, ROUGH)
        assert a == make_course(s)
        assert (a.start, a.target) == (b.start, b.target)  # weather draws come last
        assert a.wind == 0 and a.mass == 1
        assert abs(a.target.cx - a.start.cx) >= 8 - 1e-9
        sides.add(a.target.cx > a.start.cx)
    assert sides == {True, False}


def test_centre_throttle_holds_and_zero_rests():
    d = Drone(course())
    hold(d, 0.0, 0.0, 2)  # exactly the weight: it stays on the pad
    assert not d.done and abs(d.y - (1.0 + LEG)) < 1e-9
    hold(d, -1.0, 0.0, 2)
    assert not d.done and abs(d.y - (1.0 + LEG)) < 1e-9


def test_throttle_climbs_and_tilt_moves_sideways():
    d = Drone(course())
    hold(d, 0.5, 0.0, 1.0)
    assert d.y > 2.5 and abs(d.x - 3.0) < 1e-9
    hold(d, 0.0, 1.0, 1.0)
    assert d.th > 0.4 and d.vx > 1.0  # leaning right pushes it right
    hold(d, 0.0, -1.0, 2.0)
    assert not d.done and d.vx < 0


def test_falling_in_the_water():
    d = Drone(course(target_x=20.0))
    hold(d, 0.4, 0.0, 1.0)
    hold(d, 0.05, 1.0, 1.2)  # off the pad to the right, over the water
    hold(d, -1.0, 0.0, 5.0)
    assert d.outcome == "water"


def test_hitting_a_pillar_side_crashes():
    d = Drone(course(target_x=8.0, target_top=6.0))
    hold(d, 0.25, 0.0, 0.8)  # low hop, then straight into the tall pillar
    hold(d, 0.1, 1.0, 4.0)
    assert d.outcome == "crashed" and d.y < 6.0


def test_hard_and_soft_touchdowns():
    for drop, want in ((0.3, "landed"), (3.0, "crashed")):
        d = Drone(course())
        d.x, d.y = 12.0, 1.0 + LEG + drop  # over the target, falling
        d.motor, d.grounded = 0.5, False
        d.vy = -0.5 if want == "landed" else -4.0
        hold(d, 0.0, 0.0, 3.0)
        assert d.outcome == want, (drop, d.outcome)


def test_tilted_touchdown_crashes():
    d = Drone(course())
    d.x, d.y, d.motor, d.vy, d.grounded = 12.0, 1.0 + LEG + 0.15, 0.5, -0.5, False
    d.th = 0.5
    hold(d, 0.0, 1.0, 1.0)
    assert d.outcome == "crashed"


def test_timeout():
    d = Drone(course())
    hold(d, -1.0, 0.0, 30)
    assert d.outcome == "timeout" and math.isclose(d.time, 20.0)


def test_autopilot_lands_in_calm_air():
    assert all(fly(Drone(make_course(s)), autopilot).outcome == "landed" for s in range(60))


def test_observation_and_rewards():
    d = Drone(make_course(3))
    o = observe(d)
    assert len(o) == N_OBS and all(abs(v) < 5 for v in o)
    before, sb = d.distance(), d.stick
    d.step(1.0, 0.0)
    assert d.distance() != before
    r_plain = reward(d, before, sb, Reward())
    r_smooth = reward(d, before, sb, Reward(smooth=0.5))
    assert math.isclose(r_plain - r_smooth, 0.5 * (2.0**2))  # throttle went -1 -> 1


def test_env_reports_outcome():
    env = DroneFlight()
    env.reset(seed=0, options={"course": 5})
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(np.array([-1.0, 0.0], np.float32))
        done = term or trunc
    assert info["outcome"] == "timeout" and trunc and env.drone.course == make_course(5, CALM)


def test_lifts_off_with_a_payload_while_tilting():
    c = course()
    c.mass = 1.3
    d = Drone(c)
    hold(d, 1.0, 1.0, 1.0)  # full throttle and full tilt from the pad
    assert not d.grounded and d.y > 1.0 + LEG + 0.5 and d.vx > 0


def test_tilt_on_the_ground_does_nothing():
    d = Drone(course())
    hold(d, 0.0, 1.0, 1.0)
    assert d.grounded and d.th == 0.0 and abs(d.x - 3.0) < 1e-9


def test_stick_rate_limits_how_fast_sticks_move():
    from drone_env import Physics

    d = Drone(course(), Physics(stick_rate=0.25))
    d.step(1.0, -1.0)
    assert d.stick == (-0.75, -0.25)
    for _ in range(10):
        d.step(1.0, -1.0)
    assert d.stick == (1.0, -1.0)


def test_crash_kinds():
    d = Drone(course())
    d.x, d.y, d.motor, d.vy, d.grounded = 12.0, 1.0 + LEG + 0.3, 0.5, -4.0, False
    hold(d, 0.0, 0.0, 1.0)
    assert d.outcome == "crashed" and d.crash == "hard"
    d = Drone(course(target_x=8.0, target_top=6.0))
    hold(d, 0.25, 0.0, 0.8)
    hold(d, 0.1, 1.0, 4.0)
    assert d.outcome == "crashed" and d.crash == "side"
