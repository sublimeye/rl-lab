import numpy as np
from gymnasium.utils.env_checker import check_env

from beaver_env import (
    CLAY,
    DEN,
    LOG,
    LOW,
    SOIL,
    SPRING,
    BeaverGame,
    Board,
    Game,
    RiverBot,
    RiverGame,
    Rules,
    make_board,
    observe,
    play,
    river_legal,
    warden_action,
)

N = 9


def at(x, y):
    return y * N + x


def corridor(rules=None):
    """A straight soil corridor: spring at (0,0), den at (8,0), rock elsewhere."""
    t = [0] * (N * N)  # rock
    for x in range(N):
        t[at(x, 0)] = SOIL
    t[at(0, 0)] = SPRING
    t[at(8, 0)] = DEN
    for x in range(N):
        t[at(x, 1)] = SOIL  # a walking lane for the beaver below the corridor
    return Board(t, at(0, 0), at(8, 0), at(8, 1))


def test_env_api():
    check_env(BeaverGame(), skip_render_check=True)
    check_env(RiverGame(), skip_render_check=True)


def test_boards_are_deterministic_and_valid():
    for s in range(50):
        a, b = make_board(s), make_board(s)
        assert a.tiles == b.tiles and a.start == b.start
        assert a.tiles[a.spring] == SPRING and a.tiles[a.den] == DEN
        g = Game(a)
        assert g.digs_needed()[a.den] >= 3  # the river has to work for it
        assert not g.wet[a.den]


def test_doing_nothing_always_floods():
    rng = np.random.default_rng(0)
    for s in range(100):
        g = play(make_board(s), RiverBot(rng), lambda g: 0)
        assert g.flooded


def test_warden_beats_some_rivers():
    rng = np.random.default_rng(0)
    dry = sum(not play(make_board(s), RiverBot(rng), warden_action).flooded for s in range(100))
    assert 10 <= dry <= 60


def test_water_moves_one_tile_per_round_and_respects_reach():
    g = Game(corridor(Rules(reach=5)), Rules(reach=5))
    for x in range(1, 8):
        assert g.dig(at(x, 0)) == (x <= 5)  # beyond reach there is no strength to dig
        g.flow()
        assert g.wet[at(x, 0)] == (x <= 5)
    assert not g.flooded


def test_clay_takes_two_digs():
    b = corridor()
    b.tiles[at(1, 0)] = CLAY
    g = Game(b)
    g.dig(at(1, 0))
    assert g.tiles[at(1, 0)] == SOIL
    g.dig(at(1, 0))
    assert g.tiles[at(1, 0)] == LOW


def test_dig_only_next_to_water():
    g = Game(corridor())
    assert not g.dig(at(3, 0))
    assert g.dig(at(1, 0))
    assert not g.dig(at(2, 0))  # (1,0) is open but not wet until the water moves
    g.flow()
    assert g.dig(at(2, 0))


def test_dam_cuts_and_drains():
    g = Game(corridor())
    for x in range(1, 6):
        g.dig(at(x, 0))
        g.flow()
    assert all(g.wet[at(x, 0)] for x in range(1, 6))
    g.beaver_at = at(3, 1)
    g.beaver(5)  # log up, onto (3,0)
    assert g.tiles[at(3, 0)] == LOG and g.logs == Rules().logs - 1
    g.flow()
    assert g.wet[at(2, 0)] and not g.wet[at(4, 0)] and not g.wet[at(5, 0)]


def test_no_dams_near_the_spring_or_on_soil():
    g = Game(corridor())
    g.dig(at(1, 0))
    g.flow()
    g.beaver_at = at(1, 1)
    assert not g.beaver_legal()[5]  # too close to the spring
    g.beaver_at = at(4, 1)
    assert not g.beaver_legal()[5]  # (4,0) is soil


def test_den_floods():
    g = Game(corridor())
    for x in range(1, 8):
        g.dig(at(x, 0))
        g.flow()
    assert not g.flooded
    g.flow()
    assert g.flooded and g.done


def test_rounds_run_out():
    g = Game(corridor())
    while not g.done:
        g.flow()
    assert g.round == Rules().rounds and not g.flooded


def test_beaver_walks_but_not_through_rock():
    g = Game(corridor())
    g.beaver_at = at(4, 1)
    g.beaver(3)  # down: rock
    assert g.beaver_at == at(4, 1)
    g.beaver(4)  # left
    assert g.beaver_at == at(3, 1)


def test_observation_and_masks():
    g = Game(make_board(3))
    o = observe(g)
    assert len(o) == 12 * 81 + 2 and all(0.0 <= v <= 1.0 for v in o)
    legal = river_legal(g)
    assert legal[-1] and sum(legal) >= 2
    assert all(g.can_dig(i) for i in range(81) if legal[i])


def test_log_only_on_low():
    b = corridor()
    b.tiles[at(4, 0)] = LOW
    g = Game(b)
    g.beaver_at = at(4, 1)
    assert g.beaver_legal()[5]
    g.beaver(5)
    assert g.tiles[at(4, 0)] == LOG


def test_no_dams_next_to_the_den():
    b = corridor()
    b.tiles[at(7, 0)] = LOW
    b.tiles[at(6, 0)] = LOW
    g = Game(b)
    g.beaver_at = at(7, 1)
    assert not g.beaver_legal()[5]  # (7,0) touches the den
    g.beaver_at = at(6, 1)
    assert g.beaver_legal()[5]
