"""The gauntlet's Fleet.step (post 14) in PyTorch, batched over the planner's copies (post 16).

The fair planner (016-flawless/fair_planner.py) spends ~90% of its time stepping copies of the
simulator. This is that step as tensor ops, for a GPU: every copy is one row, every branch of
the numpy code (sitting on a pillar or flying, a frog's strike coming due, a contact with a
pad) is a mask instead of an index list, so the step has no data-dependent control flow and
compiles (torch.compile) into a few fused kernels.

What it covers is what the copies need:
- the physics, battery, pursuers (birds, missiles) already in flight, frogs, contacts, limits;
- fair_planner._ghost_step's extra checks (the margins round birds, missiles and frogs, and the
  frog windows), when the copies carry `ghost`.
It does not spawn waypoint encounters: the planner's copies know nothing of waypoints not yet
reached (`sim.enc_at[:] = 1e9`), so none ever comes due in them. `from_numpy` refuses a fleet
with one pending. The real flight stays on the numpy Fleet.

State is a dict of tensors (`S`, changed by a step) and one of per-copy constants (`C`, the
course, the air and mass the copy assumes, its frogs' timing, its ghost windows). `step` is a
pure function: it returns a new S.

Float64 on the CPU matches numpy to round-off (test_torch_fleet.py); float32 is for the GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "014-gauntlet"))
sys.path.insert(0, str(HERE.parent / "015-teacher"))
sys.path.insert(0, str(HERE.parent / "016-flawless"))

import fair_planner as fp  # noqa: E402  (also patches expert's moves, as the planner has them)

from gauntlet_env import (  # noqa: E402
    BACK,
    BIRD,
    BIRD_N,
    BIRD_TURN,
    CRASHED,
    FLYING,
    FOOT_BAND,
    FROG,
    FROG_HIT,
    FROG_JITTER,
    FROG_LEAD,
    FROG_REACH,
    GOLDEN,
    GONE_R,
    HOLD,
    K_BIRDS,
    K_PURSUERS,
    LANDED,
    LEAP,
    LEG,
    LOST,
    MAX_ENC,
    MISSILE,
    MISSILE_N,
    MISSILE_TURN,
    P_HIT,
    POINTS,
    TIMEOUT,
    WATER,
    Fleet,
    G,
    Physics,
)

TWO_PI = 2 * math.pi
for _v in (P_HIT, fp.MARGIN_P):  # one value per kind (birds, then missiles): see step
    assert len(set(_v[:K_BIRDS])) == 1 and len(set(_v[K_BIRDS:])) == 1
W_BOB = 2 * math.pi / 1.7

# per-copy constants: (name, kind) with kind f (float), i (int), b (bool)
CONST = {
    **{
        k: "f"
        for k in "s_x0 s_x1 s_top t_x0 t_x1 t_top dir w_lo w_hi wind gust period phase "
        "mass lock sw_ratio".split()
    },  # fmt: skip
    "end_tick": "i",
    "sw_after": "i",
    "frog_p": "f",
    "frog_on": "b",
    "p_spd": "f",
    "p_end": "f",
    "p_hostile": "b",
    "p_enc": "i",
}
STATE = {
    **{k: "f" for k in "x y vx vy th om motor stick0 stick1 charge".split()},
    **{k: "i" for k in "ticks outcome crash grounded ground_pad hit_enc".split()},
    "touch": "f",
    "p_on": "b",
    "p_x": "f",
    "p_y": "f",
    "p_psi": "f",
    "p_steer": "b",
    "p_done": "b",
    "beaten": "i",
    "frog_n": "i",
    "frog_s": "f",
    "frog_sx": "f",
    "frog_sy": "f",
    "frog_tx": "f",
    "frog_ty": "f",
    "frog_past": "b",
}
INT = torch.int32


def _dt(kind: str, dtype):
    return {"f": dtype, "i": INT, "b": torch.bool}[kind]


def _pending(f: Fleet) -> np.ndarray:
    k = np.minimum(f.enc_next, MAX_ENC - 1)
    return (f.enc_next < f.n_enc) & (f.enc_at[np.arange(f.n), k] < 1e8)


def from_numpy(f: Fleet, device="cpu", dtype=torch.float64):
    """(S, C) tensors for the drones of numpy Fleet f (ghost windows too, if f has them)."""
    if _pending(f).any():
        raise ValueError("a waypoint is still pending; the torch step spawns no encounters")

    def t(v, kind):
        return torch.tensor(np.array(v), device=device, dtype=_dt(kind, dtype))  # a copy

    S = {k: t(getattr(f, k), kind) for k, kind in STATE.items()}
    C = {k: t(getattr(f, k), kind) for k, kind in CONST.items()}
    gh = getattr(f, "ghost", None)
    if gh is not None:
        C["ghost"] = t(gh, "f")
    return S, C


def take(S: dict, C: dict, rows) -> tuple[dict, dict]:
    """Copies `rows` of every drone (repeats allowed), as Fleet.take."""
    return {k: v[rows] for k, v in S.items()}, {k: v[rows] for k, v in C.items()}


def params(ph: Physics, battery: bool = True) -> dict:
    """The scalars a step needs (Python floats: they compile into the kernels)."""
    return {**vars(ph), "battery": battery}


# --- pieces -----------------------------------------------------------------------------------
def _time(S, P, dtype):
    return S["ticks"].to(dtype) * P["dt"]


def thrust_scale(charge):
    return 1.0 - 0.03 * (1.0 - charge) - 0.2 * torch.exp(-charge / 0.06)


def strike_time(C, n):
    """When strike n of each frog comes, (N, 2)."""
    p = C["frog_p"]
    nf = n.to(p.dtype)
    after = C["sw_after"][:, None].to(p.dtype)
    switched = after + (nf - after) * C["sw_ratio"][:, None]
    beats = torch.where(n <= C["sw_after"][:, None], nf, switched)
    jitter = FROG_JITTER * (2.0 * torch.remainder(nf * GOLDEN + p[..., 4], 1.0) - 1.0)
    return p[..., 3] + p[..., 2] * beats + jitter


def frog_pos(S, C, P):
    """Frogs' (x, y, vx, vy), each (N, 2), as Fleet.frog_pos."""
    p = C["frog_p"]
    t = _time(S, P, p.dtype)[:, None]
    px = p[..., 0] + 0.0 * t
    py = p[..., 1] + 0.3 * torch.sin(W_BOB * t + p[..., 5])
    pvy = (0.3 * W_BOB) * torch.cos(W_BOB * t + p[..., 5])
    u = t - S["frog_s"]
    sx, sy, tx, ty = S["frog_sx"], S["frog_sy"], S["frog_tx"], S["frog_ty"]
    a = torch.clamp(u / LEAP, 0.0, 1.0)
    b = torch.clamp((u - LEAP - HOLD) / BACK, 0.0, 1.0)
    leap = (u >= 0) & (u < LEAP)
    back = (u >= LEAP + HOLD) & (u < LEAP + HOLD + BACK)
    hold = (u >= LEAP) & (u < LEAP + HOLD)
    sa, sb = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
    da, db = 6 * a * (1 - a) / LEAP, 6 * b * (1 - b) / BACK
    w = torch.where
    x = w(leap, sx + (tx - sx) * sa, w(hold, tx, w(back, tx + (px - tx) * sb, px)))
    y = w(leap, sy + (ty - sy) * sa, w(hold, ty, w(back, ty + (py - ty) * sb, py)))
    zero = torch.zeros_like(u)
    vx = w(leap, (tx - sx) * da, w(back, (px - tx) * db, zero))
    vy = w(leap, (ty - sy) * da, w(back, (py - ty) * db, w(hold, zero, pvy)))
    return x, y, vx, vy


def _end(S, mask, outcome):
    S["touch"] = torch.where(mask[:, None], torch.stack([S["vx"], S["vy"]], 1), S["touch"])
    S["outcome"] = torch.where(mask, outcome, S["outcome"])


def _frogs_move(S, C, P, m):
    p = C["frog_p"]
    on = C["frog_on"]
    ahead = C["dir"] * (S["x"] - p[:, 0, 0])
    past0 = m & on[:, 0] & (ahead > FROG_REACH + 1.0)
    S["frog_past"] = S["frog_past"] | torch.stack([past0, torch.zeros_like(past0)], 1)
    t = _time(S, P, p.dtype)
    st = strike_time(C, S["frog_n"])
    due = m[:, None] & on & (t[:, None] >= st)
    fx, fy, _, _ = frog_pos(S, C, P)
    col = p[..., 0]
    aim_x = (S["x"] + S["vx"] * FROG_LEAD)[:, None]
    aim_y = (S["y"] + S["vy"] * FROG_LEAD)[:, None]
    tx = torch.minimum(torch.maximum(aim_x, col - FROG_REACH), col + FROG_REACH)
    ty = torch.clamp(aim_y, 0.5, P["ceiling"] - 0.5).expand_as(col)
    S["frog_s"] = torch.where(due, st, S["frog_s"])
    S["frog_sx"] = torch.where(due, fx, S["frog_sx"])
    S["frog_sy"] = torch.where(due, fy, S["frog_sy"])
    S["frog_tx"] = torch.where(due, tx, S["frog_tx"])
    S["frog_ty"] = torch.where(due, ty, S["frog_ty"])
    S["frog_n"] = S["frog_n"] + due.to(INT)


def _pursuers_move(S, C, P, m, bird):
    dt = P["dt"]
    on = S["p_on"] & m[:, None]
    px, py, psi, sp = S["p_x"], S["p_y"], S["p_psi"], C["p_spd"]
    t = _time(S, P, px.dtype)[:, None]
    rx, ry = S["x"][:, None] - px, S["y"][:, None] - py
    rvx = S["vx"][:, None] - sp * torch.cos(psi)
    rvy = S["vy"][:, None] - sp * torch.sin(psi)
    r2 = torch.clamp(rx * rx + ry * ry, min=1e-6)
    r = torch.sqrt(r2)
    ldot = (rx * rvy - ry * rvx) / r2
    closing = -(rx * rvx + ry * rvy) / r
    steer = S["p_steer"]
    commit = on & steer & bird & (closing > 0) & (r < C["lock"][:, None] * closing)
    steer = steer & ~commit
    # (scalars into torch.where would make float32 tensors: 0.45 -> 0.44999998)
    gain = torch.where(bird, px.new_tensor(BIRD_N), px.new_tensor(MISSILE_N))
    turn = torch.where(bird, px.new_tensor(BIRD_TURN), px.new_tensor(MISSILE_TURN))
    turned = torch.minimum(torch.maximum(gain * ldot, -turn), turn) * dt
    psi = psi + torch.where(steer & on, turned, torch.zeros_like(turned))
    px = torch.where(on, px + sp * torch.cos(psi) * dt, px)
    py = torch.where(on, py + sp * torch.sin(psi) * dt, py)
    S["p_steer"], S["p_psi"], S["p_x"], S["p_y"] = steer, psi, px, py
    passed = on & (closing < 0) & (r < 6.0) & (~steer | ~bird)
    leaves = on & ((t >= C["p_end"]) | (py < -0.5) | (py > P["ceiling"] + 4.0) | (r > GONE_R))
    count = (passed | leaves) & ~S["p_done"]
    S["p_done"] = S["p_done"] | count
    beaten = torch.stack([(count & bird).sum(1), (count & ~bird).sum(1)], 1)
    S["beaten"] = S["beaten"] + beaten.to(INT)
    S["p_on"] = S["p_on"] & ~leaves


def _threats_hit(S, C, P, h, bird, hit_r2):
    dx = S["p_x"] - S["x"][:, None]
    dy = S["p_y"] - S["y"][:, None]
    hit = h[:, None] & S["p_on"] & (dx * dx + dy * dy < hit_r2)
    any_hit = hit.any(1)
    k = hit.to(torch.int8).argmax(1)  # the first slot hit
    S["hit_enc"] = torch.where(any_hit, C["p_enc"].gather(1, k[:, None])[:, 0], S["hit_enc"])
    is_bird = bird[k]
    _end(S, any_hit & is_bird, BIRD)
    _end(S, any_hit & ~is_bird, MISSILE)
    live = h & (S["outcome"] == FLYING)
    fx, fy, _, _ = frog_pos(S, C, P)
    dx, dy = fx - S["x"][:, None], fy - S["y"][:, None]
    fh = live & (C["frog_on"] & (dx * dx + dy * dy < FROG_HIT * FROG_HIT)).any(1)
    _end(S, fh, FROG)


def _limits(S, C, P, m):
    x = S["x"]
    lost = m & ((x < C["w_lo"] - 0.5) | (x > C["w_hi"] + 0.5) | (S["y"] > P["ceiling"]))
    S["outcome"] = torch.where(lost, LOST, S["outcome"])
    late = m & ~lost & (S["ticks"] >= C["end_tick"])
    S["outcome"] = torch.where(late, TIMEOUT, S["outcome"])


def _contacts(S, C, P, fly, feet_y0):
    c, s = torch.cos(S["th"]), torch.sin(S["th"])
    x, y = S["x"], S["y"]
    pts = [(x + lx * c + ly * s, y - lx * s + ly * c) for lx, ly in POINTS]
    decided = ~fly
    landing = torch.full_like(S["ticks"], -1)
    pads = ((C["s_x0"], C["s_x1"], C["s_top"]), (C["t_x0"], C["t_x1"], C["t_top"]))
    vy = S["vy"]
    for k, (px, py) in enumerate(pts):
        for p, (x0, x1, top) in enumerate(pads):
            inside = ~decided & (x0 < px) & (px < x1) & (py < top)
            if k < 2:
                from_above = feet_y0[k] >= top - FOOT_BAND
                landing = torch.where(inside & from_above & (vy <= 0.0), p, landing)
                side = inside & ~from_above
            else:
                side = inside
            S["crash"] = torch.where(side, 2, S["crash"])
            _end(S, side, CRASHED)
            decided = decided | side
        wet = ~decided & (py < 0.0)
        _end(S, wet, WATER)
        decided = decided | wet
    down = ~decided & (landing >= 0)
    gentle = (
        (S["vy"] >= -P["land_vy"])
        & (S["vx"].abs() <= P["land_vx"])
        & (S["th"].abs() <= P["land_tilt"])
    )
    hard = down & ~gentle
    S["crash"] = torch.where(hard, 1, S["crash"])
    _end(S, hard, CRASHED)
    ok = down & gentle
    over = ok & (landing == 1) & (C["t_x0"] <= S["x"]) & (S["x"] <= C["t_x1"])
    fp1 = torch.where(over, C["frog_on"][:, 1], S["frog_past"][:, 1])
    S["frog_past"] = torch.stack([S["frog_past"][:, 0], fp1], 1)
    _end(S, over, LANDED)
    rest = ok & ~over
    S["grounded"] = torch.where(rest, 1, S["grounded"])
    S["ground_pad"] = torch.where(rest, landing, S["ground_pad"])
    zero = torch.zeros_like(S["th"])
    S["th"] = torch.where(rest, zero, S["th"])
    S["om"] = torch.where(rest, zero, S["om"])
    top = torch.where(landing == 1, C["t_top"], C["s_top"])
    S["y"] = torch.where(rest, top + LEG, S["y"])
    S["vy"] = torch.where(rest, zero, S["vy"])
    S["vx"] = torch.where(rest, S["vx"] * 0.85, S["vx"])


def _tick(S, C, P, live, thr, tilt, bird, hit_r2):
    """One physics step (1/60 s) for the copies in mask `live` (all flying)."""
    dt = P["dt"]
    dtype = S["x"].dtype
    cmd = (thr + 1.0) / 2.0
    charge = S["charge"]
    if P["battery"]:
        cmd = torch.where(charge > 0.0, cmd, torch.zeros_like(cmd))
    motor = torch.where(live, S["motor"] + (cmd - S["motor"]) * dt / P["motor_tau"], S["motor"])
    S["motor"] = motor
    acc = motor * P["thrust_max"] * G
    if P["battery"]:
        acc = acc * thrust_scale(charge)
        drained = torch.clamp(charge - 0.0085 * motor**1.5 * dt, min=0.0)
        S["charge"] = torch.where(live, drained, charge)
    acc = acc / C["mass"]
    sit = live & (S["grounded"] == 1) & (acc <= G)
    fly = live & ~sit
    # still sitting on a pillar: friction slows any slide
    vx_sit = S["vx"] * 0.85
    S["vx"] = torch.where(sit, vx_sit, S["vx"])
    S["x"] = torch.where(sit, S["x"] + vx_sit * dt, S["x"])
    # flying
    S["grounded"] = torch.where(fly, 0, S["grounded"])
    th0 = S["th"]
    c0, s0 = torch.cos(th0), torch.sin(th0)
    feet_y0 = [S["y"] - lx * s0 + ly * c0 for lx, ly in POINTS[:2]]
    alpha = P["kp"] * (tilt * P["tilt_max"] - th0) - P["kd"] * S["om"]
    alpha = torch.clamp(alpha, -P["alpha_max"], P["alpha_max"])
    alpha = alpha * torch.clamp(0.3 + 1.4 * motor, max=1.0)
    om = S["om"] + alpha * dt
    th = th0 + om * dt
    vx, vy = S["vx"], S["vy"]
    speed = torch.sqrt(vx * vx + vy * vy)
    t = _time(S, P, dtype)
    air = C["wind"] + C["gust"] * torch.sin(TWO_PI * t / C["period"] + C["phase"])
    ax = acc * torch.sin(th) - P["drag_lin"] * vx - P["drag_quad"] * vx * speed
    ay = acc * torch.cos(th) - G - P["drag_lin"] * vy - P["drag_quad"] * vy * speed
    ax = ax + air / C["mass"]
    vx = vx + ax * dt
    vy = vy + ay * dt
    S["om"] = torch.where(fly, om, S["om"])
    S["th"] = torch.where(fly, th, th0)
    S["vx"] = torch.where(fly, vx, S["vx"])
    S["vy"] = torch.where(fly, vy, S["vy"])
    S["x"] = torch.where(fly, S["x"] + vx * dt, S["x"])
    S["y"] = torch.where(fly, S["y"] + vy * dt, S["y"])
    S["ticks"] = S["ticks"] + live.to(INT)
    # threats (both kinds of row), then contacts (flying rows), hits and limits
    _frogs_move(S, C, P, live)
    _pursuers_move(S, C, P, live, bird)
    _contacts(S, C, P, fly, feet_y0)
    h = live & (S["outcome"] == FLYING)
    _threats_hit(S, C, P, h, bird, hit_r2)
    _limits(S, C, P, live & (S["outcome"] == FLYING))


def _ghost(S, C, P, bird, near_r):
    """fair_planner._ghost_step's checks after the step: margins round birds, missiles and
    frogs, and the frog windows (C['ghost']: (N, 2 frogs, 3 strikes, [from, to]))."""
    flying = S["outcome"] == FLYING
    near = torch.hypot(S["p_x"] - S["x"][:, None], S["p_y"] - S["y"][:, None]) < near_r
    caught = flying[:, None] & S["p_on"] & near
    by_missile = (caught & ~bird).any(1)  # numpy keeps the last slot caught: a missile's
    by_bird = (caught & bird).any(1)
    S["outcome"] = torch.where(by_missile, MISSILE, torch.where(by_bird, BIRD, S["outcome"]))
    fx, fy, _, _ = frog_pos(S, C, P)
    d = torch.hypot(fx - S["x"][:, None], fy - S["y"][:, None])
    close = (C["frog_on"] & (d < FROG_HIT + fp.MARGIN_F)).any(1)
    S["outcome"] = torch.where((S["outcome"] == FLYING) & close, FROG, S["outcome"])
    gh = C["ghost"]
    t = _time(S, P, gh.dtype)[:, None, None]
    inside = (S["x"][:, None] - C["frog_p"][..., 0]).abs() < fp.ZONE
    live = (S["outcome"] == FLYING)[:, None] & C["frog_on"] & ~S["frog_past"] & inside
    hit = (live[..., None] & (t >= gh[..., 0]) & (t <= gh[..., 1])).any(2).any(1)
    S["outcome"] = torch.where(hit, FROG, S["outcome"])


def step(S: dict, C: dict, actions: torch.Tensor, P: dict) -> dict:
    """One stick update (3 physics steps) for every copy still flying; a new S. With
    C['ghost'], fair_planner's ghost checks follow, as in its copies."""
    S = dict(S)
    dev = S["x"].device
    bird = torch.arange(K_PURSUERS, device=dev) < K_BIRDS
    x = S["x"]

    def per_kind(v):  # (slots,) from a numpy per-slot array: made on the device (no copies
        # from the CPU, which a CUDA graph can't capture)
        return torch.where(bird, x.new_full((), float(v[0])), x.new_full((), float(v[-1])))

    p_hit = per_kind(P_HIT)
    a = torch.clamp(actions, -1.0, 1.0)
    thr, tilt = a[:, 0], a[:, 1]
    start = S["outcome"] == FLYING
    for _ in range(P["substeps"]):
        live = S["outcome"] == FLYING  # outcomes never return to flying: numpy's live list
        _tick(S, C, P, live, thr, tilt, bird, p_hit * p_hit)
    S["stick0"] = torch.where(start, thr, S["stick0"])
    S["stick1"] = torch.where(start, tilt, S["stick1"])
    if "ghost" in C:
        near_r = p_hit + per_kind(fp.MARGIN_P)
        _ghost(S, C, P, bird, near_r)
    return S
