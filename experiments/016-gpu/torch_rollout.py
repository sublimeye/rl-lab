"""The planner's look-ahead in PyTorch: post 15's controller (with 016's gentle touchdown) and
Planner.rollout, on torch_fleet's step.

rollout(S, C, P, plans, H) flies every copy its plan (m1, t1, m2, t2, m3) for H updates and
returns (updates flown, closest pass, progress summed), as expert.Planner.rollout does.
"""

from __future__ import annotations

import math

import torch
import torch_fleet as tf

from gauntlet_env import FROG_HIT, LANDED, LEG, G  # noqa: F401

fp = tf.fp
expert = fp.expert
FLYING = tf.FLYING


def tables(device, dtype):
    """The move table (kind, a, b) as tensors (fair_planner's, with its extra moves)."""
    return (
        torch.as_tensor(expert.MK, device=device),
        torch.as_tensor(expert.MA, device=device, dtype=dtype),
        torch.as_tensor(expert.MB, device=device, dtype=dtype),
    )


def goal(S, C):
    return (C["t_x0"] + C["t_x1"]) / 2, C["t_top"] + LEG


def distance(S, C):
    gx, gy = goal(S, C)
    dx, dy = gx - S["x"], gy - S["y"]
    return torch.sqrt(dx * dx + dy * dy)


def control(S, C, P, m, T):
    """expert.control as fair_planner patches it (_gentle_control): sticks (N, 2) for move m."""
    MK, MA, MB = T
    kind, a, b = MK[m], MA[m], MB[m]
    x, y, vx, vy = S["x"], S["y"], S["vx"], S["vy"]
    w = torch.where
    gx, gy = goal(S, C)
    dx = gx - x
    d = C["dir"]
    half = (C["t_x1"] - C["t_x0"]) / 2
    want_vx = d * b
    want_y = a
    three, one = x.new_full((), 3.0), x.new_full((), 1.0)
    over = (dx.abs() < half - 0.15) & (vx.abs() < 1.2)
    near = dx.abs() < half + 1.0
    low = (y - LEG < C["t_top"] + 0.1) & (dx.abs() > half + 0.2) & (dx.abs() < half + 4.0)
    land_vx = w(low, torch.zeros_like(x), torch.minimum(torch.maximum(2.0 * dx, -a), a))
    land_y = w(over, gy - 0.6, w(near, gy + 0.35, gy + 0.5))
    is_land = kind == expert.LAND
    want_vx = w(is_land, land_vx, want_vx)
    want_y = w(is_land, land_y, want_y)
    vy_lim = w(is_land & over, x.new_full((), 1.1), three)
    tilt_lim = w(is_land & near, x.new_full((), 0.6), one)
    is_wait = kind == expert.WAIT
    want_vx = w(is_wait, torch.clamp(1.2 * (gx - d * a - x), -4.0, 4.0), want_vx)
    want_y = w(is_wait, gy + b, want_y)
    vyc = torch.minimum(torch.maximum(1.5 * (want_y - y), -vy_lim), vy_lim)
    level = w(S["grounded"] == 1, one, torch.zeros_like(x))
    vyc = w(kind == expert.LEVEL, level, vyc)
    t = S["ticks"].to(x.dtype) * P["dt"]
    air = C["wind"] + C["gust"] * torch.sin(2 * math.pi * t / C["period"] + C["phase"])
    lean = -torch.atan2(air / C["mass"], x.new_full((), G)) / P["tilt_max"]
    tilt = torch.minimum(torch.maximum(lean + 0.35 * (want_vx - vx), -tilt_lim), tilt_lim)
    hover = C["mass"] / (P["thrust_max"] * tf.thrust_scale(S["charge"]) * torch.cos(S["th"]))
    thr = torch.clamp(2.0 * hover - 1.0 + 0.6 * (vyc - vy), -1.0, 1.0)
    jink = kind == expert.JINK
    thr = w(jink, a, thr)
    tilt = w(jink, d * b, tilt)
    # _gentle_control: touching down tilts at most 80% of the touchdown limit
    over2 = ((gx - x).abs() < half + 0.3) & (y - LEG - C["t_top"] < 0.6)
    cap = 0.8 * P["land_tilt"] / P["tilt_max"]
    tilt = w(is_land & over2, torch.clamp(tilt, -cap, cap), tilt)
    return torch.stack([thr, tilt], 1)


def clearance(S, C, P):
    x, y = S["x"][:, None], S["y"][:, None]
    big = x.new_full((), 1e9)
    d = torch.where(S["p_on"], torch.hypot(S["p_x"] - x, S["p_y"] - y), big).amin(1)
    fx, fy, _, _ = tf.frog_pos(S, C, P)
    dist = torch.where(C["frog_on"], torch.hypot(fx - x, fy - y) - 0.05, big).amin(1)
    return torch.minimum(d, dist)


def rollout_body(S, C, P, plans, T, u, acc):
    """One update of Planner.rollout (u: the update, a tensor). acc = (alive, dmin, gain, d0)."""
    alive, dmin, gain, d0 = acc
    flying = S["outcome"] == FLYING
    m1, t1, m2, t2, m3 = plans.unbind(1)
    m = torch.where(u < t1, m1, torch.where(u < t2, m2, m3))
    act = control(S, C, P, m, T)
    S = tf.step(S, C, act, P)
    # numpy stops at the first update with nothing flying (landed copies then get d0 for each
    # update left, the others nothing more): the same, without a branch
    none = ~flying.any()
    landed = S["outcome"] == LANDED
    zero = torch.zeros_like(d0)
    gain = gain + torch.where(landed, d0, torch.where(none, zero, d0 - distance(S, C)))
    alive = alive + flying.to(alive.dtype)
    dmin = torch.where(none, dmin, torch.minimum(dmin, clearance(S, C, P)))
    return S, (alive, dmin, gain, d0)


def rollout(S, C, P, plans, H, T, body=rollout_body):
    d0 = distance(S, C)
    acc = (torch.zeros_like(d0), torch.full_like(d0, 1e9), torch.zeros_like(d0), d0)
    for u in range(H):
        S, acc = body(S, C, P, plans, T, torch.tensor(u, device=d0.device), acc)
    alive, dmin, gain, _ = acc
    return S, alive, dmin, gain
