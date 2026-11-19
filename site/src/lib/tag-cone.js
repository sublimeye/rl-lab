// Browser port of experiments/003-tag/cone.py (post 007): a seeker that faces somewhere,
// sees only inside a cone, turns its head at a limited speed, notices things after a
// reaction time and moves slower backwards. site/test/tag-cone.test.mjs replays episodes
// exported from Python and checks every state.
import { ACTIONS, logits, move } from './tag-sim.js';
import { seekbotDirection, seekObserve, sees, Tracker } from './tag-sight.js';

export const LOOKS = 9; // 0 = hold, then turn toward ACTIONS[1..8]'s directions
export const HUMAN = { fov: 100, turn: 300, reaction: 0.25, backpedal: 0.5 };
export const ALLROUND = { fov: 360, turn: 1080, reaction: 0, backpedal: 1 };
const TURN_MAX = 1080;
const REACTION_MAX = 0.4;
const TOUCH = 0.75;
const RAD = Math.PI / 180; // same as Python's math.radians
const D = Math.SQRT1_2;
const UX = [1, D, 0, -D, -1, -D, 0, D];
const UY = [0, D, 1, D, 0, -D, -1, -D];

export const obsSize = (nRays = 16) => 49 - 16 + nRays + 14;

/** Angle into (-pi, pi] (JS % is C's fmod, like Python's math.fmod). */
export function wrap(a) {
  let r = (a + Math.PI) % (2 * Math.PI);
  if (r <= 0) r += 2 * Math.PI;
  return r - Math.PI;
}

export function turnToward(facing, target, step) {
  const d = wrap(target - facing);
  if (Math.abs(d) <= step) return wrap(target);
  return wrap(facing + (d > 0 ? step : -step));
}

export const lookAngle = (look) => ((look - 1) * Math.PI) / 4;

/** Top-speed multiplier when moving along (dx, dy) while facing `facing`. */
export function speedFactor(dx, dy, facing, backpedal) {
  const n = Math.hypot(dx, dy);
  if (n < 1e-9) return 1;
  const c = (dx * Math.cos(facing) + dy * Math.sin(facing)) / n;
  return 1 - ((1 - backpedal) * (1 - c)) / 2;
}

/** cone.ConeTracker: sight limited to a cone, noticing after `reaction` seconds. */
export class ConeTracker extends Tracker {
  constructor(grid, speed, dt, senses) {
    super(grid, speed, dt);
    this.s = senses;
    this.k = 1 + Math.floor(senses.reaction / dt + 1e-9);
    this.cosHalf = Math.cos((senses.fov / 2) * RAD);
  }

  reset(me, prey, facing = 0) {
    this.facing = facing;
    this.dwell = new Int32Array(this.g.n);
    this.streak = 0;
    super.reset(me, prey);
  }

  inCone(dx, dy, fx = Math.cos(this.facing), fy = Math.sin(this.facing)) {
    if (this.s.fov >= 360) return true;
    const d2 = dx * dx + dy * dy;
    return dx * fx + dy * fy >= this.cosHalf * Math.sqrt(d2) || d2 < TOUCH * TOUCH;
  }

  view(me, prey) {
    const g = this.g;
    const w = g.nav.w;
    const { lookX, lookY, lookStart } = g;
    const fx = Math.cos(this.facing);
    const fy = Math.sin(this.facing);
    this.los = new Uint8Array(g.n);
    this.vis = new Uint8Array(g.n);
    for (let c = 0; c < g.n; c++) {
      if (!g.nav.free[c]) continue;
      let los = 1;
      let inView = 1;
      for (let i = lookStart[c]; i < lookStart[c + 1]; i++) {
        if (!sees(me.x, me.y, lookX[i], lookY[i], w)) {
          los = 0;
          inView = 0;
          break;
        }
        if (inView && !this.inCone(lookX[i] - me.x, lookY[i] - me.y, fx, fy)) inView = 0;
      }
      this.los[c] = los;
      this.dwell[c] = inView ? this.dwell[c] + 1 : 0;
      this.vis[c] = inView && this.dwell[c] >= this.k ? 1 : 0;
    }
    const view =
      sees(me.x, me.y, prey.x, prey.y, w) && this.inCone(prey.x - me.x, prey.y - me.y, fx, fy);
    this.streak = view ? this.streak + 1 : 0;
    this.seen = view && (this.streak >= this.k || (this.ever && this.since <= this.k));
  }

  /** Per compass sector: the share of the map's weight it could check by turning that way. */
  lookable(me) {
    const g = this.g;
    const out = new Array(8).fill(0);
    let total = 0;
    for (let c = 0; c < g.n; c++) total += this.weight[c];
    if (!(total > 0)) return out;
    for (let c = 0; c < g.n; c++) {
      const wt = this.los[c] ? this.weight[c] : 0;
      if (!(wt > 0)) continue;
      const dx = g.cx[c] - me.x;
      const dy = g.cy[c] - me.y;
      let best = 0;
      let bs = -Infinity;
      for (let d = 0; d < 8; d++) {
        const v = dx * UX[d] + dy * UY[d];
        if (v > bs) {
          bs = v;
          best = d;
        }
      }
      out[best] += wt;
    }
    for (let d = 0; d < 8; d++) out[d] /= total;
    return out;
  }
}

/** cone.cone_observe: post 6's 49 inputs, facing, the four settings, lookable(). */
export function coneObserve(me, prey, tr, w, rayFn, nRays = 16) {
  const out = seekObserve(me, prey, tr, 'map', w, rayFn, nRays);
  const s = tr.s;
  out.push(Math.cos(tr.facing), Math.sin(tr.facing));
  out.push(s.fov / 360, s.turn / TURN_MAX, s.reaction / REACTION_MAX, s.backpedal);
  out.push(...tr.lookable(me));
  return out;
}

/** Best move and best look of a two-headed network. */
export function act2(net, input) {
  const z = logits(net, input);
  const k = ACTIONS.length;
  let m = 0;
  let l = 0;
  for (let i = 1; i < k; i++) if (z[i] > z[m]) m = i;
  for (let i = 1; i < LOOKS; i++) if (z[k + i] > z[k + l]) l = i;
  return [m, l];
}

/**
 * cone.auto_look: where a scripted chaser (kind: 'search' | 'lastseen' | null for post 6's
 * seeker) turns its head. null = hold.
 */
export function autoLook(kind, grid, tr, c, e) {
  if (tr.seen) return Math.atan2(e.y - c.y, e.x - c.x);
  let t = null;
  if (kind === 'search') {
    let any = false;
    for (const v of tr.weight) any ||= v > 0;
    if (any) t = grid.nav.aim[tr.bestSpot(c)];
  } else if (kind === 'lastseen' && tr.ever) {
    t = [tr.last[0], tr.last[1]];
  }
  if (t && Math.hypot(t[0] - c.x, t[1] - c.y) > 0.3) return Math.atan2(t[1] - c.y, t[0] - c.x);
  if (Math.hypot(c.vx, c.vy) > 0.5) return Math.atan2(c.vy, c.vx);
  return null;
}

/**
 * One chaser step in cone.ConeArena.step's order, after the runner's direction is known:
 * pick the move, turn the head, then move both and look. chaser: { kind: 'net', net } |
 * { kind: 'post6', net } | { kind: 'search' | 'lastseen' | 'xray' }. Returns [move, look]
 * (look -1 for automatic heads).
 */
export function chaserStep(chaser, ratio, grid, tr, c, e, ed, w, rayFn) {
  let cd;
  let target = null;
  let picked = [0, -1];
  if (chaser.kind === 'net') {
    picked = act2(chaser.net, coneObserve(c, e, tr, w, rayFn));
    cd = ACTIONS[picked[0]];
    if (picked[1]) target = lookAngle(picked[1]);
  } else {
    if (chaser.kind === 'post6') {
      const z = logits(chaser.net, seekObserve(c, e, tr, 'map', w, rayFn));
      let m = 0;
      for (let i = 1; i < ACTIONS.length; i++) if (z[i] > z[m]) m = i;
      cd = ACTIONS[m];
      picked = [m, -1];
    } else {
      cd = seekbotDirection(chaser.kind, ratio, grid, tr, c, e, w);
    }
    target = autoLook(chaser.kind === 'post6' ? null : chaser.kind, grid, tr, c, e);
  }
  if (target !== null) tr.facing = turnToward(tr.facing, target, tr.s.turn * RAD * w.dt);
  const f = speedFactor(cd[0], cd[1], tr.facing, tr.s.backpedal);
  move(e, ed[0], ed[1], w.maxSpeed, w);
  move(c, cd[0], cd[1], w.maxSpeed * ratio * f, w);
  tr.update(c, e);
  return picked;
}

/** Distance to the nearest wall or obstacle from (x, y) at `angle` (for drawing the cone). */
export function rayAt(x, y, w, angle, reach = 20) {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const tx = dx > 0 ? (w.width - x) / dx : dx < 0 ? -x / dx : Infinity;
  const ty = dy > 0 ? (w.height - y) / dy : dy < 0 ? -y / dy : Infinity;
  let best = Math.min(tx, ty, reach);
  for (const [cx, cy, r] of w.pillars) {
    const ox = x - cx;
    const oy = y - cy;
    const b = ox * dx + oy * dy;
    const disc = b * b - (ox * ox + oy * oy - r * r);
    if (disc < 0) continue;
    const t = -b - Math.sqrt(disc);
    if (t >= 0 && t < best) best = t;
  }
  for (const [x0, y0, x1, y1] of w.blocks ?? []) {
    const t1 = (x0 - x) / dx;
    const t2 = (x1 - x) / dx;
    const t3 = (y0 - y) / dy;
    const t4 = (y1 - y) / dy;
    const tmin = Math.max(Math.min(t1, t2), Math.min(t3, t4));
    const tmax = Math.min(Math.max(t1, t2), Math.max(t3, t4));
    if (tmax >= tmin && tmin >= 0 && tmin < best) best = tmin;
  }
  return best;
}
