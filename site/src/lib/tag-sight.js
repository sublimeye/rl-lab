// Browser port of experiments/003-tag/sight.py (post 006): line of sight, the chaser's
// possibility map, its inputs, and the scripted hider and search bot. Keep them in step:
// site/test/tag-seek.test.mjs replays episodes exported from Python and checks every state.
import { chaserDirection, NavGrid, squareHits } from './tag-sim.js';

export const SIGHTS = ['xray', 'none', 'last', 'map'];
const NEAR_SCALE = 3;
const FORGET = 2;
const NEVER = 10000;
const BIG = 30000;
const NB = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
  [0, -1],
  [1, -1],
];

export function obsSize(sight, rays = 16) {
  return 4 + rays + 6 + (sight === 'last' || sight === 'map' ? 6 : 0) + (sight === 'map' ? 17 : 0);
}

/** Does the segment from a to b miss every obstacle? Grazing counts as blocked. */
export function sees(ax, ay, bx, by, w) {
  const dx = bx - ax;
  const dy = by - ay;
  const L2 = dx * dx + dy * dy;
  for (const [cx, cy, r] of w.pillars) {
    const t = L2 === 0 ? 0 : Math.min(1, Math.max(0, ((cx - ax) * dx + (cy - ay) * dy) / L2));
    const qx = ax + t * dx - cx;
    const qy = ay + t * dy - cy;
    if (qx * qx + qy * qy < r * r) return false;
  }
  // Slab test per block, x then y (unrolled: this runs ~2,000 times a step).
  for (const [x0, y0, x1, y1] of w.blocks ?? []) {
    let lo = 0;
    let hi = 1;
    if (Math.abs(dx) < 1e-12) {
      if (ax < x0 || ax > x1) continue;
    } else {
      let t1 = (x0 - ax) / dx;
      let t2 = (x1 - ax) / dx;
      if (t1 > t2) [t1, t2] = [t2, t1];
      lo = Math.max(lo, t1);
      hi = Math.min(hi, t2);
    }
    if (Math.abs(dy) < 1e-12) {
      if (ay < y0 || ay > y1) continue;
    } else {
      let t1 = (y0 - ay) / dy;
      let t2 = (y1 - ay) / dy;
      if (t1 > t2) [t1, t2] = [t2, t1];
      lo = Math.max(lo, t1);
      hi = Math.min(hi, t2);
    }
    if (lo <= hi) return false;
  }
  return true;
}

/** The bot grid of a map plus what sight and tracking need (sight.Grid). */
export class Grid {
  constructor(nav) {
    this.nav = nav;
    const { nx, ny, free } = nav;
    const n = nx * ny;
    this.n = n;
    this.cx = new Float64Array(n);
    this.cy = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      this.cx[k] = ((k % nx) + 0.5) * nav.cell;
      this.cy[k] = (Math.floor(k / nx) + 0.5) * nav.cell;
    }
    const counts = new Int32Array(n);
    for (let t = 0; t < n; t++) {
      let c = 0;
      for (let s = 0; s < n; s++) if (nav.dist[t * n + s] < BIG) c++;
      counts[t] = c;
    }
    let top = 0;
    for (const c of counts) top = Math.max(top, c);
    this.main = new Uint8Array(n);
    let size = 0;
    for (let k = 0; k < n; k++) {
      this.main[k] = free[k] && counts[k] === top ? 1 : 0;
      size += this.main[k];
    }
    this.mainSize = size;
    // first-step neighbours in ACTIONS order, -1 where blocked (no corner cutting)
    this.nb = new Int32Array(n * 8).fill(-1);
    for (let c = 0; c < n; c++) {
      if (!free[c]) continue;
      const j = Math.floor(c / nx);
      const i = c % nx;
      NB.forEach(([di, dj], d) => {
        const ii = i + di;
        const jj = j + dj;
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || !free[jj * nx + ii]) return;
        if (di && dj && !(free[j * nx + ii] && free[jj * nx + i])) return;
        this.nb[c * 8 + d] = jj * nx + ii;
      });
    }
    // Look points: the quarter centres and the centre of each open cell, where a whole
    // square fits. A cell counts as seen only when all of them are in sight.
    const w = nav.w;
    const h = 0.3;
    const q = nav.cell / 4;
    const offs = [
      [-q, -q],
      [q, -q],
      [-q, q],
      [q, q],
      [0, 0],
    ];
    this.look = [];
    for (let c = 0; c < n; c++) {
      if (!free[c]) {
        this.look.push([]);
        continue;
      }
      let pts = offs
        .map(([ox, oy]) => [this.cx[c] + ox, this.cy[c] + oy])
        .filter(
          ([x, y]) =>
            h <= x && x <= w.width - h && h <= y && y <= w.height - h && !squareHits(x, y, h, w),
        );
      if (!pts.length) pts = [[this.cx[c], this.cy[c]]];
      this.look.push(pts);
    }
    this.lookStart = new Int32Array(n + 1);
    const flat = this.look.flat();
    this.lookX = Float64Array.from(flat, (p) => p[0]);
    this.lookY = Float64Array.from(flat, (p) => p[1]);
    for (let c = 0; c < n; c++) this.lookStart[c + 1] = this.lookStart[c] + this.look[c].length;
  }

  /** Open cells whose look points are all in line of sight from (x, y). */
  visible(x, y) {
    const out = new Uint8Array(this.n);
    const w = this.nav.w;
    const { lookX, lookY, lookStart } = this;
    for (let k = 0; k < this.n; k++) {
      if (!this.nav.free[k]) continue;
      let all = 1;
      for (let i = lookStart[k]; i < lookStart[k + 1]; i++) {
        if (!sees(x, y, lookX[i], lookY[i], w)) {
          all = 0;
          break;
        }
      }
      out[k] = all;
    }
    return out;
  }

  /** m plus every cell one grid step from it. */
  dilate(m) {
    const out = m.slice();
    for (let c = 0; c < this.n; c++) {
      if (!m[c]) continue;
      for (let d = 0; d < 8; d++) {
        const k = this.nb[c * 8 + d];
        if (k >= 0) out[k] = 1;
      }
    }
    return out;
  }
}

/** The chaser's knowledge during one round (sight.Tracker). */
export class Tracker {
  constructor(grid, speed, dt) {
    this.g = grid;
    this.stepLen = speed * dt;
    this.cell = grid.nav.cell;
    this.dt = dt;
  }

  reset(me, prey) {
    this.ever = false;
    this.since = 0;
    this.last = [0, 0, 0, 0];
    this.budget = 0;
    this.possible = this.g.main.slice();
    this.age = new Int32Array(this.g.n).fill(NEVER);
    this.look(me, prey);
  }

  update(me, prey) {
    this.since++;
    for (let k = 0; k < this.g.n; k++) this.age[k] = Math.min(this.age[k] + 1, NEVER);
    this.budget += this.stepLen;
    while (this.budget >= this.cell) {
      this.budget -= this.cell;
      const p = this.g.dilate(this.possible);
      for (let k = 0; k < this.g.n; k++) p[k] &= this.g.main[k];
      this.possible = p;
    }
    this.look(me, prey);
  }

  look(me, prey) {
    const g = this.g;
    this.vis = g.visible(me.x, me.y);
    for (let k = 0; k < g.n; k++) if (this.vis[k]) this.age[k] = 0;
    this.seen = sees(me.x, me.y, prey.x, prey.y, g.nav.w);
    if (this.seen) {
      this.ever = true;
      this.since = 0;
      this.last = [prey.x, prey.y, prey.vx, prey.vy];
      const p = new Uint8Array(g.n);
      p[g.nav.cellOf(prey.x, prey.y)] = 1;
      // it may be anywhere in that cell, so it can be in a neighbour almost at once
      this.possible = g.dilate(p);
      for (let k = 0; k < g.n; k++) this.possible[k] &= g.main[k];
      this.budget = 0;
    } else {
      let any = false;
      for (let k = 0; k < g.n; k++) {
        if (this.vis[k]) this.possible[k] = 0;
        any ||= this.possible[k] === 1;
      }
      if (!any) for (let k = 0; k < g.n; k++) this.possible[k] = g.main[k] && !this.vis[k] ? 1 : 0;
    }
    this.weight = new Float64Array(g.n);
    for (let k = 0; k < g.n; k++) {
      if (this.possible[k]) this.weight[k] = 1 - Math.exp(-this.age[k] * (this.dt / FORGET));
    }
  }

  /** Per first step: share of the weight that way, and the best spot that way. */
  byDirection(me) {
    const g = this.g;
    const n = g.n;
    const dist = g.nav.dist;
    const c = g.nav.cellOf(me.x, me.y);
    const mass = new Array(8).fill(0);
    const near = new Array(8).fill(0);
    let total = 0;
    for (let t = 0; t < n; t++) {
      const wt = this.weight[t];
      if (!(wt > 0)) continue;
      let first = 0;
      let md = BIG;
      for (let d = 0; d < 8; d++) {
        const k = g.nb[c * 8 + d];
        const L = k >= 0 ? dist[t * n + k] : BIG;
        if (L < md) {
          md = L;
          first = d;
        }
      }
      if (md >= BIG) continue;
      total += wt;
      mass[first] += wt;
      const s = wt * Math.exp(-(1 + md) * (this.cell / NEAR_SCALE));
      if (s > near[first]) near[first] = s;
    }
    if (total > 0) for (let d = 0; d < 8; d++) mass[d] /= total;
    return [mass, near];
  }

  /** The cell the search bot heads for. */
  bestSpot(me) {
    const n = this.g.n;
    const c = this.g.nav.cellOf(me.x, me.y);
    let best = 0;
    let bs = -Infinity;
    for (let t = 0; t < n; t++) {
      const L = this.g.nav.dist[t * n + c];
      const s = L < BIG ? this.weight[t] * Math.exp(-L * (this.cell / NEAR_SCALE)) : 0;
      if (s > bs) {
        bs = s;
        best = t;
      }
    }
    return best;
  }

  weightSum() {
    let s = 0;
    for (const v of this.weight) s += v;
    return s;
  }
}

/** The chaser's inputs, same order and scaling as sight.seek_observe. */
export function seekObserve(me, prey, tr, sight, w, rayFn, nRays = 16) {
  const ms = w.maxSpeed;
  const out = [(me.x / w.width) * 2 - 1, (me.y / w.height) * 2 - 1, me.vx / ms, me.vy / ms];
  out.push(...rayFn(me.x, me.y, w, nRays));
  if (sight === 'xray' || tr.seen) {
    const dx = prey.x - me.x;
    const dy = prey.y - me.y;
    out.push(1, dx / w.width, dy / w.height, prey.vx / ms, prey.vy / ms);
    out.push(Math.hypot(dx, dy) / Math.hypot(w.width, w.height));
  } else {
    out.push(0, 0, 0, 0, 0, 0);
  }
  if (sight === 'last' || sight === 'map') {
    if (tr.ever) {
      const [lx, ly, lvx, lvy] = tr.last;
      out.push(1, (lx - me.x) / w.width, (ly - me.y) / w.height, lvx / ms, lvy / ms);
      out.push(Math.min((tr.since * w.dt) / 20, 1));
    } else {
      out.push(0, 0, 0, 0, 0, 0);
    }
  }
  if (sight === 'map') {
    const [mass, near] = tr.byDirection(me);
    out.push(tr.weightSum() / tr.g.mainSize);
    for (let d = 0; d < 8; d++) out.push(mass[d], near[d]);
  }
  return out;
}

/**
 * The scripted hider (sight.hider_direction). st = { target: null, arrived: false }, kept
 * per round; steps = steps played so far.
 */
export function hiderDirection(st, steps, grid, tr, e, c, replan = 10) {
  const nav = grid.nav;
  const n = grid.n;
  if (tr.seen || st.target === null || (!st.arrived && steps % replan === 0)) {
    st.arrived = false;
    st.target = null;
    const ck = nav.cellOf(c.x, c.y);
    const hk = nav.cellOf(e.x, e.y);
    let bs = -Infinity;
    for (let t = 0; t < n; t++) {
      if (!grid.main[t] || tr.vis[t]) continue;
      const dc = nav.dist[t * n + ck];
      const dh = nav.dist[t * n + hk];
      const s = (dc - dh) * 1000 + dc;
      if (s > bs) {
        bs = s;
        st.target = t;
      }
    }
  }
  if (st.target === null) {
    const dx = e.x - c.x;
    const dy = e.y - c.y;
    const m = Math.hypot(dx, dy) || 1;
    return [dx / m, dy / m];
  }
  const [tx, ty] = nav.aim[st.target];
  if (st.arrived || Math.hypot(tx - e.x, ty - e.y) < 0.15) {
    st.arrived = true;
    return [0, 0];
  }
  return nav.direction(e.x, e.y, tx, ty, 0.36);
}

/** Scripted chasers (sight.seekbot_direction): 'xray', 'lastseen' or 'search'. */
export function seekbotDirection(kind, ratio, grid, tr, c, e, w) {
  if (kind === 'xray' || tr.seen) {
    const spec = { kind: 'intercept', ratio, reaction: 1, dirs8: false, lead: 1 };
    return chaserDirection(spec, c, e, w, grid.nav);
  }
  const nav = grid.nav;
  if (kind === 'lastseen') {
    if (!tr.ever) return [0, 0];
    const [tx, ty] = tr.last;
    if (Math.hypot(tx - c.x, ty - c.y) < 0.2) return [0, 0];
    return nav.direction(c.x, c.y, tx, ty, 0.36);
  }
  let any = false;
  for (const v of tr.weight) any ||= v > 0;
  if (!any) return [0, 0];
  const [tx, ty] = nav.aim[tr.bestSpot(c)];
  return nav.direction(c.x, c.y, tx, ty, 0.36);
}

export { NavGrid };
