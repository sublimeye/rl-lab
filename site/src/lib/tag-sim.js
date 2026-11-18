// Browser port of experiments/003-tag/tag_env.py and maps.py. Keep them in step:
// site/test/tag-sim.test.mjs replays trajectories exported from Python and checks every state,
// and checks that the same seed builds the same random map.

const D = Math.SQRT1_2;
/** 0 = stand still, then 8 directions counter-clockwise from east (same order as Python). */
export const ACTIONS = [
  [0, 0],
  [1, 0],
  [D, D],
  [0, 1],
  [-D, D],
  [-1, 0],
  [-D, -D],
  [0, -1],
  [D, -D],
];

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * world = { width, height, pillars: [[cx, cy, r]], blocks?: [[x0, y0, x1, y1]], half, dt, accel,
 * drag, maxSpeed, maxSteps }
 */
export function body(x, y, vx = 0, vy = 0) {
  return { x, y, vx, vy };
}

function edges(b, w) {
  const h = w.half;
  if (b.x < h || b.x > w.width - h) {
    b.x = clamp(b.x, h, w.width - h);
    b.vx = 0;
  }
  if (b.y < h || b.y > w.height - h) {
    b.y = clamp(b.y, h, w.height - h);
    b.vy = 0;
  }
}

/** One physics step: accelerate along (ax, ay), cap speed, move, collide. Mutates b. */
export function move(b, ax, ay, maxSpeed, w) {
  b.vx = (b.vx + ax * w.accel * w.dt) * w.drag;
  b.vy = (b.vy + ay * w.accel * w.dt) * w.drag;
  const speed = Math.hypot(b.vx, b.vy);
  if (speed > maxSpeed) {
    b.vx *= maxSpeed / speed;
    b.vy *= maxSpeed / speed;
  }
  b.x += b.vx * w.dt;
  b.y += b.vy * w.dt;
  edges(b, w);
  for (const [cx, cy, r] of w.pillars) {
    const qx = clamp(cx, b.x - w.half, b.x + w.half);
    const qy = clamp(cy, b.y - w.half, b.y + w.half);
    let dx = qx - cx;
    let dy = qy - cy;
    let d = Math.hypot(dx, dy);
    if (d >= r) continue;
    let push;
    if (d < 1e-9) {
      dx = b.x - cx;
      dy = b.y - cy;
      d = Math.hypot(dx, dy) || 1;
      push = r + w.half - d;
    } else {
      push = r - d;
    }
    const nx = dx / d;
    const ny = dy / d;
    b.x += nx * push;
    b.y += ny * push;
    const vn = b.vx * nx + b.vy * ny;
    if (vn < 0) {
      b.vx -= vn * nx;
      b.vy -= vn * ny;
    }
  }
  const h = w.half;
  for (const [x0, y0, x1, y1] of w.blocks ?? []) {
    const ox = Math.min(b.x + h - x0, x1 - b.x + h);
    const oy = Math.min(b.y + h - y0, y1 - b.y + h);
    if (ox <= 0 || oy <= 0) continue;
    if (ox < oy) {
      if (b.x < (x0 + x1) / 2) {
        b.x = x0 - h;
        b.vx = Math.min(b.vx, 0);
      } else {
        b.x = x1 + h;
        b.vx = Math.max(b.vx, 0);
      }
    } else if (b.y < (y0 + y1) / 2) {
      b.y = y0 - h;
      b.vy = Math.min(b.vy, 0);
    } else {
      b.y = y1 + h;
      b.vy = Math.max(b.vy, 0);
    }
  }
  edges(b, w);
}

/** Unit direction toward (tx, ty), going around the nearest blocking pillar. */
export function steer(fx, fy, tx, ty, w, clearance) {
  let gx = tx - fx;
  let gy = ty - fy;
  const glen = Math.hypot(gx, gy);
  if (glen < 1e-9) return [0, 0];
  gx /= glen;
  gy /= glen;
  let best = null;
  for (const [cx, cy, r] of w.pillars) {
    const R = r + clearance;
    const px = cx - fx;
    const py = cy - fy;
    const dist = Math.hypot(px, py);
    const along = px * gx + py * gy;
    if (along <= 0 || along - R > glen) continue;
    if (Math.abs(px * gy - py * gx) >= R) continue;
    if (best === null || dist < best[0]) best = [dist, px, py, R];
  }
  if (best === null) return [gx, gy];
  const [dist, px, py, R] = best;
  const ux = px / dist;
  const uy = py / dist;
  const alpha = dist <= R ? Math.PI / 2 : Math.asin(R / dist);
  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  const left = [ux * ca - uy * sa, ux * sa + uy * ca];
  const right = [ux * ca + uy * sa, -ux * sa + uy * ca];
  return left[0] * gx + left[1] * gy >= right[0] * gx + right[1] * gy ? left : right;
}

/**
 * chaser = { kind: 'pursuit' | 'intercept', ratio, reaction, dirs8, lead }
 * nav: a NavGrid on random maps (plans around any obstacle); without it, pillar steering.
 */
export function chaserDirection(c, me, prey, w, nav = null) {
  let tx = prey.x;
  let ty = prey.y;
  if (c.kind === 'intercept') {
    const dist = Math.hypot(prey.x - me.x, prey.y - me.y);
    const t = Math.min((c.lead * dist) / (w.maxSpeed * c.ratio), 1.5);
    tx = clamp(prey.x + prey.vx * t, w.half, w.width - w.half);
    ty = clamp(prey.y + prey.vy * t, w.half, w.height - w.half);
  }
  let [dx, dy] = nav
    ? nav.direction(me.x, me.y, tx, ty, w.half * 1.2)
    : steer(me.x, me.y, tx, ty, w, w.half * 1.5);
  if (c.dirs8 && (dx || dy)) {
    const ang = Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) * (Math.PI / 4);
    dx = Math.cos(ang);
    dy = Math.sin(ang);
  }
  return [dx, dy];
}

export function caught(e, c, w) {
  return Math.abs(e.x - c.x) < 2 * w.half && Math.abs(e.y - c.y) < 2 * w.half;
}

/** The 9 inputs plus `nRays` wall distances, same order and scaling as Python's observe(). */
export function observe(e, c, w, nRays = 0) {
  const dx = c.x - e.x;
  const dy = c.y - e.y;
  const out = [
    (e.x / w.width) * 2 - 1,
    (e.y / w.height) * 2 - 1,
    e.vx / w.maxSpeed,
    e.vy / w.maxSpeed,
    dx / w.width,
    dy / w.height,
    c.vx / w.maxSpeed,
    c.vy / w.maxSpeed,
    Math.hypot(dx, dy) / Math.hypot(w.width, w.height),
  ];
  return nRays ? out.concat(rays(e.x, e.y, w, nRays)) : out;
}

/** Policy network forward pass: tanh hidden layers, linear output. Returns the logits. */
export function logits(net, input) {
  let x = input;
  net.layers.forEach((layer, i) => {
    const out = new Array(layer.b.length);
    for (let o = 0; o < out.length; o++) {
      const row = layer.w[o];
      let s = layer.b[o];
      for (let k = 0; k < x.length; k++) s += row[k] * x[k];
      out[o] = i < net.layers.length - 1 ? Math.tanh(s) : s;
    }
    x = out;
  });
  return x;
}

/** The action the network rates highest. */
export function act(net, input) {
  const z = logits(net, input);
  let best = 0;
  for (let i = 1; i < z.length; i++) if (z[i] > z[best]) best = i;
  return best;
}

/** A random free start point (clear of every obstacle). rand() returns [0, 1). */
export function freePoint(w, rand) {
  for (;;) {
    const x = w.half + rand() * (w.width - 2 * w.half);
    const y = w.half + rand() * (w.height - 2 * w.half);
    if (w.blocks) {
      if (!squareHits(x, y, w.half + 0.15, w)) return [x, y];
    } else if (w.pillars.every(([cx, cy, r]) => Math.hypot(x - cx, y - cy) > r + w.half * 1.5)) {
      return [x, y];
    }
  }
}

/** The network's action probabilities (softmax of the logits). */
export function probs(net, input) {
  const z = logits(net, input);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Draw an action from the network's probabilities, like it does while training. */
export function sampleAct(net, input, rand = Math.random) {
  const p = probs(net, input);
  let r = rand();
  for (let i = 0; i < p.length; i++) {
    r -= p[i];
    if (r < 0) return i;
  }
  return p.length - 1;
}

// ------------------------------------------------------------ random maps (maps.py)

export const FAMILIES = ['pillars', 'blocks', 'walls', 'maze', 'mixed'];
const GAP = 1.2;
const WALL = 0.3;
const TRIES = 60;

/** Same sequence as maps.mulberry32 in Python: floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Primitives: ['c', cx, cy, r] or ['r', x0, y0, x1, y1].
function primGap(p, q) {
  if (p[0] === 'c' && q[0] === 'c') return Math.hypot(p[1] - q[1], p[2] - q[2]) - p[3] - q[3];
  if (p[0] === 'r' && q[0] === 'r') {
    const dx = Math.max(0, q[1] - p[3], p[1] - q[3]);
    const dy = Math.max(0, q[2] - p[4], p[2] - q[4]);
    return Math.hypot(dx, dy);
  }
  const [c, r] = p[0] === 'c' ? [p, q] : [q, p];
  const dx = Math.max(0, r[1] - c[1], c[1] - r[3]);
  const dy = Math.max(0, r[2] - c[2], c[2] - r[4]);
  return Math.hypot(dx, dy) - c[3];
}

function bounds(p) {
  return p[0] === 'c' ? [p[1] - p[3], p[2] - p[3], p[1] + p[3], p[2] + p[3]] : p.slice(1);
}

function makeShape(kind, rand, W, H) {
  const u = (lo, hi) => lo + rand() * (hi - lo);
  if (kind === 'pillar') {
    const r = u(0.5, 1.2);
    return [['c', u(GAP + r, W - GAP - r), u(GAP + r, H - GAP - r), r]];
  }
  if (kind === 'block') {
    const bw = u(0.8, 2.4);
    const bh = rand() < 0.5 ? bw : u(0.8, 2.4);
    const x0 = u(GAP, W - GAP - bw);
    const y0 = u(GAP, H - GAP - bh);
    return [['r', x0, y0, x0 + bw, y0 + bh]];
  }
  const maze = kind === 'maze';
  if (rand() < (maze ? 0.5 : 0.65)) {
    const n = maze ? u(2.0, 5.0) : u(1.6, 4.0);
    if (rand() < 0.5) {
      const x0 = u(GAP, W - GAP - n);
      const y0 = u(GAP, H - GAP - WALL);
      return [['r', x0, y0, x0 + n, y0 + WALL]];
    }
    const x0 = u(GAP, W - GAP - WALL);
    const y0 = u(GAP, H - GAP - n);
    return [['r', x0, y0, x0 + WALL, y0 + n]];
  }
  const hi = maze ? 3.6 : 2.8;
  const a = u(1.2, hi);
  const b = u(1.2, hi);
  const x0 = u(GAP, W - GAP - a);
  const y0 = u(GAP, H - GAP - b);
  const flipX = rand() < 0.5;
  const flipY = rand() < 0.5;
  const hy = flipY ? y0 + b - WALL : y0;
  const vx = flipX ? x0 + a - WALL : x0;
  return [
    ['r', x0, hy, x0 + a, hy + WALL],
    ['r', vx, y0, vx + WALL, y0 + b],
  ];
}

function plan(family, rand) {
  if (family === 'pillars') return Array(2 + Math.floor(rand() * 3)).fill('pillar');
  if (family === 'blocks') return Array(2 + Math.floor(rand() * 3)).fill('block');
  if (family === 'walls') return Array(4 + Math.floor(rand() * 4)).fill('wall');
  if (family === 'maze') return Array(12).fill('maze');
  if (family === 'mixed') {
    const n = Math.floor(rand() * 3);
    const extra = [];
    for (let i = 0; i < n; i++) extra.push(['pillar', 'block', 'wall'][Math.floor(rand() * 3)]);
    return ['pillar', 'block', 'wall', 'wall', ...extra];
  }
  throw new Error(`unknown map family ${family}`);
}

function fits(shape, placed, W, H) {
  for (const p of shape) {
    const [x0, y0, x1, y1] = bounds(p);
    if (x0 < GAP || y0 < GAP || x1 > W - GAP || y1 > H - GAP) return false;
  }
  return placed.every((s) => s.every((q) => shape.every((p) => primGap(p, q) >= GAP)));
}

/** The map for (seed, family): { width, height, pillars, blocks }. Same as maps.generate. */
export function generateMap(seed, family, W = 12, H = 8) {
  const rand = mulberry32(seed);
  for (let attempt = 0; attempt < 20; attempt++) {
    const placed = [];
    for (const kind of plan(family, rand)) {
      for (let t = 0; t < TRIES; t++) {
        const shape = makeShape(kind, rand, W, H);
        if (fits(shape, placed, W, H)) {
          placed.push(shape);
          break;
        }
      }
    }
    const prims = placed.flat();
    const map = {
      width: W,
      height: H,
      pillars: prims.filter((p) => p[0] === 'c').map((p) => p.slice(1)),
      blocks: prims.filter((p) => p[0] === 'r').map((p) => p.slice(1)),
    };
    if (placed.length && navigable(map)) return map;
  }
  throw new Error(`no navigable map for seed ${seed}`);
}

/** Does a square of half-size h centred at (x, y) overlap any obstacle? */
export function squareHits(x, y, h, w) {
  for (const [cx, cy, r] of w.pillars) {
    const qx = clamp(cx, x - h, x + h);
    const qy = clamp(cy, y - h, y + h);
    if (Math.hypot(qx - cx, qy - cy) < r) return true;
  }
  return (w.blocks ?? []).some(
    ([x0, y0, x1, y1]) => x + h > x0 && x - h < x1 && y + h > y0 && y - h < y1,
  );
}

function openCells(w, cell, h) {
  const nx = Math.round(w.width / cell);
  const ny = Math.round(w.height / cell);
  const free = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      free[j * nx + i] = squareHits((i + 0.5) * cell, (j + 0.5) * cell, h, w) ? 0 : 1;
    }
  }
  return { nx, ny, free };
}

/** Flood fill over the spots a square fits: every one of them must be reachable. */
export function navigable(w, cell = 0.1, h = 0.3) {
  const { nx, ny, free } = openCells(w, cell, h);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * cell;
      const y = (j + 0.5) * cell;
      if (x < h || x > w.width - h || y < h || y > w.height - h) free[j * nx + i] = 0;
    }
  }
  const start = free.indexOf(1);
  if (start < 0) return false;
  const seen = new Uint8Array(free.length);
  seen[start] = 1;
  const q = [start];
  let count = 1;
  for (let qi = 0; qi < q.length; qi++) {
    const k = q[qi];
    const i = k % nx;
    for (const n of [i > 0 ? k - 1 : -1, i < nx - 1 ? k + 1 : -1, k - nx, k + nx]) {
      if (n >= 0 && n < free.length && free[n] && !seen[n]) {
        seen[n] = 1;
        count++;
        q.push(n);
      }
    }
  }
  return count === free.reduce((a, b) => a + b, 0);
}

/** Distance to the nearest wall or obstacle in n directions, as a fraction of `reach`. */
export function rays(x, y, w, n = 16, reach = 6) {
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    const dx = Math.cos((2 * Math.PI * k) / n);
    const dy = Math.sin((2 * Math.PI * k) / n);
    const tx = dx > 0 ? (w.width - x) / dx : dx < 0 ? -x / dx : Infinity;
    const ty = dy > 0 ? (w.height - y) / dy : dy < 0 ? -y / dy : Infinity;
    let best = Math.min(tx, ty);
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
    out[k] = Math.min(best / reach, 1);
  }
  return out;
}

/** Is the segment at least `cl` away from every obstacle? */
export function segmentClear(ax, ay, bx, by, w, cl) {
  const dx = bx - ax;
  const dy = by - ay;
  const L2 = dx * dx + dy * dy;
  for (const [cx, cy, r] of w.pillars) {
    const t = L2 === 0 ? 0 : clamp(((cx - ax) * dx + (cy - ay) * dy) / L2, 0, 1);
    if (Math.hypot(ax + t * dx - cx, ay + t * dy - cy) < r + cl) return false;
  }
  for (const [x0, y0, x1, y1] of w.blocks ?? []) {
    let lo = 0;
    let hi = 1;
    let hit = true;
    for (const [o, d, mn, mx] of [
      [ax, dx, x0 - cl, x1 + cl],
      [ay, dy, y0 - cl, y1 + cl],
    ]) {
      if (Math.abs(d) < 1e-12) {
        if (o < mn || o > mx) {
          hit = false;
          break;
        }
        continue;
      }
      let t1 = (mn - o) / d;
      let t2 = (mx - o) / d;
      if (t1 > t2) [t1, t2] = [t2, t1];
      lo = Math.max(lo, t1);
      hi = Math.min(hi, t2);
      if (lo > hi) {
        hit = false;
        break;
      }
    }
    if (hit) return false;
  }
  return true;
}

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

/** Grid path planning for the scripted bots, same as maps.NavGrid. */
export class NavGrid {
  constructor(w, cell = 0.5, h = 0.2) {
    this.w = w;
    this.cell = cell;
    const { nx, ny, free } = openCells(w, cell, h);
    Object.assign(this, { nx, ny, free });
    const n = nx * ny;
    this.dist = new Int16Array(n * n).fill(32767);
    const q = new Int32Array(n);
    for (let s = 0; s < n; s++) {
      if (!free[s]) continue;
      const d = this.dist.subarray(s * n, (s + 1) * n);
      d[s] = 0;
      q[0] = s;
      let tail = 1;
      for (let head = 0; head < tail; head++) {
        const c = q[head];
        for (const k of this.neighbours(c)) {
          if (d[k] === 32767) {
            d[k] = d[c] + 1;
            q[tail++] = k;
          }
        }
      }
    }
    this.near = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      if (free[k]) {
        this.near[k] = k;
        continue;
      }
      const j = Math.floor(k / nx);
      const i = k % nx;
      let best = -1;
      let bd = Infinity;
      for (let m = 0; m < n; m++) {
        if (!free[m]) continue;
        const dd = (Math.floor(m / nx) - j) ** 2 + ((m % nx) - i) ** 2;
        if (dd < bd) {
          bd = dd;
          best = m;
        }
      }
      this.near[k] = best;
    }
    // Aim points: the first spot, in order of distance from each cell's centre (ties in row
    // order), where a real square fits. Same as maps.NavGrid.
    const f = 0.05;
    const fine = openCells(w, f, 0.3 + 0.02);
    const per = Math.round(cell / f);
    const key = (k) => (2 * (k % per) - per + 1) ** 2 + (2 * Math.floor(k / per) - per + 1) ** 2;
    const order = [...Array(per * per).keys()].sort((p, q) => key(p) - key(q) || p - q);
    this.aim = [];
    for (let k = 0; k < n; k++) {
      const j = Math.floor(k / nx);
      const i = k % nx;
      let pt = [(i + 0.5) * cell, (j + 0.5) * cell];
      for (const m of order) {
        const sj = Math.floor(m / per);
        const si = m % per;
        if (fine.free[(j * per + sj) * fine.nx + i * per + si]) {
          pt = [(i * per + si + 0.5) * f, (j * per + sj + 0.5) * f];
          break;
        }
      }
      this.aim.push(pt);
    }
  }

  /** Open cells one step from c, no cutting corners. */
  *neighbours(c) {
    const { nx, ny, free } = this;
    const j = Math.floor(c / nx);
    const i = c % nx;
    for (const [di, dj] of NB) {
      const ii = i + di;
      const jj = j + dj;
      if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || !free[jj * nx + ii]) continue;
      if (di && dj && !(free[j * nx + ii] && free[jj * nx + i])) continue;
      yield jj * nx + ii;
    }
  }

  cellOf(x, y) {
    const i = clamp(Math.floor(x / this.cell), 0, this.nx - 1);
    const j = clamp(Math.floor(y / this.cell), 0, this.ny - 1);
    return this.near[j * this.nx + i];
  }

  center(k) {
    return [((k % this.nx) + 0.5) * this.cell, (Math.floor(k / this.nx) + 0.5) * this.cell];
  }

  nextCell(c, t) {
    const n = this.nx * this.ny;
    let best = c;
    let bd = this.dist[t * n + c];
    for (const k of this.neighbours(c)) {
      if (this.dist[t * n + k] < bd) {
        best = k;
        bd = this.dist[t * n + k];
      }
    }
    return best;
  }

  /** Unit direction from (fx, fy) toward (tx, ty) along a shortest open path. */
  direction(fx, fy, tx, ty, cl) {
    let gx;
    let gy;
    if (segmentClear(fx, fy, tx, ty, this.w, cl)) {
      gx = tx - fx;
      gy = ty - fy;
    } else {
      const c = this.cellOf(fx, fy);
      const t = this.cellOf(tx, ty);
      const n1 = this.nextCell(c, t);
      if (n1 === c) {
        gx = tx - fx;
        gy = ty - fy;
      } else {
        const n2 = this.nextCell(n1, t);
        let [px, py] = this.aim[n2];
        if (!segmentClear(fx, fy, px, py, this.w, cl * 0.5)) [px, py] = this.aim[n1];
        gx = px - fx;
        gy = py - fy;
      }
    }
    const m = Math.hypot(gx, gy);
    return m > 1e-9 ? [gx / m, gy / m] : [0, 0];
  }
}
