// Browser port of experiments/003-tag/tag_env.py. Keep the two in step:
// site/test/tag-sim.test.mjs replays a trajectory exported from Python and checks every state.

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

/** world = { width, height, pillars: [[cx, cy, r]], half, dt, accel, drag, maxSpeed, maxSteps } */
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

/** chaser = { kind: 'pursuit' | 'intercept', ratio, reaction, dirs8, lead } */
export function chaserDirection(c, me, prey, w) {
  let tx = prey.x;
  let ty = prey.y;
  if (c.kind === 'intercept') {
    const dist = Math.hypot(prey.x - me.x, prey.y - me.y);
    const t = Math.min((c.lead * dist) / (w.maxSpeed * c.ratio), 1.5);
    tx = clamp(prey.x + prey.vx * t, w.half, w.width - w.half);
    ty = clamp(prey.y + prey.vy * t, w.half, w.height - w.half);
  }
  let [dx, dy] = steer(me.x, me.y, tx, ty, w, w.half * 1.5);
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

/** The evader's 9 inputs, same order and scaling as Python's observe(). */
export function observe(e, c, w) {
  const dx = c.x - e.x;
  const dy = c.y - e.y;
  return [
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

/** A random free start point (not inside a pillar). rand() returns [0, 1). */
export function freePoint(w, rand) {
  for (;;) {
    const x = w.half + rand() * (w.width - 2 * w.half);
    const y = w.half + rand() * (w.height - 2 * w.half);
    if (w.pillars.every(([cx, cy, r]) => Math.hypot(x - cx, y - cy) > r + w.half * 1.5)) {
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
