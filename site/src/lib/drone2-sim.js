// Browser port of experiments/011-drone2/fleet_env.py for one drone: post 10's physics plus a
// moving pad, birds, a battery, sensor delay and memory, and the two ways of seeing (exact
// "state" and the depth-camera "vision"). Same operations in the same order as the numpy
// version, so a flight here matches Python's (test/drone2-sim.test.mjs checks it).

export const G = 9.81;
const POINTS = [
  [-0.22, -0.18],
  [0.22, -0.18],
  [-0.35, 0.04],
  [0.35, 0.04],
];
export const LEG = 0.18;
const FOOT_BAND = 0.15;
export const K_BIRDS = 3;
const BIRD_HIT = 0.55;
export const N_RAYS = 32;
export const RAY_SPAN = 150 * (Math.PI / 180);
export const RAY_RANGE = 15.0;
export const OUTCOMES = [
  "flying",
  "landed",
  "crashed",
  "water",
  "lost",
  "timeout",
  "bird",
];

export const PHYSICS = {
  dt: 1 / 60,
  substeps: 3,
  tMax: 20.0,
  thrustMax: 2.0,
  motorTau: 0.08,
  tiltMax: 0.5236,
  kp: 70.0,
  kd: 14.0,
  alphaMax: 40.0,
  dragLin: 0.15,
  dragQuad: 0.04,
  landVy: 1.5,
  landVx: 1.5,
  landTilt: 0.35,
  width: 24.0,
  ceiling: 13.0,
};

export const WEATHER = {
  calm: { wind: 0, gust: 0, heavy: 0 },
  rough: { wind: 1.5, gust: 1.0, heavy: 0.35 },
};

/** fleet_env.FULL / FULL_VISION; switch pieces off for the game's options. */
export const FULL = {
  padMove: 0.7,
  birds: true,
  battery: true,
  noise: 1.0,
  delay: 1,
  stack: 3,
  obs: "state",
};

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeCourse(seed, weather, feat, ph = PHYSICS) {
  const rand = mulberry32(seed);
  const u = (lo, hi) => lo + (hi - lo) * rand();
  let sx = u(2.0, 5.0);
  const sh = u(0.5, 4.0);
  const tw = u(1.6, 2.6);
  let tx = u(sx + 8.0, ph.width - 2.5);
  const th = u(0.5, 7.0);
  if (rand() < 0.5) {
    sx = ph.width - sx;
    tx = ph.width - tx;
  }
  const wind = u(-1.0, 1.0) * weather.wind;
  const gust = u(0.0, 1.0) * weather.gust;
  const period = u(2.0, 5.0);
  const phase = u(0.0, 2 * Math.PI);
  const mass = 1.0 + u(0.0, 1.0) * weather.heavy;
  const moves = rand() < feat.padMove;
  let amp = u(0.8, 2.5);
  const room = Math.min(
    tx - tw / 2 - 1.0,
    ph.width - 1.0 - (tx + tw / 2),
    Math.abs(tx - sx) - 1.0 - tw / 2 - 1.5,
  );
  amp = moves ? Math.max(0.0, Math.min(amp, room)) : 0.0;
  const padPeriod = u(4.0, 9.0);
  const padPhase = u(0.0, 2 * Math.PI);
  const lo = feat.minBirds ?? 0;
  const nBirds = feat.birds ? lo + Math.floor(rand() * (K_BIRDS + 1 - lo)) : 0;
  const birds = [];
  for (let k = 0; k < K_BIRDS; k++) {
    const by = u(2.0, 11.0);
    const fast = rand() < 0.5;
    const speed = fast ? u(4.0, 7.0) : u(1.0, 3.0);
    const dir = rand() < 0.5 ? 1.0 : -1.0;
    let bx0 = u(0.0, ph.width + 4.0);
    const bob = u(0.0, 0.5);
    const bobPeriod = u(1.5, 3.0);
    const bobPhase = u(0.0, 2 * Math.PI);
    const xNow = -2.0 + (bx0 % (ph.width + 4.0));
    if (Math.abs(xNow - sx) < 3.0 && Math.abs(by - (sh + LEG)) < 3.0)
      bx0 += (ph.width + 4.0) / 2;
    if (k < nBirds)
      birds.push([by, speed * dir, bx0, bob, bobPeriod, bobPhase]);
  }
  const charge = feat.battery ? u(0.06, 1.0) : 1.0;
  return {
    start: [sx - 1.0, sx + 1.0, sh],
    target: [tx - tw / 2, tx + tw / 2, th],
    wind,
    gust,
    period,
    phase,
    mass,
    amp,
    padPeriod,
    padPhase,
    birds,
    charge,
  };
}

export const thrustScale = (c) =>
  1.0 - 0.03 * (1.0 - c) - 0.2 * Math.exp(-c / 0.06);

export function newDrone(course, feat, ph = PHYSICS) {
  const d = {
    course,
    feat,
    ph,
    x: (course.start[0] + course.start[1]) / 2,
    y: course.start[2] + LEG,
    vx: 0,
    vy: 0,
    th: 0,
    om: 0,
    motor: 0,
    stick: [-1, 0],
    charge: course.charge,
    ticks: 0,
    outcome: 0,
    crash: 0,
    grounded: 1,
    groundPad: 0,
    touch: [0, 0],
    padX0: 0,
    padX1: 0,
    padVx: 0,
  };
  pad(d);
  return d;
}

export const time = (d) => d.ticks * d.ph.dt;
export const flying = (d) => d.outcome === 0;

function pad(d) {
  const c = d.course;
  const t = d.ticks * d.ph.dt;
  const w = (2 * Math.PI) / c.padPeriod;
  const off = c.amp * Math.sin(w * t + c.padPhase);
  d.padX0 = c.target[0] + off;
  d.padX1 = c.target[1] + off;
  d.padVx = c.amp * w * Math.cos(w * t + c.padPhase);
}

function pyMod(a, m) {
  let r = a % m;
  if (r !== 0 && r < 0 !== m < 0) r += m;
  return r;
}

/** Birds' [x, y, vx, vy] now. */
export function birdPos(d) {
  const t = d.ticks * d.ph.dt;
  const span = d.ph.width + 4.0;
  return d.course.birds.map(([by, bvx, bx0, bob, bobP, bobPh]) => {
    const wb = (2 * Math.PI) / bobP;
    return [
      -2.0 + pyMod(bx0 + bvx * t, span),
      by + bob * Math.sin(wb * t + bobPh),
      bvx,
      bob * wb * Math.cos(wb * t + bobPh),
    ];
  });
}

export function points(d) {
  const c = Math.cos(d.th);
  const s = Math.sin(d.th);
  return POINTS.map(([lx, ly]) => [
    d.x + lx * c + ly * s,
    d.y - lx * s + ly * c,
  ]);
}

function end(d, outcome) {
  d.touch = [d.vx, d.vy];
  d.outcome = outcome;
}

function birdsHit(d) {
  if (!d.feat.birds || d.outcome !== 0) return;
  for (const [bx, by] of birdPos(d)) {
    const dx = bx - d.x;
    const dy = by - d.y;
    if (dx * dx + dy * dy < BIRD_HIT * BIRD_HIT) {
      end(d, 6);
      return;
    }
  }
}

function limits(d) {
  if (d.outcome !== 0) return;
  const ph = d.ph;
  if (d.x < -0.5 || d.x > ph.width + 0.5 || d.y > ph.ceiling) d.outcome = 4;
  else if (d.ticks >= Math.round(ph.tMax / ph.dt)) d.outcome = 5;
}

function contacts(d, before) {
  const ph = d.ph;
  const pts = points(d);
  const c = d.course;
  const pads = [
    [c.start[0], c.start[1], c.start[2]],
    [d.padX0, d.padX1, c.target[2]],
  ];
  let landing = -1;
  for (let k = 0; k < 4; k++) {
    const [px, py] = pts[k];
    for (let p = 0; p < 2; p++) {
      const [x0, x1, top] = pads[p];
      if (!(x0 < px && px < x1 && py < top)) continue;
      const fromAbove = k < 2 && before[k][1] >= top - FOOT_BAND;
      if (fromAbove) {
        if (d.vy <= 0.0) landing = p;
      } else {
        d.crash = 2;
        end(d, 2);
        return;
      }
    }
    if (py < 0.0) {
      end(d, 3);
      return;
    }
  }
  if (landing < 0) return;
  const relVx = d.vx - (landing === 1 ? d.padVx : 0.0);
  const gentle =
    d.vy >= -ph.landVy &&
    Math.abs(relVx) <= ph.landVx &&
    Math.abs(d.th) <= ph.landTilt;
  if (!gentle) {
    d.crash = 1;
    end(d, 2);
  } else if (landing === 1 && d.padX0 <= d.x && d.x <= d.padX1) end(d, 1);
  else {
    d.grounded = 1;
    d.groundPad = landing;
    d.th = 0.0;
    d.om = 0.0;
    d.y = (landing === 1 ? c.target[2] : c.start[2]) + LEG;
    d.vy = 0.0;
    d.vx *= 0.85;
  }
}

export function tick(d, throttle, tilt) {
  const ph = d.ph;
  const co = d.course;
  const dt = ph.dt;
  let cmd = (throttle + 1.0) / 2.0;
  if (d.feat.battery) cmd = d.charge > 0.0 ? cmd : 0.0;
  d.motor += ((cmd - d.motor) * dt) / ph.motorTau;
  let acc = d.motor * ph.thrustMax * G;
  if (d.feat.battery) {
    acc = acc * thrustScale(d.charge);
    d.charge = Math.max(0.0, d.charge - 0.0085 * d.motor ** 1.5 * dt);
  }
  acc = acc / co.mass;
  if (d.grounded === 1 && acc <= G) {
    d.vx *= 0.85;
    d.x += d.vx * dt;
    const old = d.padX0;
    d.ticks += 1;
    pad(d);
    d.x += d.groundPad === 1 ? d.padX0 - old : 0.0;
    birdsHit(d);
    limits(d);
    return;
  }
  d.grounded = 0;
  const before = points(d);
  let alpha = ph.kp * (tilt * ph.tiltMax - d.th) - ph.kd * d.om;
  alpha = Math.min(Math.max(alpha, -ph.alphaMax), ph.alphaMax);
  alpha *= Math.min(1.0, 0.3 + 1.4 * d.motor);
  d.om += alpha * dt;
  d.th += d.om * dt;
  const speed = Math.sqrt(d.vx * d.vx + d.vy * d.vy);
  const t = d.ticks * dt;
  const air =
    co.wind + co.gust * Math.sin((2 * Math.PI * t) / co.period + co.phase);
  let ax =
    acc * Math.sin(d.th) - ph.dragLin * d.vx - ph.dragQuad * d.vx * speed;
  const ay =
    acc * Math.cos(d.th) - G - ph.dragLin * d.vy - ph.dragQuad * d.vy * speed;
  ax += air / co.mass;
  d.vx += ax * dt;
  d.vy += ay * dt;
  d.x += d.vx * dt;
  d.y += d.vy * dt;
  d.ticks += 1;
  pad(d);
  contacts(d, before);
  birdsHit(d);
  limits(d);
}

const clip1 = (v) => Math.max(-1.0, Math.min(1.0, v));

/** One stick update (20 a second). */
export function step(d, throttle, tilt) {
  if (d.outcome !== 0) return;
  throttle = clip1(throttle);
  tilt = clip1(tilt);
  for (let i = 0; i < d.ph.substeps; i++) {
    if (d.outcome !== 0) break;
    tick(d, throttle, tilt);
  }
  d.stick = [throttle, tilt];
}

export function goal(d) {
  return [(d.padX0 + d.padX1) / 2, d.course.target[2] + LEG];
}

export function distance(d) {
  const [gx, gy] = goal(d);
  const dx = gx - d.x;
  const dy = gy - d.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function surface(d, x) {
  const c = d.course;
  if (c.start[0] < x && x < c.start[1]) return c.start[2];
  if (d.padX0 < x && x < d.padX1) return c.target[2];
  return 0.0;
}

// --- what the pilot sees ---------------------------------------------------------------------
function obsV010(d) {
  const c = d.course;
  const [gx, gy] = goal(d);
  const under = d.y - LEG - surface(d, d.x);
  const half = (c.target[1] - c.target[0]) / 2;
  const sCx = (c.start[0] + c.start[1]) / 2;
  return [
    (gx - d.x) / 10,
    (gy - d.y) / 10,
    d.vx / 5,
    d.vy / 5,
    d.th / d.ph.tiltMax,
    d.om / 5,
    d.motor * 2 - 1,
    half / 1.3,
    (sCx - d.x) / 10,
    (c.start[2] + LEG - d.y) / 10,
    d.y / 10,
    Math.max(-1.0, Math.min(2.0, under / 5)),
    d.stick[0],
    d.stick[1],
  ];
}

function nearestBirds(d) {
  const out = [];
  const bp = d.feat.birds ? birdPos(d) : [];
  const near = bp
    .map((b, i) => {
      const dx = b[0] - d.x;
      const dy = b[1] - d.y;
      return { i, dx, dy, vx: b[2], vy: b[3], d2: dx * dx + dy * dy };
    })
    .sort((a, b) => a.d2 - b.d2 || a.i - b.i);
  for (let k = 0; k < 2; k++) {
    const b = near[k];
    if (b) out.push(b.dx / 10, b.dy / 10, b.vx / 5, b.vy / 5, 1);
    else out.push(2.0, 0, 0, 0, 0);
  }
  return out;
}

const STATE_SIGMA = [
  0.005, 0.005, 0.02, 0.02, 0.04, 0.02, 0.02, 0, 0.005, 0.005, 0.005, 0.01, 0,
  0, 0.02, 0.01, 0.005, 0.005, 0.02, 0.02, 0, 0.005, 0.005, 0.02, 0.02, 0,
];
const VISION_SIGMA = [
  0.02, 0.02, 0.04, 0.02, 0.02, 0.01, 0.005, 0, 0, 0.005, 0.005,
];

/** The depth camera: { depth (0..1), pad (0/1), bird (0/1), angle (world, radians) } per ray. */
export function rays(d) {
  const out = [];
  const c = d.course;
  const span = 2 * RAY_SPAN;
  const step = span / (N_RAYS - 1);
  const bp = d.feat.birds ? birdPos(d) : [];
  const boxes = [
    [c.start[0], c.start[1], c.start[2], 0],
    [d.padX0, d.padX1, c.target[2], 1],
  ];
  for (let k = 0; k < N_RAYS; k++) {
    const body = k === N_RAYS - 1 ? RAY_SPAN : k * step + -RAY_SPAN;
    const a = body - d.th;
    const dx = Math.sin(a);
    const dy = -Math.cos(a);
    let best = RAY_RANGE;
    let kind = 0;
    const tw = dy < 0 ? -d.y / dy : Infinity;
    best = Math.min(best, tw > 0 ? tw : Infinity);
    for (const [x0, x1, top, kk] of boxes) {
      const tx1 = (x0 - d.x) / dx;
      const tx2 = (x1 - d.x) / dx;
      const ty1 = (-1.0 - d.y) / dy;
      const ty2 = (top - d.y) / dy;
      const tIn = Math.max(Math.min(tx1, tx2), Math.min(ty1, ty2));
      const tOut = Math.min(Math.max(tx1, tx2), Math.max(ty1, ty2));
      if (tIn <= tOut && tIn > 0 && tIn < best) {
        best = tIn;
        kind = kk;
      }
    }
    for (const [bx, by] of bp) {
      const cx = bx - d.x;
      const cy = by - d.y;
      const b = dx * cx + dy * cy;
      const cc = cx * cx + cy * cy - 0.3 * 0.3;
      const disc = b * b - cc;
      const t = b - Math.sqrt(Math.max(disc, 0.0));
      if (disc >= 0 && t > 0 && t < best) {
        best = t;
        kind = 2;
      }
    }
    out.push({
      depth: best / RAY_RANGE,
      pad: kind === 1 ? 1 : 0,
      bird: kind === 2 ? 1 : 0,
      angle: a,
      dist: best,
    });
  }
  return out;
}

const gauss = () =>
  Math.sqrt(-2 * Math.log(1 - Math.random())) *
  Math.cos(2 * Math.PI * Math.random());

/** One reading for the pilot (before delay and memory). noise: scale or 0; gps: [ex, ey]. */
export function sense(d, noise = 0, gps = [0, 0]) {
  if (d.feat.obs === "state") {
    const o = [
      ...obsV010(d),
      d.padVx / 3,
      d.charge * 2 - 1,
      ...nearestBirds(d),
    ];
    return noise ? o.map((v, i) => v + gauss() * noise * STATE_SIGMA[i]) : o;
  }
  const [gx, gy] = goal(d);
  let own = [
    d.vx / 5,
    d.vy / 5,
    d.th / d.ph.tiltMax,
    d.om / 5,
    d.motor * 2 - 1,
    d.charge * 2 - 1,
    d.y / 10,
    d.stick[0],
    d.stick[1],
    (gx + gps[0] - d.x) / 10,
    (gy + gps[1] - d.y) / 10,
  ];
  const r = rays(d);
  let depth = r.map((x) => x.depth);
  if (noise) {
    own = own.map((v, i) => v + gauss() * noise * VISION_SIGMA[i]);
    depth = depth.map((v) =>
      Math.max(0, Math.min(1, v + gauss() * noise * 0.01)),
    );
  }
  return [...own, ...depth, ...r.map((x) => x.pad), ...r.map((x) => x.bird)];
}

/** Sensor delay line plus the pilot's memory of its last `stack` readings. */
export function newEyes(d, noise = 0) {
  const gps = noise ? [gauss() * 0.3 * noise, gauss() * 0.3 * noise] : [0, 0];
  const first = sense(d, noise, gps);
  return {
    noise,
    gps,
    pending: Array.from({ length: d.feat.delay }, () => first),
    frames: Array.from({ length: d.feat.stack }, () => first),
  };
}

export function look(d, eyes) {
  return eyes.frames.flat().map((v) => Math.max(-5, Math.min(5, v)));
}

export function push(d, eyes) {
  let now = sense(d, eyes.noise, eyes.gps);
  if (eyes.pending.length) {
    eyes.pending.push(now);
    now = eyes.pending.shift();
  }
  eyes.frames.unshift(now);
  eyes.frames.pop();
  return look(d, eyes);
}

/** The exported actor (tanh MLP; output clipped to the stick range). */
export function netForward(net, obs) {
  let h = obs.map((v) => Math.max(-5, Math.min(5, v)));
  const n = net.layers.length;
  net.layers.forEach((L, i) => {
    const out = new Array(L.b.length);
    for (let r = 0; r < L.b.length; r++) {
      const row = L.w[r];
      let z = L.b[r];
      for (let c = 0; c < row.length; c++) z += row[c] * h[c];
      out[r] = i < n - 1 ? Math.tanh(z) : z;
    }
    h = out;
  });
  return h.map(clip1);
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

function decodeHalves(b64) {
  const bin =
    typeof atob === "function"
      ? atob(b64)
      : Buffer.from(b64, "base64").toString("binary");
  const out = new Array(bin.length / 2);
  for (let i = 0; i < out.length; i++)
    out[i] = halfToFloat(
      bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8),
    );
  return out;
}

/** A model file's layers ({shape, w, b} as float16 base64) as plain rows. */
export function unpackNet(model) {
  return {
    ...model,
    layers: model.layers.map((L) => {
      const [rows, cols] = L.shape;
      const flat = decodeHalves(L.w);
      const w = [];
      for (let r = 0; r < rows; r++)
        w.push(flat.slice(r * cols, (r + 1) * cols));
      return { w, b: decodeHalves(L.b) };
    }),
  };
}
