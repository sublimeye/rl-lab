// Browser port of experiments/012-dodge/dodge_env.py for one drone: post 11's physics, battery,
// moving pad and sensor pipeline, plus the threats (birds, falling rocks, a proportional-
// navigation missile), the dodge-school arena and the two ways of seeing ("state": the 4 most
// urgent threats; "vision": a 4-channel depth camera). Same operations in the same order as the
// numpy version, so a flight here matches Python's (test/dodge-sim.test.mjs checks it).

export const G = 9.81;
const POINTS = [
  [-0.22, -0.18],
  [0.22, -0.18],
  [-0.35, 0.04],
  [0.35, 0.04],
];
export const LEG = 0.18;
const FOOT_BAND = 0.15;

export const K_BIRDS = 4;
export const K_ROCKS = 2;
export const K_THREATS = K_BIRDS + K_ROCKS + 1; // birds, rocks, the missile
export const K_SEE = 4;
/** Each threat slot's kind: 0 bird, 1 rock, 2 missile. */
export const KIND = [0, 0, 0, 0, 1, 1, 2];
export const KINDS = ["bird", "rock", "missile"];
/** Contact: the drone's centre within this of the threat's centre. */
export const HIT = [0.55, 0.55, 0.55, 0.55, 0.5, 0.5, 0.6];
/** The size the camera sees. */
export const LOOK = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.25];

export const BIRD_MARGIN = 6.0;
export const ROCK_TOP = 16.0;
export const ROCK_SPREAD = 1.6;
const GOLDEN = 0.6180339887498949;
export const MISSILE_SPEED = 11.0;
export const MISSILE_TURN = 0.45;
export const MISSILE_N = 3.0;
export const MISSILE_FUEL = 4.0;
export const ARENA_T = 12.0;
export const HOLD_R = 1.5;

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
  "rock",
  "missile",
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
  stickRate: 0.0,
};

export const WEATHER = {
  calm: { wind: 0, gust: 0, heavy: 0 },
  rough: { wind: 1.5, gust: 1.0, heavy: 0.35 },
};

/** dodge_env.Threats presets: how many of each threat a course gets. */
const threatsOf = (o) => ({
  birds: [0, 3],
  rocks: [0, 2],
  rockEvery: [3.0, 6.0],
  missile: 0.5,
  launch: [2.0, 7.0],
  missileSpeed: [MISSILE_SPEED, MISSILE_SPEED],
  birdScale: 1.0,
  aimBirds: false,
  arena: 0.0,
  ...o,
});
export const THREATS = {
  none: threatsOf({ birds: [0, 0], rocks: [0, 0], missile: 0.0 }),
  mix: threatsOf({}),
  school: threatsOf({
    birds: [0, 4],
    rocks: [0, 2],
    rockEvery: [2.0, 4.0],
    missile: 0.7,
    launch: [1.0, 6.0],
    arena: 0.5,
  }),
  arena: threatsOf({
    birds: [2, 4],
    rocks: [1, 2],
    rockEvery: [2.0, 4.0],
    missile: 0.8,
    launch: [1.0, 6.0],
    arena: 1.0,
  }),
  bird: threatsOf({ birds: [3, 3], rocks: [0, 0], missile: 0.0, arena: 1.0 }),
  aimed: threatsOf({
    birds: [1, 3],
    rocks: [0, 0],
    missile: 0.0,
    arena: 1.0,
    aimBirds: true,
  }),
  rock: threatsOf({
    birds: [0, 0],
    rocks: [2, 2],
    rockEvery: [2.0, 3.0],
    missile: 0.0,
    arena: 1.0,
  }),
  missile: threatsOf({
    birds: [0, 0],
    rocks: [0, 0],
    missile: 1.0,
    launch: [1.0, 4.0],
    arena: 1.0,
  }),
};

/** dodge_env.FULL (obs "state"; set obs: "vision" for FULL_VISION). */
export const FULL = {
  padMove: 0.7,
  battery: true,
  noise: 1.0,
  delay: 1,
  stack: 3,
  obs: "state",
  threats: THREATS.mix,
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

/** One course; every draw happens in make_course's order, used or not. */
export function makeCourse(seed, weather, feat, ph = PHYSICS) {
  const rand = mulberry32(seed);
  const tr = feat.threats;
  const u = (lo, hi) => lo + (hi - lo) * rand();
  const count = ([lo, hi]) => lo + Math.floor(rand() * (hi + 1 - lo));
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
  // post 12
  const arena = rand() < tr.arena;
  const gx = u(6.0, ph.width - 6.0);
  const gy = u(4.0, 9.0);
  const span = ph.width + 2 * BIRD_MARGIN;
  const nBirds = count(tr.birds);
  const birds = [];
  for (let k = 0; k < K_BIRDS; k++) {
    let by = u(2.0, 11.0);
    const fast = rand() < 0.5;
    const speed = fast ? u(4.0, 7.0) : u(1.0, 3.0);
    const dir = rand() < 0.5 ? 1.0 : -1.0;
    let bx0 = u(0.0, span);
    let bob = u(0.0, 0.5);
    const bobPeriod = u(1.5, 3.0);
    const bobPhase = u(0.0, 2 * Math.PI);
    const xNow = -BIRD_MARGIN + (bx0 % span);
    if (Math.abs(xNow - sx) < 3.0 && Math.abs(by - (sh + LEG)) < 3.0)
      bx0 += span / 2;
    if (tr.aimBirds && arena) {
      // through the point at tC (from the same draws)
      const v = speed * dir * tr.birdScale;
      const tC = 2.0 + (8.0 * (xNow + BIRD_MARGIN)) / span;
      by = gy + ((by - 6.5) / 4.5) * 0.3;
      bob = 0.1 * bob;
      const a = gx + BIRD_MARGIN - v * tC;
      bx0 = ((a % span) + span) % span;
    }
    if (k < nBirds)
      birds.push([by, speed * dir * tr.birdScale, bx0, bob, bobPeriod, bobPhase]);
  }
  const nRocks = count(tr.rocks);
  const rocks = [];
  for (let k = 0; k < K_ROCKS; k++) {
    const every = u(tr.rockEvery[0], tr.rockEvery[1]);
    const first = u(0.5, 0.5 + every);
    const drift = u(-1.0, 1.0);
    const c = rand();
    if (k < nRocks) rocks.push([every, first, drift, c]);
  }
  const hasMissile = rand() < tr.missile;
  const launch = u(tr.launch[0], tr.launch[1]);
  const side = rand() < 0.5 ? 1.0 : 0.0;
  const mY = u(1.0, 8.0);
  const mSpeed = u(tr.missileSpeed[0], tr.missileSpeed[1]);
  const charge = feat.battery ? u(0.06, 1.0) : 1.0;
  let target;
  if (arena) {
    // no landing: the target pad leaves the world
    target = [-100.0, -100.0, 0.0];
    amp = 0.0;
  } else target = [tx - tw / 2, tx + tw / 2, th];
  return {
    start: [sx - 1.0, sx + 1.0, sh],
    target,
    wind,
    gust,
    period,
    phase,
    mass,
    amp,
    padPeriod,
    padPhase,
    arena,
    goal: [gx, gy],
    birds,
    rocks,
    missile: hasMissile ? [launch, side, mY, mSpeed] : null,
    charge,
  };
}

export const thrustScale = (c) =>
  1.0 - 0.03 * (1.0 - c) - 0.2 * Math.exp(-c / 0.06);

const EMPTY_BIRD = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
const EMPTY_ROCK = [1.0, 0.0, 0.0, 0.0];

export function newDrone(course, feat, ph = PHYSICS) {
  const c = course;
  const m = c.missile ?? [0.0, 0.0, 0.0, MISSILE_SPEED];
  const d = {
    course,
    feat,
    ph,
    x: (c.start[0] + c.start[1]) / 2,
    y: c.start[2] + LEG,
    vx: 0,
    vy: 0,
    th: 0,
    om: 0,
    motor: 0,
    stick: [-1, 0],
    charge: c.charge,
    ticks: 0,
    outcome: 0,
    crash: 0,
    grounded: 1,
    groundPad: 0,
    touch: [0, 0],
    padX0: 0,
    padX1: 0,
    padVx: 0,
    arena: c.arena ? 1 : 0,
    endTick: Math.round((c.arena ? ARENA_T : ph.tMax) / ph.dt),
    // threat slots, as in Fleet: 4 birds, 2 rock droppers, the missile
    bird: Array.from({ length: K_BIRDS }, (_, k) => c.birds[k] ?? EMPTY_BIRD),
    birdOn: Array.from({ length: K_BIRDS }, (_, k) => k < c.birds.length),
    rockP: Array.from({ length: K_ROCKS }, (_, k) => c.rocks[k] ?? EMPTY_ROCK),
    rockOn: Array.from({ length: K_ROCKS }, (_, k) => k < c.rocks.length),
    rockN: [0, 0], // rocks dropped so far
    rockT: [-1e3, -1e3], // when the current one was dropped
    rockX: [0.0, 0.0], // where
    mOn: c.missile ? 1 : 0,
    mT0: m[0],
    mSide: m[1],
    mY0: m[2],
    mSpeed: m[3],
    mState: 0, // 0 waiting, 1 flying, 2 gone
    mX: 0.0,
    mY: 0.0,
    mPsi: 0.0,
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

/** Python's (and np.mod's) modulo: the result takes the divisor's sign. */
function pyMod(a, m) {
  let r = a % m;
  if (r !== 0 && r < 0 !== m < 0) r += m;
  return r;
}

/** Every threat slot now: [{ x, y, vx, vy, on, kind }] x 7 (birds, rocks, missile). */
export function threats(d) {
  const t = d.ticks * d.ph.dt;
  const span = d.ph.width + 2 * BIRD_MARGIN;
  const out = [];
  for (let k = 0; k < K_BIRDS; k++) {
    const [by, bvx, bx0, bob, bobP, bobPh] = d.bird[k];
    const wb = (2 * Math.PI) / bobP;
    out.push({
      x: -BIRD_MARGIN + pyMod(bx0 + bvx * t, span),
      y: by + bob * Math.sin(wb * t + bobPh),
      vx: bvx,
      vy: bob * wb * Math.cos(wb * t + bobPh),
      on: d.birdOn[k],
      kind: 0,
    });
  }
  for (let k = 0; k < K_ROCKS; k++) {
    const p = d.rockP[k];
    const fall = t - d.rockT[k];
    const ry = ROCK_TOP - 0.5 * G * fall * fall;
    out.push({
      x: d.rockX[k] + p[2] * fall,
      y: ry,
      vx: p[2],
      vy: -G * fall,
      on: d.rockOn[k] && d.rockN[k] > 0 && ry > -1.0,
      kind: 1,
    });
  }
  out.push({
    x: d.mX,
    y: d.mY,
    vx: d.mSpeed * Math.cos(d.mPsi),
    vy: d.mSpeed * Math.sin(d.mPsi),
    on: d.mState === 1,
    kind: 2,
  });
  return out;
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

const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

/** Rocks drop and the missile launches and steers (the tick's clock already moved on). */
function threatsMove(d) {
  const ph = d.ph;
  const t = d.ticks * ph.dt;
  for (let k = 0; k < K_ROCKS; k++) {
    const p = d.rockP[k];
    if (!(d.rockOn[k] && t >= p[1] + d.rockN[k] * p[0])) continue;
    const n = d.rockN[k];
    d.rockT[k] = p[1] + n * p[0];
    const spread = 2.0 * pyMod(n * GOLDEN + p[3], 1.0) - 1.0;
    d.rockX[k] = clip(d.x + ROCK_SPREAD * spread, 0.5, ph.width - 0.5);
    d.rockN[k] = n + 1;
  }
  if (d.mState === 1) {
    let psi = d.mPsi;
    const rx = d.x - d.mX;
    const ry = d.y - d.mY;
    const rvx = d.vx - d.mSpeed * Math.cos(psi);
    const rvy = d.vy - d.mSpeed * Math.sin(psi);
    const ldot = (rx * rvy - ry * rvx) / Math.max(rx * rx + ry * ry, 1e-6);
    psi = psi + clip(MISSILE_N * ldot, -MISSILE_TURN, MISSILE_TURN) * ph.dt;
    d.mPsi = psi;
    d.mX += d.mSpeed * Math.cos(psi) * ph.dt;
    d.mY += d.mSpeed * Math.sin(psi) * ph.dt;
    const tf = d.ticks * ph.dt;
    const mx = d.mX;
    const my = d.mY;
    if (
      tf - d.mT0 > MISSILE_FUEL ||
      my < 0.0 ||
      mx < -8.0 ||
      mx > ph.width + 8.0 ||
      my > ph.ceiling + 6.0
    )
      d.mState = 2;
  }
  if (d.mOn === 1 && d.mState === 0 && t >= d.mT0) {
    d.mX = d.mSide > 0.5 ? ph.width + 3.0 : -3.0;
    d.mY = d.mY0;
    d.mPsi = Math.atan2(d.y - d.mY, d.x - d.mX);
    d.mState = 1;
  }
}

function threatsHit(d) {
  if (d.outcome !== 0) return;
  const all = threats(d);
  for (let k = 0; k < K_THREATS; k++) {
    const h = all[k];
    const dx = h.x - d.x;
    const dy = h.y - d.y;
    if (h.on && dx * dx + dy * dy < HIT[k] * HIT[k]) {
      end(d, 6 + KIND[k]); // BIRD, ROCK, MISSILE
      return;
    }
  }
}

function limits(d) {
  if (d.outcome !== 0) return;
  const ph = d.ph;
  if (d.x < -0.5 || d.x > ph.width + 0.5 || d.y > ph.ceiling) d.outcome = 4;
  else if (d.ticks >= d.endTick) d.outcome = 5;
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
    threatsMove(d);
    threatsHit(d);
    limits(d);
    return;
  }
  d.grounded = 0;
  const before = points(d);
  let alpha = ph.kp * (tilt * ph.tiltMax - d.th) - ph.kd * d.om;
  alpha = clip(alpha, -ph.alphaMax, ph.alphaMax);
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
  threatsMove(d);
  contacts(d, before);
  threatsHit(d);
  limits(d);
}

const clip1 = (v) => clip(v, -1.0, 1.0);

/** One stick update (20 a second). */
export function step(d, throttle, tilt) {
  if (d.outcome !== 0) return;
  throttle = clip1(throttle);
  tilt = clip1(tilt);
  const r = d.ph.stickRate;
  if (r > 0) {
    throttle = d.stick[0] + clip(throttle - d.stick[0], -r, r);
    tilt = d.stick[1] + clip(tilt - d.stick[1], -r, r);
  }
  for (let i = 0; i < d.ph.substeps; i++) {
    if (d.outcome !== 0) break;
    tick(d, throttle, tilt);
  }
  d.stick = [throttle, tilt];
}

/** Where the pilot is sent: the pad's centre, or the arena's point in the air. */
export function goal(d) {
  if (d.arena === 1) return [d.course.goal[0], d.course.goal[1]];
  return [(d.padX0 + d.padX1) / 2, d.course.target[2] + LEG];
}

export function distance(d) {
  const [gx, gy] = goal(d);
  const dx = gx - d.x;
  const dy = gy - d.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Arena: within HOLD_R of the point (what the hold reward counts). */
export const holding = (d) =>
  d.arena === 1 && (d.outcome === 0 || d.outcome === 5) && distance(d) < HOLD_R;

function surface(d, x) {
  const c = d.course;
  if (c.start[0] < x && x < c.start[1]) return c.start[2];
  if (d.padX0 < x && x < d.padX1) return c.target[2];
  return 0.0;
}

// --- what the pilot sees ---------------------------------------------------------------------
export const N_BASE = 18;
export const N_SLOT = 8;
export const N_STATE = N_BASE + K_SEE * N_SLOT;
export const N_OWN = 13;
export const N_VISION = N_OWN + 4 * N_RAYS;

function baseInputs(d) {
  const c = d.course;
  const [gx, gy] = goal(d);
  const under = d.y - LEG - surface(d, d.x);
  const half = d.arena === 1 ? 0.0 : (c.target[1] - c.target[0]) / 2;
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
    clip(under / 5, -1.0, 2.0),
    d.stick[0],
    d.stick[1],
    d.padVx / 3,
    d.charge * 2 - 1,
    d.arena * 1.0,
    d.mState === 1 ? 1.0 : 0.0,
  ];
}

/** The most urgent threats first (distance over closing speed; receding ones count double their
 * distance), K_SEE slots of dx, dy, vx, vy, present, bird, rock, missile. */
export function threatSlots(d) {
  const keyed = threats(d).map((h, i) => {
    const dx = h.x - d.x;
    const dy = h.y - d.y;
    const r = Math.sqrt(dx * dx + dy * dy);
    const closing =
      -(dx * (h.vx - d.vx) + dy * (h.vy - d.vy)) / Math.max(r, 1e-6);
    const key = h.on ? r / Math.max(closing, 0.5) : Infinity;
    return { i, dx, dy, vx: h.vx, vy: h.vy, kind: h.kind, key };
  });
  // np.argsort(kind="stable"): ties keep slot order
  keyed.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : a.i - b.i));
  const out = [];
  for (let s = 0; s < K_SEE; s++) {
    const h = keyed[s];
    if (Number.isFinite(h.key))
      out.push(
        h.dx / 10,
        h.dy / 10,
        h.vx / 5,
        h.vy / 5,
        1,
        h.kind === 0 ? 1 : 0,
        h.kind === 1 ? 1 : 0,
        h.kind === 2 ? 1 : 0,
      );
    else out.push(2.0, 0, 0, 0, 0, 0, 0, 0);
  }
  return out;
}

const STATE_SIGMA = [
  0.005, 0.005, 0.02, 0.02, 0.04, 0.02, 0.02, 0, 0.005, 0.005, 0.005, 0.01, 0,
  0, 0.02, 0.01, 0, 0,
  ...Array.from({ length: K_SEE }, () => [0.005, 0.005, 0.02, 0.02, 0, 0, 0, 0]).flat(),
];
const VISION_SIGMA = [
  0.02, 0.02, 0.04, 0.02, 0.02, 0.01, 0.005, 0, 0, 0.005, 0.005, 0, 0,
];

/** The depth camera, per ray: { depth (0..1), pad, threat, shadow (0/1), angle (world,
 * radians), dist (m) }. A shadow is where a ray ends on a surface's top under a falling rock. */
export function rays(d) {
  const out = [];
  const c = d.course;
  const span = 2 * RAY_SPAN;
  const step = span / (N_RAYS - 1);
  const all = threats(d);
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
    let kind = 0; // 0 nothing/surface, 1 marked pad, 2 threat
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
    for (let j = 0; j < K_THREATS; j++) {
      const h = all[j];
      if (!h.on) continue;
      const cx = h.x - d.x;
      const cy = h.y - d.y;
      const b = dx * cx + dy * cy;
      const cc = cx * cx + cy * cy - LOOK[j] * LOOK[j];
      const disc = b * b - cc;
      const t = b - Math.sqrt(Math.max(disc, 0.0));
      if (disc >= 0 && t > 0 && t < best) {
        best = t;
        kind = 2;
      }
    }
    const hx = d.x + best * dx;
    const hy = d.y + best * dy;
    let shadow = 0;
    if (kind !== 2 && best < RAY_RANGE) {
      let sTop = 0.0;
      if (c.start[0] < hx && hx < c.start[1]) sTop = c.start[2];
      else if (d.padX0 < hx && hx < d.padX1) sTop = c.target[2];
      if (Math.abs(hy - sTop) < 1e-6)
        for (let j = K_BIRDS; j < K_BIRDS + K_ROCKS; j++) {
          const h = all[j];
          if (h.on && Math.abs(h.x - hx) < 0.45 && h.y > hy) shadow = 1;
        }
    }
    out.push({
      depth: best / RAY_RANGE,
      pad: kind === 1 ? 1 : 0,
      threat: kind === 2 ? 1 : 0,
      shadow,
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
    const o = [...baseInputs(d), ...threatSlots(d)];
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
    d.arena * 1.0,
    d.mState === 1 ? 1.0 : 0.0,
  ];
  const r = rays(d);
  let depth = r.map((x) => x.depth);
  if (noise) {
    own = own.map((v, i) => v + gauss() * noise * VISION_SIGMA[i]);
    depth = depth.map((v) => clip(v + gauss() * noise * 0.01, 0.0, 1.0));
  }
  return [
    ...own,
    ...depth,
    ...r.map((x) => x.pad),
    ...r.map((x) => x.threat),
    ...r.map((x) => x.shadow),
  ];
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
  return eyes.frames.flat().map((v) => clip(v, -5, 5));
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
  let h = obs.map((v) => clip(v, -5, 5));
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
