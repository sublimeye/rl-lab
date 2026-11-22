// Browser port of experiments/014-gauntlet/gauntlet_env.py for one drone: a 60-120 m flight from
// the start pillar to a landing pad, with encounters at waypoints (aimed birds that home and then
// commit, the missile, crossing flocks), post 13's air-frogs (one guarding the pad, sometimes one
// mid-route), a 40 m sight range, and the pilot's inputs (3 stacked frames, 50 ms lag, 3 s of frog
// history). Same operations in the same order as the numpy version, so a flight here matches
// Python's (test/gauntlet-sim.test.mjs checks it).

export const G = 9.81;
const POINTS = [
  [-0.22, -0.18],
  [0.22, -0.18],
  [-0.35, 0.04],
  [0.35, 0.04],
];
export const LEG = 0.18;
const FOOT_BAND = 0.15;

export const SIGHT = 40.0; // threats, frogs and the pad are seen within this distance
export const MARGIN = 12.0; // world beyond the start pillar and the pad

// pursuers: 6 bird slots, then 2 missile slots
export const K_BIRDS = 6;
export const K_MISSILES = 2;
export const K_PURSUERS = K_BIRDS + K_MISSILES;
/** Each pursuer slot's kind: 0 bird, 1 missile. */
export const P_KIND = [...Array(K_BIRDS).fill(0), ...Array(K_MISSILES).fill(1)];
export const P_HIT = P_KIND.map((k) => (k === 0 ? 0.55 : 0.6));
export const BIRD_TURN = 2.0;
export const BIRD_N = 3.0;
export const BIRD_LIFE = 14.0;
export const MISSILE_SPEED = 11.0;
export const MISSILE_TURN = 0.45;
export const MISSILE_N = 3.0;
export const MISSILE_FUEL = 4.0;
export const GONE_R = 50.0;

export const ENCOUNTERS = ["calm", "bird", "two birds", "missile", "bird + missile", "flock"];
const [CALM, BIRD1, BIRD2, MISSILE1, COMBO, FLOCK] = [0, 1, 2, 3, 4, 5];
export const MAX_ENC = 16;
export const N_ENC_RAND = 8;

export const OUTCOMES = [
  "flying",
  "landed",
  "crashed",
  "water",
  "lost",
  "timeout",
  "bird",
  "missile",
  "frog",
];
const [LANDED, CRASHED, WATER, LOST, TIMEOUT, BIRD, MISSILE, FROG] = [1, 2, 3, 4, 5, 6, 7, 8];

export const K_FROGS = 2; // slot 0 mid-route, slot 1 guarding the pad
export const FROG_HIT = 0.6;
export const FROG_REACH = 2.0;
export const FROG_LEAD = 0.35;
export const LEAP = 0.35;
export const HOLD = 0.5;
export const BACK = 1.0;
export const FROG_JITTER = 0.1;
export const HIST = 16;
export const HIST_EVERY = 4;
const GOLDEN = 0.6180339887498949;
const BOB_W = (2 * Math.PI) / 1.7;

export const PHYSICS = {
  dt: 1 / 60,
  substeps: 3,
  tMax: 60.0,
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
  ceiling: 13.0,
};

export const WEATHER = {
  calm: { wind: 0, gust: 0, heavy: 0 },
  rough: { wind: 1.5, gust: 1.0, heavy: 0.35 },
};

export const TEST_MIX = [0.1, 0.2, 0.2, 0.15, 0.2, 0.15];

/** gauntlet_env.FULL: the test course. */
export const FULL = {
  battery: true,
  noise: 1.0,
  delay: 1,
  stack: 3,
  memory: "hist",
  route: [60.0, 120.0],
  first: [8.0, 14.0],
  spacing: [18.0, 28.0],
  mix: TEST_MIX,
  routeFrog: 0.5,
  guard: 1.0,
  frogPeriod: [3.0, 4.0],
  missileSpeed: [MISSILE_SPEED, MISSILE_SPEED],
  birdLock: 1.0,
  presence: 1.0,
  predict: false,
  switch: null,
};
/** No threats at all. */
export const CLEAR = { ...FULL, mix: [1.0, 0, 0, 0, 0, 0], routeFrog: 0.0, guard: 0.0 };
/** evaluate.only(k): encounter kind k at every waypoint, no frogs. */
export const only = (k) => ({
  ...FULL,
  mix: ENCOUNTERS.map((_, i) => (i === k ? 1.0 : 0.0)),
  routeFrog: 0.0,
  guard: 0.0,
});
/** evaluate.TESTS["bird + missile"]. */
export const COMBO_TEST = only(4);

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One course; every draw happens in make_course's order. */
export function makeCourse(seed, weather, feat, ph = PHYSICS) {
  const rand = mulberry32(seed);
  const u = (lo, hi) => lo + (hi - lo) * rand();
  const length = u(feat.route[0], feat.route[1]);
  const sh = u(0.5, 4.0);
  const tw = u(1.6, 2.6);
  const th = u(0.5, 7.0);
  const dir = rand() < 0.5 ? 1.0 : -1.0;
  const width = length + 2 * MARGIN;
  const sx = dir > 0 ? MARGIN : width - MARGIN;
  const tx = sx + dir * length;
  const wind = u(-1.0, 1.0) * weather.wind;
  const gust = u(0.0, 1.0) * weather.gust;
  const period = u(2.0, 5.0);
  const phase = u(0.0, 2 * Math.PI);
  const mass = 1.0 + u(0.0, 1.0) * weather.heavy;
  const charge = feat.battery ? u(0.5, 1.0) : 1.0;
  const mSpeed = u(feat.missileSpeed[0], feat.missileSpeed[1]);
  const frogs = [];
  for (let k = 0; k < K_FROGS; k++) {
    const present = rand() < (k === 0 ? feat.routeFrog : feat.guard) * feat.presence;
    const frac = u(0.35, 0.65);
    let yPerch = u(3.0, 9.0);
    const fPeriod = u(feat.frogPeriod[0], feat.frogPeriod[1]);
    const first = u(1.0, 1.0 + fPeriod);
    const c = rand();
    const bobPhase = u(0.0, 2 * Math.PI);
    let xCol;
    if (k === 0) xCol = sx + dir * length * frac;
    else {
      xCol = tx;
      yPerch = th + 2.5 + (yPerch - 3.0) / 3.0;
    }
    frogs.push(present ? [xCol, yPerch, fPeriod, first, c, bobPhase] : null);
  }
  // waypoints: kind = np.searchsorted(cum, r, side="right"), clipped to the last kind
  let total = 0;
  for (const w of feat.mix) total += w;
  const cum = [];
  let acc = 0;
  for (const w of feat.mix) cum.push((acc += w / total));
  const encounters = [];
  let at = u(feat.first[0], feat.first[1]);
  while (at < length - 12.0 && encounters.length < MAX_ENC) {
    const r = rand();
    let kind = cum.findIndex((v) => v > r);
    if (kind < 0) kind = cum.length - 1;
    if (rand() >= feat.presence) kind = CALM;
    const params = Array.from({ length: N_ENC_RAND }, () => rand());
    if (frogs[0] !== null) {
      // keep encounters clear of the mid-route frog's column
      const col = Math.abs(frogs[0][0] - sx);
      if (col - 12.0 < at && at < col + 4.0) kind = CALM;
    }
    encounters.push([at, kind, params]);
    at += u(feat.spacing[0], feat.spacing[1]);
  }
  return {
    start: [sx - 1.0, sx + 1.0, sh],
    target: [tx - tw / 2, tx + tw / 2, th],
    dir,
    length,
    world: [0.0, width],
    wind,
    gust,
    period,
    phase,
    mass,
    charge,
    mSpeed,
    birdLock: feat.birdLock,
    frogs,
    encounters,
    switch: feat.switch ?? null,
  };
}

export const thrustScale = (c) => 1.0 - 0.03 * (1.0 - c) - 0.2 * Math.exp(-c / 0.06);

const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const clip1 = (v) => clip(v, -1.0, 1.0);
const EMPTY_FROG = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
const fill = (v) => Array.from({ length: K_PURSUERS }, () => v);

/** Fleet.set_course for one drone. */
export function newDrone(course, feat, ph = PHYSICS) {
  const c = course;
  return {
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
    endTick: Math.round(ph.tMax / ph.dt),
    encNext: 0,
    encCount: ENCOUNTERS.map(() => 0), // met so far, per kind
    hitEnc: -1, // the encounter kind whose pursuer hit the drone
    // pursuers: 6 birds, 2 missiles
    pOn: fill(false),
    pX: fill(0.0),
    pY: fill(0.0),
    pPsi: fill(0.0),
    pSpd: fill(1.0),
    pSteer: fill(false), // still homing (an aimed bird stops when it commits)
    pEnd: fill(0.0), // leaves at this time
    pHostile: fill(false), // an aimed bird or the missile
    pDone: fill(false), // counted as beaten
    pEnc: fill(0), // the encounter kind it came from
    beaten: [0, 0], // aimed birds, missiles got away from
    // frogs: x_col, y_perch, period, first, c, bob_phase
    frogP: c.frogs.map((f) => f ?? EMPTY_FROG),
    frogOn: c.frogs.map((f) => f !== null),
    frogN: [0, 0],
    frogS: [-1e3, -1e3],
    frogSx: [0.0, 0.0],
    frogSy: [0.0, 0.0],
    frogTx: [0.0, 0.0],
    frogTy: [0.0, 0.0],
    frogPast: [false, false], // got past it (the pad's: landed)
    swAfter: c.switch ? c.switch[0] : 10 ** 6,
    swRatio: c.switch ? c.switch[1] : 1.0,
  };
}

export const time = (d) => d.ticks * d.ph.dt;
export const flying = (d) => d.outcome === 0;
const startX = (d) => (d.course.start[0] + d.course.start[1]) / 2;

/** Metres flown along the route (from the start pillar's centre). */
export const progress = (d) => d.course.dir * (d.x - startX(d));

// --- encounters --------------------------------------------------------------------------
function spawn(d, slot, x, y, psi, spd, steer, hostile, end, enc) {
  d.pOn[slot] = true;
  d.pX[slot] = x;
  d.pY[slot] = y;
  d.pPsi[slot] = psi;
  d.pSpd[slot] = spd;
  d.pSteer[slot] = steer;
  d.pHostile[slot] = hostile;
  d.pDone[slot] = !hostile;
  d.pEnd[slot] = end;
  d.pEnc[slot] = enc;
}

/** A free slot for a bird (kind 0) or a missile (1); if all are taken, the one leaving soonest. */
function slotOf(d, kind) {
  let best = -1;
  for (let k = 0; k < K_PURSUERS; k++) {
    if (P_KIND[k] !== kind) continue;
    if (!d.pOn[k]) return k;
    if (best < 0 || d.pEnd[k] < d.pEnd[best]) best = k;
  }
  return best;
}

/** A point `dist` from the drone at `ang` above the line ahead (or behind), 1-12.5 m up. */
function from(d, ang, dist, behind) {
  const fx = behind ? -d.course.dir : d.course.dir;
  let y = d.y + dist * Math.sin(ang);
  if (!(1.0 <= y && y <= 12.5)) {
    ang = -ang;
    y = d.y + dist * Math.sin(ang);
  }
  y = Math.min(Math.max(y, 1.0), 12.5);
  const x = d.x + fx * dist * Math.cos(ang);
  return [x, y];
}

function closingOf(d, x, y, speed) {
  const dx = x - d.x;
  const dy = y - d.y;
  const r = Math.max(Math.hypot(dx, dy), 1e-6);
  return speed + Math.max(0.0, (d.vx * dx + d.vy * dy) / r);
}

/** Heading onto a collision course with the drone as it flies now (straight at it if none). */
function aim(d, x, y, speed) {
  let rx = d.x - x;
  let ry = d.y - y;
  const vx = d.vx;
  const vy = d.vy;
  const a = vx * vx + vy * vy - speed * speed;
  const b = 2 * (rx * vx + ry * vy);
  const c = rx * rx + ry * ry;
  let t = -1.0;
  if (Math.abs(a) < 1e-9) t = b < 0 ? -c / b : -1.0;
  else {
    const disc = b * b - 4 * a * c;
    if (disc >= 0) {
      const q = Math.sqrt(disc);
      const ts = [(-b - q) / (2 * a), (-b + q) / (2 * a)].filter((s) => s > 0);
      t = ts.length ? Math.min(...ts) : -1.0;
    }
  }
  if (t > 0) {
    rx = rx + vx * t;
    ry = ry + vy * t;
  }
  return Math.atan2(ry, rx);
}

function bird(d, ang, dist, speed, behind, enc, t) {
  const [x, y] = from(d, ang, dist, behind);
  const slot = slotOf(d, 0);
  const psi = aim(d, x, y, speed);
  spawn(d, slot, x, y, psi, speed, true, true, t + BIRD_LIFE, enc);
  return dist / closingOf(d, x, y, speed);
}

function missile(d, ang, dist, behind, enc, t) {
  const [x, y] = from(d, ang, dist, behind);
  const spd = d.course.mSpeed;
  const psi = aim(d, x, y, spd);
  spawn(d, slotOf(d, 1), x, y, psi, spd, true, true, t + MISSILE_FUEL, enc);
}

function match(d, tArrive, ang, speed, behind) {
  const [x, y] = from(d, ang, 20.0, behind);
  return Math.min(Math.max(tArrive * closingOf(d, x, y, speed), 12.0), 45.0);
}

function encounter(d) {
  const [, kind, c] = d.course.encounters[d.encNext];
  d.encNext += 1;
  d.encCount[kind] += 1;
  const t = d.ticks * d.ph.dt;
  const deg = Math.PI / 180;
  if (kind === BIRD1) {
    bird(d, (2 * c[0] - 1) * 50 * deg, 28 + 10 * c[1], 4 + 3 * c[2], false, kind, t);
  } else if (kind === BIRD2) {
    const ta = bird(d, (10 + 40 * c[0]) * deg, 28 + 10 * c[1], 4 + 3 * c[2], false, kind, t);
    const ang = -(10 + 40 * c[3]) * deg;
    const spd = 4 + 3 * c[4];
    const behind = c[5] < 0.3;
    const dist = match(d, ta, ang, spd, behind);
    bird(d, ang, dist, spd, behind, kind, t);
  } else if (kind === MISSILE1) {
    missile(d, (2 * c[0] - 1) * 40 * deg, 30 + 8 * c[2], c[1] < 0.3, kind, t);
  } else if (kind === COMBO) {
    const tb = bird(d, (2 * c[0] - 1) * 50 * deg, 28 + 10 * c[1], 4 + 3 * c[2], false, kind, t);
    const ang = (2 * c[3] - 1) * 40 * deg;
    const behind = c[4] < 0.3;
    missile(d, ang, match(d, tb, ang, d.course.mSpeed, behind), behind, kind, t);
  } else if (kind === FLOCK) {
    const n = 2 + (c[0] < 0.5 ? 1 : 0);
    const spd = 4 + 3 * c[7];
    const psi = d.course.dir > 0 ? Math.PI : 0.0; // flying back along the route
    for (let j = 0; j < n; j++) {
      const y = 1.5 + 10.5 * c[1 + j];
      const x = d.x + d.course.dir * (24 + 14 * c[4 + j]);
      spawn(d, slotOf(d, 0), x, y, psi, spd, false, false, t + BIRD_LIFE, kind);
    }
  }
}

function encounters(d) {
  const enc = d.course.encounters;
  if (d.encNext < enc.length && progress(d) >= enc[d.encNext][0]) encounter(d);
}

// --- pursuers ------------------------------------------------------------------------------
/** Every pursuer slot now: [{ x, y, vx, vy, on, kind, steer, hostile, enc }] x K_PURSUERS. */
export function pursuers(d) {
  return d.pOn.map((on, k) => ({
    x: d.pX[k],
    y: d.pY[k],
    vx: d.pSpd[k] * Math.cos(d.pPsi[k]),
    vy: d.pSpd[k] * Math.sin(d.pPsi[k]),
    on,
    kind: P_KIND[k],
    steer: d.pSteer[k],
    hostile: d.pHostile[k],
    enc: d.pEnc[k],
  }));
}

function pursuersMove(d) {
  if (!d.pOn.some(Boolean)) return;
  const dt = d.ph.dt;
  const t = d.ticks * dt;
  const lock = d.course.birdLock;
  for (let k = 0; k < K_PURSUERS; k++) {
    if (!d.pOn[k]) continue;
    const isBird = P_KIND[k] === 0;
    let psi = d.pPsi[k];
    const sp = d.pSpd[k];
    const rx = d.x - d.pX[k];
    const ry = d.y - d.pY[k];
    const rvx = d.vx - sp * Math.cos(psi);
    const rvy = d.vy - sp * Math.sin(psi);
    const r2 = Math.max(rx * rx + ry * ry, 1e-6);
    const r = Math.sqrt(r2);
    const ldot = (rx * rvy - ry * rvx) / r2;
    const closing = -(rx * rvx + ry * rvy) / r;
    let steer = d.pSteer[k];
    const commit = steer && isBird && closing > 0 && r < lock * closing;
    steer = steer && !commit;
    const gain = isBird ? BIRD_N : MISSILE_N;
    const turn = isBird ? BIRD_TURN : MISSILE_TURN;
    psi = psi + (steer ? clip(gain * ldot, -turn, turn) * dt : 0.0);
    const px = d.pX[k] + sp * Math.cos(psi) * dt;
    const py = d.pY[k] + sp * Math.sin(psi) * dt;
    d.pSteer[k] = steer;
    d.pPsi[k] = psi;
    d.pX[k] = px;
    d.pY[k] = py;
    // beaten: an aimed bird or the missile that has passed (receding, close) or leaves
    const passed = closing < 0 && r < 6.0 && (!steer || !isBird);
    const leaves =
      t >= d.pEnd[k] || py < -0.5 || py > d.ph.ceiling + 4.0 || r > GONE_R;
    if ((passed || leaves) && !d.pDone[k]) {
      d.pDone[k] = true;
      d.beaten[P_KIND[k]] += 1;
    }
    if (leaves) d.pOn[k] = false;
  }
}

// --- frogs (post 13) -------------------------------------------------------------------------
/** Python's (and np.mod's) modulo: the result takes the divisor's sign. */
function pyMod(a, m) {
  let r = a % m;
  if (r !== 0 && r < 0 !== m < 0) r += m;
  return r;
}

export function strikeTime(d, k, n) {
  const p = d.frogP[k];
  const after = d.swAfter;
  const beats = n <= after ? n : after + (n - after) * d.swRatio;
  const jitter = FROG_JITTER * (2.0 * pyMod(n * GOLDEN + p[4], 1.0) - 1.0);
  return p[3] + p[2] * beats + jitter;
}

export function perch(d, k) {
  const p = d.frogP[k];
  const t = d.ticks * d.ph.dt;
  return [p[0] + 0.0 * t, p[1] + 0.3 * Math.sin(BOB_W * t + p[5])];
}

/** Every frog slot now: [{ x, y, vx, vy, on, perchX, perchY, phase }] x K_FROGS (empty slots
 * computed too, as in Python). phase: "rest", "leap", "hold" or "back". */
export function frogs(d) {
  const t = d.ticks * d.ph.dt;
  const out = [];
  for (let k = 0; k < K_FROGS; k++) {
    const [px, py] = perch(d, k);
    const pvy = 0.3 * BOB_W * Math.cos(BOB_W * t + d.frogP[k][5]);
    const u = t - d.frogS[k];
    const sx = d.frogSx[k];
    const sy = d.frogSy[k];
    const tx = d.frogTx[k];
    const ty = d.frogTy[k];
    const a = clip(u / LEAP, 0.0, 1.0);
    const b = clip((u - LEAP - HOLD) / BACK, 0.0, 1.0);
    const leap = u >= 0 && u < LEAP;
    const back = u >= LEAP + HOLD && u < LEAP + HOLD + BACK;
    const hold = u >= LEAP && u < LEAP + HOLD;
    const sa = a * a * (3 - 2 * a);
    const sb = b * b * (3 - 2 * b);
    const da = (6 * a * (1 - a)) / LEAP;
    const db = (6 * b * (1 - b)) / BACK;
    let f;
    if (leap)
      f = { x: sx + (tx - sx) * sa, y: sy + (ty - sy) * sa, vx: (tx - sx) * da, vy: (ty - sy) * da, phase: "leap" };
    else if (hold) f = { x: tx, y: ty, vx: 0.0, vy: 0.0, phase: "hold" };
    else if (back)
      f = { x: tx + (px - tx) * sb, y: ty + (py - ty) * sb, vx: (px - tx) * db, vy: (py - ty) * db, phase: "back" };
    else f = { x: px, y: py, vx: 0.0, vy: pvy, phase: "rest" };
    out.push({ ...f, on: d.frogOn[k], perchX: px, perchY: py });
  }
  return out;
}

export function nextStrike(d) {
  const t = d.ticks * d.ph.dt;
  return d.frogN.map((n, k) => strikeTime(d, k, n) - t);
}

function frogsMove(d) {
  if (!d.frogOn.some(Boolean)) return;
  // the mid-route frog is behind the drone once it's past its reach
  if (d.frogOn[0] && d.course.dir * (d.x - d.frogP[0][0]) > FROG_REACH + 1.0) d.frogPast[0] = true;
  const t = d.ticks * d.ph.dt;
  const due = d.frogN.map((n, k) => d.frogOn[k] && t >= strikeTime(d, k, n));
  if (!due.some(Boolean)) return;
  const now = frogs(d);
  for (let k = 0; k < K_FROGS; k++) {
    if (!due[k]) continue;
    d.frogS[k] = strikeTime(d, k, d.frogN[k]);
    d.frogSx[k] = now[k].x;
    d.frogSy[k] = now[k].y;
    const col = d.frogP[k][0];
    d.frogTx[k] = clip(d.x + d.vx * FROG_LEAD, col - FROG_REACH, col + FROG_REACH);
    d.frogTy[k] = clip(d.y + d.vy * FROG_LEAD, 0.5, d.ph.ceiling - 0.5);
    d.frogN[k] += 1;
  }
}

function threatsMove(d) {
  frogsMove(d);
  encounters(d);
  pursuersMove(d);
}

function end(d, outcome) {
  d.touch = [d.vx, d.vy];
  d.outcome = outcome;
}

function threatsHit(d) {
  if (d.outcome !== 0) return;
  for (let k = 0; k < K_PURSUERS; k++) {
    const dx = d.pX[k] - d.x;
    const dy = d.pY[k] - d.y;
    if (d.pOn[k] && dx * dx + dy * dy < P_HIT[k] * P_HIT[k]) {
      d.hitEnc = d.pEnc[k];
      end(d, P_KIND[k] === 0 ? BIRD : MISSILE);
      return;
    }
  }
  if (!d.frogOn.some(Boolean)) return;
  for (const f of frogs(d)) {
    const dx = f.x - d.x;
    const dy = f.y - d.y;
    if (f.on && dx * dx + dy * dy < FROG_HIT * FROG_HIT) {
      end(d, FROG);
      return;
    }
  }
}

// --- physics ---------------------------------------------------------------------------------
export function points(d) {
  const c = Math.cos(d.th);
  const s = Math.sin(d.th);
  return POINTS.map(([lx, ly]) => [d.x + lx * c + ly * s, d.y - lx * s + ly * c]);
}

function limits(d) {
  if (d.outcome !== 0) return;
  const w = d.course.world;
  if (d.x < w[0] - 0.5 || d.x > w[1] + 0.5 || d.y > d.ph.ceiling) d.outcome = LOST;
  else if (d.ticks >= d.endTick) d.outcome = TIMEOUT;
}

function contacts(d, before) {
  const ph = d.ph;
  const c = d.course;
  const pts = points(d);
  const pads = [c.start, c.target];
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
        end(d, CRASHED);
        return;
      }
    }
    if (py < 0.0) {
      end(d, WATER);
      return;
    }
  }
  if (landing < 0) return;
  const gentle =
    d.vy >= -ph.landVy && Math.abs(d.vx) <= ph.landVx && Math.abs(d.th) <= ph.landTilt;
  if (!gentle) {
    d.crash = 1;
    end(d, CRASHED);
  } else if (landing === 1 && c.target[0] <= d.x && d.x <= c.target[1]) {
    d.frogPast[1] = d.frogOn[1];
    end(d, LANDED);
  } else {
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
    // still sitting on a pillar: friction slows any slide
    d.vx *= 0.85;
    d.x += d.vx * dt;
    d.ticks += 1;
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
  const air = co.wind + co.gust * Math.sin((2 * Math.PI * t) / co.period + co.phase);
  let ax = acc * Math.sin(d.th) - ph.dragLin * d.vx - ph.dragQuad * d.vx * speed;
  const ay = acc * Math.cos(d.th) - G - ph.dragLin * d.vy - ph.dragQuad * d.vy * speed;
  ax += air / co.mass;
  d.vx += ax * dt;
  d.vy += ay * dt;
  d.x += d.vx * dt;
  d.y += d.vy * dt;
  d.ticks += 1;
  threatsMove(d);
  contacts(d, before);
  threatsHit(d);
  limits(d);
}

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
  const c = d.course;
  return [(c.target[0] + c.target[1]) / 2, c.target[2] + LEG];
}

export function distance(d) {
  const [gx, gy] = goal(d);
  return Math.hypot(gx - d.x, gy - d.y);
}

/** Whether the pad is within the sight range (what the pilot's "in sight" input says). */
export const padSeen = (d) => Math.abs(goal(d)[0] - d.x) < SIGHT;

function surface(d, x) {
  const c = d.course;
  if (c.start[0] < x && x < c.start[1]) return c.start[2];
  if (c.target[0] < x && x < c.target[1]) return c.target[2];
  return 0.0;
}

/** Fleet.frog_failed: hit by a frog, or out of time near one it hadn't got past yet. */
export function frogFailed(d) {
  if (d.outcome === FROG) return true;
  if (d.outcome !== TIMEOUT) return false;
  return d.frogOn.some((on, k) => on && !d.frogPast[k] && Math.abs(d.frogP[k][0] - d.x) < 10);
}

// --- what the pilot sees ---------------------------------------------------------------------
export const N_BASE = 18;
export const K_SEE = 4;
export const N_SLOT = 7;
export const N_PREDICT = 3;
export const N_FROG = 8;
export const N_HIST = 3;
export const HIST_LEN = (HIST - 1) * HIST_EVERY + 1;
const FAR = SIGHT / 10; // an empty slot reads as this far away (x 10 m)

export const slotWidth = (predict) => N_SLOT + (predict ? N_PREDICT : 0);
export const frameWidth = (predict) => N_BASE + K_SEE * slotWidth(predict) + K_FROGS * N_FROG;
/** The network's input width for these features. */
export const inputWidth = (feat) =>
  frameWidth(feat.predict) * feat.stack + (feat.memory === "hist" ? HIST * K_FROGS * N_HIST : 0);

function baseInputs(d, hide = false) {
  const c = d.course;
  const [gx, gy] = goal(d);
  const dx = gx - d.x;
  const seen = Math.abs(dx) < SIGHT;
  const under = d.y - LEG - surface(d, d.x);
  const half = (c.target[1] - c.target[0]) / 2;
  const sDx = startX(d) - d.x;
  const sSeen = Math.abs(sDx) < SIGHT;
  return [
    clip(dx, -SIGHT, SIGHT) / 10,
    seen ? (gy - d.y) / 10 : 0.0,
    d.vx / 5,
    d.vy / 5,
    d.th / d.ph.tiltMax,
    d.om / 5,
    d.motor * 2 - 1,
    half / 1.3,
    clip(sDx, -SIGHT, SIGHT) / 10,
    sSeen ? (c.start[2] + LEG - d.y) / 10 : 0.0,
    d.y / 10,
    clip(under / 5, -1.0, 2.0),
    d.stick[0],
    d.stick[1],
    seen ? 1.0 : 0.0,
    d.charge * 2 - 1,
    !hide && d.pOn.some((on, k) => on && P_KIND[k] === 1) ? 1.0 : 0.0, // a missile is out
    c.dir,
  ];
}

/** The most urgent pursuers in sight first (distance / closing speed): K_SEE slots of dx, dy,
 * vx, vy, present, bird, missile; with predict also the closest approach (in how long / 3 s, the
 * pursuer's offset from the drone then / 5 m). hide: the blindfold (every slot empty). */
export function threatSlots(d, hide = false, predict = false) {
  const keyed = pursuers(d).map((h, i) => {
    const dx = h.x - d.x;
    const dy = h.y - d.y;
    const r = Math.sqrt(dx * dx + dy * dy);
    const on = h.on && r < SIGHT && !hide;
    const closing = -(dx * (h.vx - d.vx) + dy * (h.vy - d.vy)) / Math.max(r, 1e-6);
    const key = on ? r / Math.max(closing, 0.5) : Infinity;
    return { i, dx, dy, vx: h.vx, vy: h.vy, kind: h.kind, key };
  });
  // np.argsort(kind="stable"): ties keep slot order
  keyed.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : a.i - b.i));
  const out = [];
  for (let s = 0; s < K_SEE; s++) {
    const h = keyed[s];
    const have = Number.isFinite(h.key);
    if (have) out.push(h.dx / 10, h.dy / 10, h.vx / 5, h.vy / 5, 1, h.kind === 0 ? 1 : 0, h.kind === 1 ? 1 : 0);
    else out.push(FAR, 0, 0, 0, 0, 0, 0);
    if (!predict) continue;
    if (!have) {
      out.push(2.0, 2.0, 0.0);
      continue;
    }
    const rvx = h.vx - d.vx;
    const rvy = h.vy - d.vy;
    const v2 = Math.max(rvx * rvx + rvy * rvy, 1e-6);
    const t = clip(-(h.dx * rvx + h.dy * rvy) / v2, 0.0, 6.0);
    out.push(t / 3, clip((h.dx + rvx * t) / 5, -2.0, 2.0), clip((h.dy + rvy * t) / 5, -2.0, 2.0));
  }
  return out;
}

/** Per frog whose perch is in sight: dx, dy to the frog, its velocity, present, 0 (no tell),
 * dx, dy to its perch. */
export function frogInputs(d, hide = false) {
  const out = [];
  for (const f of frogs(d)) {
    const on = f.on && Math.hypot(f.perchX - d.x, f.perchY - d.y) < SIGHT && !hide;
    if (on)
      out.push(
        (f.x - d.x) / 10,
        (f.y - d.y) / 10,
        f.vx / 10,
        f.vy / 10,
        1,
        0,
        (f.perchX - d.x) / 10,
        (f.perchY - d.y) / 10,
      );
    else out.push(FAR, 0.0, 0.0, 0.0, 0, 0, FAR, 0.0);
  }
  return out;
}

/** gauntlet_env.sigma: sensor noise per input (x the flight's noise scale). */
export function sigma(predict) {
  const slot = [0.005, 0.005, 0.02, 0.02, 0, 0, 0, ...(predict ? [0.01, 0.01, 0.01] : [])];
  return [
    0.005, 0.005, 0.02, 0.02, 0.04, 0.02, 0.02, 0, 0.005, 0.005, 0.005, 0.01, 0, 0, 0,
    0.01, 0, 0,
    ...Array.from({ length: K_SEE }, () => slot).flat(),
    ...Array.from({ length: K_FROGS }, () => [0.005, 0.005, 0.02, 0.02, 0, 0, 0.005, 0.005]).flat(),
  ];
}

const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

/** One frame of what the pilot senses (observe). hide: { threats, frogs } blindfolds. */
export function sense(d, noise = 0, hide = {}) {
  const predict = !!d.feat.predict;
  const o = [...baseInputs(d, !!hide.threats), ...threatSlots(d, !!hide.threats, predict), ...frogInputs(d, !!hide.frogs)];
  if (!noise) return o;
  const sg = sigma(predict);
  return o.map((v, i) => (sg[i] ? v + gauss() * noise * sg[i] : v));
}

/** From a sensed frame: each frog's offset from its perch and whether it's seen. */
export function frogOffsets(seen, predict = false) {
  const at = N_BASE + K_SEE * slotWidth(predict);
  const out = [];
  for (let k = 0; k < K_FROGS; k++) {
    const fr = seen.slice(at + k * N_FROG, at + (k + 1) * N_FROG);
    const on = fr[4];
    out.push((fr[0] - fr[6]) * on, (fr[1] - fr[7]) * on, on);
  }
  return out;
}

/** The sensor delay line, the last `stack` frames and (memory "hist") the last HIST_LEN updates
 * of frog offsets, newest first. noise: scale (0 for none); hide: { threats, frogs }. */
export function newEyes(d, noise = 0, hide = {}) {
  const first = sense(d, noise, hide);
  const memory = d.feat.memory === "hist";
  return {
    noise,
    hide,
    pending: Array.from({ length: d.feat.delay }, () => first),
    frames: Array.from({ length: d.feat.stack }, () => first),
    hist: memory ? Array.from({ length: HIST_LEN }, () => frogOffsets(first, d.feat.predict)) : [],
  };
}

/** What the network reads: the stacked frames, then every HIST_EVERY-th history row, +-5. */
export function look(d, eyes) {
  const parts = eyes.frames.flat();
  if (d.feat.memory === "hist") for (let r = 0; r < HIST_LEN; r += HIST_EVERY) parts.push(...eyes.hist[r]);
  return parts.map((v) => clip(v, -5, 5));
}

export function push(d, eyes) {
  let now = sense(d, eyes.noise, eyes.hide);
  if (eyes.pending.length) {
    eyes.pending.push(now);
    now = eyes.pending.shift();
  }
  eyes.frames.unshift(now);
  eyes.frames.pop();
  if (d.feat.memory === "hist") {
    eyes.hist.unshift(frogOffsets(now, d.feat.predict));
    eyes.hist.pop();
  }
  return look(d, eyes);
}

/** The exported actor: tanh hidden layers, a linear output clipped to the stick range. */
export function netForward(net, obs) {
  let h = obs;
  const n = net.layers.length;
  net.layers.forEach((L, i) => {
    const out = new Array(L.b.length);
    for (let r = 0; r < L.b.length; r++) {
      const row = L.w[r];
      let z = 0;
      for (let c = 0; c < row.length; c++) z += row[c] * h[c];
      z += L.b[r];
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
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const out = new Array(bin.length / 2);
  for (let i = 0; i < out.length; i++)
    out[i] = halfToFloat(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
  return out;
}

/** A model file ({label, predict, layers: [{shape, w, b}] as float16 base64}) as plain rows,
 * plus the features its input width implies (memory "hist" or "none"). */
export function unpackNet(model) {
  const layers = model.layers.map((L) => {
    const [rows, cols] = L.shape;
    const flat = decodeHalves(L.w);
    const w = [];
    for (let r = 0; r < rows; r++) w.push(flat.slice(r * cols, (r + 1) * cols));
    return { w, b: decodeHalves(L.b) };
  });
  const width = model.layers[0].shape[1];
  const predict = !!model.predict;
  const memory = width === inputWidth({ ...FULL, predict }) ? "hist" : "none";
  return { ...model, predict, memory, width, layers };
}
