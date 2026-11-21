// Browser port of experiments/010-drone/drone_env.py: the same physics, line for line and in the
// same order, so a flight here matches the Python one (test/drone-sim.test.mjs checks it).

export const G = 9.81;
const FEET = [
  [-0.22, -0.18],
  [0.22, -0.18],
];
const TIPS = [
  [-0.35, 0.04],
  [0.35, 0.04],
];
const POINTS = [...FEET, ...TIPS];
export const LEG = 0.18;
const FOOT_BAND = 0.15;

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
  windy: { wind: 1.5, gust: 1.0, heavy: 0 },
  heavy: { wind: 0, gust: 0, heavy: 0.35 },
  rough: { wind: 1.5, gust: 1.0, heavy: 0.35 },
};

/** Same sequence as drone_env.mulberry32: floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pad = (x0, x1, top) => ({
  x0,
  x1,
  top,
  cx: (x0 + x1) / 2,
  half: (x1 - x0) / 2,
});

export function makeCourse(seed, weather = WEATHER.calm, ph = PHYSICS) {
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
  return {
    start: pad(sx - 1.0, sx + 1.0, sh),
    target: pad(tx - tw / 2, tx + tw / 2, th),
    wind,
    gust,
    period,
    phase,
    mass,
  };
}

export function newDrone(course, ph = PHYSICS) {
  return {
    course,
    ph,
    x: course.start.cx,
    y: course.start.top + LEG,
    vx: 0,
    vy: 0,
    th: 0,
    om: 0,
    motor: 0,
    ticks: 0,
    stick: [-1, 0],
    grounded: true,
    outcome: null,
    crash: null,
    touch: [0, 0],
  };
}

export const time = (d) => d.ticks * d.ph.dt;

export function goal(d) {
  const t = d.course.target;
  return [t.cx, t.top + LEG];
}

export function distance(d) {
  const [gx, gy] = goal(d);
  const dx = gx - d.x;
  const dy = gy - d.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function points(d) {
  const c = Math.cos(d.th);
  const s = Math.sin(d.th);
  return POINTS.map(([lx, ly]) => [
    d.x + lx * c + ly * s,
    d.y - lx * s + ly * c,
  ]);
}

export function surface(d, x) {
  for (const p of [d.course.start, d.course.target])
    if (p.x0 < x && x < p.x1) return p.top;
  return 0.0;
}

/** The wind's push (m/s^2) at the current tick. */
export function air(d) {
  const co = d.course;
  const t = d.ticks * d.ph.dt;
  return co.wind + co.gust * Math.sin((2 * Math.PI * t) / co.period + co.phase);
}

function end(d, outcome) {
  d.touch = [d.vx, d.vy];
  d.outcome = outcome;
}

function contacts(d, before) {
  const ph = d.ph;
  let landing = null;
  const pts = points(d);
  for (let k = 0; k < pts.length; k++) {
    const [px, py] = pts[k];
    for (const p of [d.course.start, d.course.target]) {
      if (p.x0 < px && px < p.x1 && py < p.top) {
        if (k < 2 && before[k][1] >= p.top - FOOT_BAND) {
          if (d.vy <= 0.0) landing = p;
        } else {
          d.crash = "side";
          end(d, "crashed");
          return;
        }
      }
    }
    if (py < 0.0) {
      end(d, "water");
      return;
    }
  }
  if (landing === null) return;
  const gentle =
    d.vy >= -ph.landVy &&
    Math.abs(d.vx) <= ph.landVx &&
    Math.abs(d.th) <= ph.landTilt;
  if (!gentle) {
    d.crash = "hard";
    end(d, "crashed");
  } else if (
    landing === d.course.target &&
    landing.x0 <= d.x &&
    d.x <= landing.x1
  )
    end(d, "landed");
  else {
    d.grounded = true;
    d.th = 0.0;
    d.om = 0.0;
    d.y = landing.top + LEG;
    d.vy = 0.0;
    d.vx *= 0.85;
  }
}

/** One physics step (1/60 s). */
export function tick(d, throttle, tilt) {
  const ph = d.ph;
  const co = d.course;
  const dt = ph.dt;
  const cmd = (throttle + 1.0) / 2.0;
  d.motor += ((cmd - d.motor) * dt) / ph.motorTau;
  const acc = (d.motor * ph.thrustMax * G) / co.mass;
  if (d.grounded && acc <= G) {
    d.vx *= 0.85;
    d.x += d.vx * dt;
    d.ticks += 1;
    limits(d);
    return;
  }
  d.grounded = false;
  const before = points(d);
  let alpha = ph.kp * (tilt * ph.tiltMax - d.th) - ph.kd * d.om;
  alpha = Math.max(-ph.alphaMax, Math.min(ph.alphaMax, alpha));
  alpha *= Math.min(1.0, 0.3 + 1.4 * d.motor);
  d.om += alpha * dt;
  d.th += d.om * dt;
  const speed = Math.sqrt(d.vx * d.vx + d.vy * d.vy);
  const wind = air(d);
  let ax =
    acc * Math.sin(d.th) - ph.dragLin * d.vx - ph.dragQuad * d.vx * speed;
  const ay =
    acc * Math.cos(d.th) - G - ph.dragLin * d.vy - ph.dragQuad * d.vy * speed;
  ax += wind / co.mass;
  d.vx += ax * dt;
  d.vy += ay * dt;
  d.x += d.vx * dt;
  d.y += d.vy * dt;
  d.ticks += 1;
  contacts(d, before);
  if (d.outcome === null) limits(d);
}

function limits(d) {
  const ph = d.ph;
  if (d.x < -0.5 || d.x > ph.width + 0.5 || d.y > ph.ceiling)
    d.outcome = "lost";
  else if (d.ticks >= Math.round(ph.tMax / ph.dt)) d.outcome = "timeout";
}

const clip1 = (v) => Math.max(-1.0, Math.min(1.0, v));

/** One stick update (20 per second): hold the sticks for `substeps` physics steps. */
export function step(d, throttle, tilt) {
  throttle = clip1(throttle);
  tilt = clip1(tilt);
  const r = d.ph.stickRate;
  if (r > 0) {
    const [t0, s0] = d.stick;
    throttle = t0 + Math.max(-r, Math.min(r, throttle - t0));
    tilt = s0 + Math.max(-r, Math.min(r, tilt - s0));
  }
  for (let i = 0; i < d.ph.substeps; i++) {
    tick(d, throttle, tilt);
    if (d.outcome !== null) break;
  }
  d.stick = [throttle, tilt];
}

export function observe(d) {
  const [gx, gy] = goal(d);
  const s = d.course.start;
  const under = d.y - LEG - surface(d, d.x);
  return [
    (gx - d.x) / 10,
    (gy - d.y) / 10,
    d.vx / 5,
    d.vy / 5,
    d.th / d.ph.tiltMax,
    d.om / 5,
    d.motor * 2 - 1,
    d.course.target.half / 1.3,
    (s.cx - d.x) / 10,
    (s.top + LEG - d.y) / 10,
    d.y / 10,
    Math.max(-1.0, Math.min(2.0, under / 5)),
    d.stick[0],
    d.stick[1],
  ];
}

/** The hand-written flight controller (drone_env.autopilot). */
export function autopilot(d) {
  const ph = d.ph;
  const [gx, gy] = goal(d);
  const s = d.course.start;
  const t = d.course.target;
  const cruise = Math.max(s.top, t.top) + LEG + 1.8;
  const dx = gx - d.x;
  let vxWant;
  let vyWant;
  if (Math.abs(dx) > 0.35 || Math.abs(d.vx) > 0.4) {
    const yWant = cruise;
    vxWant = d.y > cruise - 1.2 ? Math.max(-4.0, Math.min(4.0, 0.9 * dx)) : 0.0;
    vyWant = Math.max(-2.0, Math.min(2.0, 1.5 * (yWant - d.y)));
  } else {
    vxWant = 0.6 * dx;
    vyWant = -Math.max(0.5, Math.min(1.2, 0.8 * (d.y - gy)));
  }
  const tilt = clip1(0.45 * (vxWant - d.vx));
  const hover = 1.0 / (ph.thrustMax * Math.max(0.5, Math.cos(d.th)));
  const frac = hover + 0.18 * (vyWant - d.vy);
  return [clip1(2 * frac - 1), tilt];
}

/** The exported actor ({act, squash, layers: [{w, b}]}) on an observation: the two sticks. */
export function netForward(net, obs) {
  let h = obs.map((v) => Math.max(-5, Math.min(5, v)));
  const n = net.layers.length;
  net.layers.forEach((L, i) => {
    const out = new Array(L.b.length);
    for (let r = 0; r < L.b.length; r++) {
      const row = L.w[r];
      let z = L.b[r];
      for (let c = 0; c < row.length; c++) z += row[c] * h[c];
      out[r] =
        i < n - 1 ? (net.act === "tanh" ? Math.tanh(z) : Math.max(z, 0)) : z;
    }
    h = out;
  });
  return net.squash ? h.map(Math.tanh) : h.map(clip1);
}

export const netPilot = (net) => (d) => netForward(net, observe(d));

/** The physics a network was trained with (its model file's "physics": {stick_rate}). */
export const physicsOf = (net) => ({
  ...PHYSICS,
  stickRate: net?.physics?.stick_rate ?? 0,
});
