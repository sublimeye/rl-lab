// Draws post 10's drone course (side view) and the two controller sticks. Colours come from the
// design tokens in global.css (read once per theme by readColors).

const TOKENS = [
  "paper",
  "ink",
  "muted",
  "rule",
  "floor",
  "sky",
  "water",
  "stone",
  "flag",
  "lava",
  "timeout",
  "wall",
  "shade",
  "leaf",
];

export function readColors(el) {
  const cs = getComputedStyle(el);
  return Object.fromEntries(
    TOKENS.map((t) => [t, cs.getPropertyValue(`--${t}`).trim()]),
  );
}

/** Sizes the canvas to its CSS box at device resolution; returns the CSS size. */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (
    canvas.width !== Math.round(w * dpr) ||
    canvas.height !== Math.round(h * dpr)
  ) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  return { w, h, dpr };
}

const WORLD = { x0: -0.25, x1: 24.25, y0: -0.9, y1: 13.0 };

/** World-to-screen mapping: the whole course when it fits, else a camera following the drone. */
export function camera(w, h, at, prev) {
  const scale = Math.max(
    Math.min(w / (WORLD.x1 - WORLD.x0), h / (WORLD.y1 - WORLD.y0)),
    22,
  );
  const vw = w / scale;
  const vh = h / scale;
  const clampC = (c, lo, hi, span) =>
    hi - lo <= span
      ? (lo + hi) / 2
      : Math.max(lo + span / 2, Math.min(hi - span / 2, c));
  let cx = clampC(at.x, WORLD.x0, WORLD.x1, vw);
  let cy = clampC(at.y, WORLD.y0, WORLD.y1, vh);
  if (prev) {
    // ease the camera so it doesn't jerk with the drone
    cx = prev.cx + (cx - prev.cx) * 0.12;
    cy = prev.cy + (cy - prev.cy) * 0.12;
  }
  return {
    scale,
    cx,
    cy,
    vw,
    vh,
    X: (x) => (x - cx) * scale + w / 2,
    Y: (y) => h / 2 - (y - cy) * scale,
  };
}

function pillar(ctx, cam, p, c, isTarget) {
  const x = cam.X(p.x0);
  const w = (p.x1 - p.x0) * cam.scale;
  const top = cam.Y(p.top);
  const bottom = cam.Y(WORLD.y0);
  ctx.fillStyle = c.stone;
  ctx.fillRect(x, top, w, bottom - top);
  // a lighter deck on top
  ctx.fillStyle = c.wall;
  ctx.fillRect(x - 2, top - 4, w + 4, 5);
  if (isTarget) {
    const cx = x + w / 2;
    ctx.strokeStyle = c.flag;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(x + 3, top - 4);
    ctx.lineTo(x + w - 3, top - 4);
    ctx.stroke();
    // landing lights at both edges
    ctx.fillStyle = c.flag;
    for (const lx of [x + 3, x + w - 3]) {
      ctx.beginPath();
      ctx.arc(lx, top - 6, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.font = `700 ${Math.max(10, Math.min(15, w / 5))}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = c.paper;
    ctx.fillText("LAND", cx, top + 6);
  }
}

function water(ctx, cam, w, h, c, t) {
  const top = cam.Y(0);
  ctx.fillStyle = c.water;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let px = 0; px <= w + 8; px += 8) {
    const wx = (px - w / 2) / cam.scale + cam.cx;
    ctx.lineTo(px, top + Math.sin(wx * 1.7 + t * 1.6) * 1.6);
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

/** The drone at (x, y, tilt th), rotors blurred by motor speed. */
function drone(ctx, cam, s, c, t) {
  const k = cam.scale;
  ctx.save();
  ctx.translate(cam.X(s.x), cam.Y(s.y));
  ctx.rotate(s.th); // + leans right: on screen (y down) that's clockwise, same sign
  // rotor wash
  if (s.motor > 0.05 && !s.crashed) {
    ctx.strokeStyle = c.muted;
    ctx.lineWidth = 1;
    ctx.globalAlpha = Math.min(0.55, s.motor * 0.7);
    for (const side of [-1, 1]) {
      for (let i = 0; i < 3; i++) {
        const ox = side * 0.3 * k + (i - 1) * 0.06 * k;
        const len = (0.25 + s.motor * 0.6) * k;
        const ph = (t * 9 + i * 0.33) % 1;
        ctx.beginPath();
        ctx.moveTo(ox, 0.04 * k + ph * len * 0.5);
        ctx.lineTo(ox, 0.04 * k + ph * len * 0.5 + len * 0.35);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = c.ink;
  ctx.fillStyle = c.ink;
  ctx.lineCap = "round";
  // legs
  ctx.lineWidth = Math.max(2, 0.04 * k);
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(side * 0.12 * k, 0.02 * k);
    ctx.lineTo(side * 0.22 * k, 0.18 * k);
    ctx.moveTo(side * 0.27 * k, 0.18 * k);
    ctx.lineTo(side * 0.17 * k, 0.18 * k);
    ctx.stroke();
  }
  // arms
  ctx.lineWidth = Math.max(3, 0.06 * k);
  ctx.beginPath();
  ctx.moveTo(-0.3 * k, -0.02 * k);
  ctx.lineTo(0.3 * k, -0.02 * k);
  ctx.stroke();
  // body
  const bw = Math.max(9, 0.28 * k);
  const bh = Math.max(5, 0.13 * k);
  ctx.beginPath();
  ctx.roundRect(-bw / 2, -bh * 0.75, bw, bh, bh / 2.2);
  ctx.fill();
  // a light so you can tell front from back: it faces the way the drone leans
  ctx.fillStyle = s.crashed ? c.lava : c.timeout;
  ctx.beginPath();
  ctx.arc(0, -bh * 0.22, Math.max(1.8, 0.03 * k), 0, Math.PI * 2);
  ctx.fill();
  // rotors: a blur disc whose opacity follows the motor
  for (const side of [-1, 1]) {
    const rx = side * 0.3 * k;
    const ry = -0.06 * k;
    ctx.fillStyle = c.ink;
    ctx.fillRect(rx - 1.5, ry, 3, 0.06 * k);
    ctx.globalAlpha = 0.3 + 0.5 * Math.min(1, s.motor * 1.4);
    ctx.beginPath();
    ctx.ellipse(
      rx,
      ry,
      Math.max(5, 0.12 * k),
      Math.max(1.6, 0.03 * k),
      0,
      0,
      Math.PI * 2,
    );
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

/**
 * The whole scene. s: interpolated {x, y, th, motor}; opts: {trail: [[x, y]], wind (m/s^2),
 * t (seconds, for animation), outcome}. Optional (post 12): course.target null for no landing
 * pad; opts.aim [x, y] for the off-screen arrow (default: the pad); opts.under(ctx, cam), drawn
 * on top of the water and pillars but under the path, rays and drone.
 */
export function drawScene(ctx, size, cam, course, s, c, opts) {
  const { w, h, dpr } = size;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = c.sky;
  ctx.fillRect(0, 0, w, h);
  // faint height marks every 2 m
  ctx.strokeStyle = c.rule;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.6;
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillStyle = c.muted;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  for (let m = 2; m <= 12; m += 2) {
    const y = Math.round(cam.Y(m)) + 0.5;
    if (y < 0 || y > h) continue;
    ctx.setLineDash([2, 6]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText(`${m} m`, 4, y - 2);
  }
  ctx.globalAlpha = 1;
  // wind streaks drifting with the wind
  if (opts.wind && Math.abs(opts.wind) > 0.05) {
    ctx.strokeStyle = c.muted;
    ctx.globalAlpha = Math.min(0.5, Math.abs(opts.wind) / 4 + 0.1);
    ctx.lineWidth = 1;
    const speed = opts.wind * 1.4;
    // opts.follow (a long course): the streaks tile the camera's view instead of a 24 m world
    const span = opts.follow ? Math.ceil(cam.vw / 26 + 1) * 26 : 26;
    const base = opts.follow ? Math.floor(cam.cx / 26) * 26 - span / 2 : -1;
    for (let i = 0; i < span; i++) {
      const wy = WORLD.y0 + 1 + ((i * 0.618) % 1) * 12;
      const wx = base + ((((i * 3.7 + opts.t * speed) % span) + span) % span);
      const x = cam.X(wx);
      const y = cam.Y(wy);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - Math.sign(opts.wind) * (8 + Math.abs(opts.wind) * 6), y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  water(ctx, cam, w, h, c, opts.t);
  pillar(ctx, cam, course.start, c, false);
  if (course.target) pillar(ctx, cam, course.target, c, true);
  if (opts.under) opts.under(ctx, cam);
  // the path flown so far
  if (opts.trail.length > 1) {
    ctx.strokeStyle = c.flag;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    opts.trail.forEach(([x, y], i) =>
      i ? ctx.lineTo(cam.X(x), cam.Y(y)) : ctx.moveTo(cam.X(x), cam.Y(y)),
    );
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
  if (opts.rays) drawRays(ctx, cam, s, opts.rays, c);
  if (opts.birds) for (const b of opts.birds) bird(ctx, cam, b, c, opts.t);
  drone(
    ctx,
    cam,
    {
      ...s,
      crashed: [
        "crashed",
        "water",
        "lost",
        "bird",
        "rock",
        "missile",
        "frog",
      ].includes(opts.outcome),
    },
    c,
    opts.t,
  );
  // where the pad (or opts.aim) is, when the camera can't see it
  const aim =
    opts.aim ?? (course.target ? [course.target.cx, course.target.top] : null);
  const tx = aim ? cam.X(aim[0]) : 0;
  const ty = aim ? cam.Y(aim[1]) : 0;
  if (aim && (tx < 0 || tx > w || ty < 0 || ty > h)) {
    const ax = Math.max(18, Math.min(w - 18, tx));
    const ay = Math.max(18, Math.min(h - 18, ty));
    const ang = Math.atan2(ty - ay, tx - ax);
    ctx.save();
    ctx.translate(ax, ay);
    ctx.rotate(ang);
    ctx.fillStyle = c.flag;
    ctx.beginPath();
    ctx.moveTo(10, 0);
    ctx.lineTo(-6, -7);
    ctx.lineTo(-6, 7);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}

/**
 * One controller stick: a round gimbal, its axis slot, a fading trail of recent positions and
 * the knob. v: [x, y] in -1..1 (y up); axis: 'y' (throttle) or 'x' (tilt).
 */
export function drawStick(ctx, size, v, axis, trail, c, active) {
  const { w, h, dpr } = size;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const r = Math.min(w, h) / 2 - 3;
  const cx = w / 2;
  const cy = h / 2;
  ctx.fillStyle = c.floor;
  ctx.strokeStyle = c.rule;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  // the slot this stick uses
  const reach = r - 14;
  ctx.strokeStyle = c.rule;
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  if (axis === "y") {
    ctx.moveTo(cx, cy - reach);
    ctx.lineTo(cx, cy + reach);
  } else {
    ctx.moveTo(cx - reach, cy);
    ctx.lineTo(cx + reach, cy);
  }
  ctx.stroke();
  // centre tick
  ctx.strokeStyle = c.muted;
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (axis === "y") {
    ctx.moveTo(cx - 9, cy);
    ctx.lineTo(cx + 9, cy);
  } else {
    ctx.moveTo(cx, cy - 9);
    ctx.lineTo(cx, cy + 9);
  }
  ctx.stroke();
  const P = ([x, y]) => [cx + x * reach, cy - y * reach];
  // trail: the last second of positions
  trail.forEach((p, i) => {
    const [x, y] = P(p);
    ctx.globalAlpha = ((i + 1) / trail.length) * 0.35;
    ctx.fillStyle = c.flag;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  const [kx, ky] = P(v);
  ctx.fillStyle = active ? c.flag : c.ink;
  ctx.strokeStyle = c.paper;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(kx, ky, Math.max(9, r * 0.2), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

/** A bird: two flapping wings, about 0.6 m across. b: [x, y, vx]. */
function bird(ctx, cam, b, c, t) {
  const k = cam.scale;
  const x = cam.X(b[0]);
  const y = cam.Y(b[1]);
  const flap =
    Math.sin(t * (Math.abs(b[2]) > 3.5 ? 18 : 10) + b[1] * 3) * 0.12 * k;
  const w = Math.max(7, 0.3 * k);
  ctx.strokeStyle = c.ink;
  ctx.lineWidth = Math.max(1.6, 0.05 * k);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(x - w, y - flap);
  ctx.quadraticCurveTo(x - w / 2, y - Math.abs(flap) - 0.04 * k, x, y);
  ctx.quadraticCurveTo(
    x + w / 2,
    y - Math.abs(flap) - 0.04 * k,
    x + w,
    y - flap,
  );
  ctx.stroke();
  ctx.fillStyle = c.ink;
  ctx.beginPath();
  ctx.arc(
    x + Math.sign(b[2] || 1) * 0.06 * k,
    y + 0.02 * k,
    Math.max(1.6, 0.05 * k),
    0,
    Math.PI * 2,
  );
  ctx.fill();
}

/**
 * The depth camera's rays: faint lines to what each hit (the pad in blue, birds in orange). Post
 * 12's rays also have threat (orange, like birds) and shadow (green, ending in a ring).
 */
function drawRays(ctx, cam, s, rays, c) {
  const ox = cam.X(s.x);
  const oy = cam.Y(s.y);
  ctx.lineWidth = 1;
  for (const r of rays) {
    const ex = s.x + Math.sin(r.angle) * r.dist;
    const ey = s.y - Math.cos(r.angle) * r.dist;
    const hot = r.bird || r.threat;
    const col = hot ? c.lava : r.shadow ? c.timeout : r.pad ? c.flag : c.muted;
    const marked = hot || r.shadow || r.pad;
    ctx.strokeStyle = col;
    ctx.globalAlpha = marked ? 0.55 : 0.22;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(cam.X(ex), cam.Y(ey));
    ctx.stroke();
    if (r.depth < 1) {
      ctx.globalAlpha = marked ? 0.9 : 0.45;
      ctx.beginPath();
      if (r.shadow && !hot) {
        ctx.lineWidth = 1.5;
        ctx.arc(cam.X(ex), cam.Y(ey), 3.2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.lineWidth = 1;
      } else {
        ctx.fillStyle = col;
        ctx.arc(cam.X(ex), cam.Y(ey), 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.globalAlpha = 1;
}

// --- post 14: a long course ------------------------------------------------------------------

/**
 * A camera for a course wider than the screen: about `span` m across (less if the height
 * doesn't fit: the view always shows y0..y1), following `at` with a lead of `lead` m, kept
 * inside [x0, x1], eased from `prev`. Same shape as camera().
 */
export function follow(w, h, at, prev, { span, x0, x1, y0 = WORLD.y0, y1 = WORLD.y1, lead = 0 }) {
  const scale = Math.min(w / span, h / (y1 - y0));
  const vw = w / scale;
  const vh = h / scale;
  const clampC = (c, lo, hi, s) =>
    hi - lo <= s ? (lo + hi) / 2 : Math.max(lo + s / 2, Math.min(hi - s / 2, c));
  let cx = clampC(at.x + lead, x0, x1, vw);
  let cy = clampC(at.y, y0, y1, vh);
  if (prev && Math.abs(prev.cx - cx) < vw) {
    cx = prev.cx + (cx - prev.cx) * 0.08;
    cy = prev.cy + (cy - prev.cy) * 0.08;
  }
  return {
    scale,
    cx,
    cy,
    vw,
    vh,
    X: (x) => (x - cx) * scale + w / 2,
    Y: (y) => h / 2 - (y - cy) * scale,
  };
}

/** Body and eyes of a frog at the origin: rx, ry radii; along: eyes at the front (+x). */
function frogBody(ctx, rx, ry, along, c) {
  ctx.fillStyle = c.leaf;
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = c.ink;
  ctx.globalAlpha = 0.45;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = c.paper;
  ctx.globalAlpha = 0.25;
  ctx.beginPath();
  if (along) ctx.ellipse(0, 0, rx * 0.6, ry * 0.35, 0, 0, Math.PI * 2);
  else ctx.ellipse(0, ry * 0.35, rx * 0.6, ry * 0.4, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
  const er = Math.max(2.4, Math.min(rx, ry) * 0.34);
  const eyes = along
    ? [
        [rx * 0.6, -ry * 0.5],
        [rx * 0.6, ry * 0.5],
      ]
    : [
        [-rx * 0.45, -ry * 0.8],
        [rx * 0.45, -ry * 0.8],
      ];
  for (const [ex, ey] of eyes) {
    for (const [col, k] of [
      [c.leaf, 1],
      [c.paper, 0.7],
    ]) {
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(ex, ey, er * k, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = c.shade;
    ctx.beginPath();
    ctx.arc(ex + (along ? er * 0.25 : 0), ey, er * 0.38, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Post 13's air-frog. f: {x, y, vx, vy, perchX, perchY, phase ("rest", "leap", "hold",
 * "back")}. Resting it bobs with its legs folded; leaping or floating back it stretches along
 * its velocity; holding, its legs dangle. Away from home a dashed ring marks its perch.
 */
export function drawFrog(ctx, cam, f, c) {
  const r = Math.max(7, 0.3 * cam.scale);
  ctx.save();
  if (f.phase !== "rest") {
    ctx.strokeStyle = c.leaf;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(cam.X(f.perchX), cam.Y(f.perchY), r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
  ctx.translate(cam.X(f.x), cam.Y(f.y));
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = c.leaf;
  ctx.lineWidth = Math.max(2, 0.07 * cam.scale);
  const speed = Math.hypot(f.vx, f.vy);
  ctx.beginPath();
  if ((f.phase === "leap" || f.phase === "back") && speed > 0.5) {
    ctx.rotate(-Math.atan2(f.vy, f.vx));
    const st = 1 + Math.min(0.7, speed * 0.04);
    for (const side of [-1, 1]) {
      ctx.moveTo(-0.6 * r * st, side * 0.35 * r);
      ctx.lineTo(-1.5 * r * st, side * 0.75 * r);
      ctx.lineTo(-2.2 * r * st, side * 0.55 * r);
      ctx.moveTo(0.6 * r * st, side * 0.4 * r);
      ctx.lineTo(1.3 * r * st, side * 0.6 * r);
    }
    ctx.stroke();
    frogBody(ctx, r * st, r / Math.sqrt(st), true, c);
  } else {
    for (const side of [-1, 1]) {
      if (f.phase === "hold") {
        ctx.moveTo(side * 0.6 * r, 0.45 * r);
        ctx.lineTo(side * 1.3 * r, 1.0 * r);
        ctx.lineTo(side * 1.25 * r, 1.6 * r);
        ctx.moveTo(side * 0.5 * r, -0.1 * r);
        ctx.lineTo(side * 1.25 * r, 0.25 * r);
      } else {
        ctx.moveTo(side * 0.55 * r, 0.35 * r);
        ctx.lineTo(side * 1.25 * r, -0.05 * r);
        ctx.lineTo(side * 1.05 * r, 0.8 * r);
        ctx.lineTo(side * 1.45 * r, 0.85 * r);
        ctx.moveTo(side * 0.35 * r, 0.5 * r);
        ctx.lineTo(side * 0.45 * r, 0.9 * r);
      }
    }
    ctx.stroke();
    frogBody(ctx, r, r, false, c);
  }
  ctx.restore();
}

/** A frog's reach: a faint dashed band `reach` m either side of its column, from 0.5 m up to
 * `top`, cut around the pillars in front of it. */
export function drawReach(ctx, cam, col, reach, top, pillars, size, c) {
  const x0 = cam.X(col - reach);
  const x1 = cam.X(col + reach);
  const yt = cam.Y(top);
  const yb = cam.Y(0.5);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, size.w, size.h);
  for (const q of pillars)
    ctx.rect(cam.X(q.x0) - 2, cam.Y(q.top) - 4, (q.x1 - q.x0) * cam.scale + 4, size.h);
  ctx.clip("evenodd");
  ctx.fillStyle = c.leaf;
  ctx.globalAlpha = 0.07;
  ctx.fillRect(x0, yt, x1 - x0, yb - yt);
  ctx.globalAlpha = 0.6;
  ctx.strokeStyle = c.leaf;
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  for (const x of [x0, x1]) {
    ctx.moveTo(Math.round(x) + 0.5, yt);
    ctx.lineTo(Math.round(x) + 0.5, yb);
  }
  ctx.stroke();
  ctx.restore();
}

/** Smoke behind the missile: pts oldest first, `n` the trail's full length; widens and fades. */
export function drawSmoke(ctx, cam, pts, n, c) {
  ctx.save();
  ctx.strokeStyle = c.muted;
  ctx.lineCap = "round";
  for (let i = 1; i < pts.length; i++) {
    const age = (pts.length - 1 - i) / n;
    ctx.globalAlpha = 0.45 * (1 - age);
    ctx.lineWidth = 2 + 5 * age;
    ctx.beginPath();
    ctx.moveTo(cam.X(pts[i - 1][0]), cam.Y(pts[i - 1][1]));
    ctx.lineTo(cam.X(pts[i][0]), cam.Y(pts[i][1]));
    ctx.stroke();
  }
  ctx.restore();
}

/** An arrow on the view's edge towards (x, y) on screen (when it's off the view). */
export function edgeArrow(ctx, size, x, y, color, hollow = false) {
  const ax = Math.max(14, Math.min(size.w - 14, x));
  const ay = Math.max(14, Math.min(size.h - 14, y));
  ctx.save();
  ctx.translate(ax, ay);
  ctx.rotate(Math.atan2(y - ay, x - ax));
  ctx.beginPath();
  ctx.moveTo(9, 0);
  ctx.lineTo(-5, -6);
  ctx.lineTo(-5, 6);
  ctx.closePath();
  if (hollow) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.restore();
}

/** Post 12's missile: a dart along its velocity (m: {x, y, vx, vy}) with a flickering motor. */
export function drawMissile(ctx, cam, m, c, t) {
  const len = Math.max(18, 0.8 * cam.scale);
  const wid = Math.max(3, 0.11 * cam.scale);
  ctx.save();
  ctx.translate(cam.X(m.x), cam.Y(m.y));
  ctx.rotate(-Math.atan2(m.vy, m.vx));
  ctx.fillStyle = c.lava;
  ctx.globalAlpha = 0.85;
  const flame = len * (0.28 + 0.12 * Math.sin(t * 40));
  ctx.beginPath();
  ctx.moveTo(-len / 2, -wid * 0.6);
  ctx.lineTo(-len / 2 - flame, 0);
  ctx.lineTo(-len / 2, wid * 0.6);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = c.ink;
  ctx.beginPath();
  ctx.moveTo(len / 2, 0);
  ctx.lineTo(len / 2 - wid * 2, -wid);
  ctx.lineTo(-len / 2, -wid);
  ctx.lineTo(-len / 2 - wid * 0.8, -wid * 2.2);
  ctx.lineTo(-len / 2 + wid * 1.6, -wid);
  ctx.lineTo(-len / 2 + wid * 1.6, wid);
  ctx.lineTo(-len / 2 - wid * 0.8, wid * 2.2);
  ctx.lineTo(-len / 2, wid);
  ctx.lineTo(len / 2 - wid * 2, wid);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = c.lava;
  ctx.beginPath();
  ctx.moveTo(len / 2, 0);
  ctx.lineTo(len / 2 - wid * 2, -wid);
  ctx.lineTo(len / 2 - wid * 2, wid);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/** A warning pill in the top-right corner (row 0, 1, ... downwards), with a pulsing dot. */
export function drawBadge(ctx, size, text, c, t, row = 0) {
  ctx.save();
  ctx.font = "700 12px system-ui, sans-serif";
  const bw = ctx.measureText(text).width + 30;
  const bh = 24;
  const x = size.w - bw - 8;
  const y = 8 + row * (bh + 6);
  ctx.fillStyle = c.paper;
  ctx.globalAlpha = 0.92;
  ctx.beginPath();
  ctx.roundRect(x, y, bw, bh, 12);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = c.lava;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = c.lava;
  ctx.globalAlpha = 0.45 + 0.55 * Math.abs(Math.sin(t * 5));
  ctx.beginPath();
  ctx.arc(x + 13, y + bh / 2, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = c.ink;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 22, y + bh / 2 + 0.5);
  ctx.restore();
}
