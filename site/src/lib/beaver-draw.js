// Draws a beaver-vs-river game (site/src/lib/beaver-sim.js) as an isometric board.
// Shared by BeaverGame and BeaverReplay.

import { CLAY, DEN, LOG, LOW, ROCK, SOIL, SPRING, TREE } from "./beaver-sim.js";

/** Scene colours from the CSS tokens, read from an element (follows light / dark mode). */
export function readColors(el) {
  const cs = getComputedStyle(el);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    bg: v("--sky"),
    grass: v("--grass"),
    dirt: v("--dirt"),
    clay: v("--clay"),
    bed: v("--bed"),
    water: v("--water"),
    rock: v("--stone"),
    leaf: v("--leaf"),
    wood: v("--wood"),
    beaver: v("--beaver"),
    ink: v("--ink"),
    muted: v("--muted"),
    paper: v("--paper"),
    river: v("--flag"),
    mark: v("--lava"),
    font: cs.fontFamily,
  };
}

const Z = { base: -0.55, high: 1, low: 0.3, water: 0.68 };

/** Geometry for a canvas of the given pixel size and an n x n board. */
export function layout(width, height, n) {
  const tw = Math.min(width / (n + 0.4), height / (n / 2 + 1.25));
  const th = tw / 2;
  const zu = tw * 0.3;
  const ox = width / 2;
  const oy = (height - (n * th + 2.2 * zu - Z.base * zu)) / 2 + 2.2 * zu;
  return { n, tw, th, zu, ox, oy };
}

/** Screen point of board coordinates (x, y are tile units, corner-based) at height z. */
function P(L, x, y, z) {
  return [L.ox + ((x - y) * L.tw) / 2, L.oy + ((x + y) * L.th) / 2 - z * L.zu];
}

/** The tile under a screen point, testing tiles front to back by their top face. */
export function tileAt(L, g, px, py) {
  const n = L.n;
  for (let s = 2 * n - 2; s >= 0; s--) {
    for (let x = Math.max(0, s - n + 1); x <= Math.min(n - 1, s); x++) {
      const y = s - x;
      const z = topZ(g.tiles[y * n + x]);
      const [cx, cy] = P(L, x + 0.5, y + 0.5, z);
      if (Math.abs(px - cx) / (L.tw / 2) + Math.abs(py - cy) / (L.th / 2) <= 1)
        return y * n + x;
    }
  }
  return -1;
}

const topZ = (t) => (t === LOW || t === LOG || t === SPRING ? Z.low : Z.high);

function shade(hex, k) {
  // k < 1 darkens, k > 1 lightens towards white
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex);
  if (!m) return hex;
  const c = [1, 2, 3].map((i) => parseInt(m[i], 16));
  const out = c.map((v) =>
    Math.round(k <= 1 ? v * k : v + (255 - v) * (k - 1)),
  );
  return `rgb(${out.join(",")})`;
}

function poly(ctx, pts, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}

/** A box from (x0,y0) to (x1,y1) in tile units, z0..z1 high: top, left-front and right-front faces. */
function box(ctx, L, x0, y0, x1, y1, z0, z1, top, side) {
  const a = (x, y, z) => P(L, x, y, z);
  poly(
    ctx,
    [a(x0, y1, z1), a(x1, y1, z1), a(x1, y1, z0), a(x0, y1, z0)],
    shade(side, 0.85),
  );
  poly(
    ctx,
    [a(x1, y0, z1), a(x1, y1, z1), a(x1, y1, z0), a(x1, y0, z0)],
    shade(side, 0.7),
  );
  poly(ctx, [a(x0, y0, z1), a(x1, y0, z1), a(x1, y1, z1), a(x0, y1, z1)], top);
}

function ellipse(ctx, cx, cy, rx, ry, fill) {
  ctx.beginPath();
  ctx.ellipse(cx, cy, Math.max(0.5, rx), Math.max(0.5, ry), 0, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

/**
 * g: a beaver-sim game. opts:
 *   options: Set of tiles the river may dig (outlined), hover: a tile, reach: g.r.reach,
 *   strength: steps from the spring per tile (water fades as it runs out),
 *   beaverFrom + t: animate the beaver from tile beaverFrom (t in 0..1),
 *   fresh: Set of tiles the water just reached (fade in with t),
 *   marks: { dig, log } tiles to mark.
 */
export function drawBoard(ctx, g, C, L, opts = {}) {
  const {
    options = null,
    hover = -1,
    strength = null,
    fresh = null,
    t = 1,
    marks = {},
  } = opts;
  const n = g.n;
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  ctx.lineJoin = "round";
  const reach = g.r.reach;

  // Beaver position (interpolated while it walks).
  const bx = (i) => [(i % n) + 0.5, Math.floor(i / n) + 0.5];
  let [bX, bY] = bx(g.beaverAt);
  if (opts.beaverFrom != null && opts.beaverFrom !== g.beaverAt && t < 1) {
    const [fx, fy] = bx(opts.beaverFrom);
    const e = t * t * (3 - 2 * t);
    bX = fx + (bX - fx) * e;
    bY = fy + (bY - fy) * e;
  }
  // drawn after the row of tiles it's nearest, so tiles in front of it can hide it
  const near = Math.round(bY - 0.5) * n + Math.round(bX - 0.5);
  const beaverRow = Math.round(bX - 0.5) + Math.round(bY - 0.5);

  for (let s = 0; s <= 2 * n - 2; s++) {
    for (let x = Math.max(0, s - n + 1); x <= Math.min(n - 1, s); x++) {
      const y = s - x;
      const i = y * n + x;
      const tile = g.tiles[i];
      const zt = topZ(tile);
      let top;
      let side = C.dirt;
      if (tile === SOIL || tile === DEN || tile === ROCK || tile === TREE)
        top = C.grass;
      else if (tile === CLAY) {
        top = C.clay;
        side = shade(C.clay, 0.8);
      } else top = C.bed;
      // checker tint so tiles read as a grid
      if (
        (x + y) % 2 &&
        (tile === SOIL || tile === ROCK || tile === TREE || tile === DEN)
      )
        top = shade(top, 0.96);
      box(ctx, L, x, y, x + 1, y + 1, Z.base, zt, top, side);

      if (tile === CLAY) {
        // clay: a few pebbles
        ctx.fillStyle = shade(C.clay, 0.78);
        for (const [u, v] of [
          [0.3, 0.35],
          [0.65, 0.3],
          [0.5, 0.7],
        ]) {
          const [px, py] = P(L, x + u, y + v, zt);
          ctx.fillRect(Math.round(px - 1), Math.round(py - 1), 2, 2);
        }
      }

      // where the beaver can't dam: faint hatching on the ground (under any water)
      if (g.noDam[i] && (tile === SOIL || tile === CLAY || tile === LOW))
        hatch(ctx, L, x, y, zt, C.muted);

      // water
      if (g.wet[i] && (tile === LOW || tile === SPRING || tile === DEN)) {
        const k = strength && strength[i] < 1e6 ? strength[i] / reach : 0;
        let a = 0.92 - 0.45 * k;
        if (fresh && fresh.has(i)) a *= t;
        ctx.globalAlpha = a;
        const zw = tile === DEN ? Z.high + 0.05 : Z.water;
        poly(
          ctx,
          [
            P(L, x, y, zw),
            P(L, x + 1, y, zw),
            P(L, x + 1, y + 1, zw),
            P(L, x, y + 1, zw),
          ],
          C.water,
        );
        // front faces of the water where the neighbour in front is lower (dry bed or off-board)
        const front = [
          [
            x,
            y + 1,
            (y + 1) * n + x,
            [
              P(L, x, y + 1, zw),
              P(L, x + 1, y + 1, zw),
              P(L, x + 1, y + 1, Z.low),
              P(L, x, y + 1, Z.low),
            ],
          ],
          [
            x + 1,
            y,
            y * n + x + 1,
            [
              P(L, x + 1, y, zw),
              P(L, x + 1, y + 1, zw),
              P(L, x + 1, y + 1, Z.low),
              P(L, x + 1, y, Z.low),
            ],
          ],
        ];
        for (const [fx, fy, j, pts] of front) {
          const off = fx >= n || fy >= n;
          if (off || (!g.wet[j] && topZ(g.tiles[j]) === Z.low))
            poly(ctx, pts, shade(C.water, 0.8));
        }
        ctx.globalAlpha = 1;
      }

      if (tile === SPRING) {
        const [cx, cy] = P(L, x + 0.5, y + 0.5, Z.water);
        ctx.strokeStyle = shade(C.water, 1.6);
        ctx.lineWidth = Math.max(1, L.tw / 40);
        for (const r of [0.18, 0.32]) {
          ctx.beginPath();
          ctx.ellipse(
            cx,
            cy,
            (r * L.tw) / 1.2,
            (r * L.th) / 1.2,
            0,
            0,
            Math.PI * 2,
          );
          ctx.stroke();
        }
      }

      // outline for river options, hover and the last moves
      const outline = (color, width, dash) => {
        ctx.setLineDash(dash || []);
        ctx.lineWidth = width;
        poly(
          ctx,
          [
            P(L, x, y, zt),
            P(L, x + 1, y, zt),
            P(L, x + 1, y + 1, zt),
            P(L, x, y + 1, zt),
          ],
          null,
          color,
        );
        ctx.setLineDash([]);
      };
      if (options && options.has(i)) {
        if (i === hover) {
          ctx.globalAlpha = 0.35;
          poly(
            ctx,
            [
              P(L, x, y, zt),
              P(L, x + 1, y, zt),
              P(L, x + 1, y + 1, zt),
              P(L, x, y + 1, zt),
            ],
            C.river,
          );
          ctx.globalAlpha = 1;
        }
        outline(C.river, Math.max(1.5, L.tw / 28), [
          Math.max(2, L.tw / 12),
          Math.max(2, L.tw / 16),
        ]);
      }
      if (marks.dig === i) outline(C.river, Math.max(2, L.tw / 18));

      if (tile === LOG) drawLogs(ctx, L, x, y, C, marks.log === i ? t : 1);
      if (tile === ROCK) drawRock(ctx, L, x, y, C);
      if (tile === TREE) drawTree(ctx, L, x, y, C);
      if (tile === DEN) drawDen(ctx, L, x, y, C, g.wet[i]);
      if (marks.log === i) outline(C.mark, Math.max(2, L.tw / 18));
    }
    if (s === beaverRow)
      drawBeaver(ctx, L, bX, bY, topZ(g.tiles[near]), C, g.wet[near]);
  }
}

function hatch(ctx, L, x, y, z, color) {
  ctx.save();
  ctx.beginPath();
  const pts = [
    P(L, x, y, z),
    P(L, x + 1, y, z),
    P(L, x + 1, y + 1, z),
    P(L, x, y + 1, z),
  ];
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
  ctx.closePath();
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.45;
  ctx.lineWidth = Math.max(1, L.tw / 40);
  for (let k = 1; k < 5; k++) {
    const a = P(L, x + k / 5, y, z);
    const b = P(L, x + k / 5, y + 1, z);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawLogs(ctx, L, x, y, C, grow) {
  const z0 = Z.low;
  for (let k = 0; k < 3; k++) {
    const zz = z0 + 0.22 + k * 0.28 * (k === 2 ? 0.9 : 1);
    if (k > 0 && grow < k / 3) break;
    const u0 = k === 2 ? 0.3 : 0.12;
    const v = k === 1 ? 0.62 : k === 2 ? 0.48 : 0.35;
    const a = P(L, x + u0, y + v, zz);
    const b = P(L, x + 1 - u0 + (k === 2 ? 0 : -0.02), y + v, zz);
    const r = L.zu * 0.22;
    ctx.strokeStyle = C.wood;
    ctx.lineCap = "round";
    ctx.lineWidth = r * 2;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.lineCap = "butt";
    ellipse(ctx, b[0], b[1], r * 0.85, r * 0.85, shade(C.wood, 1.45));
  }
}

function drawRock(ctx, L, x, y, C) {
  const [cx, cy] = P(L, x + 0.5, y + 0.5, Z.high);
  const w = L.tw;
  ellipse(ctx, cx, cy + w * 0.02, w * 0.36, w * 0.16, shade(C.rock, 0.55));
  ctx.beginPath();
  ctx.moveTo(cx - w * 0.34, cy);
  ctx.quadraticCurveTo(
    cx - w * 0.3,
    cy - w * 0.36,
    cx - w * 0.02,
    cy - w * 0.38,
  );
  ctx.quadraticCurveTo(cx + w * 0.3, cy - w * 0.34, cx + w * 0.34, cy);
  ctx.quadraticCurveTo(cx, cy + w * 0.14, cx - w * 0.34, cy);
  ctx.fillStyle = C.rock;
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(cx - w * 0.16, cy - w * 0.3);
  ctx.quadraticCurveTo(
    cx - w * 0.02,
    cy - w * 0.36,
    cx + w * 0.12,
    cy - w * 0.28,
  );
  ctx.strokeStyle = shade(C.rock, 1.35);
  ctx.lineWidth = Math.max(1, w / 30);
  ctx.stroke();
}

function drawTree(ctx, L, x, y, C) {
  const [cx, cy] = P(L, x + 0.5, y + 0.5, Z.high);
  const w = L.tw;
  ellipse(
    ctx,
    cx + w * 0.06,
    cy + w * 0.03,
    w * 0.3,
    w * 0.12,
    "rgba(0,0,0,0.18)",
  );
  ctx.fillStyle = C.wood;
  ctx.fillRect(cx - w * 0.04, cy - w * 0.3, w * 0.08, w * 0.3);
  ellipse(ctx, cx, cy - w * 0.5, w * 0.27, w * 0.27, shade(C.leaf, 0.85));
  ellipse(ctx, cx - w * 0.06, cy - w * 0.56, w * 0.18, w * 0.17, C.leaf);
  ellipse(
    ctx,
    cx - w * 0.1,
    cy - w * 0.6,
    w * 0.07,
    w * 0.06,
    shade(C.leaf, 1.25),
  );
}

function drawDen(ctx, L, x, y, C, wet) {
  const [cx, cy] = P(L, x + 0.5, y + 0.5, Z.high + (wet ? 0.05 : 0));
  const w = L.tw;
  ctx.beginPath();
  ctx.moveTo(cx - w * 0.4, cy + w * 0.02);
  ctx.quadraticCurveTo(cx - w * 0.36, cy - w * 0.42, cx, cy - w * 0.42);
  ctx.quadraticCurveTo(
    cx + w * 0.36,
    cy - w * 0.42,
    cx + w * 0.4,
    cy + w * 0.02,
  );
  ctx.quadraticCurveTo(cx, cy + w * 0.18, cx - w * 0.4, cy + w * 0.02);
  ctx.fillStyle = shade(C.wood, 0.85);
  ctx.fill();
  // sticks
  ctx.strokeStyle = shade(C.wood, 1.3);
  ctx.lineWidth = Math.max(1, w / 34);
  ctx.lineCap = "round";
  for (const [a, b, c, d] of [
    [-0.3, -0.12, 0.12, -0.34],
    [-0.12, -0.3, 0.3, -0.1],
    [-0.34, -0.02, 0.0, -0.22],
    [0.06, -0.22, 0.34, -0.02],
  ]) {
    ctx.beginPath();
    ctx.moveTo(cx + a * w, cy + b * w);
    ctx.lineTo(cx + c * w, cy + d * w);
    ctx.stroke();
  }
  ctx.lineCap = "butt";
  // entrance
  ellipse(
    ctx,
    cx + w * 0.05,
    cy - w * 0.02,
    w * 0.1,
    w * 0.07,
    shade(C.wood, 0.35),
  );
  if (wet) {
    ctx.globalAlpha = 0.6;
    ellipse(ctx, cx, cy + w * 0.04, w * 0.46, w * 0.16, C.water);
    ctx.globalAlpha = 1;
  }
}

function drawBeaver(ctx, L, X, Y, z, C, swimming) {
  const [cx, cy] = P(L, X, Y, swimming ? Z.water - 0.1 : z);
  const w = L.tw;
  // tail
  ctx.save();
  ctx.translate(cx - w * 0.16, cy + w * 0.03);
  ctx.rotate(-0.35);
  ellipse(ctx, 0, 0, w * 0.12, w * 0.055, shade(C.beaver, 0.55));
  ctx.restore();
  // body
  ellipse(ctx, cx, cy - w * 0.07, w * 0.17, w * 0.12, C.beaver);
  // head
  ellipse(
    ctx,
    cx + w * 0.13,
    cy - w * 0.15,
    w * 0.085,
    w * 0.075,
    shade(C.beaver, 1.1),
  );
  ellipse(
    ctx,
    cx + w * 0.09,
    cy - w * 0.22,
    w * 0.025,
    w * 0.022,
    shade(C.beaver, 0.7),
  );
  ellipse(ctx, cx + w * 0.16, cy - w * 0.17, w * 0.014, w * 0.014, "#111");
  // teeth
  ctx.fillStyle = "#f4ecd8";
  ctx.fillRect(
    cx + w * 0.2,
    cy - w * 0.13,
    Math.max(1, w * 0.02),
    Math.max(1, w * 0.03),
  );
  if (swimming) {
    ctx.globalAlpha = 0.55;
    ellipse(ctx, cx, cy - w * 0.01, w * 0.24, w * 0.07, C.water);
    ctx.globalAlpha = 1;
  }
}

/** Sizes the canvas backing store to its CSS box (device pixels). Returns the CSS-pixel size. */
export function fitCanvas(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return dpr;
}
