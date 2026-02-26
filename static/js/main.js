/* ============================================================
   Four-Bar Linkage Synthesizer — Client-side JavaScript
   Kinematics are solved directly in JS for real-time animation.
   Reward metrics are fetched from the Flask API.
   ============================================================ */

"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────

const PARAM_KEYS   = ["L1", "L2", "L3", "L4", "xO2", "yO2"];
const BOUNDS       = [[0.05, 0.4], [0.05, 0.4], [0.05, 0.4], [0.05, 0.4], [-0.1, 0.2], [-0.1, 0.2]];
const DEFAULT_PARAMS = [0.1, 0.07, 0.12, 0.09, 0.0, 0.0];
const N_STEPS      = 360;          // animation steps per revolution
const FPS_TARGET   = 60;
const ANIM_SPD     = 1;            // frames to advance per render tick
const METRICS_DEBOUNCE_MS = 350;

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  params:     [...DEFAULT_PARAMS],
  lockMask:   [false, false, false, false, false, false],
  targetC:    0.0,
  seed:       null,
  frame:      0,
  paused:     false,
  optimizing: false,
  metrics:    { reward: null, mse: null, maxDev: null, rom: null },
  // cached full-revolution data (updated after params change)
  fullPath:         [],   // [{x,y}] coupler midpoints
  rockerAnglesArr:  [],   // [deg | null]
  crankAnglesArr:   [],   // [rad]
};

// ── Four-bar kinematics (mirrors Python closure() / solve_rocker_angle()) ────

/**
 * Compute all joint positions for a given crank angle θ₂.
 * Returns null if no solution exists (linkage cannot close).
 */
function computeStep(theta2, p) {
  const [L1, L2, L3, L4, xO2, yO2] = p;
  const Bx = xO2 + L2 * Math.cos(theta2);
  const By = yO2 + L2 * Math.sin(theta2);
  const dx = Bx - L1;   // O4 = (L1, 0)
  const dy = By;
  const R  = Math.sqrt(dx * dx + dy * dy);
  const minR = Math.abs(L3 - L4);
  const maxR = L3 + L4;
  if (maxR <= 0 || R < minR - 1e-9 || R > maxR + 1e-9) return null;
  const cosP = Math.max(-1, Math.min(1, (L4 * L4 + R * R - L3 * L3) / (2 * L4 * R)));
  const phi   = Math.acos(cosP);
  const psi   = Math.atan2(dy, dx);
  const th4   = psi - phi;
  const Cx = L1 + L4 * Math.cos(th4);
  const Cy =      L4 * Math.sin(th4);
  return {
    O1:  { x: 0,   y: 0 },
    O2:  { x: xO2, y: yO2 },
    O4:  { x: L1,  y: 0 },
    B:   { x: Bx,  y: By },
    C:   { x: Cx,  y: Cy },
    mid: { x: (Bx + Cx) / 2, y: (By + Cy) / 2 },
    th4,
  };
}

/** Compute the full coupler path over 360 crank-angle steps. */
function computeFullPath(p) {
  const pts    = [];
  const rockers = [];
  const cranks  = [];
  for (let i = 0; i < N_STEPS; i++) {
    const th2 = (i / N_STEPS) * 2 * Math.PI;
    const res  = computeStep(th2, p);
    cranks.push(th2);
    if (res) {
      pts.push(res.mid);
      rockers.push(res.th4 * 180 / Math.PI);
    } else {
      rockers.push(null);
    }
  }
  return { pts, rockers, cranks };
}

// ── Canvas rendering ──────────────────────────────────────────────────────────

const canvas = document.getElementById("linkage-canvas");
const ctx    = canvas.getContext("2d");

// CSS pixel dimensions — set by resizeCanvas(), used for all layout math.
// canvas.width/height are device pixels (×dpr) and must NOT be used for layout.
let cssW = 1, cssH = 1;

/** Convert world coords (m) to canvas pixels. */
function w2c(wx, wy, origin, scale) {
  return { x: origin.x + wx * scale, y: origin.y - wy * scale };
}

/** Draw cross-hatch ground symbol. */
function drawGround(cx, cy, sz) {
  ctx.save();
  ctx.strokeStyle = "#2d4a6e";
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(cx - sz, cy + sz * 0.5);
  ctx.lineTo(cx + sz, cy + sz * 0.5);
  ctx.stroke();
  const n = 5;
  for (let i = 0; i < n; i++) {
    const x0 = cx - sz + (i / (n - 1)) * sz * 2;
    ctx.beginPath();
    ctx.moveTo(x0, cy + sz * 0.5);
    ctx.lineTo(x0 - sz * 0.35, cy + sz * 0.9);
    ctx.stroke();
  }
  ctx.restore();
}

/** Draw a circular joint. */
function drawJoint(px, py, r, fill, stroke) {
  ctx.beginPath();
  ctx.arc(px, py, r, 0, 2 * Math.PI);
  ctx.fillStyle   = fill;
  ctx.fill();
  ctx.strokeStyle = stroke || "rgba(7,11,20,0.8)";
  ctx.lineWidth   = 1.5;
  ctx.stroke();
}

/** Compute auto-scale transform to fit the mechanism in the canvas.
 *  Uses cssW/cssH (CSS pixel dimensions) — never device-pixel dimensions. */
function computeTransform(p) {
  const [L1, L2, L3, L4, xO2, yO2] = p;

  // Collect every world-space point that must stay inside the viewport:
  // fixed pivots + full sweep of B, C, and coupler midpoints.
  let minX =  Infinity, maxX = -Infinity;
  let minY =  Infinity, maxY = -Infinity;

  function expand(x, y) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  // Fixed pivots
  expand(0,   0);    // O1
  expand(xO2, yO2);  // O2
  expand(L1,  0);    // O4

  // Sweep B over full rotation; also compute C and midpoint where valid
  for (let i = 0; i < N_STEPS; i++) {
    const th2 = (i / N_STEPS) * 2 * Math.PI;
    const Bx  = xO2 + L2 * Math.cos(th2);
    const By  = yO2 + L2 * Math.sin(th2);
    expand(Bx, By);
    const res = computeStep(th2, p);
    if (res) {
      expand(res.C.x,   res.C.y);
      expand(res.mid.x, res.mid.y);
    }
  }

  // Guard against degenerate case (all points collapsed)
  if (!isFinite(minX)) { minX = -0.1; maxX = 0.3; minY = -0.2; maxY = 0.2; }

  // Add 12 % padding around the tight bounding box
  const rx = (maxX - minX) || 0.1;
  const ry = (maxY - minY) || 0.1;
  minX -= rx * 0.12;  maxX += rx * 0.12;
  minY -= ry * 0.12;  maxY += ry * 0.12;

  // Layout in CSS pixels (cssW/cssH are set by resizeCanvas)
  const pad    = 20;
  const drawW  = cssW - 2 * pad;
  const drawH  = cssH - 2 * pad;
  const scaleX = drawW / (maxX - minX);
  const scaleY = drawH / (maxY - minY);
  const scale  = Math.min(scaleX, scaleY);

  // World centre → canvas centre
  const midWorldX = (minX + maxX) / 2;
  const midWorldY = (minY + maxY) / 2;
  const ox = pad + drawW / 2 - midWorldX * scale;
  const oy = pad + drawH / 2 + midWorldY * scale;   // +y because world-y is up

  return { origin: { x: ox, y: oy }, scale };
}

/** Faint background grid — uses CSS pixel dimensions. */
function drawGrid() {
  const step = 28;
  ctx.save();
  ctx.strokeStyle = "rgba(26,54,84,0.5)";
  ctx.lineWidth   = 0.5;
  for (let x = 0; x < cssW; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cssH); ctx.stroke();
  }
  for (let y = 0; y < cssH; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cssW, y); ctx.stroke();
  }
  ctx.restore();
}

/** Main render function — called every animation frame. */
function render() {
  // Use CSS pixel dimensions for all layout; canvas.width/height are device pixels.
  ctx.clearRect(0, 0, cssW, cssH);

  // Background
  ctx.fillStyle = "#070b14";
  ctx.fillRect(0, 0, cssW, cssH);
  drawGrid();

  const p     = state.params;
  const trans = computeTransform(p);
  const { origin: O, scale: S } = trans;
  const t = (wx, wy) => w2c(wx, wy, O, S);

  // ── Coupler path trail ─────────────────────────────────
  const path = state.fullPath;
  if (path.length > 1) {
    ctx.save();
    ctx.lineWidth   = 1.5;
    ctx.strokeStyle = "rgba(244,63,94,0.55)";
    ctx.beginPath();
    const p0 = t(path[0].x, path[0].y);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < path.length; i++) {
      const pi = t(path[i].x, path[i].y);
      ctx.lineTo(pi.x, pi.y);
    }
    // close the path
    ctx.lineTo(p0.x, p0.y);
    ctx.stroke();
    ctx.restore();
  }

  // ── Target line (dashed amber) ─────────────────────────
  {
    const xLeft  = (0    - O.x) / S;   // left  canvas edge → world x
    const xRight = (cssW - O.x) / S;   // right canvas edge → world x (use CSS px)
    const tl1 = t(xLeft,  state.targetC);
    const tl2 = t(xRight, state.targetC);
    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth   = 1.5;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.moveTo(tl1.x, tl1.y);
    ctx.lineTo(tl2.x, tl2.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // ── Animated linkage at current frame ─────────────────
  const theta2 = (state.frame % N_STEPS) / N_STEPS * 2 * Math.PI;
  const res    = computeStep(theta2, p);

  if (res) {
    const { O1, O2, O4, B, C, mid } = res;
    const pO1 = t(O1.x, O1.y);
    const pO2 = t(O2.x, O2.y);
    const pO4 = t(O4.x, O4.y);
    const pB  = t(B.x,  B.y);
    const pC  = t(C.x,  C.y);
    const pM  = t(mid.x, mid.y);

    // Draw pivot offset dotted line (O1 → O2) only if offset exists
    if (Math.abs(p[4]) > 1e-5 || Math.abs(p[5]) > 1e-5) {
      ctx.save();
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = "#2d4a6e";
      ctx.lineWidth   = 1.2;
      ctx.beginPath();
      ctx.moveTo(pO1.x, pO1.y);
      ctx.lineTo(pO2.x, pO2.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Ground link (O1 → O4) — grey solid
    ctx.save();
    ctx.lineWidth   = 3;
    ctx.strokeStyle = "#334155";
    ctx.lineCap     = "round";
    ctx.beginPath();
    ctx.moveTo(pO1.x, pO1.y);
    ctx.lineTo(pO4.x, pO4.y);
    ctx.stroke();
    ctx.restore();

    // Helper: draw a thick rounded link
    function drawLink(ax, ay, bx, by, col, glow) {
      ctx.save();
      if (glow) {
        ctx.shadowColor = col;
        ctx.shadowBlur  = 8;
      }
      ctx.lineWidth   = 4.5;
      ctx.strokeStyle = col;
      ctx.lineCap     = "round";
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.restore();
    }

    // Crank   O2 → B  (cyan)
    drawLink(pO2.x, pO2.y, pB.x, pB.y, "#00d4ff", true);
    // Coupler B  → C  (purple)
    drawLink(pB.x, pB.y, pC.x, pC.y, "#a78bfa", false);
    // Rocker  O4 → C  (green)
    drawLink(pO4.x, pO4.y, pC.x, pC.y, "#22c55e", false);

    // Ground symbols
    drawGround(pO1.x, pO1.y, 9);
    drawGround(pO4.x, pO4.y, 9);

    // Joints
    drawJoint(pO1.x, pO1.y, 5,  "#475569");
    drawJoint(pO2.x, pO2.y, 5,  "#00d4ff");
    drawJoint(pO4.x, pO4.y, 5,  "#475569");
    drawJoint(pB.x,  pB.y,  4.5,"#fbbf24");
    drawJoint(pC.x,  pC.y,  4.5,"#fbbf24");

    // Coupler midpoint (moving red dot)
    ctx.save();
    ctx.shadowColor = "#f43f5e";
    ctx.shadowBlur  = 10;
    drawJoint(pM.x, pM.y, 4, "#f43f5e", "#7f1d1d");
    ctx.restore();

    // Labels
    ctx.save();
    ctx.font      = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#526480";
    ctx.fillText("O₁", pO1.x + 8,  pO1.y - 5);
    ctx.fillText("O₂", pO2.x + 8,  pO2.y - 5);
    ctx.fillText("O₄", pO4.x + 8,  pO4.y - 5);
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("B",  pB.x  + 6,  pB.y  - 4);
    ctx.fillText("C",  pC.x  + 6,  pC.y  - 4);
    ctx.restore();
  }

  // Update crank angle display
  const deg = Math.round((state.frame % N_STEPS) * (360 / N_STEPS));
  document.getElementById("angle-value").textContent = `${deg}°`;
}

// ── Animation loop ────────────────────────────────────────────────────────────

let animHandle = null;

function tick() {
  if (!state.paused) {
    state.frame += ANIM_SPD;
  }
  render();
  animHandle = requestAnimationFrame(tick);
}

// ── Rocker-vs-Crank chart (Chart.js) ─────────────────────────────────────────

let rockerChart = null;

function initChart() {
  const chartCanvas = document.getElementById("rocker-chart");
  Chart.defaults.color = "#526480";
  rockerChart = new Chart(chartCanvas, {
    type: "line",
    data: {
      labels:   [],
      datasets: [
        {
          label:       "Rocker angle (°)",
          data:        [],
          borderColor: "#00d4ff",
          borderWidth: 2,
          pointRadius: 0,
          tension:     0.35,
          fill:        {
            target: "origin",
            above:  "rgba(0,212,255,0.06)",
          },
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 200 },
      plugins: {
        legend:  { display: false },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          ticks: {
            color:        "#526480",
            maxTicksLimit: 8,
            callback: (val, idx, ticks) => {
              // Format as multiples of π/2
              const rad = (idx / N_STEPS) * 2 * Math.PI;
              const frac = rad / Math.PI;
              const fracs = [0, 0.5, 1, 1.5, 2];
              const labels = ["0", "π/2", "π", "3π/2", "2π"];
              const i = fracs.findIndex(f => Math.abs(f - frac) < 0.05);
              return i >= 0 ? labels[i] : "";
            },
          },
          grid:  { color: "rgba(26,54,84,0.5)" },
          title: { display: true, text: "Crank angle (rad)", color: "#526480", font: { size: 10 } },
        },
        y: {
          ticks: { color: "#526480" },
          grid:  { color: "rgba(26,54,84,0.5)" },
          title: { display: true, text: "Rocker angle (°)", color: "#526480", font: { size: 10 } },
        },
      },
    },
  });
}

function updateChart() {
  if (!rockerChart) return;
  const cranks  = state.crankAnglesArr;
  const rockers = state.rockerAnglesArr;
  rockerChart.data.labels              = cranks.map((_, i) => i);
  rockerChart.data.datasets[0].data   = rockers.map(v => v !== null ? v : NaN);
  rockerChart.update("none");

  // ROM badge
  const valid = rockers.filter(v => v !== null);
  const romEl = document.getElementById("rom-badge");
  if (valid.length > 0) {
    const rom = Math.max(...valid) - Math.min(...valid);
    romEl.textContent = `ROM: ${rom.toFixed(2)}°`;
  } else {
    romEl.textContent = "ROM: —";
  }
}

// ── UI: sliders + number boxes ────────────────────────────────────────────────

function initSliders() {
  PARAM_KEYS.forEach((key, i) => {
    const slider = document.getElementById(`slider-${key}`);
    const numBox = document.getElementById(`num-${key}`);
    const [lo, hi] = BOUNDS[i];

    // Sync from state
    slider.min   = lo;
    slider.max   = hi;
    slider.step  = (hi - lo) / 1000;
    slider.value = state.params[i];
    numBox.value = state.params[i].toFixed(4);
    updateSliderFill(slider);

    slider.addEventListener("input", () => {
      const v = parseFloat(slider.value);
      state.params[i] = v;
      numBox.value     = v.toFixed(4);
      updateSliderFill(slider);
      onParamChange();
    });

    numBox.addEventListener("change", () => {
      let v = parseFloat(numBox.value);
      if (isNaN(v)) v = state.params[i];
      v = Math.max(lo, Math.min(hi, v));
      state.params[i] = v;
      slider.value     = v;
      numBox.value     = v.toFixed(4);
      updateSliderFill(slider);
      onParamChange();
    });
  });
}

/** Update the filled-track colour on a slider by CSS custom property. */
function updateSliderFill(slider) {
  const lo  = parseFloat(slider.min);
  const hi  = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - lo) / (hi - lo)) * 100;
  slider.style.setProperty("--fill", `${pct.toFixed(1)}%`);
  // Use a background gradient to show the filled portion
  slider.style.background =
    `linear-gradient(to right, var(--cyan) ${pct.toFixed(1)}%, var(--border2) ${pct.toFixed(1)}%)`;
}

function refreshSliderUI() {
  PARAM_KEYS.forEach((key, i) => {
    const slider = document.getElementById(`slider-${key}`);
    const numBox = document.getElementById(`num-${key}`);
    slider.value = state.params[i];
    numBox.value  = state.params[i].toFixed(4);
    updateSliderFill(slider);
  });
}

// ── UI: lock checkboxes ───────────────────────────────────────────────────────

function initLocks() {
  PARAM_KEYS.forEach((key, i) => {
    const chk = document.getElementById(`lock-${key}`);
    chk.addEventListener("change", () => {
      state.lockMask[i] = chk.checked;
    });
  });
}

// ── UI: target & seed ─────────────────────────────────────────────────────────

function initTargetSeed() {
  document.getElementById("target-c").addEventListener("change", e => {
    state.targetC = parseFloat(e.target.value) || 0.0;
    onParamChange();
  });
  document.getElementById("seed-input").addEventListener("change", e => {
    const s = e.target.value.trim().toLowerCase();
    state.seed = (s === "" || s === "random" || s === "none") ? null : parseInt(s, 10);
  });
}

// ── Metrics API call ──────────────────────────────────────────────────────────

let metricsTimer = null;

function onParamChange() {
  // Immediately rebuild full path in JS for animation
  const { pts, rockers, cranks } = computeFullPath(state.params);
  state.fullPath        = pts;
  state.rockerAnglesArr = rockers;
  state.crankAnglesArr  = cranks;
  updateChart();

  // Debounce the API call (fetches reward / mse / max_dev from Python)
  clearTimeout(metricsTimer);
  metricsTimer = setTimeout(fetchMetrics, METRICS_DEBOUNCE_MS);
}

async function fetchMetrics() {
  try {
    const resp = await fetch("/api/metrics", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        params:    state.params,
        lock_mask: state.lockMask,
        target_c:  state.targetC,
      }),
    });
    if (!resp.ok) return;
    const data = await resp.json();

    // Update params with projected version
    state.params = data.params;
    refreshSliderUI();

    state.metrics.reward = data.reward;
    state.metrics.mse    = data.mse;
    state.metrics.maxDev = data.max_dev;
    state.metrics.rom    = data.rom;

    updateMetricsDisplay();

    // Also update chart with server-side rocker angles (should match JS)
    state.rockerAnglesArr = data.rocker_angles;
    state.crankAnglesArr  = data.crank_angles;
    updateChart();
  } catch (err) {
    console.warn("Metrics fetch failed:", err);
  }
}

function updateMetricsDisplay() {
  const { reward, mse, maxDev } = state.metrics;

  const rewardEl = document.getElementById("stat-reward");
  const mseEl    = document.getElementById("stat-mse");
  const devEl    = document.getElementById("stat-maxdev");

  if (reward === null || reward < -1e5) {
    rewardEl.textContent = "invalid";
    rewardEl.className   = "metric-value bad";
  } else {
    rewardEl.textContent = reward.toFixed(4);
    rewardEl.className   = reward > -1 ? "metric-value good" :
                           reward > -5 ? "metric-value warn" : "metric-value bad";
  }

  mseEl.textContent = mse !== null ? mse.toExponential(3) : "—";
  mseEl.className   = "metric-value " + (mse === null ? "neutral" : mse < 1e-4 ? "good" : mse < 1e-3 ? "warn" : "bad");

  devEl.textContent = maxDev !== null ? maxDev.toExponential(3) : "—";
  devEl.className   = "metric-value " + (maxDev === null ? "neutral" : maxDev < 0.01 ? "good" : maxDev < 0.05 ? "warn" : "bad");
}

// ── Optimizer buttons ─────────────────────────────────────────────────────────

function setOptimizing(running, method) {
  state.optimizing = running;
  const btns = ["btn-cma", "btn-cem", "btn-ppo", "btn-reset"].map(id => document.getElementById(id));
  btns.forEach(b => { b.disabled = running; });

  const statusEl  = document.getElementById("opt-status");
  const progressW = document.getElementById("progress-bar-wrap");

  if (running) {
    statusEl.textContent = `Running ${method} — please wait…`;
    statusEl.className   = "opt-status running";
    progressW.style.display = "block";
    animateProgress();
  } else {
    statusEl.className   = "opt-status done";
    progressW.style.display = "none";
  }
}

let _progressInterval = null;
function animateProgress() {
  clearInterval(_progressInterval);
  const bar = document.getElementById("progress-bar");
  let pct = 0;
  _progressInterval = setInterval(() => {
    pct = Math.min(pct + Math.random() * 2, 95);
    bar.style.width = `${pct}%`;
  }, 300);
}
function finishProgress() {
  clearInterval(_progressInterval);
  const bar = document.getElementById("progress-bar");
  bar.style.width = "100%";
  setTimeout(() => { bar.style.width = "0%"; }, 600);
}

async function runOptimizer(method) {
  if (state.optimizing) return;
  const labels = { cma: "CMA-ES", cem: "CEM", ppo: "PPO" };
  setOptimizing(true, labels[method]);

  try {
    const resp = await fetch(`/api/optimize/${method}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        params:    state.params,
        target_c:  state.targetC,
        seed:      state.seed,
        lock_mask: state.lockMask,
      }),
    });
    const data = await resp.json();

    if (data.error) {
      alert(`Optimization error: ${data.error}`);
      return;
    }

    state.params         = data.best_params;
    state.metrics.reward = data.reward;
    state.metrics.mse    = data.mse;
    state.metrics.maxDev = data.max_dev;

    refreshSliderUI();
    updateMetricsDisplay();
    onParamChange();          // rebuild path + chart

    const statusEl = document.getElementById("opt-status");
    statusEl.textContent = `${labels[method]} complete — reward: ${data.reward.toFixed(4)}`;
  } catch (err) {
    alert(`Optimization failed: ${err}`);
  } finally {
    finishProgress();
    setOptimizing(false, "");
  }
}

function resetAll() {
  state.params   = [...DEFAULT_PARAMS];
  state.lockMask = [false, false, false, false, false, false];
  state.targetC  = 0.0;
  state.seed     = null;

  refreshSliderUI();

  PARAM_KEYS.forEach(key => {
    document.getElementById(`lock-${key}`).checked = false;
  });
  document.getElementById("target-c").value  = "0.0";
  document.getElementById("seed-input").value = "random";
  document.getElementById("opt-status").textContent = "";
  document.getElementById("opt-status").className   = "opt-status";

  onParamChange();
}

// ── Pause / resume ────────────────────────────────────────────────────────────

document.getElementById("btn-pause").addEventListener("click", () => {
  state.paused = !state.paused;
  document.getElementById("icon-pause").style.display = state.paused ? "none"  : "";
  document.getElementById("icon-play").style.display  = state.paused ? ""      : "none";
});

// ── Canvas resize ─────────────────────────────────────────────────────────────

function resizeCanvas() {
  const parent = canvas.parentElement;
  const dpr    = window.devicePixelRatio || 1;
  const w      = parent.clientWidth  || 600;
  const h      = parent.clientHeight || 400;

  // Store CSS dimensions — all layout code uses these, NOT canvas.width/height.
  cssW = w;
  cssH = h;

  // Set device-pixel backing store
  canvas.width  = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width  = `${w}px`;
  canvas.style.height = `${h}px`;

  // Reset transform then apply DPR scale once.
  // Changing canvas.width/height resets the context, so we must re-apply.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener("resize", () => {
  resizeCanvas();
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  resizeCanvas();
  initSliders();
  initLocks();
  initTargetSeed();
  initChart();

  // Wire optimizer buttons
  document.getElementById("btn-cma").addEventListener("click",   () => runOptimizer("cma"));
  document.getElementById("btn-cem").addEventListener("click",   () => runOptimizer("cem"));
  document.getElementById("btn-ppo").addEventListener("click",   () => runOptimizer("ppo"));
  document.getElementById("btn-reset").addEventListener("click",  resetAll);

  // Initial computation
  onParamChange();

  // Start animation
  tick();
});
