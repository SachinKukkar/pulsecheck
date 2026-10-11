// Pulse lab: steps 2-4 of the pipeline on a synthetic "average green" trace,
// computed live so the reader can see why clip length, noise and flicker matter.
import { h, chartCard, line, area, dots, axis, hexAlpha, fmt, rerenderAll, SERIES } from './charts.js';

const FS = 30;
const BAND = [0.7, 3.0];

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand) { // Box-Muller
  const u = Math.max(rand(), 1e-12), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeSignal({ bpm, len, noise, flicker }) {
  const n = Math.round(len * FS);
  const rand = mulberry32(7);
  const f = bpm / 60;
  const amp = 0.3; // pulse depth in grey levels: tiny next to the 142 average
  const t = [], g = [];
  for (let i = 0; i < n; i++) {
    const ti = i / FS;
    const phi = 2 * Math.PI * f * ti;
    const beat = Math.sin(phi) + 0.35 * Math.sin(2 * phi - 0.6) + 0.1 * Math.sin(3 * phi - 1.2);
    const drift = 0.8 * Math.sin(2 * Math.PI * 0.05 * ti) + 0.3 * ti / len;
    const light = flicker * amp * Math.sin(2 * Math.PI * 1.8 * ti); // a 108 bpm lamp flicker
    t.push(ti);
    g.push(142.4 - amp * beat + drift + light + noise * amp * gauss(rand));
  }
  return { t, g };
}

// Biquad (RBJ) high-pass and low-pass, run forwards and backwards: a zero-phase band-pass.
function biquad(x, type, fc) {
  const w = (2 * Math.PI * fc) / FS, c = Math.cos(w), s = Math.sin(w) / (2 * Math.SQRT1_2);
  const [b0, b1, b2] = type === 'hp' ? [(1 + c) / 2, -(1 + c), (1 + c) / 2] : [(1 - c) / 2, 1 - c, (1 - c) / 2];
  const a0 = 1 + s, a1 = -2 * c, a2 = 1 - s;
  const y = new Array(x.length);
  let x1 = x[0], x2 = x[0], y1 = type === 'hp' ? 0 : x[0], y2 = y1;
  for (let i = 0; i < x.length; i++) {
    const v = (b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
  }
  return y;
}
function filtfilt(x, type, fc) { return biquad(biquad(x, type, fc).reverse(), type, fc).reverse(); }
function bandpass(x) {
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  let y = x.map((v) => v - mean);
  y = filtfilt(filtfilt(y, 'hp', BAND[0]), 'hp', BAND[0]);
  return filtfilt(filtfilt(y, 'lp', BAND[1]), 'lp', BAND[1]);
}

// Power at arbitrary frequencies by direct DFT (cheap at these sizes, no padding rules to explain).
function powerAt(x, freqs) {
  const n = x.length;
  const w = x.map((v, i) => v * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)))); // Hann
  return freqs.map((f) => {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) { const a = (2 * Math.PI * f * i) / FS; re += w[i] * Math.cos(a); im -= w[i] * Math.sin(a); }
    return re * re + im * im;
  });
}

let charts = null;
let current = null;

function compute(params) {
  const { t, g } = makeSignal(params);
  const filtered = bandpass(g);
  const n = g.length;
  const step = FS / n; // Hz between plain FFT bins = 1 / clip length
  const kMax = Math.floor(4 / step);
  const binF = Array.from({ length: kMax + 1 }, (_, k) => k * step);
  const binP = powerAt(filtered, binF);
  const fineF = Array.from({ length: 401 }, (_, i) => i * 0.01);
  const fineP = powerAt(filtered, fineF);
  const inBand = (f) => f >= BAND[0] && f <= BAND[1];
  let kBest = -1;
  binF.forEach((f, k) => { if (inBand(f) && (kBest < 0 || binP[k] > binP[kBest])) kBest = k; });
  let iBest = -1;
  fineF.forEach((f, i) => { if (inBand(f) && (iBest < 0 || fineP[i] > fineP[iBest])) iBest = i; });
  const norm = Math.max(...fineP, ...binP, 1e-12);
  return { t, g, filtered, n, step, binF, binP: binP.map((p) => p / norm), fineF, fineP: fineP.map((p) => p / norm), kBest, fineBest: fineF[iBest], params };
}

function readout(r) {
  const el = document.getElementById('lab-readout');
  const est = r.binF[r.kBest] * 60;
  const off = Math.abs(est - r.params.bpm);
  const parts = [h('div', {}, h('strong', { text: `${r.params.len} s × ${FS} frames a second = ${r.n} numbers. ` }),
    `Plain FFT steps are 1/${r.params.len} s = ${fmt(r.step, 3)} Hz apart, i.e. ${fmt(r.step * 60, 1)} bpm. `,
    `The strongest step in the band is k = ${r.kBest}: ${fmt(r.binF[r.kBest], 2)} Hz × 60 = `, h('strong', { text: `${fmt(est, 1)} bpm` }),
    ` (true ${r.params.bpm}). Interpolating between steps gives ${fmt(r.fineBest * 60, 1)} bpm.`)];
  if (off > r.step * 60) {
    parts.push(h('div', { style: { marginTop: '6px' } }, h('strong', { text: 'Off by more than one step. ' }),
      r.params.flicker > 0.5 ? 'The 108 bpm lamp flicker is stronger than the pulse, and the green channel alone cannot tell light from blood. POS fixes this by combining all three colours.' : 'Noise is drowning the pulse: a longer clip or steadier light helps.'));
  }
  el.replaceChildren(...parts);
}

function render(params) {
  current = compute(params);
  readout(current);
  if (!charts) {
    const root = document.getElementById('lab-charts');
    const signalCard = chartCard({
      title: 'What the computer is given',
      takeaway: 'Average green of a skin patch, frame by frame. The pulse is a ripple of a fraction of a grey level.',
      height: 230,
      table: () => ({ columns: ['frame', 'seconds', 'average green'], rows: current.g.map((v, i) => [i + 1, fmt(current.t[i], 3), fmt(v, 2)]) }),
      draw: (t) => ({ data: [line(current.t, current.g, t.s[0], { hovertemplate: 'frame %{customdata}<br>%{y:.2f}<extra></extra>', customdata: current.t.map((_, i) => i + 1) })],
        layout: { xaxis: axis(t, { title: { text: 'seconds' } }), yaxis: axis(t, { title: { text: 'average green' } }) } }),
    });
    const filtCard = chartCard({
      title: 'After the band-pass (0.7–3 Hz)',
      takeaway: 'Drift removed, only heartbeat-speed wiggles kept.',
      height: 230,
      table: () => ({ columns: ['seconds', 'filtered'], rows: current.filtered.map((v, i) => [fmt(current.t[i], 3), fmt(v, 3)]) }),
      draw: (t) => ({ data: [line(current.t, current.filtered, t.s[0], { hovertemplate: '%{x:.2f} s<br>%{y:.3f}<extra></extra>' })],
        layout: { xaxis: axis(t, { title: { text: 'seconds' } }), yaxis: axis(t, { title: { text: 'grey levels' } }) } }),
    });
    const specCard = chartCard({
      title: 'The Fourier transform',
      takeaway: 'Dots are the plain FFT steps; the line fills in between them. The shaded band is where a heartbeat can live.',
      height: 260,
      legend: [{ label: 'FFT steps (1/T apart)', color: SERIES(1), kind: 'dot' }, { label: 'In-between (interpolated)', color: SERIES(0) }],
      table: () => ({ columns: ['step k', 'Hz', 'bpm', 'relative power'], rows: current.binF.map((f, k) => [k, fmt(f, 3), fmt(f * 60, 1), fmt(current.binP[k], 3)]) }),
      draw: (t) => ({
        data: [area(current.fineF.map((f) => f * 60), current.fineP, t.s[0], hexAlpha(t.s[0], 0.1), { hovertemplate: '%{x:.1f} bpm<br>%{y:.2f}<extra></extra>' }),
          dots(current.binF.map((f) => f * 60), current.binP, t.s[1], t, { customdata: current.binF.map((_, k) => k), hovertemplate: 'step k=%{customdata}<br>%{x:.1f} bpm · power %{y:.2f}<extra></extra>' })],
        layout: {
          xaxis: axis(t, { title: { text: 'beats a minute' }, range: [0, 240], dtick: 30 }), yaxis: axis(t, { title: { text: 'relative power' }, range: [0, 1.15] }),
          shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: BAND[0] * 60, x1: BAND[1] * 60, y0: 0, y1: 1, fillcolor: t.grid, opacity: 0.45, line: { width: 0 }, layer: 'below' }],
          annotations: [{ x: current.binF[current.kBest] * 60, y: current.binP[current.kBest], text: `k=${current.kBest}`, showarrow: true, arrowhead: 0, arrowcolor: t.ink2, ax: 26, ay: -22, font: { color: t.ink, size: 12 } }],
        },
      }),
    });
    root.append(signalCard, filtCard);
    root.after(h('div', { style: { marginTop: '16px' } }, specCard));
    charts = true;
  } else {
    rerenderAll();
  }
}

export function initHow() {
  const ids = ['bpm', 'len', 'noise', 'flicker'];
  const read = () => Object.fromEntries(ids.map((id) => [id, Number(document.getElementById(`lab-${id}`).value)]));
  for (const id of ids) {
    const inp = document.getElementById(`lab-${id}`);
    const out = document.getElementById(`lab-${id}-out`);
    inp.addEventListener('input', () => {
      out.textContent = id === 'noise' || id === 'flicker' ? Number(inp.value).toFixed(1) : inp.value;
      render(read());
    });
  }
  render(read());
}
