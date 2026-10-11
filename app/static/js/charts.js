// Plotly wrappers that apply the site's chart rules in one place:
// 2px lines, >=8px markers with a 2px surface ring, bars capped at 24px with a
// rounded data end, hairline solid grids, text in ink tokens (never series
// colours), an HTML legend for 2+ series, and a table view for every chart.
// Colours are read from CSS custom properties, so a theme switch re-renders.

const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

export function tokens() {
  return {
    surface: css('--surface'), ink: css('--ink'), ink2: css('--ink-2'), muted: css('--muted'),
    grid: css('--grid'), axis: css('--axis'), border: css('--border'), deemph: css('--deemph'),
    s: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css(`--s${i}`)),
    seq: [100, 200, 300, 400, 500, 600, 700].map((i) => css(`--seq-${i}`)),
    good: css('--good'), critical: css('--critical'), warning: css('--warning'),
  };
}

export const SERIES = (i) => `var(--s${i + 1})`; // for HTML legends (follows the theme by itself)

const registry = new Map();
let uid = 0;
export function rerenderAll() {
  for (const [id, fn] of registry) fn() === false && registry.delete(id);
}
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(rerenderAll, 150); });

function isObj(v) { return v && typeof v === 'object' && !Array.isArray(v); }
export function merge(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b || {})) out[k] = isObj(v) && isObj(a?.[k]) ? merge(a[k], v) : v;
  return out;
}

export function axis(t, o = {}) {
  return merge({
    gridcolor: t.grid, gridwidth: 1, linecolor: t.axis, linewidth: 1, showline: false, zeroline: false,
    ticks: '', tickfont: { color: t.muted, size: 11.5 }, automargin: true,
    title: { font: { color: t.ink2, size: 12 }, standoff: 8 }, fixedrange: true,
  }, o);
}

export function baseLayout(t) {
  return {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface,
    font: { family: FONT, size: 12, color: t.ink2 },
    margin: { l: 8, r: 12, t: 8, b: 8 }, showlegend: false, hovermode: 'closest',
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { family: FONT, color: t.ink, size: 12.5 }, align: 'left' },
    xaxis: axis(t), yaxis: axis(t), bargap: 0.35,
  };
}

export const CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false };

// ---------- trace helpers ----------
export const line = (x, y, color, o = {}) => merge({ type: 'scatter', mode: 'lines', x, y, line: { color, width: 2, shape: 'linear' } }, o);
export const area = (x, y, color, wash, o = {}) =>
  merge({ type: 'scatter', mode: 'lines', x, y, line: { color, width: 2 }, fill: 'tozeroy', fillcolor: wash }, o);
export const dots = (x, y, color, t, o = {}) =>
  merge({ type: 'scatter', mode: 'markers', x, y, marker: { color, size: 9, line: { color: t.surface, width: 2 } } }, o);
export const bars = (x, y, color, o = {}) => merge({ type: 'bar', x, y, marker: { color, cornerradius: 4 } }, o);

/** Bar thickness in data units so bars render <= maxPx wide. */
export function barWidth(plotEl, n, maxPx = 24, frac = 0.7) {
  const px = Math.max((plotEl.clientWidth || 600) - 80, 100) / Math.max(n, 1);
  return Math.min(frac, maxPx / px);
}

export function hexAlpha(hex, a) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/** Sequential colourscale from the theme's blue ramp (light -> dark = low -> high). */
export const seqScale = (t) => t.seq.map((c, i) => [i / (t.seq.length - 1), c]);

const rgb = (hex) => { const n = parseInt(hex.replace('#', ''), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
const lum = ([r, g, b]) => {
  const c = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * c(r) + 0.7152 * c(g) + 0.0722 * c(b);
};
/** Ink for a label sitting on the sequential ramp at value v (0..1): whichever of black/white contrasts more. */
export function inkOn(t, v) {
  const s = t.seq, x = Math.min(Math.max(v, 0), 1) * (s.length - 1), i = Math.floor(x), f = x - i;
  const a = rgb(s[i]), b = rgb(s[Math.min(i + 1, s.length - 1)]);
  const L = lum(a.map((av, k) => av + (b[k] - av) * f));
  return (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) ? '#0b0b0b' : '#ffffff';
}

// ---------- DOM ----------
export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'style' && isObj(v)) Object.assign(e.style, v);
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return e;
}

function legendEl(items) {
  if (!items?.length || items.length < 2) return null;
  return h('div', { class: 'legend' }, items.map((it) =>
    h('span', {}, h('i', { class: `key-${it.kind || 'line'}`, style: it.kind === 'line' || !it.kind ? { color: it.color } : { background: it.color } }), it.label)));
}

function tableEl({ columns, rows }) {
  const tbl = h('table', { class: 'data' });
  tbl.append(h('thead', {}, h('tr', {}, columns.map((c) => h('th', { scope: 'col', text: c })))));
  tbl.append(h('tbody', {}, rows.map((r) => h('tr', {}, r.map((v) => h('td', { text: v == null ? '-' : v }))))));
  return tbl;
}

/**
 * A chart card: title, one-line takeaway, legend, plot, and a Table toggle.
 * draw(t, plotEl) returns { data, layout }; table() returns { columns, rows }.
 */
export function chartCard({ title, takeaway, height = 260, legend, table, draw, cls }) {
  const id = `c${++uid}`;
  const plot = h('div', { class: 'plot', style: { height: `${height}px` }, role: 'img', 'aria-label': title });
  const tbox = h('div', { class: 'tablebox', hidden: true });
  const btn = table ? h('button', { class: 'btn small toggle-table', type: 'button', 'aria-pressed': 'false', text: 'Table' }) : null;
  const card = h('div', { class: `chart-card ${cls || ''}` },
    h('header', {}, h('div', {}, h('h3', { text: title }), takeaway ? h('p', { class: 'takeaway', text: takeaway }) : null), btn),
    legendEl(legend), plot, tbox);
  const render = () => {
    if (!card.isConnected) return !card._mounted; // drop from the registry once removed from the page
    card._mounted = true;
    if (!window.Plotly || plot.hidden || plot.offsetParent === null) return true; // hidden: drawn when shown
    const t = tokens();
    const { data, layout } = draw(t, plot);
    window.Plotly.react(plot, data, merge(merge(baseLayout(t), { height }), layout || {}), CONFIG);
    return true;
  };
  if (btn) {
    btn.addEventListener('click', () => {
      const showTable = tbox.hidden;
      if (showTable && !tbox.firstChild) tbox.append(tableEl(table()));
      tbox.hidden = !showTable;
      plot.hidden = showTable;
      btn.textContent = showTable ? 'Chart' : 'Table';
      btn.setAttribute('aria-pressed', String(showTable));
      if (!showTable) render();
    });
  }
  registry.set(id, render);
  queueMicrotask(render);
  requestAnimationFrame(render);
  return card;
}

export function tile(label, value, unit, note, cls = '') {
  return h('div', { class: `tile ${cls}` }, h('div', { class: 'label', text: label }),
    h('div', { class: 'value' }, value ?? '-', unit ? h('small', { text: unit }) : null),
    note ? h('div', { class: 'note', text: note }) : null);
}

export const fmt = (v, d = 1) => (v == null || !isFinite(v) ? '-' : Number(v).toFixed(d));
export const pct = (v, d = 0) => (v == null || !isFinite(v) ? '-' : `${(100 * v).toFixed(d)}%`);
