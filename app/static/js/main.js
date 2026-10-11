import { rerenderAll, h, fmt } from './charts.js';
import { initAnalyze, leaveAnalyze } from './analyze.js';
import { initResults } from './results.js';
import { initHow } from './how.js';

const VIEWS = ['analyze', 'results', 'how', 'about'];
let howReady = false;

function store(key, value) {
  try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch { return null; }
  return null;
}

function applyTheme(theme) {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  rerenderAll();
}

function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function route() {
  const name = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'analyze';
  for (const v of VIEWS) {
    document.getElementById(`view-${v}`).classList.toggle('active', v === name);
    document.querySelector(`.nav a[data-view="${v}"]`).toggleAttribute('aria-current', v === name);
    if (v === name) document.querySelector(`.nav a[data-view="${v}"]`).setAttribute('aria-current', 'page');
  }
  if (name !== 'analyze') leaveAnalyze();
  if (name === 'results') initResults();
  if (name === 'how' && !howReady) { howReady = true; initHow(); }
  requestAnimationFrame(rerenderAll); // charts in a view that was hidden are drawn now
}

async function boot() {
  const saved = store('pulsecheck-theme');
  if (saved) applyTheme(saved);
  document.getElementById('theme-toggle').addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    store('pulsecheck-theme', next);
    applyTheme(next);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => rerenderAll());

  let info = null;
  try { info = await (await fetch('/api/info')).json(); } catch { /* the page still works without it */ }
  document.getElementById('version').textContent = info ? `v${info.version}` : '';
  initAnalyze(info);
  const m = info?.model;
  const about = document.getElementById('about-model');
  if (m) {
    about.append(h('h3', { text: 'Loaded model' }), h('p', {},
      `${m.model_label}, trained on ${m.train_source} (${m.n_train_videos} videos, ${String(m.rppg_method || 'pos').toUpperCase()} pulses) on ${m.trained_at}. `,
      `Test AUC ${fmt(m.test_video_auc, 3)}; decision threshold ${fmt(m.threshold, 2)}.`,
      m.kind === 'synthetic' ? h('strong', { text: ' Synthetic demo model: not evidence of real deepfake detection.' }) : null));
  } else {
    about.append(h('p', { class: 'fineprint', text: 'No detector is loaded in this deployment yet.' }));
  }
  window.addEventListener('hashchange', route);
  route();
}

boot();
