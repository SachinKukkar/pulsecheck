import { h, chartCard, tile, line, area, dots, bars, axis, barWidth, hexAlpha, seqScale, inkOn, fmt, pct, SERIES } from './charts.js';

const ROI_LABEL = {
  forehead_left: 'Forehead (left)', forehead_right: 'Forehead (right)', cheek_left: 'Cheek (left)',
  cheek_right: 'Cheek (right)', nose: 'Nose bridge', chin: 'Chin',
};
const ROI_SHORT = { forehead_left: 'FhL', forehead_right: 'FhR', cheek_left: 'ChL', cheek_right: 'ChR', nose: 'Nose', chin: 'Chin' };
const METHOD_LABEL = { pos: 'POS', chrom: 'CHROM', green: 'Green' };
const RECORD_SECONDS = 15;

let state = { source: null, file: null, sample: null, info: null, stream: null, recorder: null };
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ inputs

function selectTab(name) {
  for (const t of ['upload', 'record', 'sample']) {
    $(`tab-${t}`).setAttribute('aria-selected', String(t === name));
    $(`panel-${t}`).hidden = t !== name;
  }
  state.source = name === 'sample' ? (state.sample ? 'sample' : null) : name === 'upload' ? (state.file ? 'upload' : null) : (state.file?.recorded ? 'record' : null);
  if (name !== 'record') stopCamera();
  updateButton();
}

function updateButton() {
  const ready = (state.source === 'upload' && state.file) || (state.source === 'record' && state.file) || (state.source === 'sample' && state.sample);
  $('analyze-btn').disabled = !ready;
}

function setFile(file, recorded = false) {
  state.file = file;
  if (recorded) state.file.recorded = true;
  state.source = recorded ? 'record' : 'upload';
  const chip = $('file-chip');
  if (!recorded) {
    chip.hidden = false;
    chip.textContent = `${file.name}  ·  ${(file.size / 1048576).toFixed(1)} MB`;
  }
  updateButton();
}

function initUpload() {
  const dz = $('dropzone');
  const input = $('file-input');
  input.addEventListener('change', () => input.files[0] && setFile(input.files[0]));
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) setFile(f);
  });
}

async function startCamera() {
  const status = $('cam-status');
  if (!navigator.mediaDevices?.getUserMedia) { status.textContent = 'This browser cannot use the camera.'; return; }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: 'user' }, audio: false,
    });
  } catch (e) {
    status.textContent = `Camera unavailable: ${e.message}`;
    return;
  }
  const v = $('cam-video');
  v.srcObject = state.stream;
  await v.play().catch(() => {});
  $('cam-record').disabled = false;
  $('cam-start').textContent = 'Turn off camera';
  status.textContent = 'Fill the oval and keep still.';
}

function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  const b = $('cam-start');
  if (b) b.textContent = 'Turn on camera';
  if ($('cam-record')) $('cam-record').disabled = true;
}

function pickMime() {
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4;codecs=avc1', 'video/mp4'];
  return types.find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || '';
}

function record() {
  if (!state.stream) return;
  const mime = pickMime();
  // A high bit rate matters: compression is what wipes out the faint pulse.
  const rec = new MediaRecorder(state.stream, { mimeType: mime || undefined, videoBitsPerSecond: 6_000_000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const count = $('cam-count');
  let left = RECORD_SECONDS;
  count.hidden = false;
  count.textContent = `Recording ${left} s`;
  const tick = setInterval(() => { left -= 1; count.textContent = `Recording ${Math.max(left, 0)} s`; }, 1000);
  rec.onstop = () => {
    clearInterval(tick);
    count.hidden = true;
    const type = rec.mimeType || mime || 'video/webm';
    const ext = type.includes('mp4') ? 'mp4' : 'webm';
    const blob = new Blob(chunks, { type });
    setFile(new File([blob], `webcam.${ext}`, { type }), true);
    $('cam-status').textContent = `Recorded ${RECORD_SECONDS} s (${(blob.size / 1048576).toFixed(1)} MB). Press Analyze.`;
    $('cam-record').disabled = false;
  };
  $('cam-record').disabled = true;
  $('cam-status').textContent = 'Keep still…';
  rec.start(500);
  setTimeout(() => rec.state !== 'inactive' && rec.stop(), RECORD_SECONDS * 1000);
}

function initRecord() {
  $('cam-start').addEventListener('click', () => (state.stream ? stopCamera() : startCamera()));
  $('cam-record').addEventListener('click', record);
}

function initSamples(samples) {
  const list = $('sample-list');
  list.replaceChildren();
  if (!samples?.length) {
    $('tab-sample').hidden = true; // nothing to offer: no tab at all, rather than an empty one
    return;
  }
  for (const s of samples) {
    const radio = h('input', { type: 'radio', name: 'sample', value: s.file, id: `s-${s.file}` });
    radio.addEventListener('change', () => { state.sample = s.file; state.source = 'sample'; updateButton(); });
    list.append(h('label', { class: 'sample', for: `s-${s.file}` }, radio,
      h('video', { src: `samples/${s.file}`, muted: true, loop: true, playsinline: true, autoplay: true, preload: 'metadata' }),
      h('div', {}, h('strong', { text: s.title }), h('p', { class: 'fineprint', style: { margin: '4px 0 0' }, text: s.description }))));
  }
}

// ------------------------------------------------------------------ run

function showProgress(stage, frac) {
  $('progress-card').hidden = false;
  $('progress-stage').textContent = stage;
  const bar = $('progress-bar');
  const known = frac != null && frac > 0 && frac < 1;
  bar.classList.toggle('indeterminate', !known);
  bar.firstElementChild.style.width = known ? `${Math.round(frac * 100)}%` : '';
  $('progress-pct').textContent = known ? `${Math.round(frac * 100)}%` : '';
}

function showError(msg) {
  $('progress-card').hidden = true;
  $('error-card').hidden = false;
  $('error-text').textContent = msg;
}

async function run() {
  $('error-card').hidden = true;
  $('analysis').hidden = true;
  $('analyze-btn').disabled = true;
  const method = $('method').value;
  showProgress(state.source === 'sample' ? 'Starting' : 'Uploading', null);
  try {
    const fd = new FormData();
    fd.append('method', method);
    let url = '/api/analyze';
    if (state.source === 'sample') { url = '/api/analyze-sample'; fd.append('name', state.sample); }
    else fd.append('file', state.file, state.file.name);
    const resp = await fetch(url, { method: 'POST', body: fd });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(body.detail || `Upload failed (${resp.status})`);
    const result = await poll(body.job_id);
    $('progress-card').hidden = true;
    renderAnalysis(result);
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    updateButton();
  }
}

async function poll(jobId) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 600));
    const resp = await fetch(`/api/jobs/${jobId}`);
    const job = await resp.json();
    if (!resp.ok) throw new Error(job.detail || 'Lost track of the job');
    if (job.status === 'done') return job.result;
    if (job.status === 'error') throw new Error(job.error);
    const stage = job.status === 'queued' && job.queue_position ? `Waiting: ${job.queue_position} ahead of you` : job.stage;
    showProgress(stage, job.progress);
  }
}

// ------------------------------------------------------------------ render

function icon(id, size = 18) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('width', size); s.setAttribute('height', size); s.setAttribute('aria-hidden', 'true');
  const u = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  u.setAttribute('href', `#${id}`);
  s.append(u);
  return s;
}

function verdictCard(d, info) {
  if (!d) {
    return h('div', { class: 'banner info' }, icon('i-info', 20), h('div', {},
      h('p', {}, h('strong', { text: 'No deepfake verdict. ' }), 'No detector has been trained for this deployment yet, so you get the heart rate and the patch comparison only.')));
  }
  if (d.prob_fake == null) {
    return h('div', { class: 'banner warn' }, icon('i-alert', 20), h('div', {}, h('p', { text: 'The clip is too short for a verdict: at least 4 seconds of face are needed.' })));
  }
  const cls = d.verdict === 'likely fake' ? 'fake' : d.verdict === 'likely real' ? 'real' : 'uncertain';
  const ic = cls === 'fake' ? 'i-x' : cls === 'real' ? 'i-check' : 'i-help';
  const label = { fake: 'Likely fake', real: 'Likely real', uncertain: 'Uncertain' }[cls];
  const synthetic = info?.model?.kind === 'synthetic';
  const meter = h('div', { class: 'meter', role: 'img', 'aria-label': `Probability fake ${pct(d.prob_fake)}; threshold ${pct(d.threshold)}` },
    h('div', { class: 'fill', style: { width: `${Math.round(d.prob_fake * 100)}%` } }),
    h('div', { class: 'thr', style: { left: `${d.threshold * 100}%` } }, h('span', { text: `threshold ${pct(d.threshold)}` })));
  return h('div', { class: 'card' },
    h('div', { class: 'verdict' },
      h('div', { class: `badge ${cls}` }, icon(ic, 22), label),
      h('div', {},
        h('div', { class: 'row' }, h('strong', { text: `${pct(d.prob_fake)} probability fake` }),
          h('span', { class: 'fineprint', style: { margin: 0 }, text: `mean of ${d.windows.length} window vote${d.windows.length === 1 ? '' : 's'}` })),
        meter, h('div', { class: 'meter-scale' }, h('span', { text: 'real' }), h('span', { text: 'fake' })))),
    h('p', { class: 'fineprint' }, synthetic
      ? 'Demo model trained on synthetic clips only. It shows the mechanics, not real deepfake detection.'
      : `Model: ${d.model?.model_label || 'classifier'} trained on ${d.model?.train_source || 'FaceForensics++'} (test AUC ${fmt(d.model?.test_video_auc, 2)}). A screening signal, not proof.`));
}

function warningsList(ws) {
  if (!ws?.length) return null;
  const ic = { good: 'i-check', warn: 'i-alert', bad: 'i-x' };
  return h('ul', { class: 'warnings' }, ws.map((w) => h('li', { class: w.level }, icon(ic[w.level] || 'i-info', 16), h('span', { text: w.text }))));
}

function stepsChart(r) {
  const s = r.signals;
  const labels = ['1 · Average green per frame (what the computer is given)', '2 · Detrended and band-passed, 0.7–3 Hz', `3 · ${METHOD_LABEL[r.method]} pulse, all three colours combined`];
  return chartCard({
    title: 'From average green to a clean pulse',
    takeaway: 'The raw trace is mostly drift and noise; filtering and POS pull out the beat.',
    height: 380,
    table: () => ({ columns: ['seconds', 'average green', 'filtered', 'pulse'],
      rows: s.t.map((t, i) => [fmt(t, 2), fmt(s.green_raw[i], 3), fmt(s.green_filtered[i], 4), fmt(s.pulse[i], 3)]) }),
    draw: (t) => ({
      data: [
        line(s.t, s.green_raw, t.s[0], { yaxis: 'y', hovertemplate: '%{y:.2f}<extra>average green</extra>' }),
        line(s.t, s.green_filtered, t.s[0], { yaxis: 'y2', hovertemplate: '%{y:.3f}<extra>filtered</extra>' }),
        line(s.t, s.pulse, t.s[0], { yaxis: 'y3', hovertemplate: '%{y:.2f}<extra>pulse</extra>' }),
      ],
      layout: {
        hovermode: 'x unified',
        xaxis: axis(t, { anchor: 'y3', title: { text: 'seconds' }, showspikes: true, spikemode: 'across', spikecolor: t.axis, spikethickness: 1, spikedash: 'solid' }),
        yaxis: axis(t, { domain: [0.69, 0.89] }), yaxis2: axis(t, { domain: [0.37, 0.57] }), yaxis3: axis(t, { domain: [0.04, 0.24] }),
        annotations: labels.map((text, i) => ({ text, xref: 'paper', yref: 'paper', x: 0, y: [0.92, 0.6, 0.27][i], xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { color: t.ink2, size: 12 } })),
        margin: { t: 4, l: 6 },
      },
    }),
  });
}

function spectrumChart(r) {
  const sp = r.spectrum;
  const bin = r.hr.bin_spacing_bpm;
  return chartCard({
    title: 'The frequencies inside the pulse',
    takeaway: `Strongest peak in the shaded heart-rate band: ${fmt(r.hr.bpm)} bpm. Plain FFT steps are ${fmt(bin, 1)} bpm apart for this clip length; the curve is interpolated between them.`,
    height: 280,
    legend: [{ label: 'Spectrum (zero-padded)', color: SERIES(0) }, { label: `FFT steps (every ${fmt(bin, 1)} bpm)`, color: SERIES(1), kind: 'dot' }],
    table: () => ({ columns: ['bpm', 'relative power'], rows: sp.raw_bpm.map((b, i) => [fmt(b, 1), fmt(sp.raw_power[i], 3)]) }),
    draw: (t) => ({
      data: [
        area(sp.bpm, sp.power, t.s[0], hexAlpha(t.s[0], 0.1), { hovertemplate: '%{x:.1f} bpm<br>power %{y:.2f}<extra></extra>' }),
        dots(sp.raw_bpm, sp.raw_power, t.s[1], t, { marker: { size: 8 }, hovertemplate: 'FFT step %{x:.1f} bpm<br>power %{y:.2f}<extra></extra>' }),
      ],
      layout: {
        xaxis: axis(t, { title: { text: 'beats a minute' }, range: [0, 240], dtick: 30 }),
        yaxis: axis(t, { title: { text: 'relative power' }, range: [0, 1.12] }),
        shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: sp.band_bpm[0], x1: sp.band_bpm[1], y0: 0, y1: 1, fillcolor: t.grid, opacity: 0.45, line: { width: 0 }, layer: 'below' }],
        annotations: [{ x: r.hr.bpm, y: 1.0, text: `${fmt(r.hr.bpm)} bpm`, showarrow: true, arrowhead: 0, arrowcolor: t.ink2, ax: 34, ay: -18, font: { color: t.ink, size: 12.5 } }],
      },
    }),
  });
}

function timelineChart(r) {
  const tl = r.hr.timeline;
  if (!tl.t.length) return null;
  return chartCard({
    title: 'Heart rate over the clip',
    takeaway: 'Estimated on sliding 10 s windows, one per second.',
    height: 230,
    table: () => ({ columns: ['seconds', 'bpm', 'SNR (dB)'], rows: tl.t.map((x, i) => [fmt(x, 1), fmt(tl.bpm[i]), fmt(tl.snr_db[i], 1)]) }),
    draw: (t) => {
      const lo = Math.min(...tl.bpm.filter(Number.isFinite)), hi = Math.max(...tl.bpm.filter(Number.isFinite));
      return {
        data: [line(tl.t, tl.bpm, t.s[0], { mode: 'lines+markers', marker: { size: 8, color: t.s[0], line: { color: t.surface, width: 2 } },
          customdata: tl.snr_db, hovertemplate: '%{x:.1f} s · <b>%{y:.1f} bpm</b><br>SNR %{customdata:.1f} dB<extra></extra>' })],
        layout: { xaxis: axis(t, { title: { text: 'window centre, seconds' } }), yaxis: axis(t, { title: { text: 'bpm' }, range: [Math.floor(lo - 8), Math.ceil(hi + 8)] }) },
      };
    },
  });
}

function roiChart(r) {
  const names = r.rois.names;
  const n = names.length;
  const gap = 0.035;
  const hgt = (1 - gap * (n - 1)) / n;
  return chartCard({
    title: 'Step 5 · Does every patch share the same beat?',
    takeaway: 'Each patch’s own pulse, aligned in time. On a real face they rise and fall together.',
    height: 64 * n + 50,
    legend: names.map((nm, i) => ({ label: `${ROI_LABEL[nm]} · ${fmt(r.hr.per_roi[nm])} bpm`, color: SERIES(i) })),
    table: () => ({ columns: ['seconds', ...names.map((nm) => ROI_SHORT[nm])], rows: r.signals.t.map((x, i) => [fmt(x, 2), ...names.map((nm) => fmt(r.rois.pulses[nm][i], 2))]) }),
    draw: (t) => {
      const layout = { hovermode: 'x unified', margin: { t: 4, l: 6 }, annotations: [] };
      const data = names.map((nm, i) => {
        const k = i === 0 ? '' : String(i + 1);
        const top = 1 - i * (hgt + gap);
        layout[`yaxis${k}`] = axis(t, { domain: [top - hgt, top], showticklabels: false, showgrid: false });
        layout.annotations.push({ text: ROI_SHORT[nm], xref: 'paper', yref: 'paper', x: 0, y: top - hgt / 2, xanchor: 'right', xshift: -6, showarrow: false, font: { color: t.ink2, size: 11.5 } });
        return line(r.signals.t, r.rois.pulses[nm], t.s[i], { yaxis: `y${k}`, line: { width: 1.6 }, hovertemplate: `%{y:.2f}<extra>${ROI_SHORT[nm]}</extra>` });
      });
      layout.xaxis = axis(t, { anchor: `y${n}`, title: { text: 'seconds' }, showspikes: true, spikemode: 'across', spikecolor: t.axis, spikethickness: 1, spikedash: 'solid' });
      layout.margin.l = 44;
      return { data, layout };
    },
  });
}

function coherenceChart(r) {
  const names = r.rois.names.map((n) => ROI_SHORT[n]);
  const z = r.rois.coherence;
  return chartCard({
    title: 'How strongly the patches agree',
    takeaway: 'Coherence in the heart-rate band: 1 means two patches carry the same rhythm.',
    height: 300,
    table: () => ({ columns: ['', ...names], rows: z.map((row, i) => [names[i], ...row.map((v) => fmt(v, 2))]) }),
    draw: (t) => ({
      data: [{ type: 'heatmap', z, x: names, y: names, zmin: 0, zmax: 1, colorscale: seqScale(t), xgap: 2, ygap: 2,
        colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: t.muted, size: 11 }, len: 0.9 },
        hovertemplate: '%{y} vs %{x}<br><b>%{z:.2f}</b><extra></extra>' }],
      layout: {
        xaxis: axis(t, { showgrid: false, side: 'bottom' }), yaxis: axis(t, { showgrid: false, autorange: 'reversed' }),
        annotations: z.flatMap((row, i) => row.map((v, j) => ({ x: names[j], y: names[i], text: fmt(v, 2), showarrow: false,
          font: { size: 11.5, color: inkOn(t, v) } }))),
      },
    }),
  });
}

function windowsChart(d) {
  if (!d?.windows?.length) return null;
  const x = d.windows.map((w) => `${fmt(w.t0, 0)}–${fmt(w.t1, 0)} s`);
  const y = d.windows.map((w) => w.prob);
  return chartCard({
    title: 'Each window votes',
    takeaway: 'Probability of fake for each 8 s window; the verdict is their mean.',
    height: 230,
    table: () => ({ columns: ['window', 'P(fake)'], rows: x.map((xx, i) => [xx, fmt(y[i], 3)]) }),
    draw: (t, el) => ({
      data: [bars(x, y, t.s[0], { width: barWidth(el, x.length), hovertemplate: '%{x}<br><b>%{y:.2f}</b><extra></extra>' })],
      layout: {
        yaxis: axis(t, { range: [0, 1], title: { text: 'P(fake)' }, tickformat: '.0%' }), xaxis: axis(t, { showgrid: false }),
        shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: d.threshold, y1: d.threshold, line: { color: t.ink, width: 1.5 } }],
        annotations: [{ xref: 'paper', x: 1, y: d.threshold, text: 'threshold', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: t.ink2, size: 11.5 } }],
      },
    }),
  });
}

function profileTable(d) {
  if (!d?.profile?.length) return null;
  const rows = d.profile.map((p) => h('tr', {}, h('td', { text: p.label }), h('td', { text: fmt(p.value, 3) }),
    h('td', { text: fmt(p.real_q?.[1], 3) }), h('td', { text: fmt(p.fake_q?.[1], 3) })));
  return h('div', { class: 'chart-card' }, h('header', {}, h('div', {}, h('h3', { text: 'What the model looked at' }),
    h('p', { class: 'takeaway', text: 'The detector’s most important features: this clip’s value against the typical (median) real and fake training windows.' }))),
  h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['feature', 'this clip', 'typical real', 'typical fake'].map((c) => h('th', { text: c })))), h('tbody', {}, rows)));
}

export function renderAnalysis(r) {
  const root = $('analysis');
  root.replaceChildren();
  const q = r.quality;
  const pm = r.hr.per_method;
  root.append(
    h('div', { class: 'section-head', style: { marginTop: '28px' } }, h('h2', { text: 'Result' }),
      h('span', { class: 'sub', text: `${fmt(q.duration_s, 1)} s analysed · ${fmt(q.effective_fps, 0)} fps · ${r.video.original_size?.join('×') || ''}` })),
    verdictCard(r.detector, state.info),
    h('div', { class: 'kpis', style: { marginTop: '16px' } },
      tile('Heart rate', fmt(r.hr.bpm), 'bpm', `${METHOD_LABEL[r.method]} method`, 'hero-tile'),
      tile('Pulse signal', fmt(r.hr.snr_db, 1), 'dB SNR', 'above 0 dB: more power on the beat than off it'),
      tile('Face found', pct(q.face_fraction), '', `${fmt(q.face_px, 0)} px between eye corners`),
      tile('Frequency step', fmt(r.hr.bin_spacing_bpm, 1), 'bpm', 'plain FFT resolution for this clip length')),
  );
  const warn = warningsList(r.warnings);
  if (warn) root.append(h('div', { class: 'card', style: { marginTop: '16px' } }, warn));

  const roiItems = r.rois.names.map((nm, i) => h('li', {}, h('i', { class: 'key-rect', style: { background: SERIES(i) } }),
    h('span', { text: ROI_LABEL[nm] }), h('span', { class: 'v', text: `${fmt(r.hr.per_roi[nm])} bpm` })));
  root.append(h('div', { class: 'grid two', style: { marginTop: '16px' } },
    h('div', { class: 'chart-card' }, h('header', {}, h('div', {}, h('h3', { text: 'Steps 1–2 · The six skin patches' }),
      h('p', { class: 'takeaway', text: 'Landmarks place the patches; masked-out hair and brow pixels are tinted magenta.' }))),
    r.preview ? h('img', { class: 'overlay-img', src: r.preview, alt: 'Video frame with the six skin patches outlined' }) : h('p', { class: 'fineprint', text: 'No preview frame.' })),
    h('div', { class: 'chart-card' }, h('header', {}, h('div', {}, h('h3', { text: 'Heart rate by patch and by method' }),
      h('p', { class: 'takeaway', text: 'Patches that disagree with the rest are a warning sign.' }))),
    h('ul', { class: 'roi-list' }, roiItems),
    h('div', { class: 'kpis', style: { marginTop: '14px' } }, Object.entries(pm).map(([m, v]) => tile(METHOD_LABEL[m], fmt(v.bpm), 'bpm', `SNR ${fmt(v.snr_db, 1)} dB`))),
    h('p', { class: 'fineprint', text: `Estimated skin tone angle (ITA): ${fmt(r.skin.ita, 0)}° (${r.skin.category}). Lighting shifts this; it is shown only to explain signal strength.` }))));

  root.append(h('div', { class: 'stack', style: { marginTop: '16px' } }, stepsChart(r)));
  root.append(h('div', { class: 'grid two', style: { marginTop: '16px' } }, spectrumChart(r), timelineChart(r)));
  root.append(h('div', { class: 'grid two', style: { marginTop: '16px' } }, roiChart(r), coherenceChart(r)));
  const wc = windowsChart(r.detector);
  const prof = profileTable(r.detector);
  if (wc || prof) root.append(h('div', { class: 'grid two', style: { marginTop: '16px' } }, wc, prof));

  const blob = new Blob([JSON.stringify(r)], { type: 'application/json' });
  root.append(h('div', { class: 'row', style: { marginTop: '18px' } },
    h('a', { class: 'btn', href: URL.createObjectURL(blob), download: 'pulsecheck-result.json', text: 'Download result (JSON)' }),
    h('button', { class: 'btn', type: 'button', text: 'Analyze another video', onclick: () => { root.hidden = true; $('input-card').scrollIntoView({ behavior: 'smooth' }); } })));
  root.hidden = false;
  root.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

export function initAnalyze(info) {
  state.info = info;
  for (const t of ['upload', 'record', 'sample']) $(`tab-${t}`).addEventListener('click', () => selectTab(t));
  initUpload();
  initRecord();
  initSamples(info?.samples);
  $('analyze-btn').addEventListener('click', run);
  if (info?.limits) {
    $('limits-note').textContent = `Videos are processed in memory and deleted when the analysis finishes; nothing is stored. Up to ${info.limits.max_upload_mb} MB; the first ${info.limits.max_seconds} s are analysed.`;
  }
  const banner = $('model-banner');
  if (info?.model?.kind === 'synthetic') {
    banner.append(h('div', { class: 'banner warn', style: { marginBottom: '16px' } }, icon('i-alert', 20), h('div', {},
      h('p', {}, h('strong', { text: 'Demo detector. ' }), 'The real-or-fake model loaded here was trained on synthetic test clips, not on real deepfakes. Heart rate works for real; treat the verdict as a demonstration until a FaceForensics++ model is trained.'))));
  }
}

export function leaveAnalyze() { stopCamera(); }
