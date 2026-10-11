import { h, chartCard, tile, line, area, dots, bars, axis, barWidth, hexAlpha, seqScale, inkOn, fmt, pct, SERIES } from './charts.js';

const METHOD_LABEL = { pos: 'POS', chrom: 'CHROM', green: 'Green' };
const MANIP_ORDER = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'];

function grid(...cards) { return h('div', { class: 'grid two', style: { marginTop: '16px' } }, cards.filter(Boolean)); }
function sectionHead(title, sub) { return h('div', { class: 'section-head' }, h('h2', { text: title }), sub ? h('span', { class: 'sub', text: sub }) : null); }

function syntheticBanner() {
  return h('div', { class: 'banner warn', style: { marginBottom: '16px' } },
    h('div', {}, h('p', {}, h('strong', { text: 'Smoke-test numbers. ' }),
      'This section comes from synthetic clips with a painted-on pulse, generated to prove the pipeline runs end to end. It says nothing about real people or real deepfakes. It will be replaced when the real-dataset commands below are run.')));
}

function emptyCard(title, lines) {
  return h('div', { class: 'card empty' }, h('h3', { text: title }), h('p', { text: 'Not run yet. Generate it with:' }), h('pre', { text: lines.join('\n') }));
}

// ------------------------------------------------------------------ heart rate

function hrSection(rep, comp) {
  const out = h('section', { class: 'section', 'aria-label': 'Heart-rate validation' });
  if (!rep) {
    out.append(sectionHead('Heart rate'), emptyCard('Heart-rate error against a pulse oximeter', [
      'python -m pulsecheck extract ubfc --root C:/datasets/UBFC-rPPG --max-seconds 0',
      'python -m pulsecheck eval-hr']));
    return out;
  }
  const hl = rep.headline;
  out.append(sectionHead('Heart rate', `${rep.source} · ${rep.n_videos} videos · ${hl.n} windows of ${hl.window_s} s · ${METHOD_LABEL[hl.method]}`));
  if (rep.source === 'synthetic') out.append(syntheticBanner());
  out.append(h('div', { class: 'kpis' },
    tile('Mean absolute error', fmt(hl.mae, 2), 'bpm', `${hl.window_s} s windows`),
    tile('RMSE', fmt(hl.rmse, 2), 'bpm', 'punishes big misses'),
    tile('Pearson r', fmt(hl.pearson_r, 2), '', 'estimate vs oximeter'),
    tile('Within 5 bpm', pct(hl.within_5bpm), '', `${pct(hl.within_10bpm)} within 10 bpm`),
    tile('Bias', fmt(hl.bias, 2), 'bpm', `95% limits ${fmt(hl.loa_low, 1)} to ${fmt(hl.loa_high, 1)}`)));

  const sc = rep.scatter;
  const lo = Math.floor(Math.min(...sc.gt, ...sc.est) - 5), hi = Math.ceil(Math.max(...sc.gt, ...sc.est) + 5);
  const scatter = chartCard({
    title: 'Video estimate vs pulse oximeter',
    takeaway: 'Each dot is one window; on the diagonal the camera and the finger agree.',
    height: 320,
    table: () => ({ columns: ['subject', 'oximeter bpm', 'video bpm'], rows: sc.gt.map((g, i) => [sc.subject[i], fmt(g), fmt(sc.est[i])]) }),
    draw: (t) => ({
      data: [line([lo, hi], [lo, hi], t.axis, { line: { width: 1 }, hoverinfo: 'skip' }),
        dots(sc.gt, sc.est, t.s[0], t, { text: sc.subject, hovertemplate: '%{text}<br>oximeter %{x:.1f} · video <b>%{y:.1f}</b> bpm<extra></extra>' })],
      layout: { xaxis: axis(t, { title: { text: 'oximeter, bpm' }, range: [lo, hi] }), yaxis: axis(t, { title: { text: 'video, bpm' }, range: [lo, hi], scaleanchor: 'x' }) },
    }),
  });
  const mean = sc.gt.map((g, i) => (g + sc.est[i]) / 2);
  const diff = sc.gt.map((g, i) => sc.est[i] - g);
  const ba = chartCard({
    title: 'Bland–Altman agreement',
    takeaway: `Average bias ${fmt(hl.bias, 2)} bpm; 95% of differences fall between ${fmt(hl.loa_low, 1)} and ${fmt(hl.loa_high, 1)} bpm.`,
    height: 320,
    table: () => ({ columns: ['mean bpm', 'video − oximeter'], rows: mean.map((m, i) => [fmt(m), fmt(diff[i], 2)]) }),
    draw: (t) => {
      const ref = (y, label) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: y, y1: y, line: { color: label === 'bias' ? t.ink2 : t.axis, width: label === 'bias' ? 1.5 : 1 } });
      return {
        data: [dots(mean, diff, t.s[0], t, { hovertemplate: 'mean %{x:.1f} bpm<br>difference <b>%{y:.1f}</b><extra></extra>' })],
        layout: {
          xaxis: axis(t, { title: { text: 'mean of video and oximeter, bpm' } }), yaxis: axis(t, { title: { text: 'video − oximeter, bpm' } }),
          shapes: [ref(hl.bias, 'bias'), ref(hl.loa_low), ref(hl.loa_high)],
          annotations: [['bias', hl.bias], ['+1.96 SD', hl.loa_high], ['−1.96 SD', hl.loa_low]].map(([txt, y]) => ({ xref: 'paper', x: 1, y, text: txt, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: t.ink2, size: 11 } })),
        },
      };
    },
  });
  out.append(grid(scatter, ba));

  const methods = rep.methods;
  const wins = rep.windows_s;
  const maeAt = (m, w) => rep.summary.find((s) => s.method === m && s.window_s === w)?.mae ?? null;
  const byWin = chartCard({
    title: 'Error vs clip length, by method',
    takeaway: 'Longer windows give finer frequency steps; POS and CHROM cancel lighting changes that fool plain green.',
    height: 290,
    legend: methods.map((m, i) => ({ label: METHOD_LABEL[m], color: SERIES(i) })),
    table: () => ({ columns: ['window (s)', ...methods.map((m) => `${METHOD_LABEL[m]} MAE`)], rows: wins.map((w) => [w, ...methods.map((m) => fmt(maeAt(m, w), 2))]) }),
    draw: (t) => ({
      data: methods.map((m, i) => line(wins, wins.map((w) => maeAt(m, w)), t.s[i], { mode: 'lines+markers', name: METHOD_LABEL[m],
        marker: { size: 8, color: t.s[i], line: { color: t.surface, width: 2 } }, hovertemplate: `${METHOD_LABEL[m]} · %{x} s<br><b>%{y:.2f} bpm</b><extra></extra>` })),
      layout: {
        hovermode: 'x unified', xaxis: axis(t, { title: { text: 'window length, seconds' }, tickvals: wins }), yaxis: axis(t, { title: { text: 'mean absolute error, bpm' }, rangemode: 'tozero' }),
        annotations: methods.map((m, i) => ({ x: wins[wins.length - 1], y: maeAt(m, wins[wins.length - 1]), text: METHOD_LABEL[m], showarrow: false, xanchor: 'left', xshift: 8, font: { color: t.ink2, size: 11.5 } })),
        margin: { r: 54 },
      },
    }),
  });
  const ex = rep.example;
  const wave = ex ? chartCard({
    title: 'The pulse the camera saw',
    takeaway: `Ten seconds of ${ex.subject}: video pulse (POS) against the finger oximeter, both standardised.`,
    height: 290,
    legend: [{ label: 'Finger oximeter', color: SERIES(1) }, { label: 'Video (POS)', color: SERIES(0) }],
    table: () => ({ columns: ['seconds', 'oximeter', 'video'], rows: ex.t.map((x, i) => [fmt(x, 2), fmt(ex.oximeter[i], 2), fmt(ex.video_pos[i], 2)]) }),
    draw: (t) => ({
      data: [line(ex.t, ex.oximeter, t.s[1], { hovertemplate: '%{y:.2f}<extra>oximeter</extra>' }), line(ex.t, ex.video_pos, t.s[0], { hovertemplate: '%{y:.2f}<extra>video</extra>' })],
      layout: { hovermode: 'x unified', xaxis: axis(t, { title: { text: 'seconds' } }), yaxis: axis(t, { title: { text: 'standardised' } }) },
    }),
  }) : null;
  out.append(grid(byWin, wave));

  const ps = rep.per_subject;
  const perSubj = chartCard({
    title: 'Error per person',
    takeaway: 'Sorted from best to worst. A few hard videos usually drive most of the error.',
    height: 280,
    table: () => ({ columns: ['subject', 'MAE (bpm)', 'windows', 'ITA°', 'skin category'], rows: ps.map((p) => [p.subject, fmt(p.mae, 2), p.n, fmt(p.ita, 0), p.skin_category]) }),
    draw: (t, el) => ({
      data: [bars(ps.map((p) => p.subject), ps.map((p) => p.mae), t.s[0], { width: barWidth(el, ps.length, 24, 0.75),
        customdata: ps.map((p) => [p.ita, p.skin_category]), hovertemplate: '%{x}<br><b>%{y:.2f} bpm</b><br>ITA %{customdata[0]:.0f}° (%{customdata[1]})<extra></extra>' })],
      layout: { xaxis: axis(t, { showticklabels: ps.length <= 16, showgrid: false, title: { text: ps.length > 16 ? 'subjects (hover for names)' : '' } }), yaxis: axis(t, { title: { text: 'MAE, bpm' } }) },
    }),
  });
  const skin = rep.by_skin.filter((b) => b.n_subjects > 0);
  const skinCard = chartCard({
    title: 'Error by estimated skin tone',
    takeaway: skin.length <= 2
      ? 'This dataset covers too narrow a range of skin tones to say anything about fairness; that gap is the finding.'
      : 'Mean error per ITA category, with the number of people in each. ITA from video is approximate.',
    height: 280,
    table: () => ({ columns: ['ITA category', 'people', 'MAE (bpm)'], rows: rep.by_skin.map((b) => [b.category, b.n_subjects, fmt(b.mae, 2)]) }),
    draw: (t, el) => ({
      data: [bars(rep.by_skin.map((b) => b.category), rep.by_skin.map((b) => b.mae), t.s[0], { width: barWidth(el, 6),
        text: rep.by_skin.map((b) => (b.n_subjects ? `n=${b.n_subjects}` : 'none')), textposition: 'outside', textfont: { color: t.ink2, size: 11.5 }, cliponaxis: false,
        hovertemplate: '%{x}<br><b>%{y:.2f} bpm</b> · %{text}<extra></extra>' })],
      layout: { xaxis: axis(t, { showgrid: false, title: { text: 'lighter ← ITA category → darker' }, autorange: 'reversed' }), yaxis: axis(t, { title: { text: 'MAE, bpm' }, rangemode: 'tozero' }), margin: { t: 18 } },
    }),
  });
  out.append(grid(perSubj, skinCard));

  if (comp?.rows?.length) {
    const rows = comp.rows.filter((r) => r.crf != null);
    const base = comp.rows.find((r) => r.crf == null);
    out.append(h('div', { style: { marginTop: '16px' } }, chartCard({
      title: 'Compression wipes out the pulse',
      takeaway: `The same ${comp.n_videos} videos re-encoded with H.264. Higher CRF = smaller file. Uncompressed baseline: ${fmt(base?.mae_pos_10s, 2)} bpm (POS, 10 s).`,
      height: 300,
      legend: [{ label: 'POS', color: SERIES(0) }, { label: 'Green', color: SERIES(2) }],
      table: () => ({ columns: ['level', 'kbps', 'POS MAE 10 s', 'POS MAE 30 s', 'Green MAE 10 s'], rows: comp.rows.map((r) => [r.label, fmt(r.kbps, 0), fmt(r.mae_pos_10s, 2), fmt(r.mae_pos_30s, 2), fmt(r.mae_green_10s, 2)]) }),
      draw: (t) => {
        const x = rows.map((r) => r.crf);
        const mk = (key, c, name) => line(x, rows.map((r) => r[key]), c, { mode: 'lines+markers', marker: { size: 8, color: c, line: { color: t.surface, width: 2 } },
          customdata: rows.map((r) => r.kbps), hovertemplate: `${name} · CRF %{x}<br><b>%{y:.2f} bpm</b><br>~%{customdata:.0f} kbps<extra></extra>` });
        const shapes = base ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: base.mae_pos_10s, y1: base.mae_pos_10s, line: { color: t.axis, width: 1 } }] : [];
        return {
          data: [mk('mae_pos_10s', t.s[0], 'POS'), mk('mae_green_10s', t.s[2], 'Green')],
          layout: { hovermode: 'x unified', shapes, xaxis: axis(t, { title: { text: 'H.264 CRF (higher = more compressed)' }, tickvals: x }), yaxis: axis(t, { title: { text: 'MAE, bpm (10 s windows)' }, rangemode: 'tozero' }),
            annotations: base ? [{ xref: 'paper', x: 0, y: base.mae_pos_10s, text: 'uncompressed', showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { color: t.ink2, size: 11 } }] : [] },
        };
      },
    })));
  } else {
    out.append(h('div', { style: { marginTop: '16px' } }, emptyCard('Heart-rate error vs compression level', [
      'python -m pulsecheck compression-study --root C:/datasets/UBFC-rPPG --limit 20'])));
  }
  return out;
}

// ------------------------------------------------------------------ detector

function rocCard(roc, title, takeaway, color) {
  return chartCard({
    title, takeaway, height: 320,
    table: () => ({ columns: ['false-alarm rate', 'detection rate', 'threshold'], rows: roc.x.map((x, i) => [fmt(x, 3), fmt(roc.y[i], 3), fmt(roc.thr?.[i], 3)]) }),
    draw: (t) => ({
      data: [line([0, 1], [0, 1], t.axis, { line: { width: 1 }, hoverinfo: 'skip' }),
        area(roc.x, roc.y, color(t), hexAlpha(color(t), 0.1), { line: { shape: 'hv' }, hovertemplate: 'false alarms %{x:.1%}<br>caught <b>%{y:.1%}</b><extra></extra>' })],
      layout: {
        xaxis: axis(t, { title: { text: 'false-alarm rate (real called fake)' }, range: [0, 1], tickformat: '.0%' }),
        yaxis: axis(t, { title: { text: 'detection rate (fakes caught)' }, range: [0, 1.02], tickformat: '.0%' }),
        annotations: [{ x: 0.62, y: 0.55, text: 'chance', showarrow: false, textangle: -38, font: { color: t.muted, size: 11 } }],
      },
    }),
  });
}

function detectorSection(rep) {
  const out = h('section', { class: 'section', 'aria-label': 'Deepfake detector evaluation' });
  if (!rep) {
    out.append(sectionHead('Deepfake detector'), emptyCard('Detector trained and tested on FaceForensics++', [
      'python -m pulsecheck extract ffpp --root C:/datasets/FaceForensics --compression c23',
      'python -m pulsecheck extract celebdf --root C:/datasets/Celeb-DF-v2',
      'python -m pulsecheck train --cross Celeb-DF=cache/celebdf --real UBFC=cache/ubfc']));
    return out;
  }
  const tv = rep.test_video;
  const tw = rep.test_window;
  out.append(sectionHead('Deepfake detector', `${rep.source} · test split · ${tv.n_real} real and ${tv.n_fake} fake videos never trained on · ${rep.chosen_label}`));
  if (rep.source === 'synthetic') out.append(syntheticBanner());
  const ci = tv.auc_ci95 && tv.auc_ci95[0] != null ? `95% CI ${fmt(tv.auc_ci95[0], 2)}–${fmt(tv.auc_ci95[1], 2)}` : 'video level';
  out.append(h('div', { class: 'kpis' },
    tile('ROC AUC', fmt(tv.auc, 3), '', ci),
    tile('Balanced accuracy', pct(tv.balanced_accuracy, 1), '', `at threshold ${fmt(rep.threshold, 2)} (picked on val)`),
    tile('Equal error rate', pct(tv.eer, 1), '', 'false alarms = misses'),
    tile('Fakes caught', pct(tv.recall, 1), '', `${pct(tv.false_alarm_rate, 1)} of real videos flagged`),
    tile('Average precision', fmt(tv.average_precision, 3), '', `${pct(tv.n_fake / tv.n, 0)} of test videos are fake`)));

  const perM = [...rep.per_method].sort((a, b) => MANIP_ORDER.indexOf(a.method) - MANIP_ORDER.indexOf(b.method));
  const colorOf = (m) => Math.max(0, MANIP_ORDER.indexOf(m) >= 0 ? MANIP_ORDER.indexOf(m) : perM.findIndex((p) => p.method === m));
  out.append(grid(
    rocCard(tv.roc, 'ROC curve, all manipulations', `Area under the curve ${fmt(tv.auc, 3)}; 0.5 is a coin flip, 1.0 is perfect.`, (t) => t.s[0]),
    perM.length ? chartCard({
      title: 'ROC by manipulation method',
      takeaway: 'Real test videos against each method’s fakes. Identity swaps and expression edits leave different traces.',
      height: 320,
      legend: perM.map((p) => ({ label: `${p.method} · AUC ${fmt(p.auc, 2)}`, color: SERIES(colorOf(p.method)) })),
      table: () => ({ columns: ['method', 'AUC', 'detection rate', 'fake videos'], rows: perM.map((p) => [p.method, fmt(p.auc, 3), pct(p.detection_rate, 1), p.n_fake]) }),
      draw: (t) => ({
        data: [line([0, 1], [0, 1], t.axis, { line: { width: 1 }, hoverinfo: 'skip' }),
          ...perM.map((p) => line(p.roc.x, p.roc.y, t.s[colorOf(p.method)], { line: { shape: 'hv' }, name: p.method, hovertemplate: `${p.method}<br>false alarms %{x:.1%} · caught <b>%{y:.1%}</b><extra></extra>` }))],
        layout: { xaxis: axis(t, { title: { text: 'false-alarm rate' }, range: [0, 1], tickformat: '.0%' }), yaxis: axis(t, { title: { text: 'detection rate' }, range: [0, 1.02], tickformat: '.0%' }) },
      }),
    }) : null));

  const c = tv.confusion;
  const cm = [[c.tn, c.fp], [c.fn, c.tp]];
  const maxc = Math.max(...cm.flat(), 1);
  const confusion = chartCard({
    title: 'Confusion matrix (test videos)',
    takeaway: `At the frozen threshold: ${c.tp} of ${c.tp + c.fn} fakes caught, ${c.fp} of ${c.tn + c.fp} real videos wrongly flagged.`,
    height: 300,
    table: () => ({ columns: ['', 'called real', 'called fake'], rows: [['actually real', c.tn, c.fp], ['actually fake', c.fn, c.tp]] }),
    draw: (t) => ({
      data: [{ type: 'heatmap', z: cm.map((r) => r.map((v) => v / maxc)), x: ['called real', 'called fake'], y: ['actually real', 'actually fake'], zmin: 0, zmax: 1,
        colorscale: seqScale(t), showscale: false, xgap: 3, ygap: 3, customdata: cm, hovertemplate: '%{y}, %{x}<br><b>%{customdata}</b> videos<extra></extra>' }],
      layout: { xaxis: axis(t, { showgrid: false, side: 'top' }), yaxis: axis(t, { showgrid: false, autorange: 'reversed' }),
        annotations: cm.flatMap((r, i) => r.map((v, j) => ({ x: ['called real', 'called fake'][j], y: ['actually real', 'actually fake'][i], text: String(v), showarrow: false,
          font: { size: 18, color: inkOn(t, v / maxc) } }))) },
    }),
  });
  const hist = rep.test_hist;
  const centers = hist.edges.slice(0, -1).map((e, i) => (e + hist.edges[i + 1]) / 2);
  const scores = chartCard({
    title: 'How confident it is',
    takeaway: 'Video scores for real and fake test videos. Overlap is where mistakes happen.',
    height: 300,
    legend: [{ label: 'Real videos', color: SERIES(0), kind: 'rect' }, { label: 'Fake videos', color: SERIES(1), kind: 'rect' }],
    table: () => ({ columns: ['score from', 'score to', 'real', 'fake'], rows: centers.map((_, i) => [fmt(hist.edges[i], 2), fmt(hist.edges[i + 1], 2), hist.real[i], hist.fake[i]]) }),
    draw: (t, el) => ({
      data: [bars(centers, hist.real, t.s[0], { name: 'real', hovertemplate: 'score %{x:.2f}<br><b>%{y}</b> real<extra></extra>' }),
        bars(centers, hist.fake, t.s[1], { name: 'fake', hovertemplate: 'score %{x:.2f}<br><b>%{y}</b> fake<extra></extra>' })],
      layout: { barmode: 'group', bargap: 0.25, bargroupgap: 0.12, xaxis: axis(t, { title: { text: 'video score, P(fake)' }, range: [0, 1] }), yaxis: axis(t, { title: { text: 'videos' } }),
        shapes: [{ type: 'line', x0: rep.threshold, x1: rep.threshold, yref: 'paper', y0: 0, y1: 1, line: { color: t.ink, width: 1.5 } }],
        annotations: [{ x: rep.threshold, yref: 'paper', y: 1, text: 'threshold', showarrow: false, xanchor: 'left', xshift: 4, yanchor: 'top', font: { color: t.ink2, size: 11 } }] },
    }),
  });
  out.append(grid(confusion, scores));

  const ms = rep.model_selection;
  const sel = chartCard({
    title: 'Which classifier',
    takeaway: `Video-level AUC in 5-fold grouped cross-validation on the training split. ${rep.chosen_label} was chosen.`,
    height: 260,
    table: () => ({ columns: ['model', 'window AUC', 'video AUC', 'seconds'], rows: ms.map((m) => [m.label, fmt(m.window_auc, 3), fmt(m.video_auc, 3), m.seconds]) }),
    draw: (t, el) => ({
      data: [bars(ms.map((m) => m.video_auc), ms.map((m) => m.label), t.s[0], { orientation: 'h', width: barWidth(el, ms.length * 2.2),
        text: ms.map((m) => fmt(m.video_auc, 3)), textposition: 'outside', textfont: { color: t.ink2 }, cliponaxis: false, hovertemplate: '%{y}<br><b>%{x:.3f}</b><extra></extra>' })],
      layout: { xaxis: axis(t, { range: [0.4, 1.05], title: { text: 'video AUC (cross-validated)' } }), yaxis: axis(t, { showgrid: false, autorange: 'reversed' }), margin: { r: 30 } },
    }),
  });
  const sd = rep.split_demo;
  const leak = chartCard({
    title: 'Why the split matters',
    takeaway: `The same model, scored two ways. Letting one video's windows into both training and testing inflates the AUC from ${fmt(sd.grouped_split.video_auc, 3)} to ${fmt(sd.random_window_split.video_auc, 3)}.`,
    height: 260,
    legend: [{ label: 'Honest: grouped by person', color: SERIES(0), kind: 'rect' }, { label: 'Leaky: random windows', color: 'var(--deemph)', kind: 'rect' }],
    table: () => ({ columns: ['split', 'window AUC', 'video AUC'], rows: [['grouped (honest)', fmt(sd.grouped_split.window_auc, 3), fmt(sd.grouped_split.video_auc, 3)], ['random windows (leaky)', fmt(sd.random_window_split.window_auc, 3), fmt(sd.random_window_split.video_auc, 3)]] }),
    draw: (t, el) => {
      const x = ['window AUC', 'video AUC'];
      return {
        data: [bars(x, [sd.grouped_split.window_auc, sd.grouped_split.video_auc], t.s[0], { name: 'grouped', text: [fmt(sd.grouped_split.window_auc, 3), fmt(sd.grouped_split.video_auc, 3)], textposition: 'outside', textfont: { color: t.ink2 }, cliponaxis: false, hovertemplate: 'grouped · %{x}<br><b>%{y:.3f}</b><extra></extra>' }),
          bars(x, [sd.random_window_split.window_auc, sd.random_window_split.video_auc], t.deemph, { name: 'leaky', text: [fmt(sd.random_window_split.window_auc, 3), fmt(sd.random_window_split.video_auc, 3)], textposition: 'outside', textfont: { color: t.ink2 }, cliponaxis: false, hovertemplate: 'leaky · %{x}<br><b>%{y:.3f}</b><extra></extra>' })],
        layout: { barmode: 'group', bargap: 0.45, bargroupgap: 0.1, yaxis: axis(t, { range: [0.4, 1.08], title: { text: 'AUC' } }), xaxis: axis(t, { showgrid: false }), margin: { t: 14 } },
      };
    },
  });
  out.append(grid(sel, leak));

  const imp = rep.importance.slice(0, 12);
  const impCard = chartCard({
    title: 'What the model relies on',
    takeaway: 'Permutation importance: how much test AUC drops when one feature is shuffled.',
    height: 30 * imp.length + 60,
    table: () => ({ columns: ['feature', 'AUC drop', 'std'], rows: rep.importance.map((d) => [d.label, fmt(d.auc_drop, 4), fmt(d.std, 4)]) }),
    draw: (t, el) => ({
      data: [bars(imp.map((d) => d.auc_drop), imp.map((d) => d.label), t.s[0], { orientation: 'h', width: 0.55,
        error_x: { type: 'data', array: imp.map((d) => d.std), color: t.axis, thickness: 1, width: 3 }, hovertemplate: '%{y}<br><b>%{x:.4f}</b> AUC drop<extra></extra>' })],
      layout: { xaxis: axis(t, { title: { text: 'drop in AUC when shuffled' }, rangemode: 'tozero' }), yaxis: axis(t, { showgrid: false, autorange: 'reversed' }) },
    }),
  });
  const dist = rep.distributions.slice(0, 6);
  const distCard = dist.length ? chartCard({
    title: 'Real vs fake, feature by feature',
    takeaway: 'Test-window distributions of the six most important features.',
    height: 380,
    legend: [{ label: 'Real windows', color: SERIES(0) }, { label: 'Fake windows', color: SERIES(1) }],
    table: () => ({ columns: ['feature', 'bin centre', 'real density', 'fake density'], rows: dist.flatMap((d) => d.real.map((r, i) => [d.label, fmt((d.edges[i] + d.edges[i + 1]) / 2, 3), fmt(r, 3), fmt(d.fake[i], 3)])) }),
    draw: (t) => {
      const layout = { annotations: [], margin: { t: 6 } };
      const data = [];
      const cols = 2, rows = Math.ceil(dist.length / cols);
      dist.forEach((d, k) => {
        const r = Math.floor(k / cols), cIdx = k % cols, n = k + 1;
        const xs = d.edges.slice(0, -1).map((e, i) => (e + d.edges[i + 1]) / 2);
        const sx = n === 1 ? '' : String(n);
        const x0 = cIdx / cols + 0.04, x1 = (cIdx + 1) / cols - 0.04;
        const yTop = 1 - r / rows - 0.06, yBot = 1 - (r + 1) / rows + 0.06;
        layout[`xaxis${sx}`] = axis(t, { domain: [x0, x1], anchor: `y${sx}`, tickfont: { size: 10 }, nticks: 4 });
        layout[`yaxis${sx}`] = axis(t, { domain: [yBot, yTop], anchor: `x${sx}`, showticklabels: false });
        layout.annotations.push({ text: d.label, xref: 'paper', yref: 'paper', x: x0, y: yTop + 0.01, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: { color: t.ink2, size: 11.5 } });
        data.push(line(xs, d.real, t.s[0], { xaxis: `x${sx}`, yaxis: `y${sx}`, line: { shape: 'spline', width: 2 }, hovertemplate: `${d.label}<br>%{x:.3f}: real %{y:.2f}<extra></extra>` }));
        data.push(line(xs, d.fake, t.s[1], { xaxis: `x${sx}`, yaxis: `y${sx}`, line: { shape: 'spline', width: 2 }, hovertemplate: `${d.label}<br>%{x:.3f}: fake %{y:.2f}<extra></extra>` }));
      });
      return { data, layout };
    },
  }) : null;
  out.append(grid(impCard, distCard));

  const levels = chartCard({
    title: 'One window vs the whole video',
    takeaway: 'Averaging the window votes over a video helps: noise in one window is outvoted.',
    height: 240,
    table: () => ({ columns: ['level', 'AUC', 'balanced accuracy'], rows: [['window', fmt(tw.auc, 3), pct(tw.balanced_accuracy, 1)], ['video', fmt(tv.auc, 3), pct(tv.balanced_accuracy, 1)]] }),
    draw: (t, el) => ({
      data: [bars(['single 8 s window', 'whole video (mean vote)'], [tw.auc, tv.auc], t.s[0], { width: barWidth(el, 4), text: [fmt(tw.auc, 3), fmt(tv.auc, 3)], textposition: 'outside', textfont: { color: t.ink2 }, cliponaxis: false, hovertemplate: '%{x}<br><b>%{y:.3f}</b><extra></extra>' })],
      layout: { yaxis: axis(t, { range: [0.4, 1.08], title: { text: 'test AUC' } }), xaxis: axis(t, { showgrid: false }), margin: { t: 14 } },
    }),
  });
  out.append(grid(levels, protocolCard(rep)));
  return out;
}

function protocolCard(rep) {
  const d = rep.data;
  const rows = ['train', 'val', 'test'].map((s) => h('tr', {}, h('td', { text: s }), h('td', { text: d[s].videos }), h('td', { text: d[s].fake_videos }), h('td', { text: d[s].windows })));
  return h('div', { class: 'chart-card' }, h('header', {}, h('div', {}, h('h3', { text: 'Protocol' }),
    h('p', { class: 'takeaway', text: `${rep.protocol.model_selection}; threshold: ${rep.protocol.threshold}; video score: ${rep.protocol.aggregation} of ${rep.protocol.window_s} s windows (${rep.protocol.step_s} s hop); rPPG: ${METHOD_LABEL[rep.protocol.rppg_method]}.` }))),
  h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['split', 'videos', 'fake videos', 'windows'].map((c) => h('th', { text: c })))), h('tbody', {}, rows)),
  h('p', { class: 'fineprint', text: `Generated ${rep.generated_at}.` }));
}

// ------------------------------------------------------------------ CNN vs classical

function cnnSection(cnn, det) {
  const out = h('section', { class: 'section', 'aria-label': 'CNN detector' });
  if (!cnn) {
    out.append(sectionHead('CNN on PPG maps'), emptyCard('A convolutional network trained on the GPU, compared with the classical model', [
      'python -m pulsecheck train-cnn --cross Celeb-DF=cache/celebdf --real UBFC=cache/ubfc']));
    return out;
  }
  const tv = cnn.test_video, tr = cnn.training;
  out.append(sectionHead('CNN on PPG maps', `${cnn.source} · same splits · ${tr.parameters.toLocaleString()} parameters · trained on ${tr.device} in ${fmt(tr.train_minutes, 1)} min`));
  out.append(h('div', { class: 'kpis' },
    tile('ROC AUC', fmt(tv.auc, 3), '', tv.auc_ci95?.[0] != null ? `95% CI ${fmt(tv.auc_ci95[0], 2)}–${fmt(tv.auc_ci95[1], 2)}` : 'video level'),
    tile('Balanced accuracy', pct(tv.balanced_accuracy, 1), '', `threshold ${fmt(cnn.threshold, 2)} (picked on val)`),
    tile('Equal error rate', pct(tv.eer, 1), '', 'false alarms = misses'),
    tile('Fakes caught', pct(tv.recall, 1), '', `${pct(tv.false_alarm_rate, 1)} of real videos flagged`),
    tile('Best epoch', String(tr.best_epoch), `of ${tr.epochs_run}`, 'early stopping on val AUC')));

  const models = [det && { name: det.chosen_label, rep: det, color: 0 }, { name: 'CNN', rep: cnn, color: 1 }].filter(Boolean);
  const row = (m) => {
    const v = m.rep.test_video, w = m.rep.test_window;
    return [m.name, fmt(m.rep.val_video_auc, 3), fmt(v.auc, 3), pct(v.balanced_accuracy, 1), pct(v.eer, 1), fmt(v.average_precision, 3), fmt(w.auc, 3)];
  };
  const cmp = h('div', { class: 'chart-card' }, h('header', {}, h('div', {}, h('h3', { text: 'Head to head' }),
    h('p', { class: 'takeaway', text: 'Same training videos, same held-out test videos. The website uses whichever is better on validation, never on test.' }))),
  h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['model', 'val AUC', 'test AUC', 'balanced acc.', 'EER', 'AP', 'window AUC'].map((c) => h('th', { text: c })))),
    h('tbody', {}, models.map((m) => h('tr', {}, row(m).map((v) => h('td', { text: v })))))));
  const roc = chartCard({
    title: 'ROC: classical vs CNN',
    takeaway: 'Test videos, all manipulations together.',
    height: 320,
    legend: models.map((m) => ({ label: `${m.name} · AUC ${fmt(m.rep.test_video.auc, 3)}`, color: SERIES(m.color) })),
    table: () => ({ columns: ['model', 'false-alarm rate', 'detection rate'], rows: models.flatMap((m) => m.rep.test_video.roc.x.map((x, i) => [m.name, fmt(x, 3), fmt(m.rep.test_video.roc.y[i], 3)])) }),
    draw: (t) => ({
      data: [line([0, 1], [0, 1], t.axis, { line: { width: 1 }, hoverinfo: 'skip' }),
        ...models.map((m) => line(m.rep.test_video.roc.x, m.rep.test_video.roc.y, t.s[m.color], { line: { shape: 'hv' }, hovertemplate: `${m.name}<br>false alarms %{x:.1%} · caught <b>%{y:.1%}</b><extra></extra>` }))],
      layout: { xaxis: axis(t, { title: { text: 'false-alarm rate' }, range: [0, 1], tickformat: '.0%' }), yaxis: axis(t, { title: { text: 'detection rate' }, range: [0, 1.02], tickformat: '.0%' }) },
    }),
  });
  out.append(grid(cmp, roc));

  const hist = cnn.history;
  const ep = hist.map((r) => r.epoch);
  const best = { type: 'line', x0: tr.best_epoch, x1: tr.best_epoch, yref: 'paper', y0: 0, y1: 1 };
  const loss = chartCard({
    title: 'Training loss',
    takeaway: 'Class-weighted binary cross-entropy on the training windows, per epoch.',
    height: 240,
    table: () => ({ columns: ['epoch', 'train loss', 'val window AUC', 'val video AUC', 'seconds'], rows: hist.map((r) => [r.epoch, fmt(r.train_loss, 4), fmt(r.val_window_auc, 3), fmt(r.val_video_auc, 3), r.seconds]) }),
    draw: (t) => ({ data: [line(ep, hist.map((r) => r.train_loss), t.s[0], { hovertemplate: 'epoch %{x}<br><b>%{y:.4f}</b><extra></extra>' })],
      layout: { xaxis: axis(t, { title: { text: 'epoch' } }), yaxis: axis(t, { title: { text: 'loss' } }), shapes: [{ ...best, line: { color: t.axis, width: 1 } }] } }),
  });
  const val = chartCard({
    title: 'Validation AUC',
    takeaway: `Video-level AUC on the validation split; the weights from epoch ${tr.best_epoch} (line) were kept.`,
    height: 240,
    table: () => ({ columns: ['epoch', 'val video AUC'], rows: hist.map((r) => [r.epoch, fmt(r.val_video_auc, 3)]) }),
    draw: (t) => ({ data: [line(ep, hist.map((r) => r.val_video_auc), t.s[0], { hovertemplate: 'epoch %{x}<br><b>%{y:.3f}</b><extra></extra>' })],
      layout: { xaxis: axis(t, { title: { text: 'epoch' } }), yaxis: axis(t, { title: { text: 'AUC' } }), shapes: [{ ...best, line: { color: t.axis, width: 1 } }] } }),
  });
  out.append(grid(loss, val));

  const meths = MANIP_ORDER.filter((m) => cnn.per_method.some((p) => p.method === m));
  const aucOf = (rep, m) => rep?.per_method.find((p) => p.method === m)?.auc ?? null;
  const perM = chartCard({
    title: 'AUC by manipulation method',
    takeaway: 'Where each model is stronger.',
    height: 280,
    legend: models.map((m) => ({ label: m.name, color: SERIES(m.color), kind: 'rect' })),
    table: () => ({ columns: ['method', ...models.map((m) => m.name)], rows: meths.map((x) => [x, ...models.map((m) => fmt(aucOf(m.rep, x), 3))]) }),
    draw: (t) => ({
      data: models.map((m) => bars(meths, meths.map((x) => aucOf(m.rep, x)), t.s[m.color], { name: m.name, hovertemplate: `${m.name} · %{x}<br><b>%{y:.3f}</b><extra></extra>` })),
      layout: { barmode: 'group', bargap: 0.35, bargroupgap: 0.12, yaxis: axis(t, { range: [0.4, 1.02], title: { text: 'AUC' } }), xaxis: axis(t, { showgrid: false }) },
    }),
  });
  const sets = [...new Set(models.flatMap((m) => (m.rep.generalization || []).filter((g) => g.kind !== 'real_only').map((g) => g.name)))];
  const gen = sets.length ? chartCard({
    title: 'Generalisation',
    takeaway: 'AUC on datasets neither model was trained on.',
    height: 280,
    legend: models.map((m) => ({ label: m.name, color: SERIES(m.color), kind: 'rect' })),
    table: () => ({ columns: ['test set', ...models.map((m) => m.name)], rows: sets.map((s) => [s, ...models.map((m) => fmt(m.rep.generalization.find((g) => g.name === s)?.auc, 3))]) }),
    draw: (t) => ({
      data: models.map((m) => bars(sets, sets.map((s) => m.rep.generalization.find((g) => g.name === s)?.auc ?? null), t.s[m.color], { name: m.name, hovertemplate: `${m.name} · %{x}<br><b>%{y:.3f}</b><extra></extra>` })),
      layout: { barmode: 'group', bargap: 0.45, bargroupgap: 0.12, yaxis: axis(t, { range: [0, 1.02], title: { text: 'AUC' } }), xaxis: axis(t, { showgrid: false }),
        shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: t.axis, width: 1 } }] },
    }),
  }) : null;
  out.append(grid(perM, gen));
  return out;
}

// ------------------------------------------------------------------ generalisation

function generalizationSection(rep) {
  const out = h('section', { class: 'section', 'aria-label': 'Generalisation' });
  out.append(sectionHead('Generalisation', 'the real test: data the model never saw'));
  const g = rep?.generalization || [];
  if (!g.length) {
    out.append(emptyCard('Cross-dataset and false-alarm tests', [
      'python -m pulsecheck train --cross Celeb-DF=cache/celebdf --real UBFC=cache/ubfc']));
    return out;
  }
  const tv = rep.test_video;
  const ext = g.filter((x) => x.kind !== 'real_only');
  const realOnly = g.filter((x) => x.kind === 'real_only');
  const sets = [{ name: `${rep.source} (test)`, auc: tv.auc, n: tv.n, bacc: tv.balanced_accuracy }, ...ext.map((x) => ({ name: x.name, auc: x.auc, n: x.n_videos, bacc: x.balanced_accuracy }))];
  const auc = chartCard({
    title: 'AUC on each test set',
    takeaway: 'A drop from the home test set to an unseen dataset is normal, and it is the honest number to quote.',
    height: 260,
    table: () => ({ columns: ['test set', 'videos', 'AUC', 'balanced accuracy'], rows: sets.map((s) => [s.name, s.n, fmt(s.auc, 3), pct(s.bacc, 1)]) }),
    draw: (t, el) => ({
      data: [bars(sets.map((s) => s.name), sets.map((s) => s.auc), t.s[0], { width: barWidth(el, Math.max(sets.length, 3)), text: sets.map((s) => fmt(s.auc, 3)), textposition: 'outside', textfont: { color: t.ink2 }, cliponaxis: false, hovertemplate: '%{x}<br><b>%{y:.3f}</b><extra></extra>' })],
      layout: { yaxis: axis(t, { range: [0, 1.08], title: { text: 'AUC' } }), xaxis: axis(t, { showgrid: false }), margin: { t: 14 },
        shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: t.axis, width: 1 } }],
        annotations: [{ xref: 'paper', x: 1, y: 0.5, text: 'chance', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: t.muted, size: 11 } }] },
    }),
  });
  const fa = h('div', { class: 'stack' }, realOnly.map((x) => h('div', { class: 'kpis' },
    tile(`False alarms on ${x.name}`, pct(x.false_alarm_rate, 1), '', `${x.confusion.fp} of ${x.n_videos} genuine videos called fake`),
    tile('Their median score', fmt(medianFromHist(x.hist), 2), '', `threshold ${fmt(rep.threshold, 2)}`))));
  out.append(grid(auc, realOnly.length ? h('div', { class: 'chart-card' }, h('header', {}, h('div', {}, h('h3', { text: 'Genuine videos from a different camera' }),
    h('p', { class: 'takeaway', text: 'All-real recordings the detector never saw. Any “fake” verdict here is a false alarm, the failure the BBC test found.' }))), fa) : null));
  return out;
}

function medianFromHist(hist) {
  const counts = hist.real.map((r, i) => r + hist.fake[i]);
  const total = counts.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (let i = 0; i < counts.length; i++) { acc += counts[i]; if (acc >= total / 2) return (hist.edges[i] + hist.edges[i + 1]) / 2; }
  return null;
}

// ------------------------------------------------------------------ entry

let loaded = false;
export async function initResults() {
  if (loaded) return;
  loaded = true;
  const root = document.getElementById('results-root');
  try {
    const rep = await (await fetch('/api/reports')).json();
    root.replaceChildren(hrSection(rep.hr, rep.compression), detectorSection(rep.detector), cnnSection(rep.cnn, rep.detector),
      generalizationSection(rep.detector));
  } catch (e) {
    root.replaceChildren(h('div', { class: 'banner bad' }, h('p', { text: `Could not load results: ${e.message}` })));
    loaded = false;
  }
}
