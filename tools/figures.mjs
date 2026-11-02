// Draws the two study figures from results/study-summary.json into docs/, one light and one dark file each. `npm run figures`
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const OUT = path.join(ROOT, 'docs');
const { arms } = JSON.parse(fs.readFileSync(path.join(ROOT, 'results/study-summary.json'), 'utf8'));

// Backgrounds are GitHub's page colors, so a file shows no card behind it. Series are told apart by lightness, dash
// and marker shape, so hue carries nothing a color-blind reader would miss.
const THEMES = {
  light: { bg: '#ffffff', text: '#13161c', muted: '#55585f', grid: '#d8dee4', series: ['#102c84', '#13161c', '#5a5d63', '#8d9096'] },
  dark: { bg: '#0d1117', text: '#edebe5', muted: '#a3a29c', grid: '#21262d', series: ['#fde16b', '#c9c8c2', '#959490', '#62656b'] },
};
const FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace";

// The summary keeps survival curves but not budgets, and these are what the arms were run to.
const CURRICULUM_BUDGET = 8000;
const DEEP_BUDGET = 4000;

const arm = (L, u, lr, curriculum = 'uniform tiled') => {
  const found = arms.find((a) => a.L === L && a.u === u && a.lr === lr && a.curriculum === curriculum);
  if (!found) throw new Error(`no arm ${L},${u} at lr ${lr}, ${curriculum}`);
  return found;
};
const logScale = (lo, hi, a, b) => (v) => a + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (b - a);
const n = (v) => +v.toFixed(1);
const text = (x, y, str, fill, { size = 12, anchor = 'start', weight = 400 } = {}) =>
  `<text x="${n(x)}" y="${n(y)}" fill="${fill}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${str}</text>`;
const hline = (x0, x1, y, stroke) => `<line x1="${n(x0)}" x2="${n(x1)}" y1="${n(y)}" y2="${n(y)}" stroke="${stroke}"/>`;

function svg(theme, w, h, title, desc, body) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" font-family="${FONT}">`,
    `<title>${title}</title>`,
    `<desc>${desc}</desc>`,
    `<rect width="${w}" height="${h}" fill="${theme.bg}"/>`,
    ...body,
    '</svg>',
    '',
  ].join('\n');
}

function survival(theme) {
  const X = logScale(150, CURRICULUM_BUDGET, 70, 850);
  const Y = (p) => 92 + (1 - p) * 260;
  const LR = 0.003;
  const series = [
    { curriculum: 'uniform clone', slot: 3, dash: '9 3 2 3' },
    { curriculum: 'bigram clone', slot: 2, dash: '2 3' },
    { curriculum: 'bigram tiled', slot: 1, dash: '7 4' },
    { curriculum: 'uniform tiled', slot: 0, dash: '' },
  ].map((s) => ({ ...s, arm: arm(2, 2, LR, s.curriculum) }));

  const body = [
    text(0, 18, 'Seeds still without the event, by curriculum', theme.text, { size: 14, weight: 600 }),
    text(0, 38, 'Study model, lr 3e-3. Logarithmic step axis. A curve crosses half at its median.', theme.muted),
  ];
  for (const [p, label] of [[1, 'all seeds'], [0.5, 'half'], [0, 'none']]) {
    body.push(hline(70, 850, Y(p), theme.grid), text(60, Y(p) + 4, label, theme.muted, { size: 11, anchor: 'end' }));
  }
  for (const s of [500, 1000, 2000, 4000, 8000]) body.push(text(X(s), 372, s === 500 ? 'step 500' : s, theme.muted, { size: 11, anchor: 'middle' }));

  for (const { arm: a, slot, dash } of series) {
    let d = `M70 ${Y(1)}`;
    for (const [step, p] of a.curve.slice(1)) d += ` H${n(X(step))} V${n(Y(p))}`;
    const last = a.curve[a.curve.length - 1][1];
    if (last > 0) d += ` H${n(X(CURRICULUM_BUDGET))}`;
    body.push(`<path d="${d}" fill="none" stroke="${theme.series[slot]}" stroke-width="${slot === 0 ? 2.5 : 2}"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`);
    if (last > 0) body.push(`<line x1="${n(X(CURRICULUM_BUDGET))}" x2="${n(X(CURRICULUM_BUDGET))}" y1="${n(Y(last) - 6)}" y2="${n(Y(last) + 6)}" stroke="${theme.series[slot]}" stroke-width="2"/>`);
    if (a.medianStep) body.push(`<circle cx="${n(X(a.medianStep))}" cy="${n(Y(0.5))}" r="4.5" fill="${theme.series[slot]}" stroke="${theme.bg}" stroke-width="2"/>`);
  }

  const [randomClone, howeClone, howePeriodic, randomPeriodic] = series.map((s) => s.arm);
  const label = (x, y, anchor, name, note) => [text(x, y, name, theme.text, { anchor }), text(x, y + 16, note, theme.muted, { anchor })];
  body.push(
    text(850, Y(1) - 10, `random characters, clone rule. No event in ${randomClone.n} seeds by step ${CURRICULUM_BUDGET}.`, theme.text, { anchor: 'end' }),
    ...label(X(575) - 34, Y(0.62), 'end', 'random characters, periodic', `median ${randomPeriodic.medianStep}, ${randomPeriodic.n} seeds`),
    ...label(X(1125) + 14, Y(0.62), 'start', "Howe's language, periodic", `median ${howePeriodic.medianStep}, ${howePeriodic.n} seeds`),
    ...label(X(7000), Y(0.3), 'end', "Howe's language, clone rule", `median ${howeClone.medianStep}, ${howeClone.events} of ${howeClone.n} fired`),
  );
  return svg(theme, 880, 384, 'Seeds still without the event, by curriculum', 'Kaplan-Meier curves for four curricula on the study model at learning rate 3e-3, from results/study-summary.json.', body);
}

function medians(theme) {
  const Y = logScale(450, 6000, 340, 96);
  const SHAPES = {
    circle: (x, y, fill) => `<circle cx="${n(x)}" cy="${n(y)}" r="5" fill="${fill}" stroke="${theme.bg}" stroke-width="2"/>`,
    square: (x, y, fill) => `<rect x="${n(x - 4.5)}" y="${n(y - 4.5)}" width="9" height="9" fill="${fill}" stroke="${theme.bg}" stroke-width="2"/>`,
    diamond: (x, y, fill) => `<path d="M${n(x)} ${n(y - 6.5)} L${n(x + 6.5)} ${n(y)} L${n(x)} ${n(y + 6.5)} L${n(x - 6.5)} ${n(y)} Z" fill="${fill}" stroke="${theme.bg}" stroke-width="2"/>`,
  };
  // Drawn slowest family first so the untied line stays on top where it shares a point with 2 blocks at depth 2.
  const FAMILIES = [
    { name: '1 block', shape: 'diamond', slot: 2, dash: '2 3', shapes: [[2, 1], [3, 1], [4, 1], [8, 1]] },
    { name: '2 blocks', shape: 'square', slot: 1, dash: '7 4', shapes: [[2, 2], [4, 2], [8, 2]] },
    { name: 'untied', shape: 'circle', slot: 0, dash: '', shapes: [[2, 2], [3, 3], [4, 4], [8, 8]] },
  ];

  const panel = (x0, x1, lr, title, notes) => {
    const X = logScale(2, 8, x0 + 16, x1 - 10);
    const out = [text(x0, 76, title, theme.text, { size: 13, weight: 600 })];
    for (const s of [500, 1000, 2000, 4000]) out.push(hline(x0, x1, Y(s), theme.grid), text(x0 - 8, Y(s) + 4, s, theme.muted, { size: 11, anchor: 'end' }));
    for (const d of [2, 3, 4, 8]) out.push(text(X(d), 360, d === 2 ? '2 layers' : d, theme.muted, { size: 11, anchor: 'middle' }));
    for (const f of FAMILIES) {
      const stroke = theme.series[f.slot];
      const points = f.shapes.filter(([L, u]) => arms.some((a) => a.L === L && a.u === u && a.lr === lr)).map(([L, u]) => ({ L, a: arm(L, u, lr) }));
      const fired = points.filter((p) => p.a.medianStep);
      out.push(`<polyline points="${fired.map((p) => `${n(X(p.L))},${n(Y(p.a.medianStep))}`).join(' ')}" fill="none" stroke="${stroke}" stroke-width="2"${f.dash ? ` stroke-dasharray="${f.dash}"` : ''}/>`);
      for (const p of points.filter((q) => !q.a.medianStep)) {
        const prev = fired[fired.length - 1];
        const tipY = Y(DEEP_BUDGET);
        out.push(`<line x1="${n(X(prev.L))}" y1="${n(Y(prev.a.medianStep))}" x2="${n(X(p.L))}" y2="${n(tipY)}" stroke="${stroke}" stroke-width="2" stroke-dasharray="${f.dash}"/>`);
        out.push(`<path d="M${n(X(p.L))} ${n(tipY - 8)} L${n(X(p.L) + 6)} ${n(tipY + 3)} L${n(X(p.L) - 6)} ${n(tipY + 3)} Z" fill="${stroke}" stroke="${theme.bg}" stroke-width="2"/>`);
      }
      for (const { L, a } of fired) {
        if (a.ci) out.push(`<line x1="${n(X(L))}" x2="${n(X(L))}" y1="${n(Y(a.ci[0]))}" y2="${n(Y(a.ci[1]))}" stroke="${stroke}" stroke-width="1.5"/>`);
        out.push(SHAPES[f.shape](X(L), Y(a.medianStep), stroke));
      }
    }
    for (const [L, value, dx, dy, str, anchor, tone] of notes) out.push(text(X(L) + dx, Y(value) + dy, str, tone === 'muted' ? theme.muted : theme.text, { anchor }));
    return out;
  };

  const med = (L, u, lr) => arm(L, u, lr).medianStep;
  const body = [
    text(0, 18, 'Median step of the event, by depth and blocks', theme.text, { size: 14, weight: 600 }),
    text(0, 38, 'Study model. Untied means every layer has its own block. Bars are 95% intervals. Both axes are logarithmic.', theme.muted),
    ...panel(48, 440, 0.003, 'lr 3e-3', [
      [2, med(2, 2, 0.003), 0, -13, med(2, 2, 0.003), 'middle'],
      [2, med(2, 1, 0.003), 0, -13, med(2, 1, 0.003), 'middle'],
      [8, DEEP_BUDGET, 4, -14, `1 block. No event in ${arm(8, 1, 0.003).n} seeds by step ${DEEP_BUDGET}.`, 'end'],
      [8, med(8, 2, 0.003), 4, -19, `2 blocks, ${med(8, 2, 0.003)}`, 'end'],
      [8, med(8, 8, 0.003), 4, 22, `untied, ${med(8, 8, 0.003)}`, 'end'],
    ]),
    ...panel(488, 880, 0.001, 'lr 1e-3', [
      [2, med(2, 2, 0.001), 0, -13, med(2, 2, 0.001), 'middle'],
      [2, med(2, 1, 0.001), 0, -13, med(2, 1, 0.001), 'middle'],
      [4, med(4, 1, 0.001), 14, 4, `1 block, ${med(4, 1, 0.001)}`, 'start'],
      [4, med(4, 2, 0.001), 14, -6, `2 blocks, ${med(4, 2, 0.001)}`, 'start'],
      [4, med(4, 4, 0.001), 14, 4, `untied, ${med(4, 4, 0.001)}`, 'start'],
      [8, 500, 4, -8, 'depth 8 not run at this rate', 'end', 'muted'],
    ]),
  ];
  return svg(theme, 880, 372, 'Median step of the event, by depth and blocks', 'Median event step for tied and untied models at depths 2, 3, 4 and 8, at two learning rates, from results/study-summary.json.', body);
}

fs.mkdirSync(OUT, { recursive: true });
for (const [name, theme] of Object.entries(THEMES)) {
  fs.writeFileSync(path.join(OUT, `survival-by-curriculum-${name}.svg`), survival(theme));
  fs.writeFileSync(path.join(OUT, `median-by-depth-${name}.svg`), medians(theme));
}
console.log('wrote docs/survival-by-curriculum-{light,dark}.svg and docs/median-by-depth-{light,dark}.svg');
