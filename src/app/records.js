// The two optional data sections: the records table, and the seeds study when its file exists.

const NS = 'http://www.w3.org/2000/svg';

const safeLink = (href) => {
  try {
    const u = new URL(href);
    return u.protocol === 'https:' || u.protocol === 'http:' ? u.href : null;
  } catch { return null; }
};

export async function loadRecords() {
  const table = document.getElementById('recordsTable');
  const empty = document.getElementById('recordsEmpty');
  let rows = [];
  try {
    const res = await fetch('records.json', { cache: 'no-store' });
    if (res.ok) rows = await res.json();
  } catch { rows = []; }
  if (!Array.isArray(rows) || !rows.length) {
    empty.hidden = false;
    return;
  }
  rows.sort((a, b) => a.medianSeconds - b.medianSeconds);
  const head = table.createTHead().insertRow();
  [['median seconds', 'num'], ['adapter'], ['browser'], ['seeds', 'num'], ['submitter'], ['log']].forEach(([label, cls]) => {
    const th = document.createElement('th');
    th.textContent = label;
    if (cls) th.className = cls;
    head.append(th);
  });
  const body = table.createTBody();
  for (const r of rows) {
    const tr = body.insertRow();
    const add = (text, cls) => {
      const td = tr.insertCell();
      td.textContent = text;
      if (cls) td.className = cls;
      return td;
    };
    add(Number(r.medianSeconds).toFixed(2), 'num');
    add(String(r.adapter ?? ''));
    add(String(r.browser ?? ''));
    add(String(r.seeds ?? ''), 'num');
    add(String(r.submitter ?? ''));
    const logCell = add('');
    const href = safeLink(r.log);
    if (href) {
      const a = document.createElement('a');
      a.href = href;
      a.textContent = 'log';
      logCell.append(a);
    }
  }
  table.hidden = false;
}

// Kaplan-Meier step curves, one per arm, told apart by lightness and dash and labeled at their ends.
const DASHES = ['', '6 4', '2 4', '10 4 2 4', '1 3'];
const STROKES = ['--glow', '--ink', '--ink-2', '--ink-3', '--ink-3'];

const CURRICULA = {
  'uniform tiled': 'random characters, periodic',
  'bigram tiled': "Howe's language, periodic",
  'bigram clone': "Howe's language, clone rule",
  'uniform clone': 'random characters, clone rule',
};
const blocks = (a) => `${a[0]} layers, ${a[1]} block${a[1] > 1 ? 's' : ''}`;

function chart(fig, label, draw) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', label);
  fig.append(svg);
  const add = (name, attrs, content) => {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (content !== undefined) n.textContent = content;
    svg.append(n);
    return n;
  };
  const redraw = () => {
    const { width: w, height: h } = fig.getBoundingClientRect();
    if (!w) return;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.replaceChildren();
    draw(w, h, add);
  };
  new ResizeObserver(redraw).observe(fig);
  redraw();
}

function survival(fig, arms) {
  const maxStep = Math.max(...arms.flatMap((a) => a.curve.map((p) => p[0])), 8000);
  chart(fig, 'Share of seeds without the event at each training step, per curriculum', (w, h, add) => {
    const labelRoom = w < 480 ? 0 : 360;
    const m = { l: 2, t: 22, b: 24 };
    const X = (s) => m.l + (s / maxStep) * (w - m.l - labelRoom);
    const Y = (v) => m.t + (1 - v) * (h - m.t - m.b);
    add('text', { x: m.l, y: m.t - 8, class: 'note' }, 'every seed still without the event');
    add('text', { x: w - labelRoom, y: Y(0) - 7, class: 'note', 'text-anchor': 'end' }, 'none left');
    add('text', { x: m.l, y: h - 4, class: 'note' }, 'step 0');
    add('text', { x: w - labelRoom, y: h - 4, class: 'note', 'text-anchor': 'end' }, `step ${maxStep}`);
    arms.forEach((arm, i) => {
      let d = '';
      let last = 1;
      for (const [s, v] of arm.curve) {
        d += `${d ? 'L' : 'M'}${X(s).toFixed(1)} ${Y(last).toFixed(1)} L${X(s).toFixed(1)} ${Y(v).toFixed(1)} `;
        last = v;
      }
      d += `L${X(maxStep).toFixed(1)} ${Y(last).toFixed(1)}`;
      const p = add('path', { d, class: 'km' });
      p.style.stroke = `var(${STROKES[i % STROKES.length]})`;
      if (DASHES[i % DASHES.length]) p.setAttribute('stroke-dasharray', DASHES[i % DASHES.length]);
      const name = CURRICULA[arm.curriculum];
      const text = `${name}, ${arm.medianStep ?? `none by ${maxStep}`}`;
      // Each label carries a sample of its line, so identity never rests on position or color.
      const lx = labelRoom ? w - labelRoom + 8 : X(1100) + 6;
      const ly = labelRoom ? Y(last) + 4 - (last === 0 && i === 0 ? 16 : 0) : Y([0.78, 0.64, 0.5, 0.36][i]);
      const sample = add('line', { x1: lx, x2: lx + 22, y1: ly - 4, y2: ly - 4, class: 'km' });
      sample.style.stroke = p.style.stroke;
      if (DASHES[i % DASHES.length]) sample.setAttribute('stroke-dasharray', DASHES[i % DASHES.length]);
      add('text', { x: lx + 28, y: ly }, labelRoom ? text : text.replace('random characters', 'random'));
    });
  });
}

function forest(fig, arms) {
  const lo = 400;
  const hi = 9000;
  chart(fig, 'Median step of the event per depth and number of blocks, on a log scale, with the seeds behind each arm', (w, h, add) => {
    const wide = w >= 480;
    const m = { l: 150, r: 60, t: 22, b: 24 };
    const X = (s) => m.l + (Math.log(s / lo) / Math.log(hi / lo)) * (w - m.l - m.r);
    const row = (h - m.t - m.b) / arms.length;
    add('text', { x: w - 24, y: m.t - 8, class: 'note', 'text-anchor': 'end' }, 'median');
    add('text', { x: w, y: m.t - 8, class: 'note', 'text-anchor': 'end' }, 'n');
    for (const s of wide ? [500, 1000, 2000, 4000, 8000] : [500, 2000, 8000]) {
      add('line', { x1: X(s), x2: X(s), y1: m.t, y2: h - m.b, class: 'grid' });
      add('text', { x: X(s), y: h - 4, class: 'note', 'text-anchor': 'middle' }, s === 500 && wide ? 'step 500' : String(s));
    }
    arms.forEach((a, i) => {
      const y = m.t + row * (i + 0.5);
      add('text', { x: 0, y: y + 4 }, blocks([a.L, a.u]));
      add('line', { x1: X(a.ci[0]), x2: X(a.ci[1]), y1: y, y2: y, class: 'whisker' });
      add('circle', { cx: X(a.medianStep), cy: y, r: 4, class: a.L === a.u ? 'mark' : 'mark hollow' });
      if (a.prefixAtEvent[1] < 0.1) add('text', { x: X(a.ci[0]) - 8, y: y + 4, class: 'note', 'text-anchor': 'end' }, wide ? 'no prefix signature' : 'no signature');
      add('text', { x: w - 24, y: y + 4, 'text-anchor': 'end' }, String(a.medianStep));
      add('text', { x: w, y: y + 4, 'text-anchor': 'end' }, String(a.n));
    });
  });
}

function contrastTable(table, contrasts) {
  const head = table.createTHead().insertRow();
  [['how many times later', ''], ['lr 3e-3, 95% interval', 'num'], ['lr 1e-3, 95% interval', 'num']].forEach(([label, cls]) => {
    const th = document.createElement('th');
    th.textContent = label;
    if (cls) th.className = cls;
    head.append(th);
  });
  const body = table.createTBody();
  const rows = new Map();
  const picked = contrasts.filter((x) => x.kind === 'tying' || x.kind === 'untied depth');
  picked.sort((x, y) => x.kind.localeCompare(y.kind) || x.a[0] - y.a[0] || x.a[1] - y.a[1]);
  for (const c of picked) {
    const name = c.kind === 'tying' ? `${c.a[0]} layers, ${c.a[1]} block${c.a[1] > 1 ? 's' : ''} against ${c.b[1]}` : `${c.a[0]} untied layers against ${c.b[0]}`;
    rows.set(name, { ...rows.get(name), [c.lr]: c });
  }
  const span = (v, [a, b]) => `${v.toFixed(2)} (${a.toFixed(2)} to ${b.toFixed(2)})`;
  for (const [name, r] of rows) {
    const tr = body.insertRow();
    tr.insertCell().textContent = name;
    for (const lr of [0.003, 0.001]) {
      const td = tr.insertCell();
      td.className = 'num';
      td.dataset.lr = `lr ${lr === 0.003 ? '3e-3' : '1e-3'}`;
      const c = r[lr];
      const lines = c ? [`fit ${span(c.timeRatio, c.ci)}`, `KM ${span(c.kmRatio, c.kmCi)}`] : ['not run'];
      for (const text of lines) {
        const line = document.createElement('span');
        line.className = 'ratio';
        line.textContent = text;
        td.append(line);
      }
    }
  }
}

export async function loadStudy() {
  let data = null;
  try {
    const res = await fetch('results/study-summary.json', { cache: 'no-store' });
    if (res.ok) data = await res.json();
  } catch { data = null; }
  if (!data?.arms?.length) return;
  document.getElementById('study').hidden = false;
  const atTabRate = data.arms.filter((a) => a.lr === 0.003);
  const curricula = Object.keys(CURRICULA).map((c) => atTabRate.find((a) => a.L === 2 && a.u === 2 && a.curriculum === c));
  survival(document.getElementById('studyCurriculum'), curricula);
  const tiled = atTabRate.filter((a) => a.curriculum === 'uniform tiled').sort((a, b) => a.L - b.L || b.u - a.u);
  forest(document.getElementById('studyForest'), tiled);
  contrastTable(document.getElementById('studyTable'), data.contrasts);
}
