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

export async function loadStudy() {
  let data = null;
  try {
    const res = await fetch('results/study-summary.json', { cache: 'no-store' });
    if (res.ok) data = await res.json();
  } catch { data = null; }
  if (!data?.arms?.length) return;
  const section = document.getElementById('study');
  section.hidden = false;
  const fig = document.getElementById('studyFig');
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Share of seeds without the event at each training step, per configuration');
  fig.append(svg);
  const maxStep = Math.max(...data.arms.flatMap((a) => a.curve.map((p) => p[0])));
  const draw = () => {
    const { width: w, height: h } = fig.getBoundingClientRect();
    if (!w) return;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.replaceChildren();
    const labelRoom = w < 420 ? 96 : 170;
    const m = { l: 2, t: 22, b: 24 };
    const X = (s) => m.l + (s / maxStep) * (w - m.l - labelRoom);
    const Y = (v) => m.t + (1 - v) * (h - m.t - m.b);
    const text = (x, y, t, cls, anchor) => {
      const n = document.createElementNS(NS, 'text');
      n.setAttribute('x', x);
      n.setAttribute('y', y);
      if (cls) n.setAttribute('class', cls);
      if (anchor) n.setAttribute('text-anchor', anchor);
      n.textContent = t;
      svg.append(n);
    };
    text(m.l, m.t - 8, 'all seeds still without the event', 'note');
    text(m.l, Y(0) - 7, 'none', 'note');
    text(m.l, h - 4, 'step 0', 'note');
    text(w - labelRoom, h - 4, `step ${maxStep}`, 'note', 'end');
    const ends = [];
    data.arms.forEach((arm, i) => {
      let d = '';
      let last = 1;
      for (const [s, v] of arm.curve) {
        d += `${d ? 'L' : 'M'}${X(s).toFixed(1)} ${Y(last).toFixed(1)} L${X(s).toFixed(1)} ${Y(v).toFixed(1)} `;
        last = v;
      }
      d += `L${X(maxStep).toFixed(1)} ${Y(last).toFixed(1)}`;
      const p = document.createElementNS(NS, 'path');
      p.setAttribute('d', d);
      p.setAttribute('class', 'km');
      p.style.stroke = `var(${STROKES[i % STROKES.length]})`;
      if (DASHES[i % DASHES.length]) p.setAttribute('stroke-dasharray', DASHES[i % DASHES.length]);
      svg.append(p);
      ends.push({ y: Y(last), label: arm.label });
    });
    ends.sort((a, b) => a.y - b.y);
    let prevY = -Infinity;
    for (const e of ends) {
      const y = Math.max(e.y + 4, prevY + 15);
      text(w - labelRoom + 8, y, e.label);
      prevY = y;
    }
  };
  new ResizeObserver(draw).observe(fig);
  draw();

  const table = document.getElementById('studyTable');
  const head = table.createTHead().insertRow();
  [['configuration'], ['median step', 'num'], ['95% interval', 'num'], ['events', 'num'], ['seeds', 'num']].forEach(([label, cls]) => {
    const th = document.createElement('th');
    th.textContent = label;
    if (cls) th.className = cls;
    head.append(th);
  });
  const body = table.createTBody();
  for (const a of data.arms) {
    const tr = body.insertRow();
    const add = (t, cls) => { const td = tr.insertCell(); td.textContent = t; if (cls) td.className = cls; };
    add(a.label);
    add(a.medianStep === null || a.medianStep === undefined ? 'not reached' : String(a.medianStep), 'num');
    add(a.ci ? `${a.ci[0] ?? 'open'} to ${a.ci[1] ?? 'open'}` : '', 'num');
    add(String(a.events), 'num');
    add(String(a.n), 'num');
  }
}
