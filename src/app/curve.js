// Training loss and copy advantage on one axis in nats, with the chance line and the 2 nat threshold named in
// words. Series differ in lightness and weight and are labeled at their ends. The precursor and the event are
// marked on the plot where they happened.

const NS = 'http://www.w3.org/2000/svg';
const CHANCE = Math.log(96);
const Y_MAX = 5;
const MIN_DOMAIN = 300;
const CHAR = 7.2;
const LINE = 15;
const MARK_ROWS = [3.55, 3.0, 2.55, 1.55, 1.05, 0.55];

const el = (name, cls, parent) => {
  const n = document.createElementNS(NS, name);
  if (cls) n.setAttribute('class', cls);
  parent.append(n);
  return n;
};
const set = (n, attrs) => { for (const k in attrs) n.setAttribute(k, attrs[k]); };
const overlaps = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

export function createCurve(host, { threshold }) {
  const svg = el('svg', 'curve-svg', host);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Training loss and copy advantage in nats against training step');
  const axis = el('line', 'axis', svg);
  const chance = el('line', 'ref dotted', svg);
  const thresh = el('line', 'ref dashed', svg);
  const chanceText = el('text', 'note', svg);
  chanceText.textContent = 'chance, ln 96 = 4.56';
  const threshText = el('text', 'note', svg);
  threshText.textContent = `${threshold} nats, the event`;
  const stepEnd = el('text', 'note', svg);
  const stepStart = el('text', 'note', svg);
  stepStart.textContent = 'step 0';

  const markText = () => {
    const t = el('text', 'mark', svg);
    t.word = el('tspan', '', t);
    t.step = el('tspan', 'note', t);
    return t;
  };
  const marks = {
    prev: { line: el('line', 'marker', svg), text: markText(), word: 'precursor', step: null, place: null },
    event: { line: el('line', 'marker strong', svg), text: markText(), word: 'event', step: null, place: null },
  };

  const loss = el('polyline', 'series loss', svg);
  const adv = el('path', 'series adv', svg);
  const advMuted = el('path', 'series adv silenced', svg);
  const dot = el('circle', 'dot', svg);
  dot.setAttribute('r', '4');
  const lossLabel = el('text', 'end', svg);
  const advLabel = el('text', 'end strong', svg);
  const cross = el('line', 'cross', svg);
  const tip = el('text', 'tip', svg);

  let w = 0;
  let h = 0;
  let points = [];
  let domain = MIN_DOMAIN;
  let target = MIN_DOMAIN;
  let head = 0;
  let headFrom = 0;
  let from = 0;
  let span = 280;
  let finished = false;
  let hover = null;
  let last = 0;
  const m = { l: 2, r: 2, t: 8, b: 22 };

  const X = (s) => m.l + (Math.min(s, domain) / domain) * (w - m.l - m.r);
  const Y = (v) => m.t + (1 - Math.max(0, Math.min(Y_MAX, v)) / Y_MAX) * (h - m.t - m.b);
  const hline = (n, y) => set(n, { x1: m.l, x2: w - m.r, y1: y, y2: y });
  const vline = (n, x) => set(n, { x1: x, x2: x, y1: m.t, y2: h - m.b });

  function layout() {
    const box = host.getBoundingClientRect();
    if (!box.width) return;
    w = box.width;
    h = box.height;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    hline(axis, Y(0));
    hline(chance, Y(CHANCE));
    hline(thresh, Y(threshold));
    set(chanceText, { x: w - m.r, y: Y(CHANCE) - 5, 'text-anchor': 'end' });
    set(threshText, { x: w / 2, y: Y(threshold) - 5, 'text-anchor': 'middle' });
    set(stepStart, { x: m.l, y: h - 5 });
    set(stepEnd, { x: w - m.r, y: h - 5, 'text-anchor': 'end' });
    marks.prev.place = null;
    marks.event.place = null;
    draw(performance.now(), true);
  }
  new ResizeObserver(layout).observe(host);

  // Draws up to the eased head so the line extends between evals instead of jumping.
  function draw(now, force = false) {
    if (!w || !points.length) {
      if (force) {
        loss.setAttribute('points', '');
        adv.setAttribute('d', '');
        advMuted.setAttribute('d', '');
      }
      return false;
    }
    const t = Math.min(1, (now - from) / span);
    const moving = t < 1 || Math.abs(domain - target) > 0.5;
    if (!moving && !force && now - last < 200) return false;
    last = now;
    domain += (target - domain) * 0.08;
    if (Math.abs(domain - target) <= 0.5) domain = target;

    const lastP = points[points.length - 1];
    const prevP = points[points.length - 2] ?? lastP;
    head = headFrom + (lastP.step - headFrom) * t;
    const ease = lastP.step === prevP.step ? 1 : Math.max(0, Math.min(1, (head - prevP.step) / (lastP.step - prevP.step)));
    const headLoss = prevP.loss + (lastP.loss - prevP.loss) * ease;
    const headAdv = prevP.adv + (lastP.adv - prevP.adv) * ease;

    let solidD = '';
    let mutedD = '';
    let lossPts = '';
    let previous = null;
    const visible = points.filter((p) => p.step < head);
    visible.push({ step: head, loss: headLoss, adv: headAdv, silenced: lastP.silenced });
    for (const p of visible) {
      const x = X(p.step).toFixed(1);
      lossPts += `${x},${Y(p.loss).toFixed(1)} `;
      const y = Y(p.adv).toFixed(1);
      if (previous) {
        const seg = ` L${x} ${y}`;
        const start = `M${previous.x} ${previous.y}`;
        if (p.silenced) mutedD += `${previous.muted ? '' : start}${seg}`;
        else solidD += `${previous.muted === false ? '' : start}${seg}`;
      }
      previous = { x, y, muted: previous ? Boolean(p.silenced) : null };
    }
    adv.setAttribute('d', solidD);
    advMuted.setAttribute('d', mutedD);
    loss.setAttribute('points', lossPts);
    set(dot, { cx: X(head).toFixed(1), cy: Y(headAdv).toFixed(1) });
    dot.classList.toggle('silenced', Boolean(lastP.silenced));

    const ends = placeEndLabels(lastP);
    stepEnd.textContent = `step ${Math.round(domain / 100) * 100}`;
    placeMark(marks.prev, ends);
    placeMark(marks.event, ends);
    drawHover();
    return moving;
  }

  // Labels sit clear of the noisy tail of their own line: the loss label under it, and the copy advantage
  // label under it once it is the upper line, over it while it is still near zero.
  function placeEndLabels(lastP) {
    const right = w - m.r - 8;
    const short = w < 420;
    const lossWords = `${short ? 'loss' : 'training loss'} ${lastP.loss.toFixed(2)}`;
    const advWords = `${short ? 'copy adv.' : 'copy advantage'} ${lastP.adv.toFixed(2)}`;
    const reach = right - CHAR * Math.max(lossWords.length, advWords.length);
    let lossLow = Y(lastP.loss);
    let advLow = Y(lastP.adv);
    let advHigh = advLow;
    if (!lastP.silenced) {
      for (const p of points) {
        if (p.step > head || X(p.step) < reach) continue;
        lossLow = Math.max(lossLow, Y(p.loss));
        advLow = Math.max(advLow, Y(p.adv));
        advHigh = Math.min(advHigh, Y(p.adv));
      }
    }
    const bottom = Y(0) - 4;
    let b = Math.min(bottom, lossLow + 16);
    let a = lastP.adv > lastP.loss ? Math.min(bottom, advLow + 17) : advHigh - 8;
    if (Math.abs(a - b) < 16) {
      if (a <= b) a = b - 16; else a = b + 16;
    }
    set(advLabel, { x: right, y: a.toFixed(1), 'text-anchor': 'end' });
    set(lossLabel, { x: right, y: b.toFixed(1), 'text-anchor': 'end' });
    advLabel.textContent = advWords;
    lossLabel.textContent = lossWords;
    const box = (y, words) => ({ x0: right - CHAR * words.length, x1: right, y0: y - 12, y1: y + 4 });
    return [box(a, advWords), box(b, lossWords)];
  }

  // A mark's words go wherever nothing else is drawn right now, and stay there until something runs into them.
  function placeMark(mark, ends) {
    const { line, text, step, word } = mark;
    const shown = step !== null && step <= head;
    line.style.display = shown ? '' : 'none';
    text.style.display = shown ? '' : 'none';
    if (!shown) return;
    const x = X(step);
    vline(line, x);
    const width = CHAR * Math.max(word.length, `step ${step}`.length);
    const rect = (side, row) => {
      const y = Y(row);
      const x0 = side < 0 ? x - 6 - width : x + 6;
      return { x0, x1: x0 + width, y0: y - 12, y1: y + LINE + 4 };
    };
    const hit = (r) => {
      if (r.x0 < m.l || r.x1 > w - m.r) return true;
      if (ends.some((e) => overlaps(e, r))) return true;
      const other = mark === marks.prev ? marks.event : marks.prev;
      if (other.place && other.step !== null && other.step <= head) {
        const ox = X(other.step);
        const oy = Y(other.place.row);
        const ow = CHAR * Math.max(other.word.length, `step ${other.step}`.length);
        const ox0 = other.place.side < 0 ? ox - 6 - ow : ox + 6;
        if (overlaps({ x0: ox0, x1: ox0 + ow, y0: oy - 12, y1: oy + LINE + 4 }, r)) return true;
      }
      for (const p of points) {
        if (p.step > head) break;
        const px = X(p.step);
        if (px < r.x0 - 3 || px > r.x1 + 3) continue;
        const ya = Y(p.adv);
        const yl = Y(p.loss);
        if ((ya > r.y0 - 2 && ya < r.y1 + 2) || (yl > r.y0 - 2 && yl < r.y1 + 2)) return true;
      }
      return false;
    };
    if (!mark.place || hit(rect(mark.place.side, mark.place.row))) {
      let found = null;
      for (const row of MARK_ROWS) {
        for (const side of [mark === marks.prev ? -1 : 1, mark === marks.prev ? 1 : -1]) {
          if (!hit(rect(side, row))) { found = { side, row }; break; }
        }
        if (found) break;
      }
      mark.place = found ?? mark.place ?? { side: 1, row: MARK_ROWS[0] };
    }
    const { side, row } = mark.place;
    const tx = side < 0 ? x - 6 : x + 6;
    const y = Y(row);
    set(text, { 'text-anchor': side < 0 ? 'end' : 'start' });
    set(text.word, { x: tx, y });
    set(text.step, { x: tx, y: y + LINE });
    text.word.textContent = word;
    text.step.textContent = `step ${step}`;
  }

  function drawHover() {
    if (!hover || !points.length) {
      cross.style.display = 'none';
      tip.style.display = 'none';
      return;
    }
    let best = points[0];
    for (const p of points) if (Math.abs(p.step - hover) < Math.abs(best.step - hover)) best = p;
    cross.style.display = '';
    tip.style.display = '';
    vline(cross, X(best.step));
    set(tip, { x: m.l, y: h - 5 });
    const words = w < 420 ? 'adv.' : 'copy advantage';
    tip.textContent = `step ${best.step}: loss ${best.loss.toFixed(2)}, ${words} ${best.adv.toFixed(2)}`;
  }

  svg.addEventListener('pointermove', (e) => {
    const box = svg.getBoundingClientRect();
    hover = Math.max(0, Math.min(1, (e.clientX - box.left - m.l) / (w - m.l - m.r))) * domain;
    stepStart.style.display = 'none';
    stepEnd.style.display = 'none';
    drawHover();
  });
  svg.addEventListener('pointerleave', () => {
    hover = null;
    stepStart.style.display = '';
    stepEnd.style.display = '';
    drawHover();
  });

  return {
    push(point, ms, state) {
      headFrom = points.length ? points[points.length - 1].step : 0;
      points.push(point);
      marks.prev.step = state.tPrev;
      marks.event.step = state.tEvent;
      target = finished ? point.step : Math.max(MIN_DOMAIN, point.step * 1.08);
      from = performance.now();
      span = ms;
    },
    // The reading at the head changed without a training step: heads silenced after the run, or brought back.
    replaceLast(value, silenced) {
      const p = points[points.length - 1];
      if (!p) return;
      p.adv = value;
      p.silenced = silenced;
      draw(performance.now(), true);
    },
    finish() {
      finished = true;
      const p = points[points.length - 1];
      if (p) target = p.step;
    },
    draw,
    reset() {
      points = [];
      domain = target = MIN_DOMAIN;
      head = headFrom = 0;
      finished = false;
      lossLabel.textContent = '';
      advLabel.textContent = '';
      stepEnd.textContent = `step ${MIN_DOMAIN}`;
      for (const mark of Object.values(marks)) {
        mark.step = null;
        mark.place = null;
        mark.line.style.display = 'none';
        mark.text.style.display = 'none';
      }
      dot.setAttribute('cx', -20);
      draw(performance.now(), true);
    },
    layout,
  };
}
