// The composition. A source (the live GPU trainer or the recording) produces frames; this file turns frames into
// the page and the reader's actions back into source calls. The page's whole state is `phase` plus `arrived`.
import { createCurve } from './curve.js';
import { createCpuMaps, createGpuMaps, createMaps } from './attention.js';
import { createStream } from './stream.js';
import { createSteer } from './steer.js';
import { createLive } from './live.js';
import { createRecording } from './recording.js';
import { loadRecords, loadStudy } from './records.js';
import { BENCH_SEEDS, buildLog, issueUrl, medianSeconds } from '../engine/bench.js';

const $ = (id) => document.getElementById(id);
const root = document.documentElement;
const params = new URLSearchParams(location.search);

const TRAIN_TO = 3000;
const CHIP_EVERY = 400;
const REFRESH_MS = 900;

let source = null;
let stream = null;
let cfg = null;
let phase = 'boot';
let arrived = false;
let queuedStart = false;
let result = null;
let lastFrame = null;
let beforeSilence = null;
let lastChipStep = -1;
let lastRefresh = 0;
let currentSeed = 1;
let raf = 0;

const setPhase = (p) => {
  phase = p;
  root.dataset.phase = p;
  $('pause').textContent = p === 'paused' ? 'Resume' : 'Pause';
  $('pause').hidden = p === 'done' || p === 'bench';
  $('reseed').disabled = p === 'bench';
  $('again').disabled = p === 'bench';
  $('benchBtn').disabled = p === 'bench';
  $('benchBtn').style.display = source?.kind === 'recording' ? 'none' : '';
};

const setArrived = (on) => {
  arrived = on;
  if (on) root.dataset.arrived = '';
  else delete root.dataset.arrived;
};

const vendorName = (info) => {
  const v = (info.vendor || '').trim();
  if (!v) return (info.description || '').trim();
  return v[0].toUpperCase() + v.slice(1);
};
const gpuLine = (info) => [vendorName(info), info.architecture].filter(Boolean).join(' ') || 'GPU';

function explainReason(reason) {
  if (reason === 'device-lost') return 'This page lost its connection to your GPU, so training stopped. Reloading usually brings it back.';
  const why = reason === 'no-adapter' ? 'This browser has WebGPU but could not get a GPU from it' : 'This browser does not offer WebGPU';
  return `${why}, so it cannot train the network here. Try a recent Chrome or Edge, or another browser with WebGPU turned on.`;
}

function say(text, sub) {
  if ($('say').textContent !== text) $('say').textContent = text;
  if (sub !== undefined && $('sub').textContent !== sub) $('sub').textContent = sub;
}

const resultSentence = (seconds) => {
  const s = seconds.toFixed(1);
  if (source.kind === 'recording') return `Induction in ${s} seconds, recorded on an ${source.machine}.`;
  const name = vendorName(source.info);
  return `Induction in ${s} seconds on your ${name ? `${name} ` : ''}GPU.`;
};

function narrate(f) {
  const stepText = `Step ${f.step} of ${TRAIN_TO}.`;
  const adv = `Copy advantage ${f.copyAdv.toFixed(2)} nats.`;
  if (arrived) {
    say(resultSentence(result.seconds), `${stepText} ${adv}`);
    return;
  }
  const sub = `${stepText} Loss ${f.trainLoss.toFixed(2)}. ${adv}`;
  if (f.copyAdv >= 0.6) say('The second pass is getting easier than the first. Watch the lower row.', sub);
  else if (f.tPrev !== null) say('Layer 0 has started to read the previous character. That is the first half of the trick.', sub);
  else say('It is guessing. A random character cannot be predicted, so the loss sits at chance.', sub);
}

function setClock(seconds) {
  const [whole, tenth] = seconds.toFixed(1).split('.');
  if ($('clockWhole').textContent !== whole) $('clockWhole').textContent = whole;
  if ($('clockTenth').textContent !== tenth) $('clockTenth').textContent = tenth;
}
const paintClock = () => setClock(source.clockMs() / 1000);

const curve = createCurve($('curve'), { threshold: 2 });

function tick(now) {
  raf = 0;
  if (!source) return;
  if (phase === 'bench') return;
  paintClock();
  const a = maps.renderer ? maps.renderer.draw(now) : false;
  const b = curve.draw(now);
  if (phase === 'training' || a || b) raf = requestAnimationFrame(tick);
}
const kick = () => { if (!raf) raf = requestAnimationFrame(tick); };

const maps = createMaps($('maps'), {
  canvas: $('canvas'),
  explain: $('explain'),
  T: 128, H: 4, L: 2,
  getEnabled: () => arrived,
  onSilence: (masks) => silence(masks),
  onLayout: () => kick(),
});

let silenceGen = 0;
async function silence(masks) {
  const mine = ++silenceGen;
  const any = masks.some(Boolean);
  const r = await source.silence(masks);
  if (mine !== silenceGen) return;
  if (any && beforeSilence === null && lastFrame) beforeSilence = lastFrame.copyAdv;
  stream.update(r.seqLoss);
  maps.update(r.scores);
  curve.replaceLast(r.copyAdv, any);
  if (lastFrame) lastFrame = { ...lastFrame, copyAdv: r.copyAdv, silenced: any };
  if (any) {
    say(resultSentence(result.seconds), `Heads silenced. Copy advantage ${r.copyAdv.toFixed(2)} nats, was ${(beforeSilence ?? r.copyAdv).toFixed(2)}. Training is untouched.`);
  } else {
    beforeSilence = null;
    say(resultSentence(result.seconds), `All heads back. Copy advantage ${r.copyAdv.toFixed(2)} nats.`);
  }
  document.querySelectorAll('[data-layer]').forEach((b) => {
    const l = Number(b.dataset.layer);
    b.setAttribute('aria-pressed', String(masks[l] === 15));
    b.textContent = masks[l] === 15 ? `Bring back layer ${l}` : `Silence layer ${l}`;
  });
  steer.refresh();
  steer.measureChips();
  kick();
}

const steer = createSteer({
  box: $('box'),
  typed: $('typed'),
  ghost: $('ghost'),
  next: $('next'),
  chips: $('chips'),
  predict: (ids) => source.predict(ids),
  help: (count) => {
    if (!arrived) return;
    $('hint').textContent = count ? '' : (maps.masks.some(Boolean) ? 'With these heads silent, none of the test prompts come out right.' : 'Too weak to continue any test prompt yet. It is still improving.');
  },
});

function onFrame(f, ms) {
  lastFrame = f;
  stream.update(f.seqLoss);
  maps.update(f.scores);
  curve.push({ step: f.step, loss: f.trainLoss, adv: f.copyAdv, silenced: f.silenced }, ms, { tPrev: f.tPrev, tEvent: f.tEvent });
  narrate(f);
  if (f.silenced) {
    $('sub').textContent = `Step ${f.step} of ${TRAIN_TO}. Heads silenced. Copy advantage ${f.copyAdv.toFixed(2)} nats. Training is untouched.`;
  }
  const now = performance.now();
  if (now - lastRefresh > REFRESH_MS && $('box').value) {
    lastRefresh = now;
    steer.refresh();
  }
  if (arrived && f.step - lastChipStep >= CHIP_EVERY) {
    lastChipStep = f.step;
    steer.measureChips();
  }
  kick();
}

function onEvent({ seconds }) {
  result = { seconds };
  setArrived(true);
  lastChipStep = lastFrame.step;
  paintClock();
  narrate(lastFrame);
  steer.setEnabled(true, placeholders().after);
  $('hint').textContent = 'Measuring which prompts it can continue.';
  setTimeout(() => steer.measureChips(), 600);
  $('resultHead').textContent = resultSentence(seconds);
  $('resultBody').textContent = source.kind === 'recording'
    ? 'That number belongs to the recorded machine. A browser with WebGPU would measure its own.'
    : 'Each seed starts from different weights, so the event lands somewhere else. New seed tries one at random.';
  kick();
}

function onFinished({ gotEvent, steps }) {
  setPhase('done');
  curve.finish();
  if (!gotEvent) {
    say(`No event in ${steps} steps with this seed.`, 'That can happen. Try a new seed.');
    return;
  }
  const stopped = source.kind === 'recording' ? 'End of the recording' : `Stopped at step ${steps} to save your battery`;
  $('sub').textContent = `${stopped}. ${lastFrame.silenced ? 'Heads silenced. ' : ''}Copy advantage ${lastFrame.copyAdv.toFixed(2)} nats.`;
  steer.measureChips();
  kick();
}

async function onLost(reason) {
  if (!source || source.kind === 'recording') return;
  source = null;
  resetAfterLoss();
  await useRecording(reason);
}

function resetAfterLoss() {
  setArrived(false);
  curve.reset();
  maps.reset();
  $('box').value = '';
  $('typed').textContent = '';
  $('ghost').replaceChildren();
  say('', '');
}

const on = { frame: onFrame, event: onEvent, finished: onFinished, lost: onLost };

function resetView() {
  setArrived(false);
  result = null;
  lastFrame = null;
  beforeSilence = null;
  lastChipStep = -1;
  stream.clear();
  maps.reset();
  curve.reset();
  steer.reset();
  setClock(0);
  $('hint').textContent = '';
  say('', '');
  document.querySelectorAll('[data-layer]').forEach((b) => {
    b.setAttribute('aria-pressed', 'false');
    b.textContent = `Silence layer ${b.dataset.layer}`;
  });
  $('caption').textContent = captionText();
  steer.setEnabled(source.canPromptEarly, source.canPromptEarly ? placeholders().before : placeholders().closed);
}

const isNarrow = () => matchMedia('(max-width: 760px)').matches;
const captionText = () => (isNarrow()
  ? 'A random sequence and its exact repeat. Light and bar show how sure the model is.'
  : 'A random sequence, then its exact repeat, column for column. Brightness and the bar under each character show how sure the model is of it.');
// The third row is 24 columns wide on a phone, and a placeholder cannot wrap between words there.
const placeholders = () => (isNarrow()
  ? { before: 'Type. Noise for now.', after: 'Type a repeating pattern', closed: 'Opens at the event.' }
  : { before: 'Type here if you like. For now it can only answer with noise.', after: 'Type something that repeats. It will carry on.', closed: 'The text box opens at the event.' });

function begin() {
  if (!source) { queuedStart = true; $('start').textContent = 'Starting'; return; }
  resetView();
  setPhase('training');
  say('It is guessing. A random character cannot be predicted, so the loss sits at chance.', `Step 0 of ${TRAIN_TO}.`);
  source.start(currentSeed);
  kick();
}

async function restart() {
  if (!source) return;
  currentSeed = source.kind === 'recording' ? 1 : 1 + Math.floor(Math.random() * 1e6);
  resetView();
  maps.renderer?.clear();
  setPhase('training');
  say('It is guessing. A random character cannot be predicted, so the loss sits at chance.', `Seed ${currentSeed}. Step 0 of ${TRAIN_TO}.`);
  await source.restart(currentSeed);
  kick();
}

// Before start the two rows wait at the foot of the screen, under the sentence, then rise into place.
function measureCover() {
  const band = $('band');
  const coverBottom = $('cover').offsetTop + $('cover').offsetHeight;
  const rows = $('scroller').getBoundingClientRect().bottom - band.getBoundingClientRect().top;
  const foot = innerHeight - rows - 64;
  band.style.setProperty('--shift', `${Math.max(0, Math.max(coverBottom + 40, foot) - band.offsetTop)}px`);
}

async function useRecording(reason) {
  const fresh = document.createElement('canvas');
  fresh.id = 'canvas';
  fresh.setAttribute('aria-label', $('canvas').getAttribute('aria-label'));
  $('canvas').replaceWith(fresh);
  root.dataset.source = 'recording';
  setArrived(false);
  try {
    source = await createRecording(on);
  } catch {
    source = null;
    $('gpu').textContent = 'No GPU';
    $('lede').textContent = explainReason(reason);
    $('start').hidden = true;
    return;
  }
  const label = `${source.machine}, ${source.browser}`;
  $('gpu').textContent = `Recording from ${label}`;
  $('lede').textContent = `${explainReason(reason)} Here is a real run, recorded on an ${source.machine}, and the text box keeps working with its real weights on your CPU.`;
  $('start').textContent = 'Play the recording';
  $('clockUnit').textContent = 'seconds, recorded';
  $('reseed').textContent = 'Play again';
  maps.use(createCpuMaps(fresh, { T: 128, H: 4, L: 2 }), fresh);
  source.attach(maps.renderer);
  stream = createStream($('stream'), source.tokens, { onHover });
  stream.clear();
  setPhase('ready');
  maps.resize();
  measureCover();
  steer.setEnabled(false, placeholders().closed);
  $('again').textContent = 'Play the recording again';
  $('resultHead').textContent = 'This is a recording.';
  $('resultBody').textContent = `It came from an ${source.machine}. A browser with WebGPU trains it live on your own machine.`;
  if (queuedStart) { queuedStart = false; begin(); }
}

function onHover(info) {
  if (!info) { $('caption').textContent = captionText(); return; }
  const shown = info.char === ' ' ? 'space' : info.char;
  $('caption').textContent = `Column ${info.column}: ${shown}. The model gives it ${(info.p * 100).toFixed(info.p < 0.1 ? 1 : 0)}%, chance is about 1%.`;
}

async function runBench() {
  if (!source || source.kind !== 'live') return;
  const runs = [];
  resetView();
  maps.renderer?.clear();
  setPhase('bench');
  steer.setEnabled(false, placeholders().closed);
  $('benchOut').hidden = true;
  $('after').scrollIntoView({ block: 'start' });
  for (const seed of BENCH_SEEDS) {
    say(`Benchmark, seed ${seed} of ${BENCH_SEEDS.length}.`, 'The page is holding still so the numbers are clean.');
    $('resultHead').textContent = `Benchmark, seed ${seed} of ${BENCH_SEEDS.length}.`;
    $('resultBody').textContent = 'The page is holding still so the numbers are clean.';
    runs.push(await source.benchmark(seed));
  }
  const log = buildLog({ cfg, adapter: source.adapter, userAgent: navigator.userAgent, runs });
  const median = medianSeconds(runs);
  const events = runs.filter((r) => r.wallEventMs !== null).length;
  const headline = median === null ? 'No event on any seed.' : `Median ${median.toFixed(2)} seconds to the event, ${events} of ${runs.length} seeds, on your ${gpuLine(source.info)}.`;
  $('benchMedian').textContent = headline;
  $('benchLog').textContent = JSON.stringify(log);
  $('issueLink').href = issueUrl(log);
  $('benchOut').hidden = false;
  $('resultHead').textContent = 'Benchmark finished.';
  $('resultBody').textContent = 'To submit it, open the prefilled issue and paste the log.';

  // The weights are the last seed's, just past the event, so the steering surface is still honest.
  result = { seconds: runs[runs.length - 1].wallEventMs === null ? 0 : runs[runs.length - 1].wallEventMs / 1000 };
  setArrived(runs[runs.length - 1].wallEventMs !== null);
  setPhase('done');
  const r = await source.silence([0, 0]);
  stream.update(r.seqLoss);
  maps.update(r.scores);
  lastFrame = { copyAdv: r.copyAdv };
  source.freeze(result.seconds);
  say(headline, 'The model here is seed 5, just past the event.');
  steer.setEnabled(true, placeholders().after);
  steer.measureChips();
  kick();
}

$('start').addEventListener('click', begin);
$('pause').addEventListener('click', () => {
  if (phase === 'training') { source.pause(); setPhase('paused'); paintClock(); }
  else if (phase === 'paused') { source.resume(); setPhase('training'); kick(); }
});
$('reseed').addEventListener('click', restart);
$('again').addEventListener('click', () => {
  if (phase === 'ready' || phase === 'boot') begin(); else restart();
  $('live').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
});
$('benchBtn').addEventListener('click', runBench);
$('restore').addEventListener('click', () => { if (arrived) maps.restore(); });
document.querySelectorAll('[data-layer]').forEach((b) => {
  b.addEventListener('click', () => { if (arrived) maps.toggleLayer(Number(b.dataset.layer)); });
});
$('copyLog').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('benchLog').textContent);
    $('copyLog').textContent = 'Copied';
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('benchLog'));
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    $('copyLog').textContent = 'Selected, press copy';
  }
  setTimeout(() => { $('copyLog').textContent = 'Copy log'; }, 1600);
});
matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
  maps.renderer?.recolor();
  kick();
});
addEventListener('resize', () => {
  if (phase === 'boot' || phase === 'ready') measureCover();
  if (!$('caption').matches(':hover')) $('caption').textContent = captionText();
});
document.fonts?.ready.then(() => {
  if (phase === 'boot' || phase === 'ready') measureCover();
  curve.layout();
});

async function boot() {
  $('caption').textContent = captionText();
  measureCover();
  loadRecords();
  loadStudy();
  cfg = await fetch('configs/tab.json').then((r) => r.json());
  let reason = null;
  try {
    if (!navigator.gpu) throw new Error('no-webgpu');
    source = await createLive(cfg, on);
  } catch (e) {
    reason = e.message === 'no-adapter' ? 'no-adapter' : 'no-webgpu';
  }
  if (reason) {
    await useRecording(reason);
    return;
  }
  root.dataset.source = 'live';
  $('gpu').textContent = gpuLine(source.info);
  maps.use(createGpuMaps(source.device, $('canvas'), source.t.view, { T: 128, H: 4, L: 2 }), $('canvas'));
  source.attach(maps.renderer);
  stream = createStream($('stream'), source.tokens, { onHover });
  stream.clear();
  setPhase('ready');
  maps.resize();
  measureCover();
  if (params.has('bench')) { runBench(); return; }
  if (queuedStart) { queuedStart = false; begin(); }
}

boot();
