// The recorded source, for browsers without WebGPU: a real run from tools/record.mjs played in the same
// composition, with the run's final weights answering prompts through the CPU forward pass.
import { createCpuModel } from '../engine/cpu.js';
import { copyProbe } from '../engine/lang.js';
import { resolve } from '../engine/layout.js';
import { COPY_THRESHOLD } from '../engine/run.js';

const PROBE_SEQS = 4;
const ROOT = 'assets/recording/';

async function gunzip(response) {
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function createRecording(on) {
  const [meta, framesRes, weightsRes] = await Promise.all([
    fetch(`${ROOT}meta.json`).then((r) => { if (!r.ok) throw new Error('no recording'); return r.json(); }),
    fetch(`${ROOT}frames.bin.gz`),
    fetch(`${ROOT}weights.bin`),
  ]);
  const frames = await gunzip(framesRes);
  const weights = new Float32Array(await weightsRes.arrayBuffer());
  const c = resolve(meta.cfg);
  const model = createCpuModel(meta.cfg, weights);
  const frameBytes = c.L * c.H * c.T * c.T;
  const frameAt = new Map(meta.frames.map((step, i) => [step, frames.subarray(i * frameBytes, (i + 1) * frameBytes)]));
  const probeBatch = copyProbe(c.V, c.probeB, c.probeT).subarray(0, PROBE_SEQS * c.probeT);

  let view = null;
  let timer = 0;
  let index = 0;
  let paused = false;
  let baseMs = 0;
  let startedAt = 0;
  let frozen = null;
  let finished = false;
  let masks = [0, 0];

  const clockMs = () => {
    if (frozen !== null) return frozen;
    if (!startedAt) return 0;
    return paused ? baseMs : baseMs + performance.now() - startedAt;
  };

  const seqLossOf = (ev) => {
    const loss = new Float32Array(c.T);
    ev.seqLoss.forEach((x, i) => { loss[c.T / 2 - 1 + i] = x; });
    return loss;
  };

  function show(i) {
    const ev = meta.evals[i];
    const next = meta.evals[i + 1];
    const gap = next ? next.ms - ev.ms : 280;
    const bytes = frameAt.get(ev.step);
    if (bytes) {
      const nextFrame = meta.frames[meta.frames.indexOf(ev.step) + 1];
      const until = nextFrame === undefined ? ev : meta.evals.find((e) => e.step === nextFrame);
      view?.push(bytes, Math.max(gap, until.ms - ev.ms));
    }
    const isEvent = meta.tEvent === ev.step;
    const seconds = ev.ms / 1000;
    if (isEvent) frozen = ev.ms;
    on.frame({
      step: ev.step,
      seconds,
      trainLoss: ev.trainLoss,
      copyAdv: ev.copyAdv,
      scores: ev.scores,
      seqLoss: seqLossOf(ev),
      tPrev: ev.step >= meta.tPrev ? meta.tPrev : null,
      tEvent: ev.step >= meta.tEvent ? meta.tEvent : null,
      silenced: false,
    }, gap);
    if (isEvent) on.event({ seconds, tEvent: meta.tEvent, tPrev: meta.tPrev });
  }

  function schedule() {
    if (paused || finished) return;
    if (index >= meta.evals.length) {
      finished = true;
      on.finished({ steps: meta.evals[meta.evals.length - 1].step + 1, gotEvent: true });
      return;
    }
    const due = meta.evals[index].ms - clockForSchedule();
    timer = setTimeout(() => {
      show(index);
      index += 1;
      schedule();
    }, Math.max(0, due));
  }
  const clockForSchedule = () => (paused ? baseMs : baseMs + performance.now() - startedAt);

  function stopAtEnd() {
    clearTimeout(timer);
    if (finished) return;
    show(meta.evals.length - 1);
    index = meta.evals.length;
    finished = true;
    on.finished({ steps: meta.evals[meta.evals.length - 1].step + 1, gotEvent: true });
  }

  return {
    kind: 'recording',
    info: meta.adapter,
    machine: meta.machine,
    browser: meta.browser,
    tokens: copyProbe(c.V, c.probeB, c.probeT).subarray(0, c.probeT),
    c,
    threshold: COPY_THRESHOLD,
    canPromptEarly: false,
    canRestart: false,
    attach(renderer) { view = renderer; },
    clockMs,
    start() {
      clearTimeout(timer);
      index = 0;
      paused = false;
      finished = false;
      frozen = null;
      masks = [0, 0];
      baseMs = 0;
      startedAt = performance.now();
      schedule();
    },
    async restart() { this.start(); },
    get paused() { return paused; },
    pause() {
      if (paused || finished) return;
      baseMs += performance.now() - startedAt;
      paused = true;
      clearTimeout(timer);
    },
    resume() {
      if (!paused) return;
      paused = false;
      startedAt = performance.now();
      schedule();
    },
    predict: async (ids) => {
      await new Promise((r) => setTimeout(r));
      return model.predict(ids, masks);
    },
    // Silencing needs the final weights the CPU holds, so playback jumps to its end first.
    async silence(next) {
      stopAtEnd();
      masks = next;
      await new Promise((r) => setTimeout(r));
      const p = model.probe(probeBatch, PROBE_SEQS, masks);
      const bytes = new Uint8Array(frameBytes);
      for (let i = 0; i < frameBytes; i++) bytes[i] = Math.round(Math.min(1, p.view[i]) * 255);
      view?.push(bytes, 180);
      return { copyAdv: p.copyAdv, scores: p.scores, seqLoss: p.seqLoss, silenced: masks.some(Boolean) };
    },
  };
}
