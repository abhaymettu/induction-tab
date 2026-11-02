// The live source: the trainer on this machine's GPU. Owns the loop, pause, and the pacing that keeps frames
// on time. Everything the page draws comes through on.frame.
import { createTrainer, getDevice } from '../engine/trainer.js';
import { COPY_THRESHOLD, trainLoop } from '../engine/run.js';
import { benchSeed, adapterInfo } from '../engine/bench.js';
import { copyProbe } from '../engine/lang.js';

const TRAIN_TO = 3000;
const GIVE_UP = 8000;
// Enough GPU time per submit to keep throughput, little enough that a frame is never held behind it.
export const CHUNK_TARGET_MS = 11;

export async function createLive(cfg, on) {
  const { adapter, device } = await getDevice();
  let lost = false;
  device.lost.then((info) => {
    if (info.reason === 'destroyed') return;
    lost = true;
    runId++;
    on.lost('device-lost');
  });
  // Work in flight when the device goes never settles from the page's point of view.
  const unlessLost = (p) => p.catch((e) => { if (lost) return new Promise(() => {}); throw e; });
  const t = await createTrainer(device, { ...cfg, seed: 1 });
  t.reset(1);

  const firstSeq = copyProbe(t.c.V, t.c.probeB, t.c.probeT).subarray(0, t.c.probeT);
  const pStaging = () => device.createBuffer({ size: t.probeOut.size + t.c.probeT * 4 + 4, usage: 0x01 | 0x08 });
  const E = t.c.evalEvery;

  let view = null;
  let loop = null;
  let runId = 0;
  let paused = false;
  let wake = null;
  let pausedMs = 0;
  let pausedAt = 0;
  let t0 = 0;
  let frozen = null;
  let msPerStep = 12;
  let chunk = 1;
  let lastChunkAt = 0;
  let lastChunkDone = 0;
  let evalMs = 280;
  let lastEvalAt = 0;
  let lastLoss = Math.log(t.c.V);
  let masks = [0, 0];
  let dirtySeed = false;

  const asFrame = (step, probe, state, seconds) => ({
    step,
    seconds,
    trainLoss: probe.trainLoss ?? lastLoss,
    copyAdv: probe.copyAdv,
    scores: probe.scores,
    seqLoss: probe.seqLoss,
    tPrev: state.tPrev,
    tEvent: state.tEvent,
    silenced: masks.some(Boolean),
  });

  const clockMs = () => {
    if (frozen !== null) return frozen;
    if (!t0) return 0;
    return (paused ? pausedAt : performance.now()) - t0 - pausedMs;
  };

  async function run(seed) {
    const id = ++runId;
    masks = [0, 0];
    t.setAblation(masks);
    if (dirtySeed || seed !== 1) t.reset(seed);
    dirtySeed = true;
    paused = false;
    pausedMs = 0;
    frozen = null;
    lastChunkAt = 0;
    lastEvalAt = 0;
    chunk = 1;
    t0 = performance.now();
    let announced = false;
    const res = await trainLoop(device, t, {
      budget: GIVE_UP,
      chunkSteps: () => chunk,
      shouldStop: (s, done) => id !== runId || (done > TRAIN_TO && s.tEvent !== null),
      onEval: (step, probe, state) => {
        if (id !== runId) return;
        const now = performance.now();
        if (lastEvalAt) evalMs += (now - lastEvalAt - evalMs) * 0.3;
        lastEvalAt = now;
        lastLoss = probe.trainLoss ?? lastLoss;
        view?.push(Math.min(600, evalMs));
        const isEvent = state.tEvent !== null && !announced;
        const seconds = isEvent ? state.wallEvent / 1000 - pausedMs / 1000 : clockMs() / 1000;
        if (isEvent) { announced = true; frozen = seconds * 1000; }
        on.frame(asFrame(step, probe, state, seconds), Math.min(600, evalMs));
        if (isEvent) on.event({ seconds, tEvent: state.tEvent, tPrev: state.tPrev });
      },
      onChunk: async (done) => {
        const now = performance.now();
        if (lastChunkAt && done > lastChunkDone) {
          const ms = (now - lastChunkAt) / (done - lastChunkDone);
          msPerStep += (ms - msPerStep) * 0.15;
          chunk = Math.max(1, Math.min(E, Math.round(CHUNK_TARGET_MS / msPerStep)));
        }
        while (paused && id === runId) await new Promise((r) => { wake = r; });
        lastChunkAt = performance.now();
        lastChunkDone = done;
      },
    });
    if (id === runId) {
      on.finished({ steps: res.steps - 1, gotEvent: res.tEvent !== null });
    }
  }

  // A probe with no training step, so a silenced head shows in the instruments at once.
  async function probeNow() {
    const staging = pStaging();
    const enc = device.createCommandEncoder();
    t.encodeSteps(enc, 0, true);
    enc.copyBufferToBuffer(t.probeOut, 0, staging, 0, t.probeOut.size);
    enc.copyBufferToBuffer(t.probeNet.lossBuf, 0, staging, t.probeOut.size, t.c.probeT * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(1);
    const probe = t.parseProbe(staging.getMappedRange().slice(0));
    staging.destroy();
    view?.push(180);
    return probe;
  }

  const info = adapterInfo(adapter);

  return {
    kind: 'live',
    info,
    tokens: firstSeq,
    c: t.c,
    t,
    device,
    adapter,
    threshold: COPY_THRESHOLD,
    canPromptEarly: true,
    canRestart: true,
    attach(renderer) { view = renderer; },
    clockMs,
    start(seed = 1) {
      loop = unlessLost(run(seed));
      return loop;
    },
    async restart(seed) {
      runId++;
      if (paused) { paused = false; wake?.(); }
      await loop;
      this.start(seed);
    },
    freeze(seconds) { frozen = seconds * 1000; },
    get paused() { return paused; },
    pause() {
      if (paused) return;
      paused = true;
      pausedAt = performance.now();
    },
    resume() {
      if (!paused) return;
      pausedMs += performance.now() - pausedAt;
      paused = false;
      wake?.();
    },
    predict: (ids) => unlessLost(t.predict(ids)),
    async silence(next) {
      masks = next;
      t.setAblation(masks);
      const probe = await unlessLost(probeNow());
      return { copyAdv: probe.copyAdv, scores: probe.scores, seqLoss: probe.seqLoss, silenced: masks.some(Boolean) };
    },
    async benchmark(seed) {
      runId++;
      if (paused) { paused = false; wake?.(); }
      await loop;
      dirtySeed = true;
      return benchSeed(device, t, seed, GIVE_UP);
    },
  };
}
