// Drives a trainer through the induction event. Shared by the page, the speedrun check and the records.
// The instruments are those of 2609.19000: the event is copy advantage >= 2 nats on two consecutive evals,
// reported at the second, and the precursor alarm is the first eval with early-layer previous-token score >= 0.10.

export const COPY_THRESHOLD = 2;
export const PREV_THRESHOLD = 0.1;

export function onsetTracker(c) {
  const early = Math.max(1, Math.floor(c.L / 2));
  const state = { tEvent: null, tPrev: null, wallEvent: null, wallPrev: null };
  let lastOk = false;
  return {
    state,
    push(step, probe, wallMs) {
      let prev = 0;
      for (let l = 0; l < early; l++) for (const h of probe.scores[l]) prev = Math.max(prev, h.prev);
      if (state.tPrev === null && prev >= PREV_THRESHOLD) { state.tPrev = step; state.wallPrev = wallMs; }
      const ok = probe.copyAdv >= COPY_THRESHOLD;
      if (state.tEvent === null && ok && lastOk) { state.tEvent = step; state.wallEvent = wallMs; }
      lastOk = ok;
      return state;
    },
  };
}

// Steps are submitted in chunks so the page keeps breathing. A chunk ends on an eval step when one is due,
// with the probe and its readback in the same submit. Eval steps are 0, E, 2E, ... counting from zero.
export async function trainLoop(device, t, { budget, chunkSteps, onEval, onChunk, shouldStop }) {
  const E = t.c.evalEvery;
  const staging = device.createBuffer({ size: t.probeOut.size, usage: 0x01 | 0x08 });
  const t0 = performance.now();
  const tracker = onsetTracker(t.c);
  let done = 0; // steps completed; the eval for step s runs after s + 1 steps
  while (done < budget && !(shouldStop && shouldStop(tracker.state, done))) {
    const n = done === 0 ? 1 : Math.min(chunkSteps(), E - ((done - 1) % E));
    const evalDue = (done + n - 1) % E === 0;
    const enc = device.createCommandEncoder();
    t.encodeSteps(enc, n, evalDue);
    if (evalDue) enc.copyBufferToBuffer(t.probeOut, 0, staging, 0, t.probeOut.size);
    device.queue.submit([enc.finish()]);
    done += n;
    if (evalDue) {
      await staging.mapAsync(1);
      const probe = t.parseProbe(staging.getMappedRange().slice(0));
      staging.unmap();
      const wall = performance.now() - t0;
      tracker.push(done - 1, probe, wall);
      if (onEval) onEval(done - 1, probe, tracker.state, wall);
    } else {
      await device.queue.onSubmittedWorkDone();
    }
    if (onChunk) await onChunk(done, performance.now() - t0);
  }
  staging.destroy();
  return { ...tracker.state, steps: done, wallMs: performance.now() - t0 };
}
