// The record format: what a benchmark run measures and how it is logged. Shared by the page and tools/.
import { resolve } from './layout.js';
import { trainLoop } from './run.js';

export const BENCH_SEEDS = [1, 2, 3, 4, 5];
export const REPO = 'https://github.com/abhaymettu/induction-tab';

// FNV-1a over the resolved config with sorted keys, so two logs compare only if they trained the same model.
export function configHash(cfg) {
  const { seed, steps, ...rest } = resolve(cfg);
  const text = JSON.stringify(Object.keys(rest).sort().map((k) => [k, rest[k]]));
  let h = 0xcbf29ce484222325n;
  for (let i = 0; i < text.length; i++) {
    h ^= BigInt(text.charCodeAt(i));
    h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h.toString(16).padStart(16, '0');
}

export function adapterInfo(adapter) {
  const i = adapter.info ?? {};
  return {
    vendor: i.vendor ?? '',
    architecture: i.architecture ?? '',
    device: i.device ?? '',
    description: i.description ?? '',
    isFallback: Boolean(i.isFallbackAdapter ?? adapter.isFallbackAdapter),
  };
}

// One seed through the event with nothing else on the path: chunks as long as the eval interval allows.
export async function benchSeed(device, t, seed, budget = 8000) {
  t.setAblation(new Array(t.c.L).fill(0));
  t.reset(seed);
  const curve = [];
  const res = await trainLoop(device, t, {
    budget,
    chunkSteps: () => t.c.evalEvery,
    shouldStop: (s) => s.tEvent !== null,
    onEval: (step, p) => curve.push([step, +p.copyAdv.toFixed(3)]),
  });
  return {
    seed,
    tEvent: res.tEvent,
    tPrev: res.tPrev,
    wallEventMs: res.wallEvent === null ? null : Math.round(res.wallEvent),
    msPerStep: +(res.wallMs / res.steps).toFixed(2),
    curve,
  };
}

export function buildLog({ cfg, adapter, userAgent, runs }) {
  return { v: 1, configHash: configHash(cfg), adapter: adapterInfo(adapter), userAgent, runs };
}

export function medianSeconds(runs) {
  const walls = runs.map((r) => r.wallEventMs).filter((w) => w !== null).sort((a, b) => a - b);
  if (!walls.length) return null;
  const mid = walls.length >> 1;
  const ms = walls.length % 2 ? walls[mid] : (walls[mid - 1] + walls[mid]) / 2;
  return +(ms / 1000).toFixed(2);
}

// GitHub rejects very long URLs, so the prefilled issue carries the summary and asks for the full log by paste.
export function issueUrl(log, repo = REPO) {
  const median = medianSeconds(log.runs);
  const summary = {
    ...log,
    medianSeconds: median,
    runs: log.runs.map(({ curve, ...r }) => r),
  };
  const title = `Record: ${median ?? 'no event'} s on ${log.adapter.vendor || 'unknown'} ${log.adapter.architecture || 'GPU'}`;
  const body = [
    'Benchmark result from the page.',
    '',
    '```json',
    JSON.stringify(summary, null, 1),
    '```',
    '',
    'Paste the full log with curves from the page below this line.',
    '',
  ].join('\n');
  return `${repo}/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}
