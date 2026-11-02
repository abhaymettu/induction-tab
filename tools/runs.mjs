// Every run behind the published numbers that was made on this machine, by name, with its exact config.
// `node tools/runs.mjs list` shows what is left. `node tools/runs.mjs <name> [maxMinutes]` runs one; finished seeds are skipped.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const tab = JSON.parse(fs.readFileSync(path.join(ROOT, 'configs/tab.json'), 'utf8'));
const study = { V: 96, T: 64, d: 64, H: 4, B: 16, mlp: 4, ln: true, lnAffine: false, uniform: true, tile: true, pRep: 1, rMin: 4, rMax: 32, probeB: 16, probeT: 64, L: 2, unique: 2, lr: 0.003 };
const t128 = { ...study, T: 128, probeT: 128, rMin: 8, rMax: 64 };
const { rMin, rMax, ...tabNoRange } = tab;
const howe = { uniform: false, tile: false };
const configA = { V: 2048, T: 256, d: 256, H: 4, L: 2, unique: 2, B: 64, mlp: 4, ln: true, lnAffine: true, lr: 0.001, uniform: false, tile: false, pRep: 0.75, probeB: 64, probeT: 128 };

const JOBS = {
  'browser-2-2': { cfg: study, seeds: [1, 64], budget: 6000 },
  'p2-T64-L2-u2': { cfg: { ...study, evalEvery: 5 }, seeds: [101, 164], budget: 6000 },
  'p2-T64-L2-u1': { cfg: { ...study, unique: 1, evalEvery: 5 }, seeds: [101, 164], budget: 6000 },
  'p2-T128-L2-u2': { cfg: { ...t128, evalEvery: 5 }, seeds: [101, 164], budget: 6000 },
  'p2-T128-L2-u1': { cfg: { ...t128, unique: 1, evalEvery: 5 }, seeds: [101, 164], budget: 6000 },
  'tab-browser': { cfg: tab, seeds: [1, 64], budget: 3000 },
  'tab-native': { native: true, cfg: tab, seeds: [1, 64], budget: 3000 },
  'saved-run': { cfg: tab, seeds: [1, 1], budget: 3000, stopAfter: 3000 },
  'fine-2-2': { cfg: { ...study, evalEvery: 5 }, seeds: [1, 64], budget: 6000, stopAfter: 75 },
  // Added after fine-2-2: these three seeds stopped before their coarse event, so 75 steps could not read them at 25-step spacing.
  'fine-2-2-extra': { cfg: { ...study, evalEvery: 5 }, spans: [[17, 17], [23, 23], [51, 51]], budget: 6000, stopAfter: 400 },
  'crossed-rule-bigram': { cfg: { ...study, ...howe }, seeds: [1, 16], budget: 8000 },
  'crossed-rule-uniform': { cfg: { ...study, pRep: 0.75, rMax: 60 }, seeds: [1, 16], budget: 8000 },
  'crossed-range-uniform': { cfg: { ...study, rMax: 60 }, seeds: [1, 16], budget: 8000 },
  'crossed-range-bigram': { cfg: { ...study, ...howe, pRep: 0.75 }, seeds: [1, 16], budget: 8000 },
  'tab-neg-L1': { cfg: { ...tab, L: 1, unique: 1 }, seeds: [1, 8], budget: 6000 },
  'tab-neg-norep': { cfg: { ...tab, pRep: 0 }, seeds: [1, 8], budget: 6000 },
  'tab-howe-curriculum': { cfg: { ...tabNoRange, ...howe, pRep: 0.75 }, seeds: [1, 8], budget: 8000 },
  'deep-12-12': { cfg: { ...study, L: 12, unique: 12 }, seeds: [1, 16], budget: 4000 },
  'deep-12-4': { cfg: { ...study, L: 12, unique: 4 }, seeds: [1, 16], budget: 4000 },
  'deep-8-1-long': { cfg: { ...study, L: 8, unique: 1 }, seeds: [1, 8], budget: 16000 },
  'config-a': { cfg: configA, seeds: [1, 5], budget: 9000 },
};

const outOf = (name) => path.join(ROOT, 'results', `${name}.jsonl`);
const spansOf = (job) => job.spans ?? [job.seeds];
const left = (name) => {
  const file = outOf(name);
  const done = fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).length : 0;
  return spansOf(JOBS[name]).reduce((n, [a, b]) => n + b - a + 1, 0) - done;
};

const [name, maxMinutes] = process.argv.slice(2);
if (!name || name === 'list') {
  for (const n of Object.keys(JOBS)) console.log(`${n} ${left(n)}`);
  process.exit(0);
}
const job = JOBS[name];
if (!job) throw new Error(`no job ${name}`);
const env = { ...process.env, ...(job.stopAfter ? { STOP_AFTER: String(job.stopAfter) } : {}) };
for (const [a, b] of spansOf(job)) {
  const span = `${a}-${b}`;
  const run = job.native
    ? ['.venv/bin/python', ['ref/train.py', '--cfg', JSON.stringify(job.cfg), '--seeds', span, '--budget', String(job.budget), '--jobs', '4', '--threads', '2', '--curves', '--out', outOf(name)]]
    : ['node', ['tools/dynamics.mjs', JSON.stringify(job.cfg), span, String(job.budget), outOf(name), ...(maxMinutes ? [maxMinutes] : [])]];
  const res = spawnSync(run[0], run[1], { cwd: ROOT, env, stdio: ['ignore', 'pipe', 'inherit'], encoding: 'utf8', maxBuffer: 1 << 26 });
  console.log(res.stdout.trim().split('\n').slice(-1)[0]);
  if (res.status !== 0) process.exit(res.status ?? 1);
}
