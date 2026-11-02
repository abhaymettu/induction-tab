// The one command behind the claims that need no long run: gradient parity, then learning dynamics against the stored
// native runs of the same seeds. `npm run check`
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const run = (cmd, args, env = {}) => execFileSync(cmd, args, { cwd: ROOT, stdio: 'inherit', env: { ...process.env, ...env } });
run('node', ['tools/parity.mjs']);
const native = fs.readFileSync(path.join(ROOT, 'results/study-lr3e-3.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)).filter((r) => r.cfg.L === 2 && r.cfg.unique === 2);
const cfg = native[0].cfg;
const tmp = path.join(ROOT, 'scratch/check-dynamics.jsonl');
fs.rmSync(tmp, { force: true });
const SEEDS = 8;
run('node', ['tools/dynamics.mjs', JSON.stringify(cfg), `1-${SEEDS}`, '6000', tmp]);
const browser = fs.readFileSync(tmp, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const diffs = browser.map((b) => b.t_event - native.find((n) => n.seed === b.seed).t_event).sort((a, b) => a - b);
const median = diffs[diffs.length >> 1];
console.log(`native minus browser event steps over ${SEEDS} seeds: ${diffs.join(' ')}; median ${median}`);
if (Math.abs(median) > 25) { console.error('FAIL: learning dynamics differ by more than one eval'); process.exit(1); }
console.log('PASS');
