// Prints the README's numbers that no analysis script reports: the negative controls, the anchor misses, the stalled
// seeds at context 128, and config A. Reads stored runs only, so it trains nothing. `npm run read-runs`
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const rows = (file) => fs.readFileSync(path.join(ROOT, 'results', file), 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line));
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

console.log('Negative controls, results/study-negatives.jsonl');
const negatives = rows('study-negatives.jsonl');
for (const [name, runs] of [['one layer', negatives.filter((r) => r.cfg.L === 1)], ['no repeats', negatives.filter((r) => r.cfg.pRep === 0)]]) {
  const anyEval = Math.max(...runs.flatMap((r) => r.curve.map((c) => c.copyAdv)));
  const lastEval = Math.max(...runs.map((r) => r.curve[r.curve.length - 1].copyAdv));
  console.log(`  ${name}: ${runs.length} seeds, ${runs.filter((r) => r.t_event !== null).length} events by step ${runs[0].budget}, highest copy advantage ${anyEval.toFixed(3)} at any eval, ${lastEval.toFixed(3)} at the last eval`);
}

console.log('Anchor misses, 3 layers, 3 blocks, lr 1e-3, results/study-lr1e-3.jsonl');
const untied3 = rows('study-lr1e-3.jsonl').filter((r) => r.cfg.L === 3 && r.cfg.unique === 3);
const misses = untied3.filter((r) => r.t_event !== null && r.t_pv === null);
console.log(`  ${misses.length} of ${untied3.length} runs fired the event with the anchor never firing`);
for (const r of misses) {
  const top = [0, 1, 2].map((layer) => Math.max(...r.curve.map((c) => c.prev[layer])).toFixed(3));
  console.log(`  seed ${r.seed}, event ${r.t_event}, top previous-token score by layer ${top.join(' ')}`);
}

console.log('Seeds with no event at context 128, width 64, results/p2-T128-*.jsonl');
const context128 = ['p2-T128-L2-u2.jsonl', 'p2-T128-L2-u1.jsonl'].flatMap(rows);
const stalled = context128.filter((r) => r.t_event === null);
console.log(`  ${stalled.length} of ${context128.length} runs had no event by step ${context128[0].budget}: ${stalled.map((r) => `seed ${r.seed}, ${r.cfg.unique} block${r.cfg.unique > 1 ? 's' : ''}`).join('; ')}`);

console.log('Config A, results/config-a.jsonl');
const configA = rows('config-a.jsonl');
for (const r of configA) {
  console.log(`  seed ${r.seed}: event ${r.t_event}, anchor ${r.t_pv}, fraction ${(r.t_pv / r.t_event).toFixed(3)}, ${(1000 * r.wall_s / r.steps_run).toFixed(0)} ms per step, ${(r.wall_s / 60).toFixed(0)} minutes`);
}
const fired = configA.filter((r) => r.t_event !== null && r.t_pv !== null);
console.log(`  seeds ${configA.length}, event median ${median(fired.map((r) => r.t_event))}, fraction median ${median(fired.map((r) => r.t_pv / r.t_event)).toFixed(3)}`);
