# AGENTS.md

`induction-tab` trains a transformer in a browser tab on hand-written WGSL. There is no build step. The page is static files served from the repo root, and `README.md` holds the findings.

## Set up

```sh
npm install
python3 -m venv .venv && .venv/bin/pip install torch numpy
```

GPU tools need Google Chrome. `npm run parity` calls `.venv/bin/python` by that path, so the venv must live at the repo root.

## Layout

- `src/engine/` is the trainer. `kernels.js` holds every WGSL kernel, `trainer.js` wires them into dispatch lists, and `run.js` drives a run through the event. `layout.js` holds the parameter layout and config defaults. `init.js`, `prng.js`, and `lang.js` hold init, randomness, and data. `bench.js` holds the record format and `configHash`. `cpu.js` is the forward pass in plain JS for browsers without WebGPU.
- `src/app/` is the page. `main.js` is the composition. `live.js` is the GPU source and `recording.js` is the recorded one. `stream.js`, `attention.js`, `curve.js`, and `steer.js` draw the rows, the maps, the curve, and the typed prompt. `records.js` draws the records and study sections.
- `ref/` is the PyTorch twin. `induction_ref/` mirrors layout, init, PRNG, and data bit for bit. `train.py` is the native trainer, `parity_ref.py` feeds `npm run parity`, and `speed_native.py` times native steps.
- `tools/` holds the checks, the analysis, and `figures.mjs`.
- `results/` holds stored runs as JSONL with one run per row, and the summaries made from them. `records/` holds benchmark logs, and `records.json` is the table the page reads.
- `research/` holds the plans (`PREREG.md`, `PREREG-2.md`, `PREREG-3.md`), the novelty audits (`NOVELTY.md`, `NOVELTY-2.md`), the upstream spec, and upstream code.
- `configs/` holds `tab.json` (the frozen tab model), `defaults.json` (the same defaults as `layout.js`), `config-a.json`, `heads-looped.json`, and the study arms in `search/`.
- `assets/recording/` is the run the no-WebGPU path plays, written by `node tools/record.mjs`. `assets/fonts/` holds the self-hosted IBM Plex Mono and its license.
- `docs/` holds the README's images. `npm run figures` writes the SVGs from `results/study-summary.json`.

## Run things

| Command | Needs | What it does |
|---|---|---|
| `npm run serve` | nothing | Serves the repo root and prints a local URL for the page, using `tools/serve.mjs`, the server every headless check uses |
| `npm run parity` | GPU, venv | Compares gradients and parameters against PyTorch on five configs and writes `results/parity-<adapter>.json` |
| `npm run check` | GPU, venv | Runs parity, then trains 8 seeds and compares event steps against the stored native runs |
| `npm run bench` | GPU | Trains seeds 1 to 5 of the tab config and prints step, seconds, and ms per step |
| `npm run breakdown` | GPU | Prints where one training step spends its time |
| `npm run verify-record -- <log>` | GPU | Accepts or rejects a benchmark log, including a rerun of its seeds |
| `node tools/verify-page.mjs` | GPU | Drives the real page through start, event, prompt, silencing, and frame times |
| `node tools/verify-flows.mjs` | GPU | Covers pause, new seed, keyboard, benchmark log, data sections, reduced motion, and sizes |
| `node tools/verify-fallback.mjs` | browser | Runs the no-WebGPU path and a lost device mid-run |
| `node tools/verify-network.mjs` | GPU | Runs the page through the event, a typed prompt, the benchmark, and the no-WebGPU path, and exits nonzero if any request or tag points at another host |
| `node tools/verify-look.mjs` | GPU | Writes stills and a real-time clip for judging the page by eye |
| `node tools/verify-perf.mjs` | GPU | Takes turns between a tight loop and the real page and reports throughput and frame times |
| `npm run speed-native` | GPU, venv | Times PyTorch on CPU and MPS and rewrites `results/native-speed.json` |
| `npm run heads` | venv | Trains 8 seeds with PyTorch on CPU and rewrites `results/heads-looped-2-1.json` |
| `npm run study` | venv | Rewrites `results/study-summary.json` and `results/study-table.md` from the study runs. Rows from `deep-8-1-long` replace the (8,1) arm |
| `.venv/bin/python tools/analyze_third.py` | venv | Rewrites `results/third-summary.json` from the third plan's runs |
| `node tools/runs.mjs list` | nothing | Lists every named run on this machine and how many seeds it still needs |
| `node tools/runs.mjs <name> [maxMinutes]` | GPU, venv for `tab-native` | Runs one named job and appends to `results/<name>.jsonl`, skipping finished seeds |
| `npm run compare` | venv | Rewrites `results/native-vs-browser.json` |
| `npm run compare-tab` | venv | Rewrites `results/tab-native-vs-browser.json` |
| `npm run anchor` | venv | Rewrites `results/anchor-summary.json` |
| `npm run read-runs` | nothing | Prints negative controls, anchor misses, stalled seeds, and config A from stored runs |
| `npm run figures` | nothing | Rewrites the SVGs in `docs/` from `results/study-summary.json` |

`npm run study` and `npm run compare` reproduce their stored files byte for byte on the M4 Pro, so `git diff results/` after a rerun is a real check.

## Traps

- Only `channel: 'chrome'` reaches Metal when headless. Bundled Chromium gets SwiftShader, which runs but says nothing about speed.
- `configHash` in `src/engine/bench.js` hashes `configs/tab.json` resolved against the defaults in `layout.js`. Changing either one invalidates every row in `records.json`. The same defaults live in `configs/defaults.json` for the Python side, so change them together.
- Tile 32 in the GEMM kernel is the measured choice. It beat 8 and 16 on Metal, and smaller tiles ran up to 6 times slower. Measure before you change it.
- Timing measured while something else uses the GPU reads high. The clock in `docs/hero-*.webp` is an example, 6.9 s against 5.2 s clean. Take `bench`, `breakdown`, and record logs on an idle GPU.
- The page fetches `results/study-summary.json` and `records.json` at load. Moving either one blanks a section of the page without an error.
- `dynamics.mjs` and `train.py` append to their output file and skip seeds it already holds. Use a fresh path for a clean rerun.
- `npm run bench` and `npm run breakdown` print only. Redirect to `results/` by hand so one machine's numbers do not overwrite another's by accident.
- The study model (d=64, T=64) is not the tab model (d=128, T=128). Say which one a number came from.
- Config A with Howe's anchor and event is one seed. Call it one draw, not a replication. Everything else extends. Follow the wording in `research/NOVELTY-2.md`, except its sentence that the speed is not a measured curriculum effect, which `results/third-summary.json` supersedes. Say steps, not speed, when comparing arms.
- The page must load nothing from another host, a hard rule. Fonts are self-hosted. Run `node tools/verify-network.mjs` after any change to what the page loads.
- Copy advantage is not copying alone. A median 30 percent of it at the event is the first pass rising above chance.
