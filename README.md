# induction-tab

Press Start and watch a transformer learn to copy on your own GPU, about 5 seconds on an M4 Pro, then type into it and silence its heads.

**[abhaymettu.github.io/induction-tab](https://abhaymettu.github.io/induction-tab/)**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/hero-dark.webp">
  <source media="(prefers-color-scheme: light)" srcset="docs/hero-light.webp">
  <img alt="The page after training. The repeated row of random characters is lit, every layer 1 attention map shows a sharp line in its lower half, and a typed abcabc prompt continues itself." src="docs/hero-light.webp">
</picture>

<sub>Seed 1 after 3000 steps on an M4 Pro. Another job was using the GPU during this capture, so its clock reads 7.1 s. The clean benchmark for this seed is 5.2 s.</sub>

## What you see

The page trains a two-layer transformer while you watch, in your browser, with no ML library and nothing to install. It trains only on random characters repeating at random periods, so the one thing to learn is that characters repeat.

For the first few hundred steps the model has learned nothing. Then three things happen close together.

- **The lower row lights up.** It is the upper row again, and brightness shows how sure the model is of each character. The model has seen those characters once, in a sequence it never trained on.
- **The attention maps draw lines.** A head in layer 0 draws a line just under the diagonal, because it reads the previous character. Heads in layer 1 draw a parallel line lower down, where they find this character's earlier copy. This is the induction circuit from Olsson et al. 2022.
- **The copy advantage crosses 2 nats.** Copy advantage is the loss on the first pass through a repeated random sequence minus the loss on the second, so it measures copying alone. The event is 2 nats or more on two consecutive evals, reported at the second. Gunner Howe defined it.

Type something that repeats and the model carries on, so `abcabcabcabcab` gives `cabcabc`. Press a head to silence it. Silence all of layer 0 or all of layer 1 and copying dies. A browser with no WebGPU plays a recording of a real run and answers prompts with the saved weights on the CPU.

On an M4 Pro in Chrome, with nothing else on the GPU, seeds 1 to 5 reach the event at steps 500, 475, 600, 425, and 475. That is 5.2, 4.9, 6.2, 4.4, and 4.9 s at 10.3 ms per step, a median of 4.9 s. Matrix multiplies take 8.44 of the 10.04 ms in a replayed step. Native PyTorch is faster on the same machine and config. MPS takes 5.5 ms per step, about 1.9 times faster than this trainer, and 8 CPU threads take 13.9 ms.

## Why it is correct

The trainer is hand-written WGSL for the forward pass, the backward pass, and AdamW. Two checks compare it to PyTorch, and the negative controls test the instruments.

**Gradients match.** `npm run parity` feeds the same weights and batches to PyTorch autograd and to the trainer for 20 steps, then compares every parameter tensor. The worst per-tensor gradient relative L2 is about 1e-6.

**Whole runs match.** The study's two-layer model ran 64 seeds both ways, from the same weights on the same batches, one on Metal and the other PyTorch on a CPU. Both reach the same event step and the same anchor step on 64 of 64 seeds, with evals 25 steps apart. The anchor is Howe's precursor, the first eval where a first-layer head reaches a previous-token score of 0.10. The worst copy advantage difference across 1539 evals is 3.2e-6.

**Negative controls show no event.** One layer has no head beneath it to read the previous character, and a model that never sees a repeat has nothing to copy. Each ran 8 seeds for 6000 steps with zero events. One layer peaked at 1.20 nats of copy advantage. No repeats peaked at 0.067, at step 0, and ended at or below 0.003.

## What the study found

The study model is smaller than the tab's, width 64 and context 64 (d=64, T=64), for cost. It ran on a rented 4 vCPU cloud VM with PyTorch on CPU. Arms ran 64 seeds unless noted and are written (layers, blocks), where a block is one set of weights, so (8,1) is the same block applied 8 times. The plans are in [`research/PREREG.md`](research/PREREG.md) and [`research/PREREG-2.md`](research/PREREG-2.md), committed before results were read. About 70% of the first plan's 3e-3 arms had already run, unread, when it was committed. Every arm is in [`results/study-table.md`](results/study-table.md).

### Most of the speed is the curriculum

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/survival-by-curriculum-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/survival-by-curriculum-light.svg">
  <img alt="Survival curves by curriculum at lr 3e-3. Random characters made periodic have a median event step of 575, Howe's language made periodic 1025, Howe's language with the clone rule 7500 with 12 of 16 seeds fired, and random characters with the clone rule fire on no seed by step 8000." src="docs/survival-by-curriculum-light.svg">
</picture>

These arms share one model and one learning rate and change the curriculum. Random characters made periodic reach the event at a median of step 575. Howe's bigram language made periodic takes 1025. Howe's language with Howe's clone rule takes 7500, with 12 of 16 seeds fired by step 8000. Random characters under the clone rule fired on 0 of 16 by step 8000. Periodic means every sequence tiles at a random period. The clone rule copies the start of 75% of sequences once, at a random offset.

Howe's curriculum against random characters made periodic is a time ratio of 13.1. The trainer is about 1.9 times slower than native MPS, so it is not where the speed comes from. Training on random repeats is also not new. shaikabdul185-arch, xanderchinxyz, and gama do it too.

### Tying weights delays the event at these depths

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/median-by-depth-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/median-by-depth-light.svg">
  <img alt="Median event step against depth at two learning rates. Untied models stay nearly flat, from 575 to 700 at lr 3e-3. Models with 2 blocks climb to 1725 at depth 8. Models with 1 block climb to 1825 at depth 4 at lr 3e-3 and 4950 at lr 1e-3, and at depth 8 show no event in 32 seeds by step 4000." src="docs/median-by-depth-light.svg">
</picture>

On random-repeat data at lr 3e-3 and depths 2 to 4, tying weights delayed induction at equal depth (time ratios 1.55, 2.20, 2.83; 64 seeds per arm). At lr 1e-3 the same ratios are 1.55, 2.09, 4.07. At depth 8 and lr 3e-3, 2 blocks against untied is 2.39, and 1 block had no events in 32 seeds by step 4000, so its ratio is above 5.71. We did not find this reported within our coverage. The closest work finds the opposite direction on other tasks and greater depth ([2610.31421](https://arxiv.org/abs/2610.31421), [2610.24135](https://arxiv.org/abs/2610.24135)), and we do not claim a contradiction.

A tied block sums gradients over its passes, so one learning rate may act larger on it, and we did not separate that from a structural effect. Time ratios come from a censored log-normal fit per arm.

Untied depth is nearly free in steps. 4 layers against 2 is 1.13 at lr 3e-3 and 0.97 at lr 1e-3, and 8 against 4 is 1.09. Each step costs more, so in floating point operations 8 against 4 is 2.15.

### The precursor gap moved, and eval spacing explains part of it

Howe's fleet puts the anchor at a steady fraction of the time to the event, inside a frozen envelope of [0.735, 0.901]. Across the study's main arms the median fraction runs from 0.727 to 0.848. The preregistered falsifier fired for three arms just under the envelope, (2,2) and (3,3) at lr 3e-3 and (2,2) at lr 1e-3.

A second plan predicted that the anchor's distance from chance caused the miss, and tested a rescaled anchor. Under that anchor (2,1) at T=64 reads 0.903 against 0.901, so the prediction failed on one arm. Eval spacing explains the miss on (2,2) at lr 3e-3 instead. At 5-step evals, T=64 (2,2) is 0.756 and (2,1) is 0.809. Deriving 25-step spacing from those same runs gives 0.727 on (2,2), which is the first plan's median. The other two arms were not rerun at 5-step evals.

Spacing does not explain Howe's curriculum on the small model at 0.628, or (8,2) at 0.683. Those two are still open.

### One replication, and the rest extends

Config A is Howe's own setup, on this trainer: V=2048, T=256, d=256, B=64, lr 1e-3, Howe's language and clone rule, Howe's anchor and event.

| Config A | Seeds | Event step | Anchor over event |
|---|---|---|---|
| Gunner Howe's fleet | 30 | median 6300, range 5650 to 7400 | median 0.844 |
| This trainer | 1 | 6225 | 0.847 |

Seed 1 took 24 minutes at 232 ms per step. The bigram table differs from Howe's, because Howe's table cannot be rebuilt without PyTorch's generator. With this few seeds, this replicates the event window and the fraction, and it does not replicate the fleet's ranking. Everything else here extends Howe's instruments to other models and curricula.

### Two exploratory reads

In a one-block model applied twice, 2 or 3 of 4 heads act as previous-token heads on pass 0 and induction heads on pass 1, on all 8 seeds. A head counts when it scores 0.10 or above on both. The strongest pass 0 head scores 0.42 to 0.47, and all four heads score 0.56 to 0.66 on pass 1. One block has to carry the whole circuit, so its heads either do both jobs or split them, and here some do both. Earlier work shows shared weights can run different operations per loop through attention routing ([2609.39892](https://arxiv.org/abs/2609.39892)) and that loop visits can differ in role ([2610.19975](https://arxiv.org/abs/2610.19975)). We did not find a named circuit role assigned to one head across passes within our coverage.

In (3,3) at lr 1e-3, 8 of 64 runs put the previous-token head in layer 1. The anchor watches only layer 0 at that depth, so it never fired on those runs.

## Rerun every number

```sh
npm install
python3 -m venv .venv && .venv/bin/pip install torch numpy
```

GPU commands need Google Chrome installed. `bench`, `breakdown`, `parity`, and `verify-record` use your GPU, `speed-native` times PyTorch on CPU and MPS, and `heads` trains with PyTorch on CPU. The rest read stored runs in `results/`. [`AGENTS.md`](AGENTS.md) has the layout and the traps.

| Claim | Number | Command | Output file |
|---|---|---|---|
| Seconds to the event, M4 Pro | median 4.9 s, 10.3 ms per step | `npm run bench` | `results/bench-m4pro.txt` |
| Where a step goes | 8.44 of 10.04 ms in matrix multiplies | `npm run breakdown` | `results/breakdown-m4pro.txt` |
| Native speed | MPS 5.5, CPU 13.9 ms per step | `npm run speed-native` | `results/native-speed.json` |
| Gradients against PyTorch | about 1e-6 relative L2 | `npm run parity` | `results/parity-apple-metal-3.json` |
| Whole runs against PyTorch | 64 of 64 seeds, worst difference 3.2e-6 | `npm run compare` | `results/native-vs-browser.json` |
| Negative controls | 0 events, peaks 1.20 and 0.067 | `npm run read-runs` | printed, from `results/study-negatives.jsonl` |
| Medians by curriculum and arm | 575, 1025, 7500 | `npm run study` | `results/study-table.md` |
| Tying and depth time ratios | 1.55, 2.20, 2.83, and the rest | `npm run study` | `results/study-table.md` |
| Gap across main arms | 0.727 to 0.848 | `npm run study` | `results/study-summary.json` |
| Gap against spacing and anchor | 0.756, 0.809, 0.727, 0.903 | `npm run anchor` | `results/anchor-summary.json` |
| Config A | event 6225, fraction 0.847 | `npm run read-runs` | printed, from `results/config-a.jsonl` |
| Heads doing both jobs | 2 or 3 of 4, 8 seeds | `npm run heads` | `results/heads-looped-2-1.json` |
| Anchor misses at (3,3) | 8 of 64 | `npm run read-runs` | printed, from `results/study-lr1e-3.jsonl` |
| Record row | 4.90 s | `npm run verify-record -- records/m4pro-chrome.json` | `records.json` |
| Figures | | `npm run figures` | `docs/*.svg` |

The stored runs came from these commands. Each row of a run file carries its config, and both trainers skip seeds their output file already holds.

```sh
# Native PyTorch on CPU
.venv/bin/python ref/train.py --cfg @configs/search/study-negatives.json --seeds 1-8 --budget 6000 --curves --jobs 4 --out fresh-negatives.jsonl
.venv/bin/python ref/train.py --cfg @configs/search/study-lr3e-3.json --seeds 1-64 --budget 6000 --curves --jobs 4 --out fresh-lr3e-3.jsonl
.venv/bin/python ref/train.py --cfg @configs/search/study-lr1e-3.json --seeds 1-64 --budget 8000 --curves --jobs 4 --out fresh-lr1e-3.jsonl
.venv/bin/python ref/train.py --cfg @configs/search/study-deep.json --seeds 1-32 --budget 4000 --curves --jobs 4 --out fresh-deep.jsonl
.venv/bin/python ref/train.py --cfg @configs/search/study-curriculum.json --seeds 1-16 --budget 8000 --curves --jobs 4 --out fresh-curriculum.jsonl

# WGSL on your GPU, with the config read from a stored file's first row by jq
node tools/dynamics.mjs "$(head -1 results/browser-2-2.jsonl | jq -c .cfg)" 1-64 6000 fresh-browser.jsonl
node tools/dynamics.mjs "$(head -1 results/p2-T64-L2-u2.jsonl | jq -c .cfg)" 101-164 6000 fresh-p2.jsonl

# More config A seeds, appended to results/config-a.jsonl
SEEDS=2-30 npm run config-a
```

## Add a record to the table

1. Close anything else using the GPU, because timing measured beside other GPU work reads high.
2. Open the page, finish a run, and press **Run the benchmark**. It trains seeds 1 to 5 with the page held still and shows a log.
3. Press **Copy log**, press **Open a prefilled GitHub issue**, and paste the log below the marked line.
4. A maintainer saves the log as `records/<gpu>-<browser>.json` and runs `npm run verify-record -- records/<gpu>-<browser>.json`. It rejects the log unless the config hash matches `configs/tab.json`, the log's curve gives its event steps, the wall times agree with the step rate, and a rerun of each seed lands within 100 steps.
5. On `ACCEPT`, the maintainer adds one row to `records.json`. If the tool says the wall times are unconfirmed, because the rerun used another adapter class, the row's `adapter` text says so.

## What is not verified

- **Other browsers and GPUs.** Safari, Firefox, real phones, and non-Apple GPUs have not been measured. Only an M4 Pro and SwiftShader have, so this repo makes no claim about an ordinary laptop. One automated Safari check timed out in a background tab (`tools/safari.mjs`).
- **The clock while you watch.** The 4.9 s comes from the benchmark, which holds the page still. A run you watch also draws, so expect its clock to read higher. We have no timing of the drawing page on an idle GPU.
- **The exact tab config against PyTorch.** Parity runs on other configs, and the whole-run match is on the study model. Neither has run on `configs/tab.json`. The committed parity files hold only the configs last run on each adapter, plain and small on Metal, small and attention-only on SwiftShader. `npm run parity` runs small, looped, attention-only, and demo.
- **How often a seed stalls.** All of seeds 1 to 5 fire on the tab config, and five seeds give no stall rate. At width 64 and context 128, 6 of 121 stored runs had no event in 6000 steps. The page says so when this happens and offers a new seed.
- **Novelty.** "Not found" means not found within the searches logged in [`research/NOVELTY-2.md`](research/NOVELTY-2.md). The two closest demos are low visibility and turned up late, so others may exist.

## What did not work

- **Howe's curriculum at this size.** On the study model it takes 7500 steps against 575, too slow for a tab, so the tab trains on random repeats. An early single run read as no signal at all. The study's 16 seeds show it does fire, late.
- **"The first we could find."** An early draft claimed this. A second search found TheoremPath's lab and gama, so the claim is gone.
- **The second plan's explanation of the gap.** Distance from chance did not explain the first plan's miss, and the rescaled anchor missed on one arm of its own.

## Credits

Other browser demos train induction heads live, including TheoremPath's lab and gama's Induction, both on the CPU or without a stated GPU backend. This one runs on the viewer's GPU in hand-written WGSL, keeps MLPs and LayerNorm, and streams previous-token and induction scores with head ablation on one page. We did not find that combination within our search coverage.

- Gunner Howe wrote [arXiv 2609.19000](https://arxiv.org/abs/2609.19000), the source of the instruments and the transition. Gunner Howe and Vin Howe are different people.
- Vin Howe made [Sequence Toy](https://sequence.toys) and [Piston](https://github.com/vinhowe/piston), which train sequence models in the browser on WebGPU.
- Olsson et al. 2022 described the circuit in [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
- [TheoremPath Induction Heads Lab](https://theorempath.com/demo/induction-heads-lab) trains a two-layer model in the browser with head ablation.
- [gama](https://github.com/pariharshyamu/gama) by pariharshyamu trains and draws its Induction model live.
- [TabGPT](https://github.com/jbingen/tabgpt) and [webgpu-llm](https://github.com/toprakdeviren/webgpu-llm) write their training kernels by hand in WGSL.
- [Little Language Machine](https://arxiv.org/abs/2601.21631) trains a small transformer client-side.
- [shaikabdul185-arch](https://github.com/shaikabdul185-arch/induction-heads-from-scratch) and [xanderchinxyz](https://github.com/xanderchinxyz/forming-induction-heads) did earlier hobby work on random repeats.

MIT, see [LICENSE](LICENSE). `research/upstream/emergence_watch` keeps its own license.
