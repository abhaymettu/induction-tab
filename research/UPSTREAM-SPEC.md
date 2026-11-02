# Upstream spec for the arXiv 2609.19000 language-model seed fleet (config A)

This file pins down the experiment behind Section 5.2 of Howe, "Capability Emergence Can Be Forecast" (2026), well enough to rebuild it on another stack. Every item cites the upstream file and line it comes from. Items marked "inferred" are my reading of the code, not something the repo states.

## Sources and commits

The code is in the public repo `github.com/gunnerhowe/Research`, subdirectory `mem-gen-delay/`. The default branch is `master`. HEAD when read (2026-11-01) was `493c4bf38bf0ddcc56986d9003b4a448c32439e4`, last pushed 2026-07-22.

Citation keys used below:

| Key | File | Commit |
|---|---|---|
| F | `mem-gen-delay/src/train_lm.py` (220 lines, the trainer that produced the fleet) | `6b057703e9efd05a057683f9bb7725d5121a31c0` |
| H | `mem-gen-delay/src/train_lm.py` (375 lines, HEAD) | `493c4bf` |
| RUN | `mem-gen-delay/scripts/run_p6r2.py` (identical at 6b05770 and HEAD) | `493c4bf` |
| S2 | `mem-gen-delay/analysis/score_r2_p6.py` (unchanged since f25aa45) | `493c4bf` |
| S4 | `mem-gen-delay/analysis/score_r4_p6.py` | `493c4bf` |
| PLAN | `mem-gen-delay/plan_p6.md` (the pre-registration log) | `493c4bf` |
| EW | `emergence-watch/emergence_watch/*.py` (MIT packaged probes) | `493c4bf` |
| PAPER | [arXiv 2609.19000](https://arxiv.org/abs/2609.19000) | n/a |

The copy of EW in `research/upstream/emergence_watch` differs from upstream in one way. Dashes in its comments and docstrings became commas, and its code is unchanged.

Why F is the fleet code. The fleet spec was frozen at 6b05770 (PLAN:105, PAPER:538). The next commit that touches `train_lm.py` is 377511b (2026-07-16 03:17), which lands after the fleet verdict commit f25aa45 (2026-07-16 03:06). Every committed `runs/grid6r2/*/summary.json` holds exactly F's argparse keys and nothing that H added (no `lang`, `p_rep` override, `scaffold`). Later edits in H sit behind flags whose defaults are no-ops. I compared the functions, and the bigram path in H draws exactly as in F.

The paper names `src/train lm.py` and `analysis/score r* p6.py` as the harnesses (PAPER:543 to 544). It calls config A "fleet A (lr1e-3, b64)" in Figure 4 (PAPER:391 to 392).

## 1. Model (config A)

| Item | Value | Citation |
|---|---|---|
| Layers | 2 (1 for the architectural negative) | F:167, RUN:27 to 32 |
| d_model | 256 | F:152 |
| Heads | 4 | F:153 |
| Head dim | 64 (d_model / heads) | F:43 to 44 |
| Block type | Attention plus MLP, not attention-only | F:47 to 48 |
| MLP | Linear(256, 1024) with bias, `nn.GELU()` (exact erf form, PyTorch default), Linear(1024, 256) with bias | F:37 |
| Attention projections | One fused `qkv` Linear(256, 768) with no bias, output `proj` Linear(256, 256) with no bias | F:35 to 36 |
| Attention scale | 1 / sqrt(64) | F:44 |
| Mask | Causal (strict upper triangle set to -inf before softmax) | F:45 to 46 |
| LayerNorm placement | Pre-LN. `x = x + proj(attn(ln1(x)))`, then `x = x + mlp(ln2(x))`, then a final `ln_f` before the unembedding | F:42, F:47 to 48, F:68 |
| LayerNorm params | `nn.LayerNorm(256)` defaults (eps 1e-5, gain 1, bias 0) | F:33 to 34, F:58 |
| Positional scheme | Learned absolute, `nn.Embedding(256, 256)`, added to token embedding | F:56, F:62 |
| Embeddings | Untied. Token `nn.Embedding(2048, 256)`, unembedding `nn.Linear(256, 2048, bias=False)` | F:55, F:59 |
| Init | PyTorch module defaults. The file has no custom init. Inferred from PyTorch defaults: embeddings N(0, 1) for both token and position tables, Linear weights and biases U(-1/sqrt(fan_in), 1/sqrt(fan_in)) | F:30 to 68 |
| Dropout | None anywhere | F:30 to 68 |
| Vocab | 2048 | F:151 |
| Context length | 256 for training. The copy probe uses only 128 positions | F:154, F:98 to 101 |

## 2. Training (config A)

| Item | Value | Citation |
|---|---|---|
| Optimizer | `torch.optim.AdamW`, one param group holding every parameter | F:170 to 171 |
| Betas | (0.9, 0.95) | F:171 |
| Eps | PyTorch default 1e-8 (not set) | F:170 |
| Weight decay | 0.01, applied to all params including embeddings, LayerNorm gains and biases | F:158, F:170 |
| Peak lr | 1e-3 (runner overrides the 3e-4 default) | RUN:11, F:157 |
| Schedule | Linear warmup `lr = 1e-3 * min(1, (step + 1) / 100)`, then constant 1e-3 to the end. No decay | F:159, F:182 to 184 |
| Batch | 64 sequences of 256 tokens (16,384 tokens per step) | F:155, F:199 |
| Loss | Mean next-token cross-entropy over all 64 x 255 prediction positions, repeated tails included. Log-softmax in fp32 | F:121 to 123, F:187 |
| Grad clipping | None | F:181 to 190 |
| Total steps | 16,000 | RUN:11 |
| dtype | fp32 (no autocast, no half casts). TF32 not set in code, so it follows the PyTorch version default | F:147 to 216 |
| Eval cadence | Every 25 steps, at `step % 25 == 0`, giving 640 records at steps 0, 25, ..., 15975 | F:160, F:191 |
| Eval timing | The eval runs after that step's optimizer update, so record `step = s` reflects s + 1 updates. Its `train_loss` is the loss of batch s computed before the update | F:185 to 202 |
| Seeds | `torch.manual_seed(seed)` before model construction (init). Data uses a separate CPU `torch.Generator` seeded `seed + 10000` | F:166, F:173 |
| Fleet | rep seeds 1 to 30, norep seeds 1 to 10, onelayer seeds 1 to 5. Seed 0 was the smoke run and is excluded | RUN:27 to 32, PLAN:108 to 112 |
| Hardware | One RTX 3080. 943 to 1021 s per positive run (from the 30 committed `summary.json` files) | PAPER:544 |

## 3. Synthetic language generator

### The bigram table

`make_language` (F:71 to 76) builds the table once from a CPU `torch.Generator` seeded `LANG_SEED = 777` (F:27). It draws two tensors in this order:

1. `succ`, shape (2048, 20), integers uniform in [0, 2048). Row i lists 20 successor slots for token i. Slots may repeat.
2. `w`, shape (2048, 20), `softmax(1.5 * N(0, 1), dim=-1)`. These are the probabilities of the 20 slots.

The table is the same for every run in the fleet. Per-seed variance comes from init and data order only (F:13 to 14).

### Sampling a training batch

`sample_batch` (F:79 to 95), called once per step with the run's data generator:

1. Draw 64 start tokens uniform in [0, 2048) (F:84).
2. For t from 0 to 255, write the current token at position t, draw a slot index from `w[current]` with `torch.multinomial`, and set the next token to `succ[current, slot]` (F:85 to 88). The 256th draw is discarded.
3. Draw a repeat mask, `rand(64) < p_rep`, with `p_rep = 0.75` for positives (F:89, F:168).
4. Draw an offset per sequence, `r = randint(16, 240)`, so r is uniform on the integers 16 to 239 (F:90). It is drawn for every sequence, repeated or not.
5. For each masked sequence, overwrite positions r to 255 with a clone of the original positions 0 to 255 - r (F:91 to 94).

"Repetition density 0.75" in the paper is this per-sequence probability `p_rep` (PAPER:369, PLAN:107, F:168). It is not a token fraction.

Inferred consequences of step 5, checked with a small simulation:

- The tail is copied from the original sequence before any overwrite. When r < 128 the second copy overwrites part of its own source, so only positions r + 1 to 2r - 1 have an earlier verbatim occurrence. The remaining tail positions continue as an ordinary bigram chain.
- The count of induction-predictable targets in a repeated sequence is min(r, 256 - r) - 1. That averages 71 per repeated sequence, about 21% of all 255 targets across a batch.
- The transition from position r - 1 to r breaks the bigram chain.

### Negatives

| Negative | Change | Citation |
|---|---|---|
| Data-ablated (`norep`, seeds 1 to 10) | `p_rep = 0.0`. Same language, same model, same steps | F:168, RUN:29 to 30 |
| Architectural (`onelayer`, seeds 1 to 5) | 1 layer instead of 2, positive data (`p_rep = 0.75`) | F:167, RUN:31 to 32 |

Inferred from F:89 to 90: `norep` still draws the repeat mask and offsets, so `norep_sK` sees exactly the token stream of `rep_sK` with the repeats left out.

### Shifted configs, for reference

| Cell | Changes from config A | Runs | Citation |
|---|---|---|---|
| Gate B | d_model 320 (head dim 80), p_rep 0.6, 20,000 steps | rep 101 to 110, norep 101 to 103 | `scripts/run_p6r5.py:12, 28 to 31`, PLAN:177 to 181 |
| Gate C | `lang_seed` 888 (new bigram table), 20,000 steps | rep 201 to 210, norep 201 to 203 | `scripts/run_p6r7r8.py:23 to 30`, PLAN:214 to 221 |
| Gap probe (R8) | lr 5e-4 or 2e-3 (seeds 301 to 305), or batch 32 or 128 (seeds 311 to 315), 32,000 steps | 20 positives | `scripts/run_p6r7r8.py:31 to 41`, PLAN:223 to 232 |
| Blind law gate (R9) | lr 7e-4, 24,000 steps | rep 321 to 325, norep 321 to 322 | PLAN:355 to 363 |

### Trap language (Section 5.5)

The trap fleet ran on the `train_lm.py` at f032a48. That is inferred from the file history: f032a48 froze the trap constants, and the next change to the file (f9ebaaa, 2026-07-19) came after the R6 verdict (2026-07-17, PLAN:313). `make_language` and `_row` there match H byte for byte. `sample_batch` in H differs only by flag-gated branches that are off by default. Citations are to H.

- The table has 2 x 2048 = 4096 rows, built the same way from seed 777 (H:119 to 123, `TRI_MODES = 2` at H:126).
- The row for the next draw is `current * 2 + (previous mod 2)` (H:135 to 138). The successor distribution therefore depends on the current token and one bit of the previous token.
- Each sequence starts with independent uniform `previous` and `current` tokens (H:153 to 154). The in-distribution probe uses the same rule (H:205 to 213).
- Runs: 24,000 steps, lr 1e-3, rep seeds 1 to 10 and norep seeds 1 to 10 (`scripts/run_p6r6.py:12, 28 to 31`, PLAN:277 to 280).
- The pre-registration text says `table[hash(prev, current)]` (PLAN:235). That was design v1, which was abandoned. The final design is the 2-mode rule above (H:127 to 132, PLAN:282 to 284).

## 4. Metric definitions as coded

All probes run under `torch.no_grad()` in `model.eval()` at every eval step, on two fixed batches that are identical in every run (F:174 to 176, F:191 to 198). The packaged MIT version in EW uses the same formulas (EW `probes.py:41 to 81`).

### Probe batches

| Batch | Construction | Size | Citation |
|---|---|---|---|
| Copy probe `pb` | Generator seed 1234. x = randint(10, 2038) of shape (64, 64), so tokens are uniform on 10 to 2037. The batch is [x ; x] | 64 sequences x 128 tokens, half length L = 64 | F:98 to 101, F:174 to 175 |
| In-distribution probe `ipb` | Generator seed 4321. 128 bigram-language sequences of 256 tokens. Sequences 0 to 63 get a repeat at fixed r = 96 using the same clone rule. Sequences 64 to 127 stay fresh | 128 x 256 | F:104 to 118 |

The copy probe is out of distribution (uniform random tokens). The in-distribution probe depends on `lang_seed`, so it changes in Gate C.

### Copy advantage (event signal)

`ce[b, i]` is the cross-entropy, in nats, of predicting token i + 1 from prefix 0..i, giving 127 values per probe sequence (F:121 to 123).

`copy_adv = mean(ce[:, 0:63]) - mean(ce[:, 63:127])` over the copy probe (F:130). The first term covers targets 1 to 63 (first half). The second covers targets 64 to 127 (second half, 64 values). Target 64 is token 0 again, but induction cannot predict it. Both means pool batch and positions.

### Event

The event is copy_adv >= 2.0 nats on two consecutive evals. `t_event` is the step of the second eval (F:161, F:205 to 211, EW `events.py:13 to 39`). With evals 25 steps apart, the first crossing is at `t_event - 25`.

### Previous-token score (alarm signal)

For each layer, take the post-softmax attention `a` (B, 4, 128, 128) on the copy probe. For each head, average `a[b, h, q, q - 1]` over the 64 sequences and over queries q = 1 to 127, which covers both halves. The layer score is the max over the 4 heads (F:135, F:141 to 143).

The alarm `t_pv` is the first eval with layer-0 (first block) score >= 0.10 (S2:66, PLAN:113 to 114). The step-0 baseline is about 0.036 (`rep_s1` log). EW generalizes "layer 0" to "max over the first half of layers", which is the same thing for 1 and 2 layers (EW `probes.py:76 to 81`).

### Prefix-matching (induction) score

On the same attention, for each head, average `a[b, h, q, q - 63]` over the 64 sequences and queries q = 64 to 126. Key q - 63 is the token after the earlier occurrence of the token at q. Query 127 is excluded. The layer score is the max over heads (F:133 to 134, F:138 to 140). The onelayer negative check is max over layers and evals < 0.05 (S2:128, S2:134).

### In-distribution copy advantage (graded behavioral ramp)

`indist_adv = mean(ce_fresh[:, 96:255]) - mean(ce_rep[:, 96:255])` on `ipb`, for targets 97 to 255 (F:195 to 197). Because r = 96 < 128, only targets 97 to 191 have a verbatim earlier copy (inferred). That caps the signal well below copy_adv. At step 15975 in `rep_s1` it is 1.29 nats against copy_adv 6.94.

`t_ind` is the first eval with indist_adv >= 0.10 (S2:67). The conjunction anchor is the first eval where layer-0 prevtok >= 0.10 and indist_adv >= 0.10 hold at the same eval (S2:68 to 69, EW `anchors.py:65 to 74`).

### Loss baseline

The baseline uses the logged single-batch `train_loss` at eval steps, not a held-out loss. Thresholds are the 1st to 99th percentiles of the pooled unique train_loss values across the 30 positives. The alarm is the first eval with train_loss <= theta, and the theta with the highest Spearman wins. It is 2.0769 (S2:89 to 98, S2:152 to 156).

### Scoring statistics

- Spearman is ranked with `np.argsort`, which breaks ties by sort order rather than averaging them (S2:37 to 44). See ambiguity 1.
- The bootstrap CI uses 10,000 resamples, `np.random.default_rng(0)` and 2.5/97.5 percentiles. It skips resamples where either side is constant (S2:47 to 55).
- Lead is `t_event - t_pv`. The fleet median is 975 and the range 825 to 1125 (S2:140 to 142).
- The conformal interval uses calibration seeds 1 to 15 and test seeds 16 to 30, alpha 0.1. The quantile is the ceil((n + 1)(1 - alpha))-th smallest absolute residual, which for n = 15 is the largest. For the offset method, delta = median(t_event - t_pv) = 975 and q = 150, giving [t_pv + 825, t_pv + 1125] (S4:26 to 58, `analysis/out6/r4_scored.json`).

## 5. Per-seed results for the 30 config A positives

Copied verbatim to `research/upstream/r2_scored.json`. The source is `mem-gen-delay/analysis/out6/r2_scored.json` at HEAD `493c4bf`, last changed in f25aa45. Git blob `4401c2fc89c48e3d35eef0b7587f3c1ab1e40e4f` matches the local copy. `paper4/gen_numbers.py:29` reads the paper's fleet numbers from this file.

Check. I downloaded all 45 `runs/grid6r2/*/metrics.jsonl` and `summary.json` files at `493c4bf` (5.2 MB, in the scratchpad, not copied) and reran the event, t_pv and t_ind rules. All 30 positives match this file. All 30 summaries log the config in sections 1 and 2. The norep layer-0 prevtok maxima and onelayer prefix maxima also match.

| seed | t_event | t_pv (anchor) | lead | t_ind |
|---|---|---|---|---|
| 1 | 6775 | 5850 | 925 | 2725 |
| 2 | 6275 | 5250 | 1025 | 2725 |
| 3 | 6625 | 5675 | 950 | 2675 |
| 4 | 6125 | 5100 | 1025 | 2750 |
| 5 | 6200 | 5250 | 950 | 2950 |
| 6 | 6200 | 5150 | 1050 | 2800 |
| 7 | 6500 | 5475 | 1025 | 2775 |
| 8 | 6525 | 5650 | 875 | 2875 |
| 9 | 5800 | 4675 | 1125 | 2925 |
| 10 | 6175 | 5200 | 975 | 2750 |
| 11 | 5850 | 4725 | 1125 | 2875 |
| 12 | 6250 | 5150 | 1100 | 2925 |
| 13 | 6700 | 5825 | 875 | 2850 |
| 14 | 6750 | 5925 | 825 | 2925 |
| 15 | 6750 | 5875 | 875 | 2750 |
| 16 | 7025 | 6150 | 875 | 2875 |
| 17 | 6250 | 5250 | 1000 | 3000 |
| 18 | 6300 | 5350 | 950 | 2750 |
| 19 | 6300 | 5350 | 950 | 2850 |
| 20 | 5650 | 4525 | 1125 | 2900 |
| 21 | 6500 | 5625 | 875 | 2900 |
| 22 | 6400 | 5400 | 1000 | 2700 |
| 23 | 6300 | 5250 | 1050 | 2950 |
| 24 | 6275 | 5300 | 975 | 2750 |
| 25 | 6675 | 5725 | 950 | 2900 |
| 26 | 6275 | 5275 | 1000 | 2825 |
| 27 | 6200 | 5225 | 975 | 2800 |
| 28 | 7400 | 6550 | 850 | 2950 |
| 29 | 6225 | 5225 | 1000 | 2850 |
| 30 | 6700 | 5700 | 1000 | 2875 |

Summary statistics, computed from the file: t_event median 6300 (5650 to 7400), t_pv median 5325, median t_pv / t_event 0.844. Negatives: norep layer-0 prevtok max 0.036 to 0.061, no events. Onelayer prefix max 0.011 to 0.014, no events.

Sanity targets from the end of `rep_s1` (step 15975): train_loss 1.857, copy_adv 6.94, indist_adv 1.29, prefix by layer [0.011, 0.865], prevtok by layer [0.957, 0.008]. At step 0: train_loss 7.79, copy_adv 0.011.

## 6. License and copied code

The repo has no root license, and GitHub reports `license: null`. `mem-gen-delay/` has no LICENSE file, so the trainer, generator, scorers and run logs carry no license. I copied no code from `mem-gen-delay/`. Sections 3 and 4 restate it in prose. The per-seed results file was copied as data at the task's request.

Five other subprojects have LICENSE files: `doob-synapses`, `emergence-watch`, `ornstein-dist`, `selection-amplify` and `spatiotemporal-extrapolation`. I read only the `emergence-watch` one, which is MIT. A license in a sibling directory does not cover `mem-gen-delay/`. `emergence-watch` is the author's packaged version of the probes, events and anchors (`pyproject.toml` declares MIT). I copied three small files from it into `research/upstream/emergence_watch/`, each with one header line naming the source path, commit and license, followed by the original bytes. The MIT text is beside them as `LICENSE`. Blob hashes of the bodies match upstream.

| Local file | Upstream path | Blob |
|---|---|---|
| `upstream/emergence_watch/probes.py` | `emergence-watch/emergence_watch/probes.py` | `1730bd5c` |
| `upstream/emergence_watch/events.py` | `emergence-watch/emergence_watch/events.py` | `55c71c55` |
| `upstream/emergence_watch/anchors.py` | `emergence-watch/emergence_watch/anchors.py` | `83455805` |
| `upstream/emergence_watch/LICENSE` | `emergence-watch/LICENSE` | `f05fd566` |

`emergence-watch` was added in 8edd010 (2026-07-18), after the fleet ran. Its formulas match F's `run_probes` and event rule (checked by reading, not by running). One difference: EW subtracts the two copy_adv means as Python floats, while F subtracts float32 tensors. EW has no data generator.

## Ambiguities and gotchas

1. **Spearman ties.** Event and anchor steps are multiples of 25. The 30 positives have only 20 distinct t_event values and 24 distinct t_pv values. Upstream ranks with `np.argsort`, so rho depends on how the platform's sort orders ties. On this file, standard average-rank Spearman gives 0.9791. numpy 2.5.3 quicksort on this Mac gives 0.9729, and random tie orders range 0.959 to 0.990. The paper's 0.977 is one of those orders. A replication should use average ranks and expect about 0.979 on upstream data, plus or minus 0.015 from ties alone.
2. **Stale docstring.** F:3 to 4 and H:3 to 4 say 75% of sequences "repeat their first half verbatim". The code uses a variable offset r in 16 to 239 (F:79 to 94). The code is what ran.
3. **The clone rule.** Copying the original prefix into positions r to 255 is not periodic repetition. For r < 128 only r - 1 targets can be predicted by induction (inferred, section 3). A stack that tiles the prefix, or inserts a copy without truncating, gives a different training signal.
4. **Init.** No explicit init, so the fleet uses PyTorch defaults, including N(0, 1) token and position embeddings. GPT-2-style stacks default to N(0, 0.02), and that likely shifts emergence timing (a guess, not measured).
5. **Weight decay on everything.** AdamW decay 0.01 also hits embeddings, LayerNorm and biases. Many stacks exclude those by default.
6. **Off-by-one eval index.** Record `step = s` is taken after s + 1 updates, and its train_loss is pre-update. This matters only for exact step matching, not at 25-step resolution.
7. **Event step convention.** `t_event` is the second of the two consecutive qualifying evals, not the first.
8. **"About 15% of training".** PAPER:19 and PLAN:134 say the 975-step lead is "~15% of training", and PAPER:333 to 334 says "~15% of the run". That is 975 / 6300 (median event step) = 15.5%. It is 6.1% of the 16,000-step budget.
9. **Environment.** No requirements file or torch version is committed (I checked `mem-gen-delay/` and `zenodo_code_bundle.zip`). `out_dir` in `summary.json` shows a Windows path (`E:\GitHub\...`). Matmul TF32 is off by default from PyTorch 1.12 on. Older versions would have used TF32 on the 3080. Determinism flags are not set.
10. **Bit-exact data is PyTorch-only.** The table, batches and probes come from PyTorch's CPU generator and `torch.multinomial`. Another stack cannot rebuild the `LANG_SEED = 777` table without PyTorch. Options are to export the table, the probe batches and the in-distribution probe once from PyTorch, or to accept a fresh table. Gate C ran a fresh table (seed 888) and gave median lead 987.5 with rho 0.952 (PAPER:371 to 374).
11. **Probe positions.** The previous-token score averages both halves of the copy probe (queries 1 to 127). The prefix score uses queries 64 to 126. copy_adv's second-half mean includes target 64, which induction cannot predict. Copy these exactly or thresholds 0.10 and 2.0 will shift slightly.
12. **What the alarm reads.** The 0.10 alarm reads uniform random probe tokens, not training text. It is a max over 4 heads of layer 0 only.
13. **Trap language naming.** The pre-registration says "hash(prev, current)" (PLAN:235). The fleet used `TRI_MODES = 2` parity selection (H:126 to 138). The trap runs used f032a48. I did not open the trap fleet's `summary.json` files to confirm their args.
14. **Gate B head dim.** Gate B keeps 4 heads at d_model 320, so head dim is 80. The paper only says "width 256 to 320".
