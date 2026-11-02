# Novelty audit, second pass: induction-tab

Written 2026-11-02 about 02:45 CST by an adversarial reviewer, after the first results. It does not redo `research/NOVELTY.md`. It attacks five new claims and closes that file's residual list. Coverage is 13 search calls (one over the 12 budget, see the search log) plus direct reads of every source quoted below. "Not found" means not found within this coverage, never that the thing does not exist.

**The main result: this pass overturns the recommended wording in `NOVELTY.md` item 1.** "The first we could find" for a tab that trains through the phase change and lets the viewer ablate heads is no longer true. Two existing browser demos train through formation live, and one of them (TheoremPath) also has head ablation. None of the eight named residuals does either.

## Verdicts

| # | Claim | Verdict | Closest prior work | What is actually new after accounting for it |
|---|---|---|---|---|
| A | A browser demo trains a model through induction-head formation and shows it live | PRIOR | TheoremPath Induction Heads Lab, https://theorempath.com/demo/induction-heads-lab (2-layer, in-browser, prefix-match score live, head ablation, about step 600, date not found). `pariharshyamu/gama` `induction/`, https://github.com/pariharshyamu/gama (created 2026-07-20; trained and drawn live in the browser). Transformer Atlas, https://github.com/Saidgurbuz/transformer-atlas (induction preset, trained model only). The eight named residuals are all negative. | Hand-written WGSL on the GPU (TheoremPath says "no GPU"; gama's README never mentions WebGPU or WGSL). A model that keeps MLPs and LayerNorm. Free human typing, if it is a real input (both labs use letter sequences). Previous-token and induction scores together with ablation on one page was not found. |
| B | A uniform random-repeat curriculum brings induction at about 500 steps against about 6300 for 2609.19000's bigram language, and the model continues human-typed text it never saw | PARTLY PRIOR | Curriculum: `shaikabdul185-arch/induction-heads-from-scratch` (uniform random repeats, 7,620 steps), `xanderchinxyz/forming-induction-heads`, gama, Transformer Atlas. Speed: TheoremPath (about 600 steps on planted patterns, lr 1e-2). Olsson et al. 2022 use random repeats only as a probe. | The curriculum is not new, and the same curriculum at lr 1e-3 took 7,620 steps, so the 500 against 6300 gap cannot be credited to the curriculum. A measured noise-to-human-text transfer was not found within this coverage. |
| C | Weight tying delays induction at equal depth (time ratio 1.55, 2.20, 2.83 at depth 2, 3, 4, lr 3e-3, n=64 per arm) | NOVEL | Opposite direction on other tasks: arXiv [2610.31421](https://www.alphaxiv.org/abs/2610.31421) (looped earlier than dense 12-layer), arXiv [2610.24135](https://www.alphaxiv.org/abs/2610.24135) (looped sooner, 45k against 55k steps). Adjacent cost at scale, not onset: arXiv [2610.40325](https://www.alphaxiv.org/abs/2610.40325). | No paper found shows tying slowing induction or copying onset. The result is new and also conflicts in direction with the closest work, so it must carry its learning rate, its task and the depth-8 arms. |
| D | One head acts as a previous-token head on pass 0 and an induction head on pass 1 of a one-block model applied twice (8 seeds, exploratory) | PARTLY PRIOR | General idea: arXiv [2609.39892](https://www.alphaxiv.org/abs/2609.39892) (shared weights route to different operations through attention) and arXiv [2610.19975](https://www.alphaxiv.org/abs/2610.19975) ("different loop visits can play different roles"). Neither names a head playing two circuit roles. | The head-level, named-circuit form. 2609.39892 and 2610.24135 both have 0 occurrences of induction head, prefix match, previous token and same head. |
| E | Howe's 0.10 anchor is 2.86 times the uniform-attention baseline (H_T - 1)/(T - 1) at T=128 | PARTLY PRIOR | Empirical baseline: arXiv [2609.19000](https://www.alphaxiv.org/abs/2609.19000) ("∼10× baseline", data-ablated scores 0.036 to 0.061 against 0.10). A chance level tied to T, for prefix match: TheoremPath ("1/T ≈ 0.07"). | The closed form for the previous-token score, and the point that a flat 0.10 means a different distance from chance at each probe length. An elementary observation, not a discovery, and valuable only if PREREG-2's test bears it out. |

## Evidence per claim

### A. Live browser demos

The named residuals, one by one. None trains through induction or shows induction heads.

| Named item | Read | Finding |
|---|---|---|
| Transformer Explainer | `gh api` README | "It runs a live GPT-2 model right in your browser". 0 mentions of induction, 0 of training. Inference. |
| bbycroft LLM Visualization | `gh api` README | "3D interactive model of a GPT-style LLM network running inference". 0 mentions of induction. |
| TensorFlow.js examples | `gh api` README index only | Trains in the browser. Nearest is `date-conversion-attention` ("Attention mechanism, RNN"). 0 mentions of induction in the index. Example code not read. |
| ONNX Runtime Web | `gh api` `js/web` README, `lib/index.ts`, `package.json` (1.32.0) | 0 lines mentioning training in any of the three. An older training package may have existed; not checked. |
| Observable notebooks | One web search | None found. Observable was not searched directly, so this is weak. |
| surfgrad (`zanussbaum/surfgrad`) | `gh api` README | "WebGPU-powered AutoGrad library", with "(in the future) machine learning models in the browser". Library only. |
| TinyGPT | Repo search and web search | Not located under that name. Unverified, not cleared. |
| jax-js (`ekzhang/jax-js`) | `gh api` README and file tree | Trains MNIST in the browser. Lists "Making a fast, general transformer inference engine or model library" as future work. 0 mentions of induction; no path matches transformer, gpt, attention or induction. |

What the residual list missed, found by searches S10 and S11:

- **TheoremPath Induction Heads Lab** (static page text read; demo not run; no publication date found, so priority over this project is unknown). "A 2-layer attention-only transformer, trained from scratch in your browser on a synthetic copying task. Watch the loss curve dive at the same step the prefix-match score climbs: that joint moment is the induction circuit forming. Right-click any head chip to ablate it and see the score collapse." Also "Pure TypeScript. Hand-written backward pass, gradient-checked against finite differences (atol 5e-3). No tensor library, no GPU." The prompt can be changed ("edit non-BOS cells"). Configuration: 2 layers, 2 heads, d=32, about 5,500 parameters, vocabulary 28, context 32, "AdamW · lr 0.01 · batch 16 · max 1500 steps". Page text has 0 occurrences of "previous-token".
- **gama `induction/`** (README and `tools/circuit.mjs` read; created 2026-07-20, last push 2026-08-11, 0 stars, no hosted URL recorded; not run). "Watch a transformer grow the circuit that lets it copy. A two-layer model is initialised, trained and drawn in the browser, live, on a task it can only solve one way." It ships `npm run gradcheck` and `npm run circuit`, and `circuit.mjs` says "So this trains the shipped model, and then trains TWO CONTROLS". README has 0 mentions of WebGPU, WGSL, ablation or natural text.
- **Transformer Atlas** (README, manifest and tree read; created 2026-10-11). Its induction preset is "2 layers trained to continue repeated random text. Look for a previous-token head in layer 0 and an induction head in layer 1." (`params` 106240, `repeat_accuracy` 0.9971). "A small trained model runs in your browser". It shows a formed circuit and a "One training step" lesson, not the phase change. `tests/browser_training.js` was not read.
- Static induction visualizers, not trainers: `kondareddy-th/deep-mind` `src/viz/inductionCircuit.js`, `sarimahsan/interplens` `ui/js/induction.js` (0 hits for backward, optimizer, Adam or gradient in either). `Fazmin/AILearningGuide` `icl.ts` uses "A hand-set induction head".

### B. Curriculum

**Random repeats as training data are prior.** Two repos match the curriculum directly:

- `shaikabdul185-arch/induction-heads-from-scratch` (README read; created 2026-10-07): "Each sequence is `L` uniformly random tokens followed by the same `L` tokens again". "**Observed transition step: 7,620** (default config, seed 0)." Vocabulary 64, d_model 64, 4 heads, attention only, LayerNorm off, lr 1e-3, batch 64, L 8 to 32. It is one seed. 0 hits for natural, english, tinystories, real text or human.
- `xanderchinxyz/forming-induction-heads` (README read; created 2026-01-21): "I trained small, attention-only transformers on the task of repeating a sequence of random tokens with varying lengths up to the specified max context." 20,000 to 60,000 epochs of one generated batch each. 0 hits for natural or english.
- gama: "Each line is a short random pattern of letters, repeated until the line is full. The pattern's **length changes every line**." Transformer Atlas trains on repeated random text.

**Olsson et al. 2022 do not train on it.** Read in full text: "we define an induction head as one which exhibits the following two properties on a repeated random sequence of tokens". Of 23 occurrences of random sequences or tokens, none describes training on them. Their small models were trained on mixed corpora ("approximately 10% python code"). They note the reverse direction: "Despite being defined narrowly as copying random sequences, induction heads can implement surprisingly abstract types of in-context learning."

Other synthetic work differs in task. arXiv 2404.07129 studies "two-layer attention-only transformers trained on a synthetic ICL task", with phase changes lasting on the order of 1e5 iterations ("≈ 7.5e4 iterations" in one condition), on a classification task, not periodic tokens. arXiv 2511.16893 finds "surface bigram repetition frequency and reliability strongly affect the formation of IHs", which is the rationale for this curriculum, already published.

**The 500 against 6300 comparison does not isolate the curriculum.** This is the sharpest attack.

- It compares configurations that differ in almost everything: this study has V=96, d=64, B=16, T=64, lr 3e-3 (PREREG); Howe's fleet A has V=2048, d=256, B=64, T=128, lr 1e-3 (UPSTREAM-SPEC, NOVELTY).
- The same kind of curriculum at lr 1e-3 took 7,620 steps (shaikabdul). Random repeats alone are therefore not fast.
- Aoyama and Wilcox give a law for updates to emergence, U_PT = e^13.26 B^-0.37 C^-0.62 (2511.16893 section 6.1, fit on natural-language data, so indicative only). Reviewer's arithmetic:

| Config | Law's prediction | Observed | Observed over predicted |
|---|---|---|---|
| Howe B=64, C=128 | 6,081 | about 6300 | 1.0 |
| shaikabdul B=64, C=64 | 9,346 | 7,620 | 0.8 |
| This project B=16, C=64 | 15,609 | about 500 | 0.03 |
| TheoremPath B=16, C=32 | 23,990 | about 600 (prefix-match, not copy advantage) | 0.03 |

Random repeats at lr 1e-3 sit on the natural-language law. The two fast runs have the high learning rates (3e-3, 1e-2) and batch 16. The slow random-repeat run has vocabulary 64, as small as theirs, so small vocabulary alone does not explain it. This points at learning rate rather than the absence of a corpus, though four runs cannot separate the factors. The study's own 1e-3 arms (PREREG, not read here) and a same-config bigram arm can settle it.

gama raises a related warning: "The learning rate decided which algorithm the model implemented." "At 3e-3 it learns the task and never forms the circuit: three seeds, eight thousand steps each", and with MLPs "no head is interpretable: best induction 0.19 against a 0.12 chance baseline". This project reports formation at 3e-3 with MLPs on a different configuration. Publish the configuration so the difference can be checked.

**Transfer.** No study or demo reporting a train-on-noise, test-on-human-text measurement was found (S3, S4, S8). One caveat: Transformer Atlas lets a viewer "**Change the text**" on its induction preset, so the capability may already be reachable in a public demo, but its README reports only synthetic-task accuracy. A sceptic will also say that token-agnostic copying is what an induction head is by definition, so continuing unseen text only where it repeats is expected, not surprising. Say what the text was and whether it repeated.

### C. Tying and onset

No paper found shows tying or looping slowing induction or copying onset. The closest work points the other way, on different tasks:

- 2610.31421 (local copy): "ICL accuracy emerges earlier for the looped, grown, and four-layer dense models than for the dense 12-layer model." Loop 4x3 and Dense 12L match in effective depth, so this is the same contrast, on Omniglot, at 12 layers.
- 2610.24135 (full text read): at lower learning rate "The 4 × 8 looped model still solves in every seed, and reaches the solution sooner on average (45k training steps against 55k)." Also "looped models solve the task where untied models do not".
- arXiv 2607.20594 (full text grepped): "untying hurts extrapolation" and "destroys learnability", with fixed effective depth 12. Final performance, not onset.
- 2610.40325 (full text grepped): "adding loops is not uniformly beneficial: gains eventually saturate, reverse, or become unstable". A cost of looping, but in final quality at scale. It has no onset measure in the passages read.

Attacks the claim must survive:

1. **Learning rate.** 2610.24135 shows this contrast moves with learning rate, and gama shows the circuit itself can depend on it. Quoting only 3e-3 is not enough; report the 1e-3 arms beside it.
2. **Effective step size (reviewer's point, no source).** A tied block sums gradients over its passes, so a fixed lr may act larger on it. A delay at 3e-3 could be an optimization effect rather than a structural one.
3. **Depth range.** Prior results sit at 12 and 32 layers, this one at 2 to 4. PREREG-2's (8,1)/(8,8) arms decide whether these conflict or a crossover reconciles them. They were not read here and must be reported with this claim.

### D. One head, two roles

2609.39892 ("Shared Weights, Selected Computations", full text grepped) owns the general idea: "weight sharing alone does not imply that every loop performs the same operation", and it asks "what routes the shared parameters to different operations?" Its answer is attention routing, "with attention routing as a causal pathway". It works at head level only as patching ("patches the attention patterns of all four second-layer heads"), and "patterns from the wrong loop" score zero, so loop-specific attention patterns are shown. Its task is graph walks. Terms: induction head 0, prefix match 0, previous token 0, same head 0. It assigns no named circuit role to a head.

2610.24135 ("Inside the Loop", full text grepped) probes the residual stream per iteration and finds a staircase, "advancing the solution by one step per iteration" as its question. "Attention head" appears 0 times. Same zero counts.

2610.19975 (LatentLoop, full text grepped): "Prior work suggests that different visits can play different roles even when they share parameters (Blayney et al., 2026; Geiping et al., 2026; Wang et al., 2026b; Zhu et al., 2025)." Its evidence is visit-level residual scale. arXiv 2604.11791 (Blayney et al., grepped) concerns cyclic fixed points; its "induction" hits are proofs by induction, and its one previous-token hit is about mixing. Geiping, Wang and Zhu were not read, so a head-level result could sit there. S6 did not surface one.

The result stands narrowly and rests on 8 exploratory seeds. A sceptic will note that a 2-layer induction circuit with tied weights almost forces something like this, or a split of heads between roles, so the interesting finding is which one occurred and how often.

### E. The anchor and the baseline

- The formula checks out against Howe's own code. `prev_token_score` in `research/upstream/emergence_watch/probes.py` averages attention to q-1 over q in 1..T-1, so uniform causal attention gives (H_T - 1)/(T - 1). This gives 0.03491 at T=128 and 0.0594 at T=64. 0.10/b(128) = 2.865 and 0.10/b(64) = 1.683, confirming PREREG-2's figures.
- Howe reports the baseline as a number: "the precursor is already at 0.37 (∼10× baseline)", and on data-ablated runs "max layer-0 previous-token score 0.036–0.061 vs. threshold 0.10". Local notes record "The step-0 baseline is about 0.036 (`rep_s1` log)" (UPSTREAM-SPEC), in line with 0.0349. The emergence-watch docstring says "the probe batch is part of the instrument". No closed form, T dependence or normalized threshold appears in the paper's grep hits (uniform, chance, baseline, harmonic, sequence length, probe length, normaliz).
- For prefix match, a T-tied chance level is already stated: TheoremPath "A randomly initialized head sits near 1/T ≈ 0.07". gama quotes "a 0.12 chance baseline". Neither concerns the previous-token score.
- S7 and S12 returned nothing relevant.

## Search log

Tool key as in `NOVELTY.md`. All orx calls used `ORX_NO_UPDATE_CHECK=1`, orx 0.2.19, telemetry off, no paid or remote backend.

| # | Tool | Query | Top hits | Verdict on the claim it targeted |
|---|---|---|---|---|
| S1 | gh-repo | `surfgrad` | zanussbaum/surfgrad; unrelated Unity surface-gradient repos | Located; negative for A |
| S2 | gh-repo | `tinygpt browser` | 0 rows | TinyGPT not located |
| S3 | orx-kw | induction heads trained on repeated random tokens | 2610.09124 Spatial Induction Heads; 2607.02800 Interpolate N-Grams; 2604.01094 Temporal Dependencies | Negative for B transfer |
| S4 | orx-emb | training only on random token sequences with repeats, copying transfers to natural text | 2610.19247 Copy-Paste LMs; 2610.00771 transfer between memorization; 2606.12058; 2511.16893 Aoyama; 2512.18634 Kawata | Aoyama's law; negative for transfer |
| S5 | orx-emb | weight tying or looping delays onset of ICL or copying at equal depth | 2610.40325; 2609.37379; 2610.31421; 2610.24135; 2610.38422; 2610.19768 | Negative for C |
| S6 | orx-emb | same attention head, different circuit roles across loop iterations | 2610.40325; 2610.19975 LatentLoop; 2610.02383; 2609.39892 | General idea prior for D; no head-level result |
| S7 | orx-kw | previous-token head score uniform attention baseline sequence length | 2610.17959; 2610.41218; unrelated | Negative for E |
| S8 | web | train only on repeated random tokens, induction heads, test on natural text | a course page; 2511.16893; 2504.03022; greaterwrong post; 2410.11474; hapax.baulab.info (summary only, not read) | Negative for B transfer |
| S9 | orx-kw | looped transformer slower to train than untied, tying slows convergence | 2610.24135; 2607.20594; 2609.40335 (embedding tying, different sense) | Negative for C |
| S10 | web | Observable or TinyGPT, live in-browser training, induction head, prefix matching | **theorempath.com induction-heads-lab**; kody-w trainer (no induction focus, summary only); Transformer Explainer; TrAVis | **Prior for A** |
| S11a | gh-code | `induction head`, TypeScript | Fazmin/AILearningGuide; sarimahsan/interplens (JS); Saidgurbuz/transformer-atlas | Static or inference demos |
| S11b | gh-code | `induction head`, JavaScript | **pariharshyamu/gama**; kondareddy-th/deep-mind; neelnanda-io/dont-go-quiet-on-me (not read) | **Prior for A** |
| S12 | orx-emb | head detection threshold depends on sequence length; chance attention to previous token | 2610.34507; 2610.34350; 2609.37879; 2609.08574; unrelated | Negative for E |

S11 ran once per language, so the total is 13 calls against a budget of 12. S3 to S7 were also re-issued once unchanged, because the first output was cut off before the full hit list; no new queries.

### Reads (not counted as searches)

| Source | Command or route | What was read |
|---|---|---|
| 2609.39892, 2610.24135 | `orx paper <id> --full` | Abstract; full text grepped for the terms in D and timing terms |
| 2209.11895 Olsson et al. | `orx paper --full` | Grepped; all 23 random-sequence hits checked for training use |
| 2404.07129, 2312.03002, 2402.11004, 2410.11474, 2606.12058, 2512.18634 | `orx paper --full` | Abstracts and targeted grep for task and data only |
| 2511.16893 Aoyama and Wilcox | `orx paper --full` | Abstract and section 6.1 |
| 2610.19975, 2604.11791, 2607.20594, 2610.40325, 2610.38422, 2610.19768 | `orx paper --full` | Grep only |
| 2609.19000, 2610.31421 | [2609.19000](https://arxiv.org/abs/2609.19000), [2610.31421](https://arxiv.org/abs/2610.31421) | Grep and passages quoted |
| `research/upstream/emergence_watch/probes.py`, `research/UPSTREAM-SPEC.md` | local | `prev_token_score`, baseline line |
| TheoremPath lab | `ctx_fetch_and_index` plus plain fetch | Static page text; demo not run |
| gama, Transformer Atlas, shaikabdul185-arch, xanderchinxyz, kondareddy-th, sarimahsan, Fazmin, surfgrad, jax-js, transformer-explainer, llm-viz, tfjs-examples, onnxruntime | `gh api` | READMEs and named files, as stated above |

Not done: no demo was run; HAPAX, the greaterwrong post and the kody-w trainer were seen only as a summary or a cached page and are not relied on; Geiping, Wang and Zhu were not read; the study's own results at 1e-3 and depth 8 were not read.

## How to phrase the claim so it survives

**Withdraw:** `NOVELTY.md` item 1's "the first we could find", and its statement that nobody found "stages, measures or names the induction phase change in a tab". Do not say a browser demo of induction formation is new.

**A. Safe wording.** "Other browser demos train induction heads live, including TheoremPath's lab and gama's Induction, both on the CPU or without a stated GPU backend. This one runs on the viewer's GPU in hand-written WGSL, keeps MLPs and LayerNorm, and streams previous-token and induction scores with head ablation on one page. We did not find that combination within our search coverage." Keep each listed feature only if the shipped page really has it, and credit both demos by name.

**B. Safe wording.** "Training on uniform random tokens made periodic forms induction heads with no corpus, as earlier hobby projects also show (shaikabdul185-arch, xanderchinxyz, gama). In our configuration (V=96, d=64, B=16, T=64, lr 3e-3) the event arrives at about 500 steps. Howe's fleet A, a different configuration, takes about 6300, so this is not a measured curriculum effect." For the attribution, run a same-config bigram arm first; until then do not say the curriculum causes the speed-up. Transfer: "A model trained only on random repeats continued [repeating/non-repeating] human-typed text. We did not find this measured elsewhere within our coverage." Fill in the bracket from what was tested.

**C. Safe wording.** "On random-repeat data at lr 3e-3 and depths 2 to 4, tying weights delayed induction at equal depth (time ratios 1.55, 2.20, 2.83; 64 seeds per arm). We did not find this reported within our coverage. The closest work finds the opposite direction on other tasks and greater depth (2610.31421, 2610.24135), and we do not claim a contradiction." Report the 1e-3 arms and the depth-8 arms in the same paragraph, whichever way they fall. Do not generalize beyond these depths or to "tying slows learning".

**D. Safe wording.** "Exploratory, 8 seeds: in a one-block model applied twice, the same head acted as a previous-token head on pass 0 and an induction head on pass 1. Earlier work shows shared weights can run different operations per loop through attention routing (2609.39892) and that loop visits can differ in role (2610.19975). We did not find a named circuit role assigned to one head across passes within our coverage." State how many of the 8 seeds show it, and how many split heads instead. Do not write "first" until Geiping, Wang and Zhu are checked.

**E. Safe wording.** "The previous-token score of uniform causal attention over T positions is (H_T - 1)/(T - 1), which is 0.0349 at T=128 and 0.0594 at T=64. Howe's 0.10 is therefore 2.86 times this baseline on Howe's 128-token probe and 1.68 times it on a 64-token probe. Howe reports the baseline empirically (about 10 times baseline at 0.37). We did not find the closed form or this dependence on probe length stated within our coverage." Present it as a small observation that explains a measurement, and claim more only if PREREG-2's anchor test passes.

**Residual risk to state.** The two closest priors for A are low-visibility (0 stars, a page with no date), and both surfaced only in the last three searches. Other small demos of this kind may exist that this coverage missed. Search engines rank them poorly, which is exactly why they were not in the first audit.
