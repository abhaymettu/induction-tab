# Third plan: closing what a hostile review found

Committed 2026-11-02 10:16 CST, before any of these runs. An independent reviewer attacked the first results. These arms answer what it found. All run on the WGSL trainer, whose runs matched PyTorch at every step checked. `node tools/runs.mjs <name>` reruns each.

## 1. The curriculum contrast was compound

The clone arms used pRep 0.75 and period range [4, 60). The tiled arms used pRep 1 and [4, 32). So the 13.1 ratio mixes the repeat rule, the repeat rate and the period range. Four new arms on the study's (2,2) model at lr 3e-3, seeds 1 to 16, budget 8000:
- `crossed-rule-bigram`: Howe's language, clone rule, pRep 1, [4, 32). Differs from bigram tiled only in the rule.
- `crossed-rule-uniform`: uniform, tiled, pRep 0.75, [4, 60). Differs from uniform clone only in the rule.
- `crossed-range-uniform`: uniform, tiled, pRep 1, [4, 60). Differs from the main arm only in the period range.
- `crossed-range-bigram`: Howe's language, clone rule, pRep 0.75, [4, 32). Differs from bigram clone only in the period range.
Report medians and ratios. No direction is claimed beyond the data. Only single-factor contrasts may be credited to a single factor.

## 2. Negative controls on the tab's own config

The earlier controls ran on the study model. `tab-neg-L1` (one layer) and `tab-neg-norep` (pRep 0), seeds 1 to 8, budget 6000. Prediction: no event in any seed.

## 3. Both passes

Copy advantage is first-pass loss minus second-pass loss, and a model that expects repeats can do worse than chance on the first pass. So the measure is not copying alone. `tab-browser` now logs both, so the event can be split into its two parts.

## 4. The tab's own model, 64 seeds, both stacks

`tab-browser` and `tab-native`, seeds 1 to 64, budget 3000. Pass: paired median difference within 25 steps. Also report how many seeds fail to reach the event by step 3000, since that is what a reader would see.

## 5. Eval spacing on the same seeds

`fine-2-2`: the study's (2,2) arm, seeds 1 to 64, evals every 5 steps, continuing 75 steps past the event. Reading its curves every 25 steps must give the study's event step on every seed, or this test is void. Then compare the gap at both spacings on the same seeds. This replaces the derived spacing in `tools/analyze_anchor.py`, which was an approximation on other seeds.

## 6. Depth

2610.31421 contrasts a 12-layer looped model with a 12-layer dense one. `deep-12-12` and `deep-12-4`, seeds 1 to 16, budget 4000, lr 3e-3. No direction is predicted. A crossover is claimed only if the ratio's interval lies wholly below 1. `deep-8-1-long`: 8 layers with 1 block, seeds 1 to 8, budget 16000, because all 32 earlier seeds stopped at 4000 with the prefix score near chance. It answers whether that arm is slow or does not form at all.

## 7. Howe's config A, more seeds

`config-a`, seeds 1 to 5, as many as lock time allows. Reported as the event window and the fraction for that many seeds, not as a rerun of a 30-seed fleet.
