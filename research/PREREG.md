# Seeds study plan

Committed 2026-11-01 about 22:00 CST, before any study result was read. By then the negatives and about 70% of the 3e-3 arms had already run on the VM, unread. What had been seen: a one-factor search around a nearby config (3 seeds each, 33 runs), and a handful of browser runs of the 2-layer model. The study arms and estimators below were fixed without those runs' looped or deeper arms, none of which had been tried.

## Question

On random characters made periodic, does tying weights across layers change when induction arrives, once tying is separated from depth and from the learning rate? Does the precursor gap of Howe (arXiv 2609.19000) hold outside Howe's 2-layer fleets?

## Design

- Curriculum and model: `tools/gen_study.py`. V=96, T=64, d=64, H=4, B=16, MLP 4x, LayerNorm without learned gains, tokens uniform, tiled repeats with period r in [4, 32), every sequence repeats.
- Arms (L layers applied, u unique blocks, block index is l mod u): (2,2) (2,1) (3,3) (3,1) (4,4) (4,2) (4,1). The (4,2) arm has the unique parameters of (2,2) and the depth of (4,4), which is what separates tying from unique depth, the confound in 2610.31421's Dense 4L.
- Learning rates 3e-3 and 1e-3, because 2610.24135 shows looped against untied can flip with learning rate. 64 seeds per arm per rate. Budget 6000 steps at 3e-3, 8000 at 1e-3. Runs that do not fire are right-censored at the budget.
- Negatives: 8 seeds each, one layer, and no repeats (pRep 0), 6000 steps. Prediction: 0 events.
- Evals every 25 steps on a fixed 16-sequence copy probe of [x ; x]. Event: copy advantage at or above 2 nats on two consecutive evals, reported at the second. Anchor: first eval with previous-token score at or above 0.10, maximum over heads and over the first max(1, floor(L/2)) applied layers. This generalizes Howe's layer 0 the way Howe's emergence-watch package does.

## Estimators

- Kaplan-Meier survival curve per arm, median with bootstrap 95% interval (2000 resamples over seeds).
- Primary contrast: accelerated failure time. Per-arm censored log-normal fit, time ratio exp(mu_a - mu_b), bootstrap 95% interval. Chosen over Cox because per-arm onsets are narrow and Cox can hit monotone likelihood.
- Contrasts, each in steps and in floating point operations (a looped arm costs the same per step as its untied twin, a deeper arm costs more):
  - tying at equal depth: (2,1)/(2,2), (3,1)/(3,3), (4,1)/(4,4), (4,2)/(4,4)
  - depth at equal unique count: (4,2)/(2,2), (3,1)/(2,1), (4,1)/(2,1)
  - untied depth: (3,3)/(2,2), (4,4)/(2,2)
- Gap: t_anchor / t_event per seed. Report median, interquartile range, range, share inside Howe's frozen envelope [0.735, 0.901], and average-rank Spearman of t_anchor against t_event. Falsifier: an arm whose median fraction falls outside the envelope.

## Replication wording

"Replicates" is used only for Howe's config A with Howe's anchor and event verbatim. Everything on this curriculum is "extends". The bigram table differs from Howe's because Howe's cannot be rebuilt without PyTorch's generator; Howe's own Gate C ran a fresh table.

## Native against browser

The same seeds of the (2,2) arm at 3e-3 run in the WGSL trainer. Pass: paired median difference in event step within one eval (25 steps), with its bootstrap interval reported. Bitwise equality is not expected, since float rounding differs and training is chaotic.
