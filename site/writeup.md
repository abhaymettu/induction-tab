# Watch a transformer learn to copy in a browser tab

[Open the tab](https://abhaymettu.github.io/induction-tab/) and press Start. The page trains a two-layer transformer while you watch, with no ML library and nothing to install, because the forward pass, the backward pass and AdamW are hand-written WGSL running on your GPU. It trains only on random characters repeating at random periods, so the one thing to learn is that characters repeat.

For the first few hundred steps the model has learned nothing. Then three things happen close together. The second copy of a random row lights up as the model starts predicting characters it has seen once. A head in layer 0 draws a line just under the diagonal of its attention map, because it reads the previous character, and heads in layer 1 draw a parallel line lower down, where they find this character's earlier copy. That is the induction circuit from Olsson et al., [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). And the copy advantage, the loss on the first pass through a repeated random sequence minus the loss on the second, reaches 2 nats on two consecutive evals, which is the event Gunner Levi Howe defined in [arXiv 2609.19000](https://arxiv.org/abs/2609.19000).

Then you can type at it. Type something that repeats and the model carries on, so `abcabcabcabcab` gives `cabcabc`. Press a head to silence it. Silence all of layer 0 or all of layer 1 and copying dies.

I wanted the transition to be something you can watch and poke, instead of a plot. A run is short enough to repeat with another seed, and you can silence a head as soon as its line appears.

## How long it takes

On an M4 Pro in Chrome, seeds 1 to 5 reach the event in a median of 4.9 s with drawing off. A run you watch also draws, so its clock reads higher. The real page with drawing on, beside another GPU job, has a median of 8.7 s, or 5.8 s on each seed's best round, and I have no reading on an idle GPU. The picture above is one watched run with another training job on the same Mac. I only measured an M4 Pro and SwiftShader, so I make no claim about other GPUs, Safari, Firefox or real phones. A browser with no WebGPU plays a recording of a real run.

A tab doesn't beat native here. On the same machine and config PyTorch on MPS takes 5.5 ms per step against 10.3 ms for this trainer. The tab trades that speed for needing nothing installed.

## Checking it against PyTorch

With the same weights and batches for 20 steps, on five configurations including the tab's own, the worst per-tensor gradient difference is about 1.1e-6 in relative L2. Then the tab's model ran 64 seeds both ways, one on Metal and the other PyTorch on a CPU, and both reach the same event step and the same anchor step on 64 of 64 seeds, compared at evals 25 steps apart. The anchor is Howe's precursor, the first eval where a first-layer head reaches a previous-token score of 0.10. Those runs stop at the event, so they say nothing about training after it.

All 64 seeds fire, at a median of step 475. Two controls, a single layer and a model that never sees a repeat, ran 8 seeds for 6000 steps each with zero events.

Copy advantage does not measure copying alone. A model that expects repeats guesses wrong on characters it has not seen, so its first pass climbs above chance, and on the tab's model at the event a median 30 percent of the advantage comes from that rise.

Other browser demos train through formation too, [TheoremPath's Induction Heads Lab](https://theorempath.com/demo/induction-heads-lab) and [gama's Induction](https://github.com/pariharshyamu/gama) among them, both on the CPU or without a stated GPU backend. [TabGPT](https://github.com/jbingen/tabgpt) and [webgpu-llm](https://github.com/toprakdeviren/webgpu-llm) also write training kernels by hand in WGSL. This one adds hand-written WGSL on your GPU for this transition, agreement with PyTorch to the same event step, and the study below, and I did not find that combination within my search coverage.

## What sets the onset

Once the trainer checked out I used Howe's event and anchor to study what sets the onset, on a smaller model, width 64 and context 64, for cost. Arms ran 64 seeds unless noted, under three plans in the repo ([first](research/PREREG.md), [second](research/PREREG-2.md), [third](research/PREREG-3.md)). Everything below is in steps, because time per step depends on model size.

Random characters, each sequence tiled at a random period, reach the event at a median of step 575. Howe's language, a synthetic one in which each character has a few likely successors, takes 1025 when tiled. Under Howe's clone rule, which copies the start of a sequence once at a random offset, it takes 7500, with 12 of 16 seeds fired by step 8000. Random characters under that rule fired on 0 of 16.

Those arms also differ in repeat probability and period range, so the third plan changed one factor at a time. On Howe's language with the rest fixed, tiling fires on 16 of 16 seeds at a median of 1025, and the clone rule on 2 of 16 by step 8000. The repeat rule is the largest single-factor contrast I can bound. Period range and language under the clone rule leave one side with no median, so I don't rank them.

Training on random repeats isn't new. [shaikabdul185-arch](https://github.com/shaikabdul185-arch/induction-heads-from-scratch), [xanderchinxyz](https://github.com/xanderchinxyz/forming-induction-heads) and gama do it too, and Zucchet et al. ([2505.17863](https://arxiv.org/abs/2505.17863)) found that repetition can greatly speed up the emergence of sparse attention, in theory and on an in-context associative recall task. The result here is the attribution, one factor at a time, on one model.

## Tying weights delays the event

On random-repeat data, at equal depth, tying weights delayed the event. I write arms as (layers, blocks), where a block is one set of weights, so (8,1) is the same block applied 8 times. The delay holds at both learning rates I tested at depths 2 to 4, with 64 seeds per arm.

| One block against untied | Depth 2 | Depth 3 | Depth 4 |
|---|---|---|---|
| lr 3e-3, fit | 1.55 | 2.31 | 3.27 |
| lr 3e-3, KM | 1.52 | 2.17 | 2.81 |
| lr 1e-3, fit | 1.55 | 2.09 | 4.07 |
| lr 1e-3, KM | 1.55 | 2.06 | 4.04 |

Each cell is a time ratio. The fit is a censored log-normal, which the plan preregistered, and KM is the ratio of Kaplan-Meier medians. The two disagree where a few runs stall. Across the 55 ratio intervals I read, at a Bonferroni level of 0.05/55 per interval, the lowest bound on tying is 1.26, so it stays above 1.

I did not find this delay reported within my search coverage. The closest work finds looped models earlier than a dense model, on other tasks, and I don't claim a contradiction with Kapl et al. ([2610.31421](https://arxiv.org/abs/2610.31421)) or Sesodia et al. ([2610.24135](https://arxiv.org/abs/2610.24135)). I also did not separate effective step size from structure, so I claim the delay at these depths, learning rates and data, and no mechanism.

(8,1) is its own case. It fires later, 8 of 8 by step 16000 at a median of 7150, with no prefix signature at the event. The prefix score, also from Olsson et al., stays at or below 0.089 there, where every other arm in every file reads at least 0.27. These runs stop at the event, so this rests on one eval per run and says nothing about what forms later. It's exploratory, on 8 seeds, and I don't explain it.

## One draw at Howe's own setup

Config A is Howe's setup, on this trainer. Seed 1 reaches the event at step 6225, inside the range of 5650 to 7400 across Howe's fleet of 30 seeds, with the anchor at a fraction of 0.847 of the event step against Howe's median of 0.844, computed from the per-seed results in [Howe's repository](https://github.com/gunnerhowe/Research). That is one draw. It isn't a rerun of the fleet and it says nothing about the fleet's spread. My bigram table differs from Howe's, because Howe's table cannot be rebuilt without PyTorch's generator.

## What did not hold

Howe's fleet puts the anchor at a steady fraction of the time to the event, inside a frozen envelope of 0.735 to 0.901. Three of my arms at depths 2 to 4 fell just under it. A second plan predicted that the anchor's distance from chance caused the miss and tested a rescaled anchor. Under it (2,1) at context 64 reads 0.903 against 0.901, so the prediction failed literally on one arm.

Eval spacing explains part of one miss instead. Rerun at 5-step evals on its own seeds, (2,2) at lr 3e-3 reads 0.748 against 0.727 at 25-step, and the other two arms were not rerun that way. Spacing does not explain Howe's curriculum on the small model at 0.628, or (8,2) at 0.683. Those are still open.

The tab's own model on Howe's language and clone rule fired on 0 of 8 seeds by step 8000, so the tab's timing belongs to its own curriculum. I stopped at 8000, so this doesn't say it never would.

## What I did not verify

All 64 seeds fire on the tab's model on an M4 Pro, which doesn't rule out a rare stall, and the small model does stall, with 5 of 64 and 2 of 64 seeds showing no event by step 6000 at width 64 and context 128. The page says so when this happens and offers a new seed. Where I say I did not find something, that means within the searches logged in [research/NOVELTY-2.md](research/NOVELTY-2.md). The two closest demos are low visibility and turned up late, so others may exist.

These are small models, the deepest at 12 layers, and nothing here transfers by itself to a large model or to natural language data. The plans are dated in git and not externally registered, and about 70% of the first plan's 3e-3 arms had already run, unread, when it was committed. The public history leaves out a private working log and copies of third-party papers and of Howe's per-seed results, with commit dates kept. The [README](README.md#rerun-every-number) lists the departures from plan and has the command and output file behind these numbers.

The paper is in the repo and has not been submitted anywhere.
