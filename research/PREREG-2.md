# Second plan: why the gap moved, and whether depth reverses the tying effect

Committed 2026-11-01 about 23:05 CST, before any of these runs. It follows from reading the first study, so everything below is a new prediction on new seeds, not a result.

## Prediction 1: the gap depends on the anchor's distance from chance

The previous-token score of uniform attention over a causal window of T positions is b(T) = (H_T - 1) / (T - 1), which is 0.0594 at T=64 and 0.0349 at T=128. Howe's anchor of 0.10 on a 128-token probe is therefore 2.86 b(128). The first study used 0.10 on a 64-token probe, only 1.68 b(64), so it fired earlier in the run. The two untied arms that left Howe's envelope may have done so for this reason alone.

Test, on seeds 101 to 164, evalEvery 5 to remove most of the 25-step quantization:
- Arms (2,2) and (2,1) at T=64 and at T=128, otherwise as in the first study (r range scaled with T).
- Anchor A: 0.10 flat. Anchor B: 2.86 b(T).
- Pass: every arm's median fraction under anchor B lies inside Howe's envelope [0.735, 0.901], and at T=128 anchors A and B are nearly the same anchor, so T=128 under A is also inside. Fail: any of those outside. If T=64 under A stays below the envelope and T=64 under B moves inside, the 0.10 threshold is shown to be probe-length specific.

## Prediction 2: the tying penalty may shrink or reverse with depth

2610.31421 found looped models acquire in-context learning earlier than a 12-layer dense model. The first study found the opposite from 2 to 4 layers. A dense model should eventually pay for depth, so there may be a crossover.

Test, seeds 1 to 32, lr 3e-3, budget 4000 steps, same family as the first study: arms (8,8), (8,2), (8,1). Report Kaplan-Meier medians and time ratios (8,1)/(8,8), (8,2)/(8,8), (8,8)/(4,4). No direction is predicted. A crossover is claimed only if a looped arm's time ratio interval lies wholly below 1.

## Exploratory, labeled as such

Per-head attention in the (2,1) arm: whether the same head is a previous-token head on the first pass and an induction head on the second, or the block splits its heads between the roles.

## Prediction 3: is the speed the curriculum's or the configuration's?

Added 2026-11-02 about 03:00 CST after the second novelty pass, before any of these runs. Same model and learning rate as the first study's (2,2) arm, seeds 1 to 16, budget 8000 steps, three curricula against its uniform-tiled result: bigram language with Howe's clone rule, uniform tokens with the clone rule, bigram language with tiled repeats. No direction is claimed beyond what these arms show. The tab's speed may be credited to the curriculum only by this comparison.
