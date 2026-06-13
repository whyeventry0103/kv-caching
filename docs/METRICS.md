# Metrics

## Decomposition

Teacher-forced paired replay measures readout distortion on a fixed reference
trace. The residual after subtracting this on-reference readout change from the
total free-run accuracy change is **accuracy change unexplained by on-reference
readout**. It is not itself a measured cause or proof of trajectory distortion.

## ADD

ADD is total variation on coarsened canonical-answer support. Excess ADD is
assessed against a per-problem permutation-test null, never split-half
subtraction. Report Miller–Madow entropy shift, the unbiased Codex pass@k
estimator, paired bootstrap uncertainty, and BH-adjusted p-values. ADD is for
sampled inference; greedy decoding uses answer flip rate.

Missing, truncated, invalid, or explicitly out-of-support answers map to one
frozen `__OTHER__` category. Truncated generations remain rows, are scored
incorrect, count in all denominators, and contribute to cell truncation rate.

## Divergence

Replay computes per-step TV on the fly and stores only scalar profiles. Full
vocabulary dumps across steps are prohibited. A top-k=256 fallback records a TV
interval induced by unknown tail overlap. AUC and fraction above each fixed tau
in `{0.05, 0.10, 0.15, 0.20}` are primary; TDP is localization only.

Report both normalized mean TV (`auc_tv`) and unnormalized total TV with trace
length. Compare total TV across trace-length regimes only with that confound
made explicit. Views include onset-aligned, content-token (numerals, operators,
answer span), and segment aggregation.

## Instruments and controls

The ladder comprises evicted-anchor attribution and latency, KV restoration,
full-cache-fork versus random-fork recovery, reverse replay, and TV-matched
noise. The whole suite receives a two-full-cache-bank negative control, where
excess ADD should be near zero and TDP percentile near one.

Every absolute budget must be accompanied by its realized retained/original KV
compression ratio.
