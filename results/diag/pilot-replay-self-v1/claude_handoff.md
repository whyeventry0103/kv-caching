# M4.1/M4.2 Gate Report for Claude

## M4.1 corrected tokenized reference

- Run: `pilot-fullcache-tokenized-v2`
- Prompt protocol: rendered chat template tokenized with
  `add_special_tokens=False`; every sidecar contains exactly one BOS.
- 20/20 result rows and 20/20 exact token-ID sidecars validated.
- Accuracy: 20/20 correct.
- Truncation: 0/20.
- Mean generated tokens: 2,407.2; range: 990–4,130.
- Stack: torch 2.7.1, transformers 4.56.2, kvpress 0.5.3.
- No serialized tensor/cache artifacts.

The first tokenized attempt exposed that all earlier M2/M3 pilot generations had
duplicated BOS because the rendered template already contained BOS and tokenizer
special tokens were added again. Those rows remain provenance/harness diagnostics,
not final evidence.

## M4.2 self-replay negative control

- Smoke run: `pilot-replay-self-smoke-v1`
  - 1 trace, 1,004 replay steps.
  - Exact TV was 0.0 at every step.
- Full run: `pilot-replay-self-v1`
  - 20/20 replay rows and 20/20 scalar-only profiles passed audit.
  - 48,144 total replay steps.
  - Global min/max/mean/summed exact TV: 0.0 / 0.0 / 0.0 / 0.0.
  - Nonzero TV steps: 0.
  - Crossings at tau={0.05,0.10,0.15,0.20}: 0 at every threshold.
  - Onset/TDP crossing: none in 20/20 rows; operational control percentile 1.0.
  - Compression events: 0.
  - No errors and no serialized tensor/cache artifacts.

Artifacts:

- `summary.json`
- `table.md`
- `self_replay_control.svg`

## Gate status

No compressed replay has started. The mandatory M4.2 stop is active pending your
review.

One protocol consequence needs an explicit ruling: the accepted M3 compression
pilot used duplicate BOS and cannot provide `total_change` against the corrected
M4 reference. A corrected 80-row compressed free-run pilot must be regenerated
before the scalar M4.4 readout/residual decomposition is exchangeable. Please rule
on sequencing that rerun relative to M4.3 compressed replay and M4.4 readout.
