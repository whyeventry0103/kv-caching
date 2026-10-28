# Review handoff: changes since `bcdcc06` (2026-10-28)

**Review range:** `bcdcc06..HEAD`.
- `3b637fe`: protected-capture adapter, Amendment T, GSM8K proposal.
- `5ad00e9`: information-calibration result.
- The commit that adds this file.

**State.** All refs agree: local, origin, backup and the GPU host. The host
was checked 2026-10-28 17:12 Dubai. GPUs idle; nothing launched since
2026-10-21; nothing approved remains in the queue.

**Authorization.** No GPU work was started for this review. No cap, window
or approval changed.

## 1. Run update

| Campaign | Result | GPU-h used / cap |
|---|---|---|
| M5C-CAL information calibration | ran 2026-10-21 01:45–03:55 Dubai, rc 0, 640/640 rows; read FINAL, **calibration FAIL** (C1, C2, C6) | 2.165 / 4 |
| M5A-RKVPP / M5A-H2OPP protected accuracy | 400/400 each, validated (unchanged since the last review) | 19.24 / 28 · 15.49 / 28 |
| Qwen3-4B (exploratory) | bank 200/200, grid / readout / reverse 400/400 each; read 2026-10-20 | not ledgered |

No other run happened. On 2026-10-26 the GPUs were briefly occupied by an
unrelated project on the same account. That project has since finished, so
there is no effect on this work.

## 2. What to review

### A. Information calibration (`5ad00e9`): instrument development, never paper evidence

**Read.** `results/diag/meeting5/info_calibration/calibration_read.json`.
- FINAL: 40/40 cells, no validation problem. Exit 4 (calibration not
  accepted).
- Input manifest `1f527da6…`; criteria fingerprint `54f9c22a…`, bound to the
  design and matched.

| Criterion | Measured | Threshold | Result |
|---|---|---|---|
| C1 full cache reads out | min MI 0.235 bits; min correct prob 0.203 | ≥ 1.5; ≥ 0.80 | FAIL |
| C2 format | full-cache OTHER 0.691 | ≤ 0.20 | FAIL |
| C3 FINAL | — | — | PASS |
| C4 end-to-end vs clone | 0.0 over 320 readouts | ≤ 0.001 | PASS |
| C5 eviction reaches facts | early-fact retention 0.172 | ≤ 0.90 | PASS |
| C6 range / floor | best compressed arm 0.026 bits | ≥ 0.1 | FAIL |

**Retention, early / late fact** (`retention_summary.json` + its script):

| Policy | Early | Late |
|---|---|---|
| Random | 0.100 | 0.350 |
| StreamingLLM | 0.000 | 0.000 |
| H2O-style | 0.046 | 0.640 |
| R-KV | 0.544 | 0.516 |

**Interpretation to check:**
- *The failure is at the reader.* The full cache itself fails C1 and C2.
  Compressed-arm floor values therefore cannot be attributed to eviction.
- *No information result exists for any pairing.* The registered Random −
  H2O-style contrast is about 0; Random − StreamingLLM is exploratory and
  also about 0.
- *Hypothesis, untested:* the answer format for a reasoning-distilled model
  on a raw prompt.
- *Pre-registered next step:* one redesign on calibration data, by dated
  pre-data amendment, under a separate approval. A second failure abandons
  the instrument.

**Reproduce on the host.** Read-only; it writes one JSON file.

```
R=~/kvcache-storage/runs/info-calibration-m5c
python scripts/report_information_probe.py --rows $R/readout --design $R/design.json \
  --out /tmp/cal.json --calibration-criteria configs/meeting5/info_calibration.yaml \
  --require-calibration-pass
```

The expected exit code is 4, with the criteria above.

### B. Protected M/Δ/ω capture adapter (`3b637fe`): built and CPU-tested, not run

- **`src/kvcache/capture_protected.py`.** `ProtectedEventTap` reads each
  prune from `press.protect.maps` and verifies it bitwise: the pre-prune keys
  gathered at the record's indices must equal the retained cache, and no
  prompt position may be evicted. `replay_condition_protected` reuses the
  unchanged `capture.replay_condition` through a scoped tap swap, restored on
  any exit. `capture.py` is untouched, because its bytes carry six generation
  identities.
- **`scripts/run_capture_pp.py`.** It refuses:
  - non-protected presses;
  - a missing cohort;
  - budgets other than 256;
  - a held config;
  - running without `--i-understand`, or off the host;
  - a foreign-identity resume;
  - a campaign without a same-identity smoke PASS.
- **Tests.** `tests/test_capture_protected.py` has 12 tests; 10 need torch and
  skip locally. They were run on the host CPU with a tiny Qwen2, where 12
  passed, and 101 passed with the capture and press tests.
  - The identity controls are exact: `random_pp` with the prompt length
    forced to 0 reproduces `random`, and `rkv_pp_v1` at 0 reproduces `rkv`.
  - A corrupted record is refused.
  - The unprotected tap refuses a protected press.
- **Cohort, frozen before data.** 23 strict-fit capture problems × 4 seeds =
  92 traces. Protected Random − StreamingLLM accuracy on these keys is +0.043
  (descriptive, from the committed condition table).
- **Proposal.** `docs/paper/CAPTURE_PP_SMOKE_TIMING_PROPOSAL.md`: smoke cap 1
  GPU-h; campaign about 13.8 GPU-h (cap 16), or about 8.2 for two policies.
  Costs are from the measured unprotected capture.
- **Known limit, reported.** At a prune interval of 4 (< R-KV's 8-entry
  window), the protected R-KV press's own guard refuses. The registered
  interval is 64; the cause at 4 was not isolated.
- **No queue job yet.** A new marker job needs a typed gate in
  `kvcache/gates.py`. That file is in the S.1 readers' fingerprint, so the
  stored S.1 reads would need re-running. It therefore lands with the
  approval.

**Reproduce the tests on the host CPU** (no GPU):

```
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_capture_protected.py tests/test_capture.py
```

### C. Amendment T (`3b637fe`; `docs/protocol/PREREGISTRATION.md`)

It was written before any information data existed.
- Random − StreamingLLM is added as an M5-C confirmation contrast, beside
  the preserved Random − H2O-style, with Holm over the two. It is labelled
  as chosen after the M5-A outcome.
- Calibration is unchanged, and every calibration comparison is exploratory.
- The protected-capture cohort is frozen.
- The full MI evidence chain is restated: a readout, then a separately
  generated later answer predicted beyond retention, age, occupancy and the
  existing proxies, then equal-budget interventions, then a natural-task
  bridge.

### D. GSM8K timing proposal (`docs/paper/GSM8K_TIMING_PROPOSAL.md`)

- 24-problem calibration split × 2 seeds, full cache: 0.82–2.76 GPU-h from
  the measured 0.0253 s/token (cap 3).
- Blocked on a pid-filter grid adapter (CPU, additive) that is not yet built.

## 3. Checks run on this tree

- Local gates, each run bare: `make lint`, `make test` (969 passed, 91
  skipped), `make dryrun`, `make audit`, and the `paper_numbers.json` diff.
- These ran at `5ad00e9`; this file adds documentation only.

## 4. Open decisions (owner)

1. The calibration redesign (approve one pre-registered redesign, or stop
   the instrument).
2. The protected-capture smoke, then the campaign.
3. The GSM8K timing run, after the adapter is built.
4. An assignee for the 42-row Qwen3 human scoring review.
