# Proposal: protected M / Δ / ω capture — smoke, then campaign (2026-10-20)

**Status: PROPOSED.** Nothing here is approved or launched.
- Both configs are `held: true`.
- No queue job exists yet (see "Approvals needed").
- The campaign runner also refuses without a same-identity smoke PASS.

## Why this is the next direct test

The protected Random − StreamingLLM accuracy gap persists (M5-A strict read:
+0.054 [+0.023, +0.090], descriptive; the registered interaction rejects).
Every M, Δ and ω we have was measured with the prompt evictable. Until
protected M, Δ and ω exist under the same conditions, we cannot say whether
the metrics still rank Random unfavourably where its protected accuracy is
higher. Protected accuracy is never combined with unprotected captures.

## What was built (CPU, 2026-10-20)

- **`src/kvcache/capture_protected.py`.**
  - *ProtectedEventTap:* it reads each prune's selection from the protected
    press's own position record (`press.protect.maps`). It then verifies the
    record bitwise: the pre-prune keys, gathered at the record's indices,
    must equal the retained cache, and no prompt position may be evicted.
  - *Same replay as the unprotected capture:* it calls the validated
    `kvcache.capture.replay_condition` with this tap swapped in for that call
    only. Every existing live control still runs:
    - bitwise equality with the full branch before the first eviction;
    - layer-0 keys at the mapped positions equal the full cache;
    - layer-0 Δ is within the algebra tolerance;
    - full/full is exactly zero.
  - *capture.py untouched:* `capture.py` is unchanged, because its bytes
    carry six generation identities with complete stored runs. The new code
    has its own identity family, `capture_pp`.
- **`scripts/run_capture_pp.py`.** It refuses:
  - non-protected presses;
  - a missing cohort source;
  - any budget other than 256;
  - a held config;
  - running without `--i-understand`, or off the remote host;
  - a resume into a foreign identity;
  - a campaign without a same-identity smoke PASS.
- **Configs.** `configs/experiments/capture_pp_smoke_m5.yaml` and
  `capture_pp_math500_b256_m5.yaml`, both covered by `make dryrun`; launcher
  alias `capturepp`.
- **Tests.** `tests/test_capture_protected.py` has 12 tests, run on the GPU
  host's CPU with a tiny eager Qwen2 (no GPU). All pass:
  - every protected policy holds every live invariant, and never evicts a
    prompt position;
  - **identity control:** `random_pp` with the prompt length forced to 0
    reproduces unprotected `random` exactly (M, Δ, event steps and the
    latent sums), and `rkv_pp_v1` at 0 reproduces `rkv`;
  - a corrupted selection record is refused;
  - the unprotected tap refuses a protected press;
  - the tap swap is scoped and restored on error;
  - the config contract holds;
  - the smoke gate refuses missing, failed or stale markers.
- **One observation, reported plainly.** In the tiny model with a prune
  interval of 4, shorter than R-KV's 8-entry recent window, the protected
  R-KV press's own guard refused ("a protected position … was evicted").
  At interval 8 it passes. The registered configs use interval 64, the
  setting the R-KV-protected smoke PASS and 400 generated rows already
  validate. The cause at interval 4 was not isolated.

## Predeclared cohort (frozen before any protected capture exists)

The 25 capture problems used by `capture-b256`, restricted to the strict-fit
cohort.
- *Excluded:* test/algebra/1349 and test/precalculus/1199, which relax B=256
  under protection.
- *Kept:* **23 problems × 4 seeds = 92 reference traces**, × 4 protected
  policies = **368 conditions**.

Protected accuracy will be compared on exactly these 92 keys. Descriptive
values from the committed M5-A condition table:

| Policy | Protected | Unprotected |
|---|---|---|
| Random | 0.554 | 0.402 |
| StreamingLLM | 0.511 | 0.272 |
| H2O-style | 0.620 | 0.196 |
| R-KV | 0.587 | 0.402 |

Random − StreamingLLM, protected, is +0.043 on these keys. That has the same
sign as the strict cohort, but rests on only 23 problems.

## Cost (from measured timings)

The unprotected `capture-b256` measured, per row: 12.08 h of condition
replay over 400 conditions (about 109 s each), plus 2.92 h of full-branch
replay over 100 traces (about 105 s each).

| Step | Work | Estimate | Proposed cap |
|---|---|---|---|
| Smoke | 2 traces × (full/full + 4 protected + 2 identity replays) | about 0.5 GPU-h plus model load | **1 GPU-h** |
| Campaign, 4 protected policies | 368 conditions + 92 full branches | about 13.8 GPU-h (about 7 h per GPU on two seed shards) | **16 GPU-h** |
| Smaller option: Random-pp and StreamingLLM-pp only | 184 conditions + 92 full branches | about 8.2 GPU-h | 10 GPU-h |

*Uncertainty, stated once:* protected caches retain the prompt, so per-step
cost may differ slightly from the unprotected capture. The smoke records
real timings before the campaign is requested.

## Read (to be frozen before any campaign data)

Paired problem-level contrasts of M, Δ and ω on the 92 keys, beside
protected accuracy on the same keys.
- *Primary pairing:* Random − StreamingLLM. Secondary pairings: Random −
  H2O-style and Random − R-KV.
- *Joint label:* the `matched_b256` rule. MISMATCH ESTABLISHED only when the
  accuracy CI and the metric CI both exclude zero in the stated directions,
  with the family correction declared up front.
- *Definitions:*
  - M is full-cache attention mass on evicted positions (selection loss);
  - Δ is attention-shape TV on the retained set (state drift);
  - ω = ‖δ⊥‖/‖δ∥‖ is a ratio, so a larger ω is not by itself more
    answer-relevant damage.
- *Reader status:* the protected matched reader is not yet built (CPU). It
  will be written and frozen with the campaign request.

## Approvals needed

1. **Smoke, cap 1 GPU-h.** On approval, these land together:
   - a campaign;
   - a `capturepp-smoke` queue job (marker `capture-pp-smoke-PASS.json`);
   - `held: false` in the smoke config;
   - **a typed gate kind** for the smoke marker.

   `tests/test_gates.py` closes the list of untyped marker jobs, so a new
   smoke needs a typed gate. `kvcache/gates.py` is in the S.1 readers'
   analysis fingerprint, so adding the gate means re-running the stored S.1
   reads on the host over unchanged rows (no GPU). That is the same
   precedent as the protected-selector and R-KV-protected smokes, and it is
   why the queue job is not added before approval.
2. After a smoke PASS and its timings: the campaign, with its cap (16 GPU-h
   proposed, or 10 for the two-policy option) and the frozen read.
