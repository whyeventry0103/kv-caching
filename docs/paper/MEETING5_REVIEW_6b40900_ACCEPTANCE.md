# Review of 6b409001: repair pass and acceptance (2026-10-14)

**Scope.** This is the bounded repair pass the review requested, covering R1–R4
and the protocol corrections.

**Owner decisions recorded in-session, 2026-10-14:**
- the essential pair plus the smoke are approved, with caps 1 / 28 / 28 GPU-h;
- the equivalence margin is ±0.03, with ±0.05 as a sensitivity;
- BBH is pinned at `9ee07bd…` / blob `936cd6d8…`, for preparation only.

Fresh-uniform, the capacity comparators, the information calibration and the
M5-B generation are **not** approved and stay held.

The four parts of this report are kept separate, as the review asked:
1. **Software acceptance** — passed, below.
2. **Real-smoke acceptance** — not yet. No GPU work was done in this pass. The
   R-KV-pp smoke runs only after the approved P1/P2/P3 work.
3. **Measured scientific outcomes** — none new. No reader change moved any
   stored number (see "Re-read of M5-eligible artifacts").
4. **Incomplete** — listed at the end.

## Remote refs

- The reviewer saw `backup/main` at `7698127`, two commits behind. That was
  confirmed, and the missing commits were pushed normally (`7698127..6b40900`).
- Both refs were verified at `6b4090018007eac7a592cf141f65c6856b14733b` before
  this pass.
- Both are pushed and re-verified at the repair commit; see the final status
  line in `docs/project/context.md`.

## Acceptance table (real CLIs and runtime gates)

**Before** is the reviewer's script, run unmodified at `6b40900`
(`review_6b40900_repro_before.json`). **After** is the same cases run through
the repaired CLIs and gates (`review_6b40900_acceptance_after.py.txt` →
`review_6b40900_repro_after.json`). Everything is under
`results/diag/meeting5/review_6b40900/`.

| Case | Before | After |
|---|---|---|
| M5-A, Random+P `correct` = `"False"` | FINAL, exit 0, accuracy 1.0 | INVALID, exit 2 |
| M5-A, one `correct` = NaN | FINAL, exit 0 | INVALID, exit 2 |
| M5-A, extra seed 999 | FINAL, exit 0 | INVALID, exit 2 |
| M5-A, duplicate expected row | PARTIAL, exit 0 | INVALID, exit 2 |
| M5-A, wrong model + bogus provenance | FINAL, exit 0 | INVALID, exit 2 |
| M5-A, torn JSON after a FINAL | exit 1, old FINAL left in place | INVALID, exit 2; old FINAL retired (`.superseded-*`) |
| M5-A, also new: wrong M5 identity, wrong prompt hash, truncated-but-correct, duplicate outside the cohort | — | INVALID, exit 2 |
| M5-A, missing row | — | PARTIAL, exit 3 |
| M5-A, schema-complete positive control | — | FINAL, exit 0 |
| M5-A, the review's minimal rows (no `trace` / `truncated` / `runmeta`) | FINAL, exit 0 | INVALID, exit 2: required fields are now enforced |
| Info, fractional `s` / `eviction_seed` | FINAL, exit 0 | INVALID, exit 2 |
| Info, impossible greedy symbol | FINAL, exit 0 | INVALID, exit 2 |
| Info, wrong provenance / failed control | FINAL, exit 0 | INVALID, exit 2 (production mode) |
| Info, missing variant | PARTIAL, exit 0 | PARTIAL, exit 3 (exit 0 only with `--partial`) |
| Info, torn JSON after a FINAL | exit 1, old FINAL left in place | INVALID, exit 2; old FINAL retired |
| Info, reference and production positive controls | FINAL | FINAL, exit 0 |
| R-KV gate, names-only marker | `(True, verified)` | refused: "lacks measurement 'prompt_len'" |
| R-KV gate, injected `raised` | `(True, verified)` | refused: "raised: injected control failure" |
| Selector gate, names-only / injected `raised` | `(True, verified)` | refused, same reasons |
| Both gates, measured positive control | — | `(True, verified)` |
| BBH, `\boxed{(A)} or \boxed{(C)}` against (C) | correct | incorrect, `conflicting_final_answers` |
| BBH, `\boxed{(C)}. Actually, the answer is (D).` | correct | incorrect, `conflicting_final_answers` |

**Archive hashes (SHA-256):**

| File | SHA-256 |
|---|---|
| before | `a516c5c04ade09865f0fffc14c5496d379ad8f8618714ecda112d9f61ab657eb` |
| after | `ac8b325c67f03ac9c1b92d1aefacc14e1081a7eee25df4744546383beec2e051` |
| reviewer script | `f85da6117d2acd1af9b406e224b0a1746f4485a003b57ea97c9e0528a21112d8` |
| after-harness | `540c564a70fe7c0fccd5c72ce5674dd56478c19551c633c0dca694cb195e4222` |

Each case is also a maintained regression test:

| Area | Tests |
|---|---|
| M5-A reader | `tests/test_policy_replication.py` |
| Information reader | `tests/test_report_information_probe.py` |
| Gates | `tests/test_protected_selectors.py`, `tests/test_rkv_pp_gate.py`, `tests/test_gates.py` |
| Graders | `tests/test_task_graders.py`, `tests/test_rescore_task_rows.py` |

## What was repaired

### R1 — M5-A reader

**Validation (shared boundary):**
- The reader goes through `read_checks.validate_rows`, which checks exact
  booleans before any conversion, required fields, and exact coverage.
- Every source run is validated in full against its own config's design
  before the 97-problem cohort is selected.
- On top of that, a row must have the declared model and benchmark, may not be
  truncated yet scored correct, and is checked for identity: legacy rows carry
  none (none is fabricated); M5 rows carry the expected identity and frozen
  prompt hash.

**Output contract:** retire, then atomic write; exits 0/3/2; a dependency
fingerprint over rows, specs and analysis code.

**Equivalence:**
- Frozen in `configs/meeting5/m5a_read.yaml`: primary ±0.03, sensitivity ±0.05.
- TOST p-values are reported.
- A zero standard error is NOT TESTABLE.
- Per-comparison claims use Holm; the joint claim uses an intersection–union
  test.

### R2 — runtime gates and production identity

**Runtime gates.** `kvcache.smoke_validation` re-derives acceptance from the
measurements, for both families:
- the traces and input hashes are recomputed from the frozen reference bank;
- coverage must be exact (no missing, duplicate or extra record);
- fields are strictly typed and finite;
- the family's predicates are re-applied.

Both smoke scripts now record the replayed trace's input hash.

**Real-data check.** On the host, the validator's trace recomputation matches
the real 2026-10-13 selector smoke exactly. That smoke's records pass every
schema, prompt-length and predicate check. Its identity and 15-record coverage
are stale, as already recorded.

**Production identity.** `scripts/run_grid_m5.py` with `kvcache.m5_runs`:
- computes the identity before the model loads;
- refuses to resume into rows or sidecars of another identity, design key or
  prompt hash;
- writes the identity, the prompt-input hash (checked before writing) and the
  capacity/retention block into every row and sidecar.

**Input manifest.**
- Problem found while wiring this: the frozen input manifest covered only 25 of
  the 100 discovery problems.
- Fix: `configs/meeting5/input_manifest_m5.json` (100 prompts, 400 traces),
  built with the existing builder plus an additive `--pids`.
- Checks: the builder's default output was rebuilt byte-identical to the
  committed manifest first, and the new manifest agrees with it on the 25
  shared problems.

**Launch wiring:**
- `launch.sh` aliases `grid_m5` and `rkvppsmoke`;
- campaigns `M5A-RKVPP-SMOKE` (1), `M5A-H2OPP` (28) and `M5A-RKVPP` (28);
- queue jobs ordered after P1/P2/P3.

**Scheduler fix (named separately).**
- Problem: GPU jobs ignored `requires_all`, which only CPU stages honoured, so
  the M5 jobs would have been eligible tonight, ahead of P1/P2/P3.
- Fix: GPU selection now uses the same dependency check, and dangling
  dependency names are refused at queue load. A regression test covers it.

**Real launch-path checks:**
- `launch.sh` → runner dry-run on the host: exit 0 for both new aliases.
- Scheduler dry-run at 02:00: all three M5 jobs report "dependency incomplete
  [...]".

**Clean-up:**
- `run_grid_rkv_pp.py` (identity-less, never used) was removed.
- Its stale remote copy was moved, not deleted, to
  `~/kv-caching-retired/2026-10-14/`, so that source sync could verify again.

### R3 — information channel

**Reader:**
- versioned production schema `info-readout-v1`, with the design's provenance
  and three passed controls;
- a separate reference mode for the analytic fixtures;
- strict integer keys;
- a greedy-consistency check against the stored top-1;
- the shared output contract;
- the registered paired Random − H2O-style contrast, from paired context
  values.

**Runner core:** `kvcache.information_probe`, backend-agnostic:
- decode-path body with eviction on;
- two independent clones per query with eviction off;
- full/full identity, clone agreement and no-readout-eviction controls;
- the retained fraction of each fact span.

**Tested on CPU:**
- *With a retrieval stub:* full cache gives 2 bits; recency eviction loses
  exactly the early fact; shared clones and readout eviction each make the
  read INVALID.
- *On the host CPU with the real transformers backend, a tiny random Qwen2 and
  a real kvpress protected press:* clones agree exactly, the readout evicts
  nothing, the body pruned, and the rows pass the production reader.

**Tokenizer and runner:**
- The encodings were checked on the **pinned** tokenizer, with no problems.
- `scripts/run_information_probe.py` provides a dry-run manifest of 320 prefix
  executions and 1,280 readouts, plus `--check-tokenizer`.
- Real execution is refused: the runner is held, and the GPU invocation is
  enabled only with its approval.

### R4 — grading

**BBH:** every final candidate in the answer region is inspected. Conflicting
options, or a box naming several options, are incorrect with the reason
recorded.

**GSM8K:** the same conflict rule applies.

**Read-time scoring:** `kvcache.task_scoring` with
`scripts/rescore_task_rows.py`:
- dispatches by benchmark, frozen revision and grader version;
- re-scores every arm from stored text under one truncation rule;
- records trace and grader hashes;
- refuses MATH-500.

**BBH pin:** recorded, and the pinned file was fetched on the host. The blob
hash was verified: 250 examples, labels (A)–(E).

### Protocol (`docs/protocol/MEETING5_INFORMATION_PROTOCOL.md` §9)

- the cache-subset statement corrected to "instantaneous crop only";
- the owner's margin and the equivalence family;
- the scope of the lower bound;
- the production-row contract;
- the earlier-information → later-answer bridge (§6.9).

## Re-read of M5-eligible artifacts (CPU, host, real runs)

- **M5-A strict cohort:**
  - all four historical sources validate in full (800 + 800 + 400 + 400 rows,
    with 0 malformed, 0 duplicated and 0 foreign), and their provenance
    controls pass;
  - the two approved arms are absent, so the read is **PARTIAL, exit 3**.
- **M5-A whole cohort:** PARTIAL, exit 3; capacity-matched = false.
- **No numerical change:** at 6b40900 these reads were already PARTIAL with no
  estimate. No estimate exists before or after.
- **Stored B=256 reads re-run** (correction to the first draft of this note,
  which said no shared analysis file was touched).
  - *Why:* `src/kvcache/gates.py` gained the `rkvpp_smoke` gate kind, and it is
    part of the analysis-code fingerprint of the S.1 and protocol-audit reads.
    The `consequence` read would otherwise refuse the stored protection read as
    stale.
  - *Run:* the five stored FINAL reads were re-run on the host over unchanged
    rows, with their recorded commands; each exited 0, FINAL.
  - *Compared:* every leaf except bookkeeping (fingerprints, retired-file
    names, reproduce lines).

  | Read | Compared leaves | Changed values |
  |---|---:|---:|
  | `protection_b256` | 4,020 | 0 |
  | `origin` | 576 | 0 |
  | `matched_b256` | 331 | 0 |
  | `reverse_b256` | 1,803 | 0 |
  | `decision_b256` | 253 | 0 |

  - `reverse_b256` also *gains* the `registered_inference` block (labelled
    EXPLORATORY at B=256). The archived copy predated that block, which was
    added in commit 7698127.
  - `decision_b256` changes only its pointer to the protection read's new
    fingerprint.
  - The analysis-code fingerprint group is the only group that changed. The
    re-reads replace the archived copies.

## Gates (each run bare, exit code shown)

See the commit message for this pass: lint, test, dryrun and audit, plus
`git diff --exit-code -- paper_numbers.json`.

## Incomplete or outstanding

- **Real-model GPU smoke of `rkv_pp_v1`.** Approved (1 GPU-h). Queued after
  P1/P2/P3.
- **P2 selector smoke rerun.** Pending; it is the first real marker judged by
  the stricter gate.
- **Information-calibration GPU execution.** Not approved; the runner refuses.
  A separate request will cover ≤ 4 GPU-h.
- **Protected capture adapter and the CAPMATCH pid-filter adapter.** Not built;
  both cover held items.
- **M5-B generation.** Not approved. The BBH loader, manifests and graders are
  ready. GSM8K first needs a calibration timing run.
