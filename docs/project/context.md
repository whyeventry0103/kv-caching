# CONTEXT HANDOFF — kv-caching research project

**Purpose of this file:** a complete, self-contained handoff. An LLM with zero prior
context should be able to read this and resume exactly where we are. Last updated
2026-09-26. Repo: `/Users/apple/dev/kv-caching` (moved off `~/Desktop` 2026-06-14 to
escape a recurring macOS TCC lockout; `~/Desktop/kv-caching` is a **symlink** to this
path, not a second copy — verified 2026-07-27), branch `main`, GitHub
`whyeventry0103/kv-caching`. For the current commit run `git rev-parse --short
HEAD` — this file no longer pins one, because a pinned hash goes stale on the
next commit and has done so repeatedly. **Push state (checked 2026-08-23):
`origin/main` is at `c4a0f1b` and `backup/main` at `b13b628`, so work HAS been
pushed to origin — the older "nothing is externalized before the reframe lands"
note no longer describes reality and the anonymization scrub is now the live
concern, not a future one.** **§0's dated bullets are the ONLY living record of
current state. §§7-9 are a frozen M4.0-era snapshot, fenced as historical below
— do not read them as current.**

This document is authoritative for *narrative and decisions*. For *code*, the repo is
ground truth — always re-read files before acting (state moves fast; this file can
lag). `docs/protocol/PREREGISTRATION.md` is authoritative for scientific protocol;
`docs/operations/remoteserver.md` is authoritative for ops.

---

## 0. TL;DR — where we are right now

  **★★★★★ 2026-11-10 PKT (follow-up) — Wording follow-up on 9f6c676; claim ledger synced. GPU jobs held.**
  - §4.6 opener rescoped: "We next examine when next-token predictions diverge during fixed-trace replay for policies that can evict the prompt." It replaces the causal "34-point loss" question.
  - Figure 3 is scoped to the four prompt-evictable policies at B=256, and "readout-invisible" is replaced by "unexplained by on-reference readout".
  - The abstract gives the tighter budget's protected ratios from the 100-problem S.1 artifact: 0.097 (Random) and 0.098 (StreamingLLM), `s1.realized.b256.*_pp.median`. The 97-problem ratios stay in §4.5 only.
  - `paper_claims.json`: the 8 abstract contexts were resynced and 2 claims added, giving 284. `CLAIM_LEDGER.md`: A6 and A11 updated (A11's "about 2×" gloss replaced by 0.434 / 0.487), A17 added, plus D, E, F5 and a new section I.
  - Gates bare: lint; test 994 passed / 91 skipped; dryrun; audit (both strict audits clean); catalogue unchanged.
  - PDF ×2: 26 pages, 0 `^!`, the same single Overfull. Pages 1, 13 and 14 inspected.
  - Open: §5 still says "readout-invisible" twice (ledger §F5; an owner wording decision).

  **★★★★★ 2026-11-10 PKT — Manuscript revision patch applied; strict-cohort compression ratios read (FINAL); GSM8K problem-ID adapter built and CPU-tested. GPU jobs held; nothing launched.**
  - **Patch.** `KVCache_Manuscript_Revision_91d94ef.patch` (owner-supplied, untracked in the repo root, not committed) applied on base `91d94ef`: blob `be2d48e` / `ed67794` matched; `git apply --check` clean. It rewrites the abstract, intro, thesis, contributions, §4.5 and conclusion, and adds the Qwen3-4B threats paragraph and 274-claim `paper_claims.json`. Its new literals were checked against their pinned keys; common-support counts 376/355 and 382/383 against `reverse_b256/b1024.json`.
  - **Strict-cohort ratios.**
    - `scripts/report_strict_compression.py` + `kvcache.strict_compression` (additive; `compression_summary`/S.1 untouched). Run on the host from the stored rows → `results/diag/meeting5/m5a/strict_realized_compression.json`, FINAL, 388 per arm.
    - Medians (unprotected / protected): Random 0.0742 / 0.1032, H2O-style 0.0677 / 0.1112, StreamingLLM 0.0980 / 0.1044, R-KV 0.0446 / 0.0582. 0 relaxed, 0 ratio-one.
    - Reproduces the S.1 Random/StreamingLLM medians exactly on the S.1 keys.
    - Added to §4.5 and pinned: `m5a.strict.ratio.*` in `build_paper_numbers.py`, plus 8 claims (282 total).
  - **PDF.** Rebuilt ×2: 26 pages, 0 `^!`, 1 Overfull (0.607 pt, Table 1), identical at HEAD. Pages 1, 13, 14, 19 rendered on the host with pdftoppm and inspected.
  - **GSM8K adapter.** `kvcache.manifest_problems` + `scripts/run_bank_manifest.py` (held, full cache only) + `configs/experiments/bank_gsm8k_timing_m5.yaml` (`held: true`, `scale_authorized: false`). 17 CPU tests; `--resolve-only` on the host resolves 24/24 IDs from the 1,319-row pinned pool.
  - **Not done.** No queue job or campaign for GSM8K. The CAPMATCH compressed-grid counterpart is not built. The claim ledger is not yet synced to the new abstract (synced in the follow-up above).
  - **Host.** Shared disk 20 GB free, below the 30 GiB launch floor.

  **★★★★★ 2026-10-27 ~00:30 PKT — M5-C calibration ran and FAILED (C1, C2, C6); queue drained; GPUs in use by another project of the same account.**
  - **Calibration.** Ran 2026-10-21 01:45–03:55 Dubai, rc 0, 2.165 of 4 GPU-h, 640/640 rows. The read is FINAL (40/40 cells).
    - Failed: C1 (full cache MI 0.235 bits, correct probability 0.203), C2 (OTHER 0.691), C6 (best compressed arm 0.026 bits).
    - Passed: C3, C4 (end-to-end difference 0.0), C5.
    - Retention, early/late fact: R-KV 0.54/0.52, H2O-style 0.05/0.64, Random 0.10/0.35, StreamingLLM 0/0.
    - The reader fails even on the full cache.
    - Next: one redesign (task/answer format, operating point) by pre-data amendment and a separate approval. No MI result exists.
  - **Queue:** nothing approved remains; every window since 10-21 had nothing to launch.
  - **Server GPUs:** at 2026-10-26 23:00 both GPUs were running `train_model.py` (another project, same account) at about 25 GB each. Our scheduler needs ≥ 28 GB free per GPU.
  - **Local disk** is nearly full again (~0.7 GB free).

  **★★★★★ 2026-10-20 ~16:30 PKT — Reviewer follow-up: protected-capture adapter built and tested; Amendment T; GSM8K proposal. Calibration still queued for 2026-10-21 01:45 (unchanged).**
  - **Protected capture (CPU).**
    - `src/kvcache/capture_protected.py` reads each protected prune from the press's own position record and verifies it bitwise. It reuses the unchanged `capture.replay_condition` via a scoped tap swap, because `capture.py` carries six identities.
    - `scripts/run_capture_pp.py` refuses when held, without `--i-understand`, off the host, on a foreign resume, or without a same-identity smoke PASS.
    - Two configs (both held) and launch alias `capturepp`. No queue job yet: a new marker job needs a typed gate in `gates.py`, whose change forces S.1 re-reads, so it lands with the approval.
    - 12 tests pass on the host CPU, including identity controls: random_pp@0 equals random and rkv_pp@0 equals rkv, exactly.
    - Observation: with interval 4 (shorter than the 8-entry window) the R-KV-protected press's own guard refuses; the registered interval is 64.
  - **Cohort frozen:** 23 strict-fit capture problems × 4 = 92 traces. Protected Random − StreamingLLM on these keys is +0.043 (descriptive).
  - **Proposal:** smoke 1 GPU-h; campaign about 13.8 GPU-h (cap 16), or about 8.2 for two policies. `docs/paper/CAPTURE_PP_SMOKE_TIMING_PROPOSAL.md`.
  - **Amendment T:** Random − StreamingLLM added as a confirmation contrast (Holm over 2, beside Random − H2O-style), before any information data. Calibration comparisons are exploratory. The MI evidence chain is restated.
  - **GSM8K:** proposal with a cap of 3 GPU-h (24 × 2 full cache; 0.82–2.76 GPU-h). Blocked on a pid-filter adapter (CPU). `docs/paper/GSM8K_TIMING_PROPOSAL.md`.
  - **Next.**
    1. Calibration read after 2026-10-21.
    2. Approvals: capture smoke, then campaign.
    3. Pid-filter adapter (CPU) for GSM8K and CAPMATCH.

  **★★★★★ 2026-10-20 12:30 Dubai — Status report; Qwen3 S.4 read (EXPLORATORY); calibration still NOT RUN (sync blocker cleared today).**
  - **Server:** unreachable 2026-10-17 → 2026-10-20 and kept `5327072`, so the approved calibration stayed held and blocked there. No launch on 10-19 or 10-20.
  - **Qwen3:** complete on unique keys (200 / 400 / 400 / 400) and read with `report_m4`.
    - StreamingLLM REPLICATES (−0.209 [−0.285, −0.125]).
    - Random is near floor (free run 0.030, truncation 0.735) and is reported verbatim.
    - Gate FAIL 0.845 stands. PAPER_FINDINGS §5r.
  - **M5-A strict re-verified** on the host: identical, including the fingerprint and the table. Protected Random − StreamingLLM +0.0541 [+0.0232, +0.0902] is descriptive; the interaction −0.137 is registered and rejects.
  - **Central question (information explains Random > StreamingLLM?): unknown.** No information data, and no protected M/Δ/ω. The registered M5-C contrast is Random − H2O-style; Random − StreamingLLM would be exploratory unless amended before confirmation.
  - **Report:** `docs/paper/STATUS_REPORT_20261020.md`.
  - **Next.**
    1. Calibration runs from 2026-10-21 01:45 (authorized).
    2. Proposals: protected capture adapter (about 12–16 GPU-h for four policies, measured basis); register Random − StreamingLLM as a contrast; GSM8K timing (≤ about 2.5 GPU-h).

  **★★★★★ 2026-10-17 ~18:50 PKT — Reviewer follow-up: condition table published, pair-specific paper and ledger, BBH metadata amended; calibration approval asked of the owner.**
  - **Condition table.** `results/diag/meeting5/m5a/m5a_read_strict.conditions.tsv` (388 rows; correct and truncated flags for 8 arms), with its sha in the read. The host re-read is bit-identical in its estimates. The standard-library reproduction from the table matches everything (`reproduce_from_table.out.txt`).
  - **Corrections, no number changed.** Protected Random − StreamingLLM +0.054 is DESCRIPTIVE, not a registered superiority claim. The "prompt-evicting policies only" restriction is withdrawn, because the StreamingLLM gap survives protection.
  - **Paper.**
    - New §4.5 (`sec:protection`): B=1024 persists, no interaction detected; the four-policy protected table; pair-specific ranking.
    - Abstract, intro bullet, §4.2 title and sentence, and threats edited.
    - PDF rebuilt ×2: 0 errors, the single Overfull is pre-existing (0.6 pt at L261–273); pages viewed.
  - **Claim ledger.** A11–A16 added; A2, B1 and C3 updated; §F1 closed, §F3 narrowed; two wordings retired.
  - **BBH manifest metadata amended** (revision is the owner's pin, plus the blob). The selection sha `146d2369…` is unchanged and asserted by a test.
  - **SSH restored**; the sync to 94720ff completed.
  - **Calibration APPROVED by the owner (2026-10-17).** Campaign M5C-CAL, cap 4 GPU-h. The job is unblocked, the config is no longer held, and the first window is 2026-10-18 01:45.
  - **Next.**
    1. Read the calibration (`report_information_probe.py --require-calibration-pass`) when it completes.
    2. Protected-capture adapter on CPU plus a smoke/timing proposal.
    3. GSM8K full-cache calibration pricing.
    4. Qwen3 continues in the windows.

  **★★★★★ 2026-10-17 17:10 Dubai — M5-A strict read FINAL; reviewer progress update and calibration approval request written; no GPU work started.**
  - **M5-A strict read: FINAL** (`results/diag/meeting5/m5a/m5a_read_strict.json`; PAPER_FINDINGS §5q). Every source was validated in full first; 97 problems × 4 seeds = 388 per arm.
    - Protected accuracy: Random 0.683, H2O-style 0.688, StreamingLLM 0.629, R-KV 0.644.
    - Unprotected accuracy: Random 0.436, H2O-style 0.255, StreamingLLM 0.245, R-KV 0.536.
    - Registered interaction t test + Holm rejects for both pairings: Random − H2O-style −0.186 (t −5.45); Random − StreamingLLM −0.137 (t −4.02).
    - Protected Random − H2O-style −0.005: not shown within ±0.03 (Holm TOST p 0.151), within ±0.05 (0.010).
    - Protected Random − StreamingLLM +0.054 [+0.023, +0.090]: not equivalent.
    - The whole-cohort read stays held (CAPMATCH absent). G1 is half met (no protected-capture adapter).
  - **Campaigns:** P1, P2, P3, both smokes and both M5 grids are complete (ledger in `results/diag/meeting5/status_20261017/status_inspection.json`).
    - Qwen3 unique expected keys: bank 200/200, grid 381/400 (Random seed 89 31/50), readout 169/400, reverse 0/400. It resumes at 01:45.
  - **Calibration request:** `docs/paper/MEETING5_CALIBRATION_APPROVAL_REQUEST.md`. Cap ≤ 4 GPU-h; estimate about 2.5 GPU-h from measured 0.0265 s/token; prerequisites met. NOT approved and NOT launched.
    - On approval, add campaign M5C-CAL, set `campaign:` on the job and remove `blocked:`, and set `held: false` — all together.
  - **Reviewer update:** `docs/paper/PROGRESS_UPDATE_20261017.md`. It also lists the paper-text and claim-ledger edits still needed.
  - **Next.**
    1. Owner decision on the calibration.
    2. Paper-text edits (abstract L71; §4.2–4.4; threats; claim ledger A2/B1/C3).
    3. The Qwen3 grid → reverse readout.

  **★★★★★ 2026-10-16 ~01:45 PKT — Review of 5616630 repaired (R1–R4, CPU); P1 reverse read FINAL (no interaction detected); Qwen hold released; calibration still HELD.**
  - **P1 reverse at B=1024: FINAL** (`results/diag/paper_audit/reverse_b1024.json`).
    - Registered t test + Holm on the protected interaction: Random +0.003 (t 0.68, Holm p 0.64), StreamingLLM −0.005 (t −1.00, Holm p 0.64). Neither rejects.
    - The trajectory contrast excludes 0 under both the compressed readout (−0.037 / −0.039) and the full-cache readout (−0.040 / −0.034). Readout changes are about 0.
    - The claim is "no interaction detected" (no equivalence test was registered). PAPER_FINDINGS §5p.
  - **Qwen hold released** per its registered condition. The three `blocked:` lines were removed. Queue order keeps both M5 grids first.
  - **Review repairs** (`docs/paper/MEETING5_REVIEW_5616630_ACCEPTANCE.md`; evidence `results/diag/meeting5/review_5616630/`):
    - R1: calibration fields validated against a frozen schedule (malformed → INVALID); criteria bound to the design; `accepted()` and `--require-calibration-pass` (exit 4).
    - R2: input manifest (actual passages/blocks/token hashes) bound into the design and rows; a changed-input or incomplete-block resume refuses before any model load.
    - R3: C6 floor rule `floor_mi_bits` 0.1, pre-data (protocol §6.8/§9).
    - R4: compression read bound to FINAL S.1 evidence (re-read FINAL, values unchanged); relabelled as nominal-budget ratios, ratio-1 counts as lower bounds, plus effective-budget ratios for the 12 relaxed conditions per policy at B=256.
  - **Calibration feasibility (host, tokenizer only):** 83 eligible reserve passages against 40 needed; all 64 variants render with 1,024-token bodies; manifest `1f527da6…`. The calibration stays HELD (separate ≤4 GPU-h approval).
  - **Runs:** grid-rkvpp256-m5 358/400 pairs at 00:17; grid-h2opp256-m5 next.
  - **Next.**
    1. H2O grid.
    2. Strict-cohort replication read (`report_policy_replication.py --cohort strict`, ±0.03) after both arms.
    3. Paper-text edits for the B=1024 forward/reverse result.
    4. Calibration approval request (owner).

  **★★★★★ 2026-10-15 ~16:00 Dubai — P1, P2, P3 and the R-KV-pp smoke all COMPLETE; M5 R-KV-protected grid RUNNING; H2O grid gate bug fixed.**
  - **Completed** (during the owner early start):
    - reverse-readout-pp1024 (P1) 800/800, 12.91 of 20 GPU-h;
    - P2 selector smoke rerun PASS at 13:19 (21 records, 0 failures, accepted by the stricter gate), 0.549 of 2 GPU-h;
    - P3 pilots: grid, readout and reverse 16/16 each, 1.39 of 6 GPU-h;
    - rkv-pp smoke PASS at 15:14 (9 records, 0 failures), 0.187 of 1 GPU-h.
  - **Bug (mine), fixed.** grid-h2opp256-m5 failed every launch (rc 1, 0 h) from 15:13. The cause: the scheduler exported KVCACHE_INPUT_MANIFEST (the M5 100-problem manifest) into the job, and the protected-selector smoke identity, whose config names no manifest, then took its manifest hash from the environment, so the VALID 13:19 PASS read as stale.
    - Fix: nightly_autorun does not export the override for runners that read their config's manifest themselves (grid_m5, rkvppsmoke, infoprobe), with a regression test.
    - No smoke rerun was needed.
    - The H2O stuck counter was reset to 0 (backup `logs/nightly_state.json.bak-20261015-1549`).
  - **Running.** grid-rkvpp256-m5 (M5A-RKVPP, cap 28) since 15:43 on both GPUs. Complete row+sidecar pairs are being published through the journal; recovery was clean.
  - **Next.**
    1. The H2O grid runs once GPUs free.
    2. Run `report_protocol_audit.py reverse --budget 1024` (P1 is complete) for the registered t test + Holm.
    3. Lift the Qwen hold once the B=1024 reads are FINAL.

  **★★★★★ 2026-10-15 ~12:30 Dubai — information-calibration preparation finished (still HELD).**
  - **Runner.** The GPU execution path in `scripts/run_information_probe.py` validates smokes, the pinned tokenizer, reserve-trace filler, a frozen design.json and resume identity, and writes blocks atomically.
  - **Controls.** An end-to-end readout control was added to `kvcache.information_probe`.
  - **Criteria.** C1–C6 frozen in `configs/meeting5/info_calibration.yaml` and protocol §6.8; evaluator `kvcache.information_calibration`; the reader takes `--calibration-criteria`.
  - **Launch wiring.** `launch.sh` alias `infoprobe`; queue job `info-calibration-m5c` is BLOCKED (separate approval, proposed ≤4 GPU-h).
  - **CPU acceptance.** `tests/test_information_runner.py`: a working channel is FINAL + PASS; a no-eviction channel FAILs C5/C6.

  **★★★★★ 2026-10-15 ~12:00 Dubai — S.1 at B=1024 FORWARD read FINAL: consequence A; owner early start running.**
  - **B=1024 forward reads.** Protected residual: Random −0.047 [−0.086, −0.015], StreamingLLM −0.040 [−0.070, −0.013]. Both are RESIDUAL PERSISTS, SHRINKS (Random's Δresidual lower bound is +0.001).
  - **Registered consequence: A** (the headline stands, at protected size).
  - **Realized compression at B=1024:** median about 0.43–0.49, i.e. about 2× not 4×; 7.7% of conditions never compressed. Details in PAPER_FINDINGS §5o. *[Corrected 2026-10-16: these are nominal-budget ratios; 7.7% have ratio 1, a lower bound on never-compressed.]*
  - **Early start (owner, 11:31).** `~/manual_ticks_1015.sh` in tmux `manual-ticks` until 01:40, then cron. P1 reverse-readout-pp1024 running on both GPUs (461/800 at launch; shard bounds 5.93 h; P1 remaining 11.968 h). Recorded in PREREGISTRATION R.16.6.
  - **Next.**
    1. P1 FINAL → `report_protocol_audit.py reverse --budget 1024` (registered t test + Holm).
    2. Combined interpretation.
    3. Lift the Qwen hold.
    4. P2 → P3 → M5.
    5. Information-calibration GPU path prep (held).

  **★★★★★ 2026-10-15 ~11:30 Dubai — run status after the 14→15 Oct window.**
  - **readout-pp1024 (S.1) is COMPLETE: 800/800.** The B=1024 forward and grid arms are both done.
  - **reverse-readout-pp1024 (P1): 461/800.** Seed 11 at 161, seed 23 at 200, seed 47 at 100, seed 89 at 0.
    - Ledger: 8.032 of 20 GPU-h used (shards 4.580 h and 3.452 h).
    - Measured rate is 57.4 rows/GPU-h, so the remaining 339 rows need about 5.9 h (projected total about 13.9, within the cap).
  - **P2 smoke, P3 pilots and the M5 jobs: not started.** They are queued behind P1.
  - **Ledger-label bug fixed today.** launch.sh recorded rc=0 for every launch, because the run block is piped into tee. The 06:10 shard's true exit was 124 (timer stop, per its log); its ledger entry still reads rc=0 and is left as recorded. Charged hours were never affected.
  - **Disk:** 40 GB free shared; our footprint is 50 GB.
  - **Next.**
    1. P1 should finish in about one window.
    2. Then run the B=1024 reads: `report_s protection --budget 1024`, `consequence`, `report_protocol_audit decision/reverse --budget 1024`.
    3. Lift the Qwen hold.
    4. Then P2, P3 and the M5 jobs, in queue order.

  **★★★★★ 2026-10-14 ~23:00 Dubai — torn-write recovery patch applied (review of 73d9031).**
  - **Fix.** A stop mid-append no longer breaks M5 resume: torn fragments are quarantined before pending journals are published.
  - **Verification.** Reproduced first (2 failing tests on 73d9031). Gates: lint 0, test 0 (925 passed), dryrun 0, audit 0. Server CPU suites: 253 passed.
  - **Next.** The approved queue continues unchanged (caps 1/28/28 after P1/P2/P3). No new scientific results yet.

  **★★★★★ 2026-10-14 ~14:30 Dubai — review of 946c76a: the two remaining execution issues are repaired (CPU only).**
  - **Smoke gates.** They now check nested values (counts are exact integers) and derive the target, layers, replay length and generation length independently. Layers come from the frozen `configs/meeting5/model_metadata.json`. The real 10-13 smoke passes the new checks.
  - **M5 runner.** It publishes row + sidecar pairs through an atomic journal (`kvcache.m5_publish`), with recovery limited to owned shards, quarantine of orphans and torn lines, and refusal of conflicts. M5 grid jobs count `grid_pairs`.
  - **Evidence.** `docs/paper/MEETING5_REVIEW_946c76a_ACCEPTANCE.md`; `results/diag/meeting5/review_946c76a/`.
  - **Unchanged.** Approvals, caps, the queue order and historical data.

  **★★★★★ 2026-10-14 ~13:30 Dubai — review of 6b409001 repaired; owner approved the M5 essential pair.**
  - **Owner decisions.**
    - A1–A3 approved: R-KV-pp smoke cap 1, H2O-pp cap 28, R-KV-pp cap 28 GPU-h. They are queued after P1/P2/P3 via `requires_all` (the scheduler now honours it for GPU jobs).
    - Margin ±0.03, with ±0.05 as sensitivity.
    - BBH pinned and fetched on the host (blob verified; 250 examples).
  - **Repairs.** R1–R4 per `docs/paper/MEETING5_REVIEW_6b40900_ACCEPTANCE.md`; before/after evidence in `results/diag/meeting5/review_6b40900/`.
  - **New production path.** M5 rows go through `scripts/run_grid_m5.py` (identity, input hashes, resume refusal) with `configs/meeting5/input_manifest_m5.json`, which covers all 100 problems; the old manifest covered 25.
  - **Re-reads.** The five stored B=256 reads were re-run because `gates.py` changed: 0 value changes.
  - **Tonight's server state.**
    - grid-pp1024 800/800.
    - readout-pp1024 345/800 (queued first).
    - P1 reverse 0/800; P2 smoke not yet rerun (the first marker judged by the stricter gate).
    - The stale remote `scripts/run_grid_rkv_pp.py` was moved to `~/kv-caching-retired/2026-10-14/`.
  - **Next.**
    1. The approved jobs run in normal windows.
    2. When B=1024 is FINAL, re-run `report_s protection --budget 1024`, then `consequence`, then `report_protocol_audit decision/reverse --budget 1024`.
    3. Lift the Qwen hold.
    4. After the essential pair, run `report_policy_replication.py --cohort strict`.
  - **Not built.** CAPMATCH adapter; protected capture; the information-calibration GPU invocation (held).

  **★★★★★ 2026-10-14 ~01:00 Dubai — Meeting 5 CPU bundle complete; one approval request awaits the owner.**
  - **No GPU used, and nothing new queued.** The approved P1/P2/P3 run in normal windows under the ledger. Ledger now: P1 20.0 h left, P2 1.7416, P3 6.0. grid-pp1024 is at 796/800; readout-pp1024 and reverse-readout-pp1024 are at 0/800.
  - **Protected R-KV `rkv_pp_v1`.**
    - Code: `src/kvcache/presses/rkv_prompt_protect.py`, with its own provenance `rkv_pp`, gate, smoke (`scripts/rkv_pp_smoke.py`) and held adapters `run_grid_rkv_pp.py` / `run_replay_rkv_pp.py`.
    - Real-model CPU parity PASS on the 1.5B model with fc-q15b over 3 traces:
      - protection off: 112/112 prunes per trace byte-identical to base R-KV, identical TV;
      - protection on: target 408 for P = 344 (else 256), pre-eviction TV 0.
    - Evidence: `results/diag/meeting5/rkv_pp_cpu_parity/`.
  - **Information instrument.** `task_information.py` reproduces the review §10.3 table exactly. Also built: `information_tasks.py` (the generator), `scripts/report_information_probe.py` (the reader). The GPU runner is NOT written.
  - **Protocol draft v0.1.** `docs/protocol/MEETING5_INFORMATION_PROTOCOL.md`, plus a PREREGISTRATION entry "Meeting 5 preparation".
  - **Related work.** `RELATED_WORK_MATRIX.md` §4, from primary sources. **Garcia 2026 (arXiv:2605.18053) already crosses protection × 7 policies including Random at a matched cap**, so M5-A is a replication/extension, not a new design.
  - **Inventory and manifests.**
    - Inventory: `results/diag/meeting5/inventory.json`. The 7 reuse cells are complete. The strict-fit cohort is 97 problems (`configs/meeting5/strict_fit_b256.json`).
    - AIME inventory: `aime_inventory.json`. There is no matched factorial.
    - Graders: the MATH grader fails GSM8K formats, so `task_graders.py` adds a GSM8K numeric grader and a BBH option grader. BBH loader: `src/kvcache/bbh.py`, TODO(human) pin.
    - Problem manifests: `configs/meeting5/problem_manifests.json`.
    - M5-A reader: `scripts/report_policy_replication.py`, PARTIAL today.
  - **Approval request.** `docs/paper/MEETING5_APPROVAL_REQUEST.md`: 1,248 rows, 50.5 / 62.8 / 83.3 GPU-h, caps totalling 89.
  - **Next, depending on the owner:**
    1. approve or reduce A1–A5;
    2. pin the BBH revision;
    3. confirm the ±0.05 margin.
  - **Still to write before the remaining requests:** the information-probe runner, the CAPMATCH pid-filter adapter, and the protected capture adapter.

  **★★★★★ 2026-10-13 22:30 Dubai — status check, no code changes.**

  **Server, nights of 7–12 Oct.** Preflight refused every launch: the shared disk had less than 30 GB free (177 refusals, caused by other users; our footprint is about 50 of 72 GB). Launches resumed 2026-10-12 09:10.

  **grid-pp1024: 796/800.** Missing 4 rows (streamingllm_pp seed 89). The rows come from two commits (474 from 1123ad0, 322 from dbdf831) whose generation-path files are byte-identical, so they are one implementation.

  **protsel-smoke: PASS (2026-10-13 09:30).**
  - Pre-eviction TV exactly 0.
  - Every post-prune size on target (256, or 417 when the prompt relaxes the budget).
  - H2O identity control: 0 mismatched heads in 1,764 layer-level prunes.
  - The typed gate accepted it, so the pilot is launchable.

  **Not started yet:** readout-pp1024, reverse-readout-pp1024, the pilots. Qwen3 remains held.

  **Report.** Progress report for the reviewer published at https://claude.ai/artifact/Uy6XizAJpZ2kPvgLTZ4Ydc.

  **NEXT.**
  - Unchanged: finish B=1024 → S.1 1024 read, consequence, margins, reverse 1024 read → lift the Qwen3 hold.
  - Owner questions are listed in the report: unprotected 1024 reverse arm, WP5 timing, WP6 endpoint, human scoring review, disk, venue.

  **★★★★★ 2026-10-06 22:00 Dubai — owner decisions recorded.**

  - **Qwen3:** option (a), exploratory.
  - **GPU work:** P1, P2 and P3 approved within their caps (20, 2 and 6 GPU-h) and queued.
  - **GSM8K:** pinned and downloaded (1,319 rows, 4.5 MB). Preparation only.
  - **Priority:** Qwen3 generation is held behind B=1024.

  **Changes.**
  - New typed gate `protsel_smoke` in `gates.py`, sharing `smoke_record_ok` with the run adapters.
  - Re-read on the host after the `gates.py` change: 0 numeric changes.
  - Host tests: 67 passed. Local gate: 726 passed.

  **NEXT.** Tonight, `grid-pp1024` resumes if the co-tenant leaves ≥24 GB free; otherwise the scheduler waits.
  - **After each window:** check GPU-hours against the caps.
  - **When grid-pp1024 and readout-pp1024 are complete:**
    - lift the Qwen3 `blocked:` lines;
    - run `report_s.py protection --budget 1024`, `consequence`, and the margins read.
  - **When reverse-readout-pp1024 is 800/800:** `report_protocol_audit.py reverse --budget 1024`.

  **★★★★★ 2026-10-06 19:40 Dubai — review of 8e1e81a: R1–R4 closed; the two runner blockers are fixed on CPU.**

  - **Eager attention.** Protected H2O now loads eager attention, through both adapters.
  - **Smoke gate.** It now accepts only an identity-matched PASS record (an empty or failed marker is refused).
  - **Smoke runner and B=1024.** The smoke runner is written (held), and the reverse reader supports B=1024 (protected 2×2 alone).
  - **Re-reads.** 0 numeric changes.

  **Last night produced nothing for B=1024.** A co-tenant held about 26 GB per GPU at 01:45. `grid-pp1024` needs 24 GB free, so the Qwen3 grid (needs 20 GB) took both GPUs. `grid-pp1024` is still at 474/800; Qwen3 is at 281/400.

  **Owner decisions pending:** Qwen3 gate option; P1 (B=1024 reverse, review suggests a 20 GPU-h cap); P2 + P3 (2 + 6 GPU-h); GSM8K pin `740312a` and download; how to prioritise B=1024 against Qwen3 tonight.

  **★★★★★ 2026-10-05 ~20:00 Dubai — review of fbf869f: bounded repair applied.**

  **R1, readers.** Values are now checked before `bool()`, in both `report_protocol_audit` and `report_s`. `matched` validates accuracy rows against the frozen design and carries a fingerprint. Host re-read: **0 numeric changes** in five reads.

  **R2, secondary grader v2.** Positive unordered wording is now required, and the multi-box check reads the whole suffix. 0 rows changed in either bank. Review sheet: `human_review_v2.tsv` (42 rows).

  **R3, planning.**
  - Readouts are priced at their source trace; seeds are parameterised.
  - The interaction is an endpoint: about 470/660 problems for ±0.025.
  - 300×4 is 6× the conditions of 100×2.

  **R4, sign table.** Fixed.

  **Interpretation.** The difference of protection improvements is +0.078 [+0.007, +0.149] for Random and −0.007 [−0.081, +0.069] for StreamingLLM. "Mainly forward" is retired.

  **Prepared and HELD.** WP4 runner adapters, provenance family, the B=1024 reverse config (P1, ~12–19 GPU-h), and a GSM8K adapter plus revision proposal (`740312a`, human pin).

  **NEXT.** Normal window: grid-pp1024, then readout-pp1024. Then `report_s.py protection --budget 1024`, `consequence`, and the margins read. Owner decisions are listed in `docs/paper/EXECUTION_STATUS.md`.

  **★★★★★ 2026-10-05 12:00 Dubai — execution plan v1.0 adopted
  (`docs/paper/EXECUTION_PLAN.md`); WP0–WP3 done, WP4 code ready, WP5/WP6 held.**
  Status table: `docs/paper/EXECUTION_STATUS.md`; claims: `CLAIM_LEDGER.md`.
  Protected reverse b256 READ (new 2×2, `results/diag/paper_audit/`): after
  protection the replay directions disagree — interaction +0.130 [+0.078,
  +0.185] Random, +0.127 [+0.065, +0.197] SLM; protection shrinks the forward
  trajectory contrast ~80% but the reverse one only ~51%/57%. "Most of the
  residual was prompt loss" RETIRED. Qwen3 bank audit FINAL: primary 169/200
  reproduced (0 mismatches), 31-error split reproduced, secondary 0.905 [0.90,
  0.96] (anchor 0.885→0.92); gate FAIL kept, disposition memo awaits owner.
  CORRECTION: arXiv:2608.01631's free-run numbers are at budget 2048 not 256
  (PAPER_FINDINGS §5m). Paper: causal "bracket" wording replaced (13 edits),
  abstract qualified, title retired. Remote: GPUs idle (window); grid-pp1024
  474/800, readout-pp1024 0, Qwen3 grid 138/400. NEXT: tonight's cron runs
  pp1024; then `report_s.py protection --budget 1024`, `report_s.py
  consequence`, and a b1024 protocol read (needs a reverse cell — not queued).

  **★★★★★ 2026-10-04 21:15 Dubai — fifth owner-requested early start
  (tick loop ~/manual_ticks_1004.sh until 01:40, jobs stop 10:43); S.1 b256
  READ FINAL: residual −0.056 (Random) / −0.078 (SLM) protected vs −0.332 /
  −0.338 unprotected, PERSISTS + SHRINKS (MEETING3_RESULTS §3g, PAPER_FINDINGS
  §5k).** Qwen3-4B bank 0.845, just below [0.85, 0.99]: STOP before any Qwen
  cell read; investigation in PREREGISTRATION (grader false negatives on
  \text{Evelyn}, set order, separately boxed parts). Owner decision needed:
  accept with note, or register a Qwen grader-sensitivity read. Running now:
  reverse-readout-pp256 (600/800). Then grid/readout-pp1024, Qwen cells.
  NOT synced (docs only; host source unchanged while jobs run).

  **★★★★★ 2026-10-03 13:15 Dubai — third review (of 8a84107) closed.**
  capture_metric_problems now validates M_event and exact array shapes and
  integer counts (5 CLI regression cases); capture reads re-read FINAL, 0 of
  5,472 numeric fields changed. Qwen3-4B bank 150/200 diagnosed: seed 89 never
  launched because readout-pp256 held both GPUs (queue order), not a failure;
  expect tail-fill to start it tonight. NEXT unchanged: readout-pp256 FINAL
  read, then reverse and 1024 cells; Qwen bank gate before any Qwen cell.

  **★★★★★ 2026-10-03 12:30 Dubai — second review (of 49c572c) closed.**
  Three boundary gaps fixed and tested through the real CLI (restoration
  statuses/exclusions/eligible sets/v2 comparators; retire-before-analysis +
  one dependency fingerprint; strict capture controls in the R.17 reads).
  Re-read: every estimate and label unchanged (0 of ~6,500 numeric fields).
  Wording narrowed: S.2 Random = prompt-plus-padding arm; StreamingLLM "not
  detected"; Llama ω "does not generalise to this model/setup"; Llama H3 is
  a mass contrast only (Llama Random 0.00 vs StreamingLLM 0.16 free-run on
  the capture keys). Report: `docs/results/REVIEW_REPAIR_REPORT.md` (second
  section). Jobs: readout-pp256 789/800; reverse-pp256, pp1024 queued;
  fc-qwen3-4b 150/200. NEXT: readout-pp256 → `report_s.py protection
  --budget 256` (FINAL expected), then reverse and 1024, then `consequence`.

  **★★★★★ 2026-10-02 18:45 Dubai — Llama-8B capture READ: ω REVERSED
  (−0.355 [−0.453, −0.275]), H3 HOLDS (−0.577), H2 NEITHER; the R.17 capture
  reads now validate behind the boundary (b256/b2048 re-read FINAL, labels
  unchanged).** Acceptance report: `docs/results/REVIEW_REPAIR_REPORT.md`.
  Mechanism draft updated (Llama reversal pinned; "not established rather than
  ruled out" for the bridge). NEXT unchanged: readout-pp256 → protection
  FINAL; Qwen bank gate before any Qwen cell.

  **★★★★★ 2026-10-02 17:30 Dubai — reader repair after the independent
  review of f69dce3; S.2 READ (FINAL).** One validation boundary
  (`kvcache.read_validation`) behind `report_s.py` (protection / origin /
  consequence) and `report_r17.py restore`: expected keys from committed
  configs + frozen pid lists (`configs/meeting3/grid100_pids.json` added),
  coverage over every row, provenance, gates, finite values, FINAL / PARTIAL
  / INVALID, exit 0/3/2, stale outputs retired. The five reviewed failures
  are reproduced in `tests/test_s_acceptance.py` (real CLI) and fail closed.
  Re-read on the host: restore-v2 FINAL (unchanged), **restore-v3 FINAL:
  Random ρ_prompt 0.367 [0.310, 0.419] PARTIAL, ρ_prompt−ρ_trace +0.147
  [+0.083, +0.200] PROMPT-SPECIFIC; StreamingLLM 0.242 [0.184, 0.295]
  INDETERMINATE, −0.030 [−0.090, +0.025] NON-SPECIFIC** (MEETING3_RESULTS
  §3e); S.1 b256 PARTIAL (grid 800/800, readout pending). MEETING4_ANSWERS
  corrected (paired contrasts, matched keys, Llama arithmetic, 3,200 rows,
  retention sim committed with device). Jobs: capture-llama8b-b256 400/400
  (NOT read yet: run `report_capture` with the Llama config and
  `report_r17 --capture-run capture-llama8b-b256`), fc-qwen3-4b 150/200
  (bank gate [0.85, 0.99] before any Qwen cell is read). NEXT: readout-pp256
  → `report_s.py protection --budget 256` (FINAL), reverse, 1024, then
  `consequence`.

  **★★★★★ 2026-09-30 21:25 Dubai — Qwen3-4B RELEASED.** Owner deleted the
  14B weights; storage 41.952/72 GiB. fc-qwen3-4b unblocked (downloads ~8 GB
  at the pinned revision on first launch). Queue order now: Llama Tier 3 →
  S.1 pp cells → restore-v3 → Qwen3-4b chain. No owner action pending.

  **★★★★★ 2026-09-30 21:30 Dubai — pending work closed out (owner: "you have
  the permission to do all of these").** Night 1 of Amendment S: grid-pp256
  391/800, restore-v3 100/200, no errors; next window 01:45. Tier 3 Llama
  RELEASED (`e069540`, synced). S.4 Qwen3-4B registered (S addendum B),
  configs queued `blocked:` — decision taken: delete the 14B weights (28 GB,
  all cells complete) rather than raise the allocation; the session's
  permission layer refused the `rm`, so the OWNER runs:
  `rm -rf ~/kvcache-storage/cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B`
  then the `blocked:` line on fc-qwen3-4b is removed, storage re-measured,
  synced. P.1 mechanism draft UPDATED with every landed read (67 pinned
  literals, audit clean, PDF x2). P.4 report: `docs/results/R17_REPORT.md`.
  Local Python: Homebrew moved python@3.14 to 3.14.7 whose pyexpat cannot load
  against the system libexpat; a source rebuild failed (Command Line Tools
  older than the SDK) and removed 3.14.3; fixed by relinking pyexpat.so to
  Homebrew's expat (`install_name_tool`, original kept as `.orig`). Durable
  fix: update Command Line Tools, then `brew reinstall python@3.14`.
  NEXT: when grid-pp256 + readout-pp256 reach 800/800 and
  reverse-readout-pp256 800/800: `report_s.py protection --budget 256`, then
  `report_m4` for the bracket; restore-v3 at 200/200: `report_s.py origin`.

  **★★★★★ 2026-09-30 02:07 Dubai — S.1 smoke PASSED; S cells running.**
  `pp-smoke-PASS.json` written 02:07, archived as
  `results/diag/amendment-s/pp_smoke.json` (sha256 `6d68a74c…`, host commit
  `15778e3`). All registered checks held on the three rule-selected traces
  (prompt lengths 353 / 52 / 17): pre-eviction TV exactly 0; post-prune cache
  equal to the target (417 for the long prompt under the relaxed-budget rule,
  256 otherwise); prompt kept at every prune in all 28 layers; both free
  generations pruned. **Identity control: max |TV difference| = 0.0 for both
  presses on all three traces**, so on the real model the unprotected
  StreamingLLM press is four sinks plus recency and the unprotected Random
  press is reproduced exactly (S addendum A, facts 1 and 2). restore-v3
  launched 01:55 on GPU1 (11 rows by 02:07); grid-pp256 is next on GPU0.
  SYNC DEFERRED on purpose: this note is committed and pushed but NOT synced,
  so the host source tree stays at `15778e3` while the S cells write rows.
  Sync with the next change the host needs (e.g. the Tier 3 release).

  **★★★★★ 2026-09-30 00:30 Dubai — Amendment S registered, built and
  queued (owner instruction 2026-09-30: "start all the 4 tasks").** Trigger:
  scoop-watch 2026-09-30 read arXiv:2609.03430 (prompt is the fragile part of
  the cache); none of our presses protects the prompt, so the residual is open
  to a prompt-loss reading. (1) S.1 prompt-protected decomposition, presses
  `random_pp` / `streamingllm_pp` (`kvcache.presses.prompt_protect`), budgets
  256 and 1024. Queue: pp-smoke -> grid-pp256 -> readout-pp256 ->
  reverse-readout-pp256 -> grid-pp1024 -> readout-pp1024 (5 cells x 800).
  (2) S.2 restore-v3 (arms frozen/prompt/trace/full), 200 rows, queued last.
  (3) Tier 3 Llama capture: release conditions met, but the three `blocked:`
  lines are STILL in configs/nightly_queue.yaml — the session's permission
  layer refused the edit twice; the owner must remove them or grant it.
  (4) Qwen3 HELD on storage (2.55 GiB free; Qwen3-4B is already pinned, its
  template is by design). Readers exist and are tested: `scripts/report_s.py`
  (protection / consequence / origin). Host tests 40 passed. NOTHING READ YET.
  NEXT: confirm `pp-smoke-PASS.json` after the 01:45 window opens; if it FAILS
  on the identity control, read S addendum A fact 1 before anything else. When
  a cell reaches 800/800: `report_s.py --runs-root ~/kvcache-storage/runs
  protection --budget 256` on the host, then `report_m4` for the bracket.

  **★★★★★ 2026-09-29 03:20 Dubai — Tier 2 COMPLETE.** restore-v2 FINAL
  200/200 (read `report_r17.py restore`, artifact
  `results/diag/meeting3/r17/restore_v2.json`): full self-restoration MIXED,
  R_fs Random 0.338 [0.320, 0.358], StreamingLLM 0.297 [0.279, 0.317];
  cohort arms Random NON-MONOTONE (lag 256 only, NON-SPECIFIC), StreamingLLM
  FLAT-NULL; 24/25 problems eligible, FC acc 0.844 [0.708, 0.958]
  (MEETING3_RESULTS §3d, PAPER_FINDINGS §5g). fork-precision FINAL, gate FAIL
  2.006 vs 2.0 (§4a). Storage 69.449/72 GiB. GPUs idle. NEXT: owner go-ahead
  to remove the three Tier 3 `blocked:` lines (q6-basis-fit-llama8b,
  capture-llama8b-smoke, capture-llama8b-b256), then sync/push; then the P.1
  draft update with the landed reads, and P.3/P.4.

  **★★★★★ 2026-09-28 20:51 Dubai — one-time owner-requested early start
  (R.16.6, the third).** tmux `manual-ticks` runs ~/manual_ticks_0928.sh
  until 2026-09-29 01:40; jobs stop 10:43. t1-tv1024 resumed from 119/200.
  Then restore-v2, then fork-precision. capture-b2048 is FINAL (R.17
  confirmatory: omega PREDICTED, H3 HOLDS, H2 NEITHER).

  **★★★★★ 2026-09-27 01:50 Dubai — R.17 non-GPU work done; GPU cells
  queued in the requested order.** Queue: q6-basis-fit-k256 ->
  capture2-smoke -> capture-b2048 (capture-v2) -> t1-tv512/1024 -> restore-v2
  -> fork-precision. Each new family (capture_v2, restore_v2, fork_precision)
  passed its torch tests on the host CPU before release. Readers are ready
  (`scripts/report_r17.py`: collapse, auc-tv, restore, fork-precision; and
  report_capture for b2048 v2). Paper: the mechanism draft is in an \iffalse
  block (`ddadc57`); ICML page budget written. NEXT: process each cell as it
  lands (worklist docs/project/r17_worklist.md); Tier 3 config and
  footprint.

  **★★★★★ 2026-09-27 — R.17 underway (target venue: proposed ICML 2027,
  pending supervisor confirmation).**
  - Registration: `ac78eab`. Worklist: docs/project/r17_worklist.md.
  - Done:
    - capture-b2048 held before its first row, for the v2 schema.
    - t1-tv cells kept: their summaries carry the collapse-test windows.
    - Reads on disk: D_acc restricted, onset effect NOT ESTABLISHED (Random
      −0.048 [−0.184, +0.075], SLM +0.026 [−0.066, +0.118]); H2 NEITHER, H3
      HOLDS (b256, exploratory); bridge 0/6 after BH; per-layer M/Δ/ω;
      restoration eligibility 8/25, FC acc 0.594 vs 0.971.
    - Paper (`a549250`): P.5 specificity retracted, reinsertion restated,
      locking reworded per R.2.
    - Q.2 ratio note; cleanup 0.4a–d.
  - Next: fp32 fork diagnosis plus precision floor (new family); capture_v2
    and the k=256 basis refit; restore_v2; Tier 1 figures; P.1/P.2.
  - GPU work runs only in cron windows.

  **★★★★★ 2026-09-26 22:30 Dubai — Task 1 D_acc FINAL.** 1,200/1,200, 0
  provenance failures, determinism control PASSED (200/200 identical per
  budget). At B = 256 (realized 2.7–4.4%), damage t0 = 0 vs 2048 is Random
  +0.110 [+0.030, +0.200] and SLM +0.190 [+0.080, +0.310] (descriptive).
  About 90% of it (+0.100 / +0.150) comes from the 36 pairs whose
  full-cache trace ends before 2048, which stay uncompressed. On the 64 long
  pairs: Random −0.035 [−0.184, +0.092], SLM +0.083 [+0.004, +0.171]. At
  B = 2048 every onset contrast is null (non-identifying, R.8.1). Write-up:
  MEETING3_RESULTS §2a. Running: fork-control-random/-slm.

  **★★★★★ 2026-09-26 11:39 Dubai — one-time owner-requested daytime start
  (R.16.6).** tmux `manual-ticks` runs ~/manual_ticks_0926.sh until
  2026-09-27 01:40, then cron takes over; jobs stop at 2026-09-27 10:43.
  dacc-t512 resumed from 320/400 on both GPUs. This is not a standing
  authorization for later out-of-window starts.

  **★★★★★ 2026-09-26 10:43 Dubai — night 2 closed cleanly.** dacc-t2048
  400/400; dacc-t512 320/400 (stopped by timer, resumable). No
  abort/traceback/storage block after 00:31. Remaining: dacc-t512 80,
  dacc-t0 400, fork controls 2×75, capture-b2048 400, t1-tv512/1024. NEXT:
  cron window 2026-09-27 01:45. post-dacc reads when t0/t512/t2048 are all
  complete.

  **★★★★★ 2026-09-26 01:25 Dubai — Q.5 exact-lag FINAL.** 200/200, 0
  provenance/content failures. Only 8/25 problems are eligible: stash =
  prompt + cohort − 256 exactly (verified on all 200 rows), so short prompts
  fall below n = 64. That leaves 30 traces (28 at lag 256). Trace-level CIs
  are likely too narrow, so treat it as exploratory. Matched reinsertion:
  Random NON-MONOTONE (lag 1 +0.0031 [+0.0004, +0.0061], NON-SPECIFIC; null
  at 4–256), StreamingLLM FLAT-NULL (all CIs include 0). Descriptively, SLM
  random-reinsert helps at every lag (+0.0092 to +0.0133). Write-up:
  MEETING3_RESULTS §3b. Running: dacc-t2048 on both GPUs.

  **★★★★★ 2026-09-26 00:45 Dubai — Tasks 2/3 primary cell FINAL (capture-b256).**
  400/400, controls VALID, 0 provenance failures; budget 256 = realized ratio
  5.77%. Q.3 collateral R-KV − H2O-style = +0.000 [+0.000, +0.000]
  INCONCLUSIVE (0 collateral of 46,451,328 evicted instances per policy;
  empirical at 256, not forced, R.16.3). Q.6 ω(Random) − ω(SLM) = +0.200
  [+0.159, +0.234] PREDICTED, exploratory. M: Random 0.529 vs 0.215–0.268.
  Write-up: docs/results/MEETING3_RESULTS.md §3a. Night 2 started early at
  the owner's request (~/manual_ticks.sh, cron from 01:45; R.16.6). Postprocess
  had published figures into the hashed source tree, so preflight aborted
  launches from 00:15. Fixed in 910d4dc; figures now go to
  results/diag/meeting3/figures/. In flight: exact-lag-r5 seed 89 (GPU0,
  150/200 before), dacc-t2048 seed 23 (GPU1, 200/400 before). NEXT: the cron
  window continues unattended to 10:43; commit each stage's outputs when its
  postprocess stage reaches FINAL.

  **★★★★★ 2026-09-25 11:30 Dubai — GPU night 1 done; second review repaired.**
  Real-model gates PASSED (D_acc parity + onset 576; basis 100 traces k=64;
  capture smoke full/full 0, layer-0 Δ 0.0). Generated: capture-b256 255/400,
  dacc-t2048 200/400 (strict readers: PARTIAL, 0 provenance failures). Review of
  cfa30cd fixed in 0a86cbe/9d46e6c/58d4dc8 (R.15): reader entry points, typed
  gates in the scheduler, provenance-v1 + frozen input manifest, full reader
  validation, transactional postprocessing; acceptance table
  docs/results/meeting3_evidence/acceptance_cfa30cd.md. Storage: TC-ViTok
  deleted on the owner's selection (77.61 GiB reclaimed); guard = combined
  trees at 72 GiB. Smoke/partial collateral is 0.0 for all policies: forced to zero at B=2048
  (7 heads, tau 0.01), NOT forced at B=256 (positive control) -- empirical
  there (R.16.3). 72 GiB explicitly approved by the owner (R.16.4). NEXT: window 2026-09-26 01:45 resumes
  capture-b256 (seeds 23/89), then post-capture-b256 reads Tasks 2/3.

  **★★★★★ 2026-09-24 23:20 Dubai — every Meeting #3 unit is now measured,
  queued behind a verified gate, or BLOCKED with a stated reason.** Built today
  and CPU-verified on the remote with the pinned stack (tiny Qwen2, mutation
  tests; full suite 501 passed): Tasks 2+3 one-replay capture (2b7a765, R.9;
  reader d135e49), Q.5 exact-lag restoration with the identity repair (9c51b9c,
  R.10), Q.1 direct fork control (4478e87, R.11). Queue for the 2026-09-25 01:45
  window onward: dacc-gate-smoke -> q6-basis-fit -> capture-smoke -> capture-b256
  -> exact-lag-r5 -> dacc-t2048/t512/t0 -> fork-control-random/slm ->
  capture-b2048 (~5-6 windows total, est.). BLOCKED: T1 TV at 512/1024 (~167 MiB
  deficit). Owner decisions open: storage-guard scope (R.8.5), space for T1 TV.
  Morning: check the three smoke markers in ~/kvcache-storage/runs, then the
  readers listed in docs/results/MEETING3_RESULTS.md §6.

  **★★★★★ 2026-09-24 (later) — Meeting #3 execution, stage checkpoint.**
  DONE: **Task 1 R.1 read** (fca22f3): paired problem-clustered CIs over raw rows,
  400 matched conditions per cell, totals equal m4 to 4 dp. Total change
  Random/SLM: 256 −0.4625 [−0.545,−0.380] / −0.6475 [−0.723,−0.570]; 512 −0.205
  [−0.273,−0.142] / −0.4075 [−0.485,−0.333]; 1024 −0.085 [−0.128,−0.045] /
  −0.1975 [−0.268,−0.138]; 2048 −0.035 [−0.062,−0.013] / −0.0775 [−0.125,−0.040];
  beneficial flip mass ≤ 0.010 everywhere. `results/diag/meeting3/t1_budget_curve.json`,
  `figures/fig8_budget_curve.{pdf,png}`. **Q.2 figure + occupancy audit** (5e567cf):
  verdict MIXED reproduced byte-identical; at B=2048 the t0=0 and t0=512 arms have
  the SAME actual exposure (cache under budget until ~step 1976), so only B=256
  identifies exposure. `results/diag/meeting3/q2_onset_grid.json`, `figures/fig9_onset_grid.*`.
  **D_acc** (622023f, reader d00bdbe): gate implemented + CPU-validated
  (mutation-tested), queued as dacc-gate-smoke (marker) -> dacc-t2048 -> dacc-t512
  -> dacc-t0, ~7.2 h per cell on 2 GPUs; first window 2026-09-25 01:45. Scheduler
  now BLOCKs a non-fitting cell instead of exiting (restoration-lag1 would have
  stranded the queue); Q.5 lag cells held (R.5). Amendment R.8 records all of this
  before any D_acc outcome, incl. the R.7 storage-wording correction (guard =
  ~/kvcache-storage only: 68.86 GiB used; 69.19 GiB if ~/kv-caching counted —
  owner to confirm). NEXT: Q.3 capture (Task 2), Q.6 fit/project (Task 3), Q.1
  direct control, Q.5 repair; morning: `report_dacc.py` over whatever landed.

  **★★★★★ 2026-09-24 — CORRECTION: Q.1 is NOT fully done.** Its pre-registered
  negative control (deferred removal at s+32, removal step must show no phase
  excess; a STOP gate for paper edits) was never implemented, queued or run,
  and the 09-15 / 09-20 notes calling Q.1 "fully done" were wrong. The paper's
  branch-A qualification rests on the three interval arms without that check.
  Recorded as PREREGISTRATION "Amendment Q.1, correction (2026-09-24)". Also
  never built: the D_acc half of Q.2 (not queued, no config) and a queued read
  cell for Q.5 (only the classifier exists).

  **★★★★★ 2026-09-23 — STATUS vs. THE THREE ORIGINAL MEETING-#3 TASKS. Task 1:
  answered (curve built today). Task 2 (attention TV): designed + coded, NEVER
  RUN. Task 3 (SVD/PCA direction): designed + coded, NEVER RUN. Queue BLOCKED on
  storage since 2026-09-21. Email now says so.**

  **(1) Task 1 — damage vs cache size: DONE.** fig8_budget_curve built from rows
  already on disk (512/1024 were never summarised per-press); m4-512/1024
  summaries archived, re-derive IDENTICAL, catalogued (922 entries). Damage
  falls monotonically 256 -> 2048 for both policies. Key finding: readout loss
  closes by 512-1024 (<=0.02 Random at 512) while the trajectory residual
  persists (-0.087 Random / -0.196 SLM at 1024, CIs exclude 0); SLM residual
  barely moves 256->512 (-0.344 -> -0.334) as its readout loss collapses. TV
  exists at 256/2048 only. Horizon side: Q.1 = locking AMPLITUDE-DEPENDENT
  (branch A qualified); Q.2 = MIXED (clause not reopened). PAPER_FINDINGS 5a/5b/5c.

  **(2) Task 2 — attention-map comparison: NOT RUN.** Metric (M lost mass /
  Delta shape-TV / GQA collateral), attn-profile-v1 columnar schema and tests
  landed 2026-09-12. No runner exists, so Q.3 was never queued. Eager attention
  required (slow path).

  **(3) Task 3 — trajectory direction: NOT RUN.** On/off-manifold split, frozen
  full-cache basis, tests landed; P.4 scope widened in Q.0. No runner, never
  queued. Estimand fixed: omega(Random) - omega(SLM) with clustered CI.

  **(4) Blocked.** restoration-lag1 (Q.5) refused every tick since 09-21: 1.14
  GiB headroom vs 1.45 needed. Smallest sufficient reclaim is the 1.5B model
  (3.32 GiB; q15b cells complete) -- a human decision, not taken.

  **(5) Fixed today.** nightly report flagged "CORE DONE" and listed
  restoration-lag1 as "next" while the guard refused it every tick; it now
  evaluates the guard and flags BLOCKED (verified on the remote, test proven to
  fail against the old report). Catalogue collision tripwire 120 -> 127 after
  examining all 14 new groups (3 structural, 11 k/400-lattice coincidences).

  **NEXT (human decisions):** free storage for Q.5; decide whether Q.3/Q.6
  runners get built; pre-register replay-512/1024 if a divergence-vs-budget
  curve is wanted. ICLR full paper due 2026-09-25. 19 commits unpushed.

  **★★★★★ 2026-09-20 — Q.1 FULLY DONE (runs + all 3 reads). Q.2 generation
  nearly done: t0 and t512 COMPLETE, t2048 347/400 and running. The Q.2 READ
  was missing and is now built and queued. ICLR abstract deadline 2026-09-18
  HAS PASSED; full paper 2026-09-25.**

  **(1) Status.** sawtooth i16/i64/i256 200/200 COMPLETE and all three
  `shape-sawtooth` reads COMPLETE (markers present). onset-t0 400/400,
  onset-t512 400/400, onset-t2048 347/400 relaunched 14:21 and running at ~93%
  on both GPUs. restoration-lag x5 not started. Nothing is stuck: t2048's
  rc=124 is the window timer and its `fails` is 0.

  **(2) THE SAME GAP AS Q.1, CAUGHT AGAIN: Q.2 had no read cell.** Generation
  was queued and nothing computed the verdict, so a drained queue would have
  looked finished with the question unanswered. Built:
  `kvcache.metrics.onset_surface` (window divergence, matched positions, OLS
  exposure slope), `scripts/report_onset_surface.py`, thin runner
  `scripts/run_onset_surface.py`, `onset-surface` dispatch entry, config, and a
  queue cell gated on ALL THREE arms. **Q.5 still has no read cell** -- lower
  priority only because its data does not exist yet.

  **(3) Design notes that matter for the Q.2 read.** Slopes are computed WITHIN
  one (budget, read position) cell and never pooled: r = B/t is matched only at
  equal t. Read positions start a full window (256 steps) after the LATEST onset
  so no arm is read across its own onset. Traces missing from any arm are
  dropped, or problem identity leaks into a quantity meant to carry exposure
  alone. A degenerate design (identical exposures) returns None, never 0.0.

  **(4) The onset gate is confirmed working from the data**, not just the tests:
  a t512-arm profile has `onset_step: 576` = 512 + one 64-step interval, exactly
  as designed.

  **(5) STORAGE GUARD BUG, fixed.** It charged a cell its FULL estimate
  regardless of progress, so t2048 at 347/400 was refused over 0.20 GB when its
  real remainder was ~0.03 GB -- a nearly finished run blocked over space it
  would never use. Now scaled by remaining work (`remaining_output_gb`). An
  unstarted cell has done==0 and is charged in full, so the guard is NOT
  weakened for anything that has not run; a test pins that, and another pins
  that an unstarted oversized cell is still refused.

  **NEXT:** (a) t2048 finishes -> `onset-surface-q2` fires automatically and
  gives the Q.2 verdict; (b) Q.5 needs ~0.35 GiB freed AND a read cell built;
  (c) **ICLR: abstract deadline passed 09-18, full paper 09-25 (5 days), paper
  is 26pp with ~24 main text against a 9-page limit and the body/appendix split
  is proposed but NOT executed**; (d) anonymized-repo URL still a placeholder.
  17 commits unpushed; push authorization NOT granted.

  **★★★★★ 2026-09-15 — Q.1 IS READ AND IT QUALIFIES A PUBLISHED RESULT. Three
  stuck cells fixed (path defect, not data). Q.2 onset-t0 resumed, 341/400.**

  **(1) Q.1 RESULT — AMPLITUDE-DEPENDENT for Random.** All arms at target_size
  256 with the interval (= per-event amplitude) swept, 100 profiles/press/arm:
  I=16 +0.0110 [-0.0252,+0.0300] excl0 NO peak 1; I=64 +0.1193 [+0.0446,+0.1410]
  excl0 YES peak 0; I=256 +0.3183 [+0.1881,+0.3361] excl0 YES peak 0.
  StreamingLLM excludes zero in NO arm, every peak at phase 9 -> INDETERMINATE,
  consistent with its withdrawal. **P.2 branch A is NOT refuted but IS qualified**
  and the paper now says so: locked at the deployed amplitude 64, not detectably
  at 16, below 64 untested.
  **TWO BINDING RULES, both pre-committed:** (a) NEVER compare excess magnitudes
  across arms -- raw amplitude scales with bucket count and the arms have 16/64/
  256 buckets, so the apparent monotone growth is the artifact the
  pre-registration forbade reading; (b) write "not detectably locked", never "not
  locked" -- I=16's interval excludes an I=64-sized effect, not a small one.

  **(2) THE BUG that stuck three cells for two days was a PATH defect, not data.**
  `run_shape_analysis.py` resolved SOURCE profiles against the launcher-scoped
  `REMOTE_RUNS_ROOT` (launch.sh re-exports it as `$BASE/$RUN_NAME`) instead of
  `KVCACHE_BASE_RUNS_ROOT`, giving `runs/shape-sawtooth-i16/replay-sawtooth-i16/
  profiles` -- run name doubled. **Amendment P.2-v2 masked the same defect**: its
  artifact was produced out of band, so the cell read complete by marker while
  its own log recorded rc=2. Fixed for both source and artifact (the marker check
  resolves against the base too). New tests simulate launch.sh's env; the old
  ones set only REMOTE_RUNS_ROOT and could not see it. Stuck counters reset.

  **(3) NOT a bug, checked:** replay-onset-t0 exiting rc=124 is the window timer.
  Resumed 19:24 and its log now reads "RESUME: 341 conditions already complete",
  so no duplicated work.

  **(4) Generation status.** sawtooth i16/i64/i256 all **200/200 COMPLETE**;
  onset-t0 341/400 running on both GPUs; onset-t512, onset-t2048 and all five
  restoration-lag cells not started. Q.5 still blocked on ~2 GiB of storage.

  **(5) TASK A's REQUESTED CURVE IS ONLY HALF AVAILABLE, and this is the gap to
  close.** Budget vs accuracy/loss: **4 points exist** (256, 512, 1024, 2048) --
  grid-diss512 and grid-diss1024 each hold 800 rows. Budget vs DIVERGENCE (TV):
  **only 2 points** (256, 2048), because grid-diss512 and grid-diss1024 have
  **0 profiles** -- they were generation-only cells. A smooth divergence-vs-budget
  curve therefore needs teacher-forced replay cells at 512 and 1024. Those are
  cheap (no generation; the traces already exist) but they are NEW measurements
  and so need a pre-registered read before being queued.

  **NEXT:** (a) decide whether to pre-register + queue replay-512/1024 for the
  Task A divergence axis; (b) Q.3/Q.6 runners still unbuilt; (c) free ~2 GiB
  before Q.5; (d) paper: anonymized-repo URL still a placeholder, body/appendix
  split proposed not executed (~24 main-text pages vs 9). Abstract due 2026-09-18.

  **★★★★★ 2026-09-14 — RUNS ARE LIVE. Q.1 arm i16 COMPLETE (200/200); i64 in
  flight on both GPUs. Q.1 READS were missing from the queue and are now added.
  Q.5 will halt on storage. Abstract due 2026-09-18, full paper 2026-09-25.**

  **(1) What has actually run.** The cron window (01:45-10:45 Dubai) fired on
  2026-09-13 and drained as far as it could:
  - `replay-sawtooth-i16` **200/200 COMPLETE**, 200 profiles.
  - `replay-sawtooth-i64` **~87/200**, relaunched 2026-09-14 00:33 across both
    GPUs (sessions `replay-sawtooth-i64-g0/g1`, ~15.5 GB and ~86% util each).
    Rate measured at ~1 profile/min over both GPUs, so ~2h to finish.
  - Everything after it: not started, no run dir.
  Nothing was stuck; the window simply closed at 10:45 mid-cell.

  **(2) A REAL GAP, now closed: the Q.1 reads had no cell.** The three sawtooth
  GENERATION cells were queued, but nothing computed the phase excess the
  pre-registered read turns on. The arms would have produced 600 profiles and
  stopped, with the verdict uncomputed and nobody notified, because a queue that
  drains looks finished. Three CPU-only `shape` cells added, each gated on its
  OWN generation cell. **One per arm, forced by the code**: phase_amplitude_excess
  raises on mixed intervals, which is also why Q.1 pre-commits to "does EACH
  arm's excess exclude zero", never the ordering across arms.

  **(3) The floor-exit guard was too broad and is now conditional on something
  detectable.** `run_shape_analysis` refused `restrict_to_floor_exit: false`
  outright. That reasoning is P.2-v2-specific: the Q.1 arms are FIXED-budget,
  have no min_budget floor and no `ratio_trace` sidecar, so the restriction is
  undefined for them and the whole-trace profile IS the pre-registered test. The
  guard now keys on whether the source writes ratio_trace sidecars, so the
  P.2-v2 refusal survives exactly where it applies. Unrestricted configs must
  state an `unrestricted_rationale` -- which window a verdict is read on is not
  allowed to be chosen by omission.

  **(4) Storage will stop the queue partway, and exactly where is known.**
  Measured 2026-09-14: **68.14 GiB of the ~70 GiB budget, 1.86 GiB headroom**.
  Projected against the chain with the 1 GiB margin: i256 OK -> onset-t0 OK ->
  onset-t512 OK -> onset-t2048 OK -> **restoration-lag1 REFUSED at 1.30 GiB**.
  So **Q.1 and Q.2 will complete and Q.5 will not start.** Freeing ~2 GiB is the
  precondition, and the only real reclaim path is dropping a model (~28 GB for
  the 14B) -- a decision, not routine cleanup. Do NOT delete from results/, and
  do not raise `est_output_gb` to get past the guard.

  **(5) No double-launch risk, verified rather than assumed.** `--dry-run` sets
  `session_names = []` and so ignores tmux entirely (landmine 20), which is why
  its "would launch" line is not evidence of anything. A real tick calls
  `tmux_session_names()`; `live_map` matched the two live sessions to GPUs {0,1}
  and `eligible_gpus` then returns [] for the next job. Checked by running those
  functions against the actual session names.

  **NEXT:** (a) let the window drain Q.1/Q.2 and read the three shape artifacts
  into PAPER_FINDINGS per the Q.1 branches; (b) free ~2 GiB before Q.5; (c) the
  paper's anonymized-repo URL is still a placeholder for an artifact that does
  not exist; (d) body/appendix split proposed but NOT executed (~24 main-text
  pages against 9); (e) Q.3/Q.6 runners still unbuilt, so those cells remain
  deliberately unqueued. 12 commits unpushed; push authorization NOT granted.

  **★★★★★ 2026-09-12/13 — MEETING #3 PRE-REGISTERED (Amendment Q) AND QUEUED;
  ICLR 2027 CONVERSION DONE. Paper is 26 pages under the venue style, ~24 of it
  main text against a 9-page limit. 11 cells queued. Abstract due 2026-09-18,
  full paper 2026-09-25.**

  **(1) Amendment Q written, every read pre-committed before queueing.** Six
  cells: Q.1 sawtooth control, Q.2 onset-staggered D_tv, Q.3 attention channels
  + GQA collateral, Q.4 attn-profile-v1 schema, Q.5 latency-swept restoration,
  Q.6 on-manifold split. Q.0 explicitly WIDENS P.4's "latent-state divergence is
  out of scope", dated, under three standing limits (structural claims only;
  exploratory by construction; basis fitted on full-cache only and frozen, with
  circularity a STOP condition).

  **(2) The Q.1 control was REDESIGNED before queueing, because the original was
  confounded.** Holding mean cache at 288 forces B = 288 - I/2, so amplitude and
  floor move together and a collapse at small amplitude would be equally
  consistent with "the cache never fell far". Run instead at FIXED FLOOR 256
  (the published cell's floor) with I in {16, 64, 256}. Mean cache then varies,
  and the direction closes the confound: I=16 has the LOWEST mean cache (264)
  and is the HARSHEST arm, so a collapse there cannot be blamed on gentleness.
  Analysis constraint from the code: phase_amplitude_excess refuses mixed
  intervals, so each arm is read separately and the verdict is whether EACH
  excess excludes zero, never the ordering of magnitudes across arms.
  **This cell can go against us: if I=16 does not exclude zero while I=256 does,
  P.2 branch A is not refuted but must be QUALIFIED in the paper.** The
  qualifying wording is drafted in Amendment Q before the read.

  **(3) Two errors in the design spec were fixed.** Task B storage was stated as
  "~4.6k scalars for an 8k trace"; 4,592 is the RECORD count at the MEDIAN trace
  (2652 steps). At the 8192 cap it is 14,336 records / 57,344 scalars -- ~12x
  low, in the direction that decides whether the cell fits. Measured: 1.03 GB
  for 800 traces as per-record dicts vs ~1.6 GB headroom, 0.41 GB columnar, so
  attn-profile-v1 is COLUMNAR and that is mandatory not stylistic. Also replaced
  "roughly six times baseline harm" with both measured ratios, 6.49x (Random)
  and 5.49x (StreamingLLM), per landmine 11.

  **(4) Infrastructure landed BEFORE queueing.** Onset gating is a NEW press
  field `compression_start_step` -- deliberately not the existing `onset_step`,
  which is a measured OUTPUT (when the first prune affected the distribution);
  reusing it would make the gate overwrite its own measurement. Seven regression
  tests, including one that asserts the gate ATTRIBUTES EXIST (landmine 12: a
  bare DecodingPress has none, and a gate that silently does not gate is what
  invalidated restoration v1 -- verified the test fails on a bare press).
  attn-profile-v1 columnar schema with a refuse-unknown-column rule; latent
  split with an immutable FrozenBasis that refuses a non-fullcache condition.
  New metric modules are in the package, not in scripts/, per scripts/CLAUDE.md.

  **(5) 11 cells queued, and the storage guard already bites.** Q.1 x3, Q.2 x3,
  Q.5 x5, chained by `requires`, 2.32 GB projected. New per-job `est_output_gb`
  + `check_disk_budget` refuses a launch that would exhaust the ~70 GB personal
  budget (1 GB margin). Simulated from 68 GB used: the six Q.1/Q.2 cells launch,
  then **restoration-lag1 is REFUSED at 1.28 GB headroom.** The Q.5 sweep needs
  ~2 GB freed first, and the only real reclaim path is dropping a model (~28 GB
  for the 14B) -- a decision, not routine cleanup. Q.3 and Q.6 are NOT queued:
  their metrics and schemas are landed and tested but their runners do not
  exist, and queueing a cell whose runner is missing dies rc=2 and is marked
  stuck (landmine 1).

  **(6) ICLR 2027 conversion done, and the page framing was wrong.** Official
  style files from media.iclr.cc are in-tree. 29 pages under article/11pt was
  never comparable to a limit defined by the venue style: under
  iclr2027_conference the paper is **26 pages, ~24 of it main text**, against 9.
  Appendices after the bibliography are unlimited, so this is RELOCATION not
  deletion; proposal in docs/strategy/ICLR2027_PAGE_BUDGET.md moves ~13 pages,
  mostly out of section 4 (9pp) and section 5 (5pp). Anonymity is STRUCTURAL:
  \iclrfinalcopy commented out makes the style file substitute "Anonymous
  authors"; verified in the built PDF that no identity string appears on any
  page. Repo URL replaced by an anonymized placeholder and BOTH mirror
  assertions removed -- **the placeholder URL is still a promise of an artifact
  that does not exist and must be replaced before submission.** Required AI use
  statement added (page-limit exempt), written to be accurate rather than
  minimal. SUBMISSION_STRATEGY Phase 6 intro rewrite APPLIED and marked applied,
  with one correction: its draft quoted 0.882 as the full-cache reference, which
  is the BANK number Amendment K designates NOT the cell comparator -- the
  applied text uses the matched references 0.903 (readout) and 0.885 (free-run).
  Phase 4's ladder band applied, then CORRECTED: the first version shaded
  FC+bracket, a position that corresponds to nothing on the axis; it now spans
  the reverse and forward rungs, which is where the bracket lives.

  **(7) SCOOP WATCH VERDICT CHANGED -- partial foreclosure.** arXiv:2608.01631
  ("Does Accuracy Equal Evidence?", Ai/He/Guo) runs a fixed-trace replay
  protocol -- structurally our forward readout -- and its Table 2 reports
  end-to-end AND fixed-trace accuracy for the same methods: AdaKV 0.233 vs
  0.400, SnapKV 0.267 vs 0.400. **Those differences ARE our residual, and the
  paper never names or computes them**; its estimand is the answer-evidence gap.
  The standing claim "the replay-decomposition instrument remains unclaimed" is
  RETRACTED. What remains unclaimed is narrower: the residual as a defined
  estimand with clustered intervals, the reverse-replay bracket, and using the
  decomposition to reconcile two literatures. Cited with a differentiating
  paragraph that states the overlap BEFORE the difference. CASK (2604.10900) and
  VeriCache (2605.17613) cited as relevant non-scoops.

  **NEXT:** (a) replace the anonymized-repo placeholder; (b) execute the
  body/appendix split against 9 pages; (c) free ~2 GB before the Q.5 sweep;
  (d) build Q.3/Q.6 runners if those cells are wanted. 10 commits unpushed;
  push authorization NOT granted.

  **★★★★★ 2026-09-08 (later) — REMOTE BACK. parity OK for the first time this
  session, the M.3 figure BUILT, and fig4's caption found to describe a figure
  the paper has never printed. Paper still 29 pages. GPUs idle, nothing queued.**

  **(1) Remote restored; `make parity` OK.** Uptime 66 days, so it never
  rebooted — sshd or the network path was down, not the box. Synced (dry-run
  reviewed first): commit, source tree, and all **98** result artifacts agree.
  All five gates now green: lint, 318 tests, dryrun, audit --strict, parity.

  **(2) fig4's caption described a curve the figure does not show — pre-existing,
  now fixed.** The committed `fig4_tvprofile` is the three-window SUMMARY bar
  chart (0.0000 / 0.0018 / 0.0884); the curve only renders with
  `--profiles-dir`, which needs the remote. The caption nonetheless said
  "compounding staircase ... the curve is shown to +400 steps of onset."
  Verified from the PNG and from text extracted out of the embedded PDF. This
  also retires the premise behind candidate C: the "figure asserts the horizon
  reading via its x-axis" problem applied to a DATA-mode rendering nobody had
  looked at, never to the printed paper.

  **(3) The planned overlay was REJECTED on measurement.** The two arms are
  perfectly paired (200 profiles, all 200 (press,seed,pid) keys shared) but sit
  ~2 orders of magnitude apart in level — first-50 mean TV 0.0036 fixed vs
  0.1972 matched; at offset 0, 0.0008 vs 0.3289 — because the matched press
  spends the early trace pinned at min_budget 32 against 256. M.3's verdict is
  drift WITHIN each arm, never level BETWEEN them, so a shared-axis overlay
  would have invited the exact unlicensed reading the figure exists to prevent.

  **(4) Built instead: `fig:ratiomatched` (fig7).** Paired above-floor window,
  each trace normalized by its own first quarter, so 1.0 = no drift and the
  plotted quantity IS the verdict: fixed rises (1.201 / 1.299), matched reverses
  (0.662 / 0.739). Artifact `results/diag/amendment-p/m3_drift_curves.json` from
  `report_m3_verdict.py --curves-out` — additive, and `m3_verdict.json` came out
  **byte-identical** (sha256) with the flag on, which doubles as a re-derivation
  check against the raw rows.

  **(5) 13 numbers went from uncited to pinned.** Printing the drift numbers made
  `audit --strict` fail with 11 unsourced literals (the gate working). Both
  drift means, all 8 CI bounds, the paired-trace count and max_target 626 are
  now catalogued from artifacts and pinned in paper_claims.json (102 → 115).

  **(6) The candidate-C reopening condition is CLOSED, not pending.** C needed a
  figure showing BOTH levels on shared axes; fig7 shows one (the across-cycle
  envelope + control). The within-cycle phase result is still unplotted, so
  **candidate A stands.** Reopen only if someone plots the phase profile
  alongside the envelope. Full reasoning: MECHANISM_CLAUSE_CANDIDATES.md
  ("Revisit of the C condition"), PAPER_FINDINGS §4y.

  **NEXT:** (a) venue conversion, then the page budget — in that order, see the
  earlier bullet's item (6); (b) 8 commits unpushed, push authorization NOT
  granted.

  **★★★★★ 2026-09-08 — MECHANISM CLAUSE DECIDED AND PROPAGATED. Candidate A,
  sharpened. Zero TODO(M.3-RESTATEMENT) markers left in the .tex. Paper still 29
  pages. No runs pending; GPUs idle; remote UNREACHABLE all session.**

  **(1) The decision.** Candidate A (regime-anchored) over B and C, but anchored
  on the MEASURED 0.0018 rather than the CONSTRUCTED 0.0000. The old thesis said
  the residual "arises from a divergence that is approximately zero at the moment
  of eviction and *compounds*" — the compounding is what M.3 delicensed, and the
  zero is an instrument property (teacher forcing makes pre-eviction TV exactly
  0.0000 by construction) already spent as §4.4's self-control. New clause:
  "...the divergence behind it is small in the steps immediately following an
  eviction and far larger later in generation, so held-fixed-trace protocols
  sample it exactly where it is smallest." §4.4 retitled "Where the divergence
  is, and where readout evaluation looks." Full rationale + rejected candidates:
  docs/project/MECHANISM_CLAUSE_CANDIDATES.md (Decision section at foot);
  dated record: PAPER_FINDINGS §4x; binding wording rule now in §5.

  **(2) C was ruled out by a BLOCKED DEPENDENCY, not by argument — this is
  reversible.** C is the better candidate IF fig4_tvprofile gets the
  ratio-matched overlay. The overlay needs make_paper_figures.py --profiles-dir,
  and neither replay-ext-diss256 nor replay-ratio-matched-m3-v2 exists locally
  (results/runs/ holds only pilots/smokes). The profiles are on the remote, which
  refused connections on port 22 all session. **If that figure is ever plotted,
  REVISIT the clause decision** — the site-4 comment in the .tex carries the
  instructions and the blocker.

  **(3) The marker set was INCOMPLETE — carry this forward.** After propagating
  the 7 markers, a sweep for the delicensed vocabulary found 3 more sites: the
  ABSTRACT ("show it grows with horizon", "pre-compounding regime" — the
  strongest unlicensed claim in the paper, unmarked), Summary of findings
  ("strong compounding"), and the trajectory-audit protocol step 4 (told future
  users to check "for compounding"; now tells them to hold dose fixed first,
  folding our own M.3 correction into the reusable protocol). A find-and-replace
  over the markers would have shipped an abstract asserting what §4.4 withdrew.

  **(4) Verification.** PDF ×2: 0 errors, 1 Overfull, 9 Underfull, 29 pages —
  identical to a baseline build of HEAD's .tex in a scratch dir, so no new
  typesetting problems and the surviving Overfull (L309-310) predates the edits.
  Gates bare: lint clean, 313 passed/14 skipped, dryrun clean, audit --strict
  clean (numbers + claims). make parity COULD NOT RUN — ssh connection refused.
  No artifact, protocol, or number changed; prose only.

  **(5) Also this session.** Retracted the false "personal email embedded in
  scripts/nightly_report.py" claim at CLAUDE.md:98 (verified: zero address
  literals, recipient from KVCACHE_REPORT_TO/_MX) — PRE_RELEASE_SCRUB had
  recorded the retraction twice and asked for an opportunistic fix that never
  landed. Corrected scrub item 2: it pointed at .tex line 493 (URL is at 634)
  and called the anonymized-mirror promise one footnote when the paragraph
  asserts it twice. Hardened make_paper_figures.py's two profile readers with
  kvcache.replay.is_profile_file (landmine 13) — not a live bug, but the
  readers were safe only because sidecars happen to lack an "estimate" key, and
  the fig4 overlay would point them at a run that HAS sidecars.

  **(6) The "29 pages vs ICLR's 9" framing is NOT like-for-like — do not cut
  content against it.** Checked 2026-09-08: the .tex is `\documentclass[11pt]
  {article}`, not the ICLR class, and no .sty/.cls exists in the repo. A page
  count under `article` 11pt is not comparable to a limit defined by
  `iclr20XX_conference.sty`, so the real sequence is: do the venue conversion
  first (SUBMISSION_STRATEGY Phases 4–6 — note Phase 6's intro rewrite is
  paste-ready and marked DO NOT APPLY TWICE), THEN re-measure, THEN decide what
  moves to an appendix. Deciding the body/appendix split from the current 29 is
  optimizing against a number that will change. Where the 29 currently sit:
  §1 p1–3, §2 p4–5, §3 p6–9, §4 p10–20, §5 (threats) p21–25, §6 p26–29.

  **NEXT:** (a) fig4_tvprofile overlay when the remote returns, then reconsider
  candidate C; (b) venue conversion, then the page budget — in that order, per
  (6); (c) 6 commits unpushed, push authorization NOT granted.

  **★★★★★ 2026-08-27 — LAST Meeting-#2 item CLOSED. Amendment P.2-v2: Random's
  eviction-locking SURVIVES ratio-matching (branch A). All four tasks plus both
  controls are now read. No runs pending; GPUs idle.**

  **(1) Result.** Restricted to the same above-floor window as the M.3 drift
  comparison: Random excess **+0.252 [0.184, 0.271], peak phase 0** (the prune
  step) — branch A, REPLICATES UNDER RATIO-MATCHING. StreamingLLM +0.076
  [0.031, 0.099] peak 1, but it stays WITHDRAWN — a fixed-budget claim that
  failed to replicate on 2026-08-12 is not revived by a different cell.
  Coverage matches the drift comparison exactly: 24 problems / 92 obs per press,
  median 75.9% of trace; excluded and counted: 8 never left the floor, 8 windows
  under 200 steps. Both restricted excesses are smaller than their whole-trace
  values (0.326 → 0.252; 0.102 → 0.076), as pre-committed on 2026-08-24.

  **(2) The framing that must travel with it.** Within a prune cycle, divergence
  is eviction-locked — mechanism, now licensed. Across cycles, the rising
  envelope is a compression-ratio effect and stays unlicensed (M.3). The
  restricted window shows both at once: phase excess excludes zero while drift
  is 0.65–0.71 and 97.8% of traces are flat. **Do not let the restored phase
  mechanism leak into the staircase claim.**

  **(3) A windowing bug moved the verdict in our favour, so it is disclosed.**
  The first run read peak 1 (branch B). The re-based onset was not preserving
  phase alignment, rotating every phase index by a measured constant 63 buckets
  across all 96 traces. Fixed, re-run, and BOTH readings are published side by
  side in the amendment. Excesses barely moved (+0.248 → +0.252); what moved is
  which bucket is the prune step, which is exactly what the branch turns on.
  M.3's drift verdict does not use the re-basing and reproduces bit-identically.

  **(4) The job had failed three times on 2026-08-25** before it ever ran: the
  orchestrator appends `--resume --seeds` to every launch and the new runner
  rejected them, so argparse exited rc=2 and the job was marked stuck. Fixed,
  with a regression test covering every command the queue can dispatch.

  **NEXT:** paper edits only. §4.4 restatement in compression-ratio terms and
  the §2 thesis mechanism-clause wording are now unblocked — both are open
  decisions, deliberately not made here. No GPU work is queued.


  **★★★★★ 2026-08-15 — BOTH powered cells landed and were read. Task 2 = NO
  RESTORATION; M.3 = the staircase is a RATIO effect. All four Meeting-#2 tasks
  now have answers. No runs pending; GPUs idle. HEAD `c4a0f1b`.
  (Record brought current 2026-08-23.)**

  **(1) Task 2 / P.5 — `restoration-diss256-v2`, 800/800, gate 800/800 PASS**
  (zero failures, zero missing, zero anchor disagreements; every treated arm
  bitwise identical, `max_abs_difference` 0.0). Effects are `none` minus arm, so
  positive would mean divergence reduced:
  Random-256 matched **-0.0013** [-0.0019,-0.0007], random -0.2485
  [-0.3330,-0.1748], relaxed -0.0014; baseline 0.0383.
  StreamingLLM-256 matched **-0.0015** [-0.0022,-0.0009], random -0.2442
  [-0.3278,-0.1709], relaxed -0.0014; baseline 0.0445.
  **NO RESTORATION** in both. matched is ~3% of baseline divergence — write it as
  *indistinguishable from no help*, never as harm. The relaxed arm does no
  better, so the null is not a budget-match artifact. The specificity control is
  the interesting part: reinserting the same NUMBER of randomly chosen evicted
  entries, with the SAME entries dropped, costs ~6x baseline — the cache is
  acutely sensitive to WHICH entries occupy it, yet restoring what a policy just
  evicted is nearly neutral.

  **(2) M.3 — `replay-ratio-matched-m3-v2`, 200/200. RATIO EFFECT.** Read
  condition (fixed in writing before the cell finished) met: above-floor coverage
  **83.5%** of 377,104 events vs 83.9% predicted; realized ratio on above-floor
  events exactly 0.038; max target 626. On the restricted, paired window (24
  pids / 92 obs per press, median 75.9% of trace retained; excluded 8
  never-left-floor + 8 too-short):
  Random drift fixed **1.201** [1.061,1.339] vs matched **0.662** [0.614,0.709];
  StreamingLLM fixed **1.299** [1.100,1.520] vs matched **0.739** [0.680,0.805].
  Both take the pre-registered FLATTENS branch. **§4.4 is to be restated in
  compression-ratio terms; no mechanistic reading of the rising profile is
  licensed.** Qualifiers: holding ratio fixed IS letting absolute budget grow
  (32→626) — the same manipulation, so this is NOT "horizon is irrelevant at
  fixed cache size"; the restriction shrinks the staircase it tests (whole-trace
  drift 2.93/2.26 vs 1.20/1.30 on the comparable window); and the matched cell
  reverses rather than merely flattens.

  **(3) Both cells were re-runs.** The first attempt at each was invalidated by a
  defect in our own instrument — P.5 by an index-space bug in
  `plan_reinsertion` (stash indices intersected with live-cache indices, which
  barred `matched` from evicting recent entries), M.3 by a feedback loop that
  computed the target from the compressed cache instead of the logical context
  and pinned 100% of 377,104 events to the `min_budget` floor. Both invalidated
  wholesale and re-run, never patched; the invalid profiles are permanent record
  under their original run names. See PREREGISTRATION, P.5 correction
  (2026-08-13) and M.3 correction (2026-08-14). A hypothesis of mine was ALSO
  wrong and is recorded: I attributed v1's large `random` effect to the
  index-space bug; v2 reproduces the aggregate to four decimals while every
  per-trace profile differs, so the fix took effect and the confound was not the
  cause.

  **(4) [RESOLVED 2026-08-27 — see the top bullet. Branch A: Random +0.252
  [0.184, 0.271], peak phase 0, on the restricted window. The whole-trace
  figures quoted below are the preliminary ones and are NOT citable.]
  As recorded on 2026-08-15 — P.2's eviction-locking under ratio matching.** Phase
  statistics on the M.3 cell (Random excess +0.326 [0.263,0.339] peak 0;
  StreamingLLM +0.102 [0.065,0.128] peak 6) were computed over the WHOLE trace
  including the floor-bound period, the harshest compression in the run, which
  would inflate phase amplitude. **Preliminary; not citable** until the phase
  analysis is redone on the same restricted window as the drift comparison.

  **NEXT, in order (superseded — (a) was done 2026-08-27):** (a) redo the phase analysis on the restricted window to
  settle (4); (b) paper edits — nothing in `kv_readout_vs_trajectory_paper.tex`
  reflects Tasks 2/3/4, the P.2 StreamingLLM withdrawal, or M.3; §4.4 needs the
  ratio restatement. No GPU work is queued.


  **★★★★★ 2026-08-12 — Meeting #2 round 2. Extended pass DONE; Tasks 3 and 4
  read; a P.2 result WITHDRAWN; a live-path press defect found and fixed;
  restoration smoke gate PASSING. HEAD `da2f0e7`.**

  **(1) `replay-ext-diss256` finished 800/800** v2 profiles (25 problems x 4
  seeds x 4 presses x budgets {256, 2048}), 12:11 remote on 2026-08-12.
  **R-KV has replay profiles for the first time**, closing P.2's stated scope
  limit. Artifacts archived to `results/diag/amendment-p/`.

  **(2) P.2 does not fully replicate, and budgets must be split.** My first run
  of the shape report pooled budgets 256 and 2048 and was wrong to: at 2048 the
  phenomenon is absent for all four presses (AUC-TV 0.013 vs 0.084-0.108; 86% of
  traces flat), so pooling dilutes toward zero. `report_profile_shape.py` gained
  `--budget` and now records the filter in its output. At budget 256, n=100 per
  press: **Random +0.1068 [+0.0395,+0.1315] peak 0** (replicates +0.0991, peak 0,
  independent sample); **StreamingLLM +0.0195 [-0.0212,+0.0499] — the prior
  +0.0384 [+0.0013,+0.0560] claim is WITHDRAWN**, not averaged (it excluded zero
  by 0.0013); H2O-style +0.0293 [-0.0306,+0.0480] null in both; **R-KV +0.0620
  [+0.0016,+0.0775] peak 9 — marginal, and its peak is not at the prune step.**
  Net: the recoverable/concentrated partition is **weaker** after adding the
  policy it most needed. Random's eviction-locking is the only phase result the
  record carries.

  **(3) Task 3 = BUDGET EFFECT ONLY** (pre-registered branch). Pooled Spearman
  rho **-0.857** [-0.893,-0.643] over 7 cells; within budget 256 **-0.200**
  [-0.400,+0.800] over 4 cells; budget 2048 has 3 cells and an interval spanning
  the whole range. Answer to "is AUC-TV proportional to the residual": **no, and
  twice over** — rank correlation is not proportionality, and the pooled
  association is a budget effect.

  **(4) Task 4's crossing test is degenerate BY CONSTRUCTION.** Pre-onset TV is
  exactly 0.0, so the zero-spread fallback reduces "first crossing" to "first
  non-zero step", and TV and JSD are non-zero on identical steps (verified: same
  zero/non-zero pattern across all 10,769 steps of a profile). Medians agree
  exactly (192 at budget 256). **No JSD onset claim is available from this test.**
  Rank displacement is **coarser** at onset (median 280-451 vs 192), the opposite
  of what `divergence.py`'s docstring claimed from mechanism rather than data;
  corrected in place.

  **(5) A defect no unit test could see.** `build_ratio_matched_press` and
  `build_restoration_press` were built on a bare `DecodingPress`, so neither had
  `enabled` or `profile_step`. Live consequences: the restoration trigger reads
  `profile_step`, so it would never fire and **all four arms would silently
  degenerate to the null**; and `press_enabled(press, False)` — what keeps the
  full-cache REFERENCE forward uncompressed — would be ignored, compressing the
  reference too. The tests passed because they inject a fake base with
  `profile_step` preset. Fixed via `instrumented=` on `build_press`, with a
  regression test that fails on the old code for exactly those two builders.

  **(6) `run_replay.py` now consumes the config blocks it never did.**
  `restoration:` loops arms as separate cells (`random.matched-256-s11` etc.),
  `matched_ratio` selects the M.3 press; the gate report and ratio trace persist
  beside each profile. Tests in `tests/test_run_replay_variants.py`.

  **(7) Restoration smoke: instrument gate PASSES.** Bitwise identical,
  `max_abs_difference` 0.0; matched arm holds 256, relaxed grows to 320 as
  designed. The smoke also caught two things the green suite did not: my
  between-prune bound had the relaxed-arm allowance on only half the check (first
  launch died on it), and **`applied_at_step` is 256, not the pre-registered
  trigger 130** — a press can only intervene during a compression event with a
  full stash, and the gap varies with prompt length. `restoration_read` now
  anchors per trace on the applied step; anchoring on the nominal one dilutes the
  effect by a trace-dependent factor.

  **(8) Smoke DONE 24/24, gate PASSED, powered cell LAUNCHED.** Zero gate
  failures; every treated arm bitwise identical (`max_abs_difference` 0.0);
  matched/random hold 256, relaxed grows to exactly 320. Live-path validation
  beyond the gate: arms are bit-identical to step 256 and first differ at **257**
  — one step after the intervention, where a cache edit can first reach the
  distribution — then differ materially per step. `applied_at_step` is **192 or
  256 depending on the trace**, never the nominal 130, which is why the read is
  now anchored per trace. Reader exercised end-to-end: `NO RESTORATION` at n=3,
  all effects within ±0.001 of zero — **a diagnostic, not a result** (P.5 forbids
  reading smoke divergence), recorded only so a near-zero powered result is not
  mistaken for a late surprise.

  **LAUNCHED 13:29 remote 2026-08-12 (outcome below, no longer in flight):**
  `restoration-diss256-g0` (seeds 11,23) and `-g1` (seeds 47,89), 800 conditions.
  → completed 800/800, then **INVALIDATED** by the `plan_reinsertion`
  index-space bug and superseded by `restoration-diss256-v2`. See the
  2026-08-15 bullet and PREREGISTRATION "Amendment P.5, correction (2026-08-13)".
  Smoke rate was ~1.8 min/condition/GPU, so expect **~12 h**, finishing around
  01:30 on 2026-08-13. Launched manually rather than waiting for the 01:45
  window — the same precedent as `rkvpub-g0/g1` — using the queue's `run_name`
  so `live_map` sees the shards; the second shard needs
  `KVCACHE_ALLOW_EXISTING_RUN=1`, which is what the orchestrator itself sets.

  **THEN (superseded — both have since run and been read; see the 2026-08-15
  bullet):** `report_restoration.py`, then `replay-ratio-matched-m3` (200).
  M.3's first attempt was also invalidated (floor collapse) and re-run as
  `-v2`; its verdict is RATIO EFFECT.


  **★★★★★ 2026-08-10 — MEETING #2 round 1. Two findings, both contradicting the
  brief; two of four tasks still unbuilt. HEAD after this: see git log.**
  **(1) Amendment P** pre-registers all five Meeting-2 reads BEFORE running:
  additivity (P.1), profile extensions (P.2), AUC-TV/residual with a refutation
  branch (P.3), extended replay measures (P.4), restoration with three arms and
  an instrument gate (P.5).
  **(2) Convention 2: nothing to migrate.** This repo never used Convention 1 —
  no `compression_error`, no `trajectory_error = Acc_free − Acc_rev` (searched
  every tracked file). Existing keys map 1:1 (`readout_change`=readout_error,
  `unexplained_residual`=residual_fwd, `reverse_readout_change`=residual_rev).
  Renaming stored keys would rewrite all 29 artifacts and void the 29/29
  re-derivation baseline for zero semantic gain — NOT done, mapping documented.
  **(3) P.1 CONTRADICTS THE BRIEF.** The proposed additivity table summed three
  terms on three different denominators (400 / 392 / 308–372) and mixed
  conventions within one row. On the COMMON population with problem-clustered
  CIs: Random **+0.039** [−0.010,+0.088] (n.s.), SLM **+0.137** [+0.060,+0.218],
  H2O **+0.139** [+0.063,+0.219], R-KV **+0.015** [−0.029,+0.060] (n.s.).
  Random and R-KV do NOT show the predicted negative interaction — signs flip
  and CIs include zero. The "partition along recoverable-vs-concentrated" is
  **not supported**; it tracked truncation rate (those two cells lose the most
  reverse-readout population: 308 and 346 of 400). Branch PARTIAL.
  **(4) P.2: eviction-locked for Random ONLY.** Aggregate phase amplitude vs
  sham: Random 0.179/0.073 peaking at phase **0** (the prune step); SLM
  0.106/0.065 peaking at 13; H2O 0.101/0.063 peaking at 21. The pooled artifact
  hid this. Trace-level drift 86–90% of traces, median ratio 2.8–3.9×.
  **(5) I withdrew one of my own statistics before reporting it.** A per-trace
  "within-cycle recovery" measure gave 85–89% "drift+recover" — a sham with
  shuffled phase labels reproduced it (0.752 vs 0.737). Noise. Removed from the
  module; eviction alignment is now aggregate-only, always against a sham.
  **(6) Task 4 confirmed to need a re-run:** stored profiles are
  `tv-profile-v1` and keep only the TV scalar per step, no distributions.
  **(7) Full horizon was ALREADY in the data** — only the figure truncates at
  ±400. `windowed_tv` computes both estimators over the full horizon.
  **NOT DONE:** the extended replay pass (P.4) and the restoration build (P.5)
  are specified and pre-registered but NOT written. Task 3's correlation needs
  P.4's profiles across cells (currently n=3, which is not a correlation).
  **NEXT:** build the extended replay measures, gate, smoke on <=5 problems,
  queue; then restoration with its retained-set instrument gate.


  **★★★★★ 2026-08-05 — VERIFICATION SWEEP. 29/29 artifacts re-derive from the raw
  rows; one stale number found and corrected; an intent gate now pins 79 numbers.**
  **(1) New: `scripts/rederive_artifacts.py`.** Re-runs every producing report on
  the remote against the raw rows and diffs the artifact field by field. This was
  the last unguarded link: `audit_paper_numbers` trusts `results/diag/`, and
  nothing checked that `results/diag/` still follows from the rows. Result:
  **29 of 29 IDENTICAL**, core cell included. Archived
  `results/diag/rederive-2026-08-05.json`.
  **(2) It found one real defect.** `report_compute.py`'s `INCLUDED` is a hardcoded
  allowlist and had drifted **9 result-bearing run dirs** behind, so the paper's
  compute statement had stopped covering the work it reports.
  **280 GPU-h / 7,200 gens -> 375 GPU-h / 10,007 gens.** Fixed in the .tex, §1b and
  §4-CURRENT; old figure moved to the superseded list. Nothing else depended on it.
  **(3) New: `scripts/audit_paper_claims.py` + `paper_claims.json`.** Value-matching
  cannot separate two quantities that round alike (the known 360-occurrence
  ambiguity). The claims file pins a literal to the QUANTITY it asserts; **79**
  load-bearing numbers verified (abstract, contributions, all of `tab:decomp`, the
  R-KV published cell, the HOTT-cap cell, compute, truncation). A regression test
  moves `rkvpub.ratio` and asserts the gate fails, so it is proven to bite. Wired
  into `make audit`.
  **(4) Residual risk, stated not hidden: 155** result-precision literals in the
  .tex are still ambiguous AND unpinned. Not known wrong — all sourced, all backing
  artifacts re-derive — but weaker than the 79. `make audit` prints the count.
  **(5) Structural checks:** PDF 0 errors / 0 undefined refs / 0 undefined cites /
  0 multiply-defined labels / 1 pre-existing Overfull (line 293) / 26 pages.
  Local-remote artifact parity: **all 75 files match by sha256** (6 previously
  local-only artifacts copied up). `make parity` still shows a commit mismatch —
  structural, `sync_source.sh` never transfers `.git`.
  **NEXT:** no runs pending. Optional: pin more of the 155; fig2 density redesign;
  D.1/D.2/D.3; page budget (26pp vs ICLR 9pp, user handling).


  **★★★★★ 2026-08-04 — THE QUEUE IS DRAINED. R-KV's published configuration reads
  REPLICATES on a one-generation margin. Every queued experimental run is done.**
  **(1) The last cell landed.** `grid_aime24_rkv_published` completed 120/120 at
  the 32768 cap against the re-validated bank (Amendment O.4). R1-Distill-Llama-8B,
  AIME-24, budget 2048, mean realized cache **0.1886**, 30 problems x 4 seeds,
  matched population. Full cache **0.3000** (36/120), R-KV **0.3083** (37/120),
  **r = 1.0278**; paired difference **+0.0083**, problem-clustered 95% CI
  **[-0.0750, +0.0917]**, 7 wins / 17 ties / 6 losses. Pre-registered branch
  **REPLICATES** (Amendment N.2: r >= 1.0 and the CI includes zero).
  **(2) The branch was chosen by code, not by judgment.** N.2's rule now lives in
  `kvcache.stats.published_config_branch` and is emitted as a field of the artifact
  by `scripts/report_published_config.py`, so the read cannot be re-cut after the
  numbers are seen. Artifact:
  `results/diag/amendment-k/classb/rkv_published_cap32768.json`.
  **(3) Say it with its precision.** One generation (37 vs 36 of 120); CI spans
  zero; n=30 problems; the bank cleared its N.3 band at exactly the floor.
  Defensible: our certified port is **not worse than full cache at ~19% of it**
  in the published configuration. NOT defensible: "we reproduced 105%". Also not
  defensible in the other direction: calling it INCONCLUSIVE because we would
  prefer that — the rule was written first and it selects REPLICATES.
  **(4) Two facts favour R-KV and are reported for that reason:** it truncates far
  more (**0.2083** vs **0.0750**, traces 18869.7 vs 14799.3 mean tokens, every
  truncation scored incorrect) and leads anyway; among non-truncated rows it leads
  **0.3895** to **0.3243**.
  **(5) An operational error, recorded.** Launching outside the queue with different
  run names (`rkvpub-g0/g1`) made the work invisible to the orchestrator, which
  counts progress by run directory; it relaunched the same cell and burned ~3h of
  GPU before its window closed, reaching 47 rows. Those rows are NOT merged (that
  would double-count problems) — but on all 47 overlapping conditions the two
  independent processes agree **exactly**: same trace, same canonical answer, same
  token count, same correctness. Sampled decoding at T=0.6 reproduced bitwise
  across processes and GPUs. The queue entry is retired with the lesson written
  into `configs/nightly_queue.yaml`.
  **(6) Queue state: `nightly_autorun.py --dry-run` on the remote now reports
  "no launchable job".** Nothing is left to run.
  **(7) Paper.** Sites updated per `docs/operations/INTEGRATION_CHECKLIST.md` §3: §2's
  "adjacent configuration" hedge replaced by the direct test; §5's
  no->100%-regime paragraph carries the cell with its precision; §5's not-measured
  item moved from "not tested in its own configuration" to "tested and
  under-powered"; `fig:aime`'s caption now scopes its no->100% claim to the 7B
  cell. `tab:audit` unchanged — what R-KV claims did not change. PDF rebuilt x2:
  0 errors, 1 Overfull (pre-existing, line 293, 1.4pt), **26 pages**.
  **(8) Gate hardening done in passing:** `classb/hottcap.json` and the new
  `classb/rkv_published_cap32768.json` were declaring the Amendment K reference
  convention but were NOT in `REFERENCE_RELATIVE_ARTIFACTS`, so the class-wide
  check was not checking them. Both added; the gate now covers 16 artifacts.
  **NEXT:** no runs pending. Remaining work is all analysis//writing:
  fig2_bracket density redesign (STILL OPEN, `docs/operations/FIGURES_PLAYBOOK.md` §9);
  D.1/D.2/D.3; A.5 playbook audit; Part F for the non-core cells; and the page
  budget (26pp vs ICLR 9pp) which the user is handling.


  **★★★★★ 2026-08-03 — AIME cell unblocked and relaunched; N.3 bank gate FAILED;
  three figure defects fixed. HEAD `3e1f042`.**
  **(1) The stuck job was a real capability gap, not a config bug.**
  `grid-aime24-rkv-published` had crashed on every launch since 2026-08-01:
  `NotImplementedError` — the R-KV window-query recompute hand-rolled unscaled
  RoPE, and R1-Distill-Llama-8B uses llama3 scaling. The guard was right to
  refuse. Fixed by delegating to transformers' `ROPE_INIT_FUNCTIONS`
  (`src/kvcache/presses/rkv.py:rope_inv_freq`). Certified three ways: a test
  locks the unscaled branch **bitwise-identical** to the old formula so no
  existing number moves; the per-event self-check against the model's own
  `position_embeddings` is retained; the real-generation smoke on
  Llama-8B/AIME-24 **passed at Jaccard 1.0000** over 448 compressions. That
  smoke deliberately does NOT write the PASS marker — the marker is R-KV's
  promotion mechanism and is defined over Qwen-7B.
  **(2) `fc-aime24-llama8b` finished 120/120 and FAILED its N.3 gate: 0.2750
  against [0.30, 0.70].** So N.2 is NOT readable and stays unread; §2's
  "adjacent configuration" hedge for R-KV still stands. Diagnosis: truncation
  0.2583 at the 16384 cap; non-truncated accuracy 0.3708 (in band); the accepted
  Qwen-7B bank does the same at the same cap (0.2208 / 0.3708). Not a pipeline
  bug — a cap artifact.
  **(3) Amendment O written BEFORE the re-run**, raising the cap to the published
  32768 with the timing disclosed: the 16384 result is permanent, both caps are
  reported, the band is not widened, and there is **ONE attempt** — a second
  failure cuts N.2/N.3. That last clause is what stops this being a search for a
  cap that passes.
  **IN FLIGHT:** `fc-aime24-llama8b-cap32768`, launched 11:32 on both GPUs
  (new run name so the 16384 rows are never touched), ~6h expected, resumable;
  cron continues it at 01:45 if the 23:30 window closes. Then
  `grid-aime24-rkv-published-cap32768` (dependency-gated).
  **(4) Figures: three recorded defects fixed, one still open.** fig2's value
  labels were struck through by their own CI whiskers on all four panels (now
  past `max(mark, ci_hi)`). fig1 showed only free-running accuracy — readout is
  now a hatched paired series at both budgets, so Fig 1 states the thesis
  (256: 0.422/0.237/0.247/0.520 free vs 0.765/0.584/0.597/0.798 readout;
  2048: they converge) instead of restating HOTT; ~38% of its area was a blank
  panel holding a text box and is reclaimed. fig4's recorded defect was **stale**
  — already fixed. **Still open: fig2's density** (4 panels, ~40 numbers) — the
  one-panel redesign is NOT done.
  **(5) Also:** cleared a stale "stuck" alert naming a retired job (state backed
  up); Fig 1's caption had described a histogram panel the committed artifact
  never had, now corrected.
  **NEXT:** read the 32768 bank against the O.4 branches when it lands. Still
  untouched from the earlier list: D.1 seven analyses, D.2 layer-1 bootstrap,
  D.3 content-token TV, **D.6 mechanical compression (25pp vs ICLR's 9pp — the
  real submission blocker)**, A.2 R-KV footnote, A.5 playbook audit, Part F
  final verification, and one `\todo{anonymized-prereg-link}`.

  **★★★★★ ITEMS 0–4 (2026-07-29). Stopped at an item boundary; items 5–10 NOT
  started.**
  **(0) Nothing launched, and that was correct** — only 11 min of remote time
  had passed, not a night. Verified rather than assumed: simulating 01:45 shows
  `grid-diss512` launching sharded across both GPUs (seeds 11,47 / 23,89,
  MAX_HOURS=9.0). Cron fires 01:45. Fixed a misleading refusal message that
  conflated "past today's stop" with "<1h to stop".
  **(1) Property tests found DEFECTS 7 AND 8.** Display rules are now *data*
  (`Entry.display_rules()`), so "no rule, no match" is assertable.
  **Defect 7 — sign-blind attribution:** `classify()` stripped the minus before
  lookup, so `−0.343` could be sourced to an entry valued `+0.343`. Residuals
  here are negative and many quantities positive.
  **Defect 8 — the bibliography-body strip NEVER WORKED:** blanking the label
  width first destroys the `\begin{thebibliography}` anchor. Masked for two
  passes because the arXiv IDs were excluded by the *separate* identifier rule.
  Counts moved: `year` 21→2, and `−0.001` (a grader-bias delta) became unsourced
  and is now catalogued for all three cells.
  **(2) ADD 1-ULP FIXED, not accepted.** Cause: `distribution_tv` summed over
  set-iteration order and the renorm masses over dict-insertion order — i.e.
  glob order. Both now sum **sorted**. Every aggregate bitwise identical; two
  consecutive remote runs now byte-identical. No "within 1 ULP" declaration
  needed; byte-diff tooling stays.
  **(3) `make integrate UNIT=<u>`** — harvest → recatalogue → before/after diff
  of every moved number → gate → figures → PDF. **Refuses a summary without
  `reference_convention`**; stops on a red audit. Does *not* edit the `.tex`.
  **`docs/operations/INTEGRATION_CHECKLIST.md` writes BOTH 512/1024 branches now** — the
  REFUTES branch withdraws §4.1 as a primary contribution and withdraws the
  reframe title outright.
  **(4) Semantic audit: no wrong numbers.** Two systematic passes over all 449
  sourced sites + hand-check of the 10 real candidates. Fixed: pre-registered
  thresholds 0.5/0.75 were being *sourced to measurements* that happened to equal
  them. Recorded, not fixed: HOTT's 0.03 classifies `sourced` not `external`
  because our own 0.03 outranks it — `matched_keys` records both.
  **NOT DONE (items 5–10):** figures, D.2 layer-1 bootstrap, D.3 content-token
  TV, D.1 seven analyses, D.6 compression, A.2 footnote, A.5 playbook audit,
  Part F final verification.

  **★★★★★ PARITY + LAUNCH PASS (2026-07-29). Parts A–C done; D–F NOT done —
  read the honest split below before planning.**
  **BACKUP EXISTS NOW.** Private mirror `backup` →
  `github.com/whyeventry0103/kv-caching-backup` (verified `isPrivate: true`),
  carrying full history **and** `results/diag`. Plus a local bundle at
  `~/kvcache-backups/`. **`origin` untouched.** Push to `backup` at every unit
  boundary. Policy in `docs/operations/remoteserver.md` §36.
  **FULL PARITY, and `make parity` keeps it.** Remote was on `3e19f8b` with no
  history and **zero** `results/`; commits transferred by **bundle over scp**
  (not by giving a shared box a private-repo credential), `git reset --mixed`
  (content hash verified identical before/after), `results/diag` rsynced.
  Now: commit ✅ source-tree hash ✅ 68/68 artifacts byte-identical ✅.
  Remote-only result files are treated as **expected new output, not drift**.
  **SEVEN CELLS QUEUED (31→38 jobs), all reads pre-registered first.**
  512/1024 (the **confirmatory** test of the resolution collapse, Amendment L);
  `grid-hottcap` (Amendment N.1 — HOTT's cap of **2048 tokens, greedy**,
  verified from their own text; we match the **cap only**, greedy stays
  **unmatched** and must be stated); `fc-aime24-llama8b` + R-KV in its
  **published** config (N.2/N.3).
  **Cron re-armed, narrower than before:** queue runner only; email and arXiv
  sweep stay OFF; `check_parity` logs either side of each batch. Armed copy
  `~/crontab.armed.2026-07-29`; disarm `crontab -r`.
  **NOT launched immediately, deliberately** — the 01:45–10:45 watcher window
  refused at 23:48 and I did not force it. Cron fires 01:45.
  **The stash is gone, but two of its ideas are now ours:** exact-value match
  outranking rounded match (fixes the coincidental-attribution defect I reported
  last pass) and an `external` bucket for numbers only a human can verify.
  Buckets: sourced 577, **external 13**, protocol 451, unsourced **0**.
  **STILL OUTSTANDING (Parts D/E/F — none started):** Unit 5.2–5.9, layer-1
  bootstrap, content-token TV, **all figure work**, the semantic prose audit,
  mechanical compression, `make integrate`, `INTEGRATION_CHECKLIST.md`, and the
  final 20-number verification. See the report's §7.

  **★★★★★ VERIFICATION PASS (2026-07-28) — three gates now, and two real
  defects found by sampling the OUTPUT rather than re-running the gate.**
  **(1) The grader audit was measuring nothing.** `diagnose_math_failures.py`
  re-scored with the *same* `is_equiv` on the *same* stored answers, so 0/1600
  disagreements was true by construction — a replay-determinism check wearing a
  grader-validity label. Now cross-checked against an **independent**
  normalization (sympify canonical form on the extracted answer; `math_verify`
  is in neither pinned stack). Real result: the two disagree on **0.6–2.8%** of
  comparable rows, and the treated arms **bracket** the FC arm (1.2 / 2.8 / 0.6
  vs 1.1%) rather than exceeding it, so the asymmetry the threat needs is
  absent. Worst-case bias in any Δ: **−0.001 / +0.014 / −0.006** — ≤0.014, no
  consistent sign, ~24× below the residuals. Candidates for human review, not
  corrections; still not human verification.
  **(2) Layer-1 bounds are NOT confidence intervals** and my "tighter, clears
  zero more comfortably" framing was wrong. They are deterministic, width
  exactly `n_undef/N`, zero sampling uncertainty. Defined explicitly in §3 now,
  with "no uncertainty statement attaches to them; every inferential claim rests
  on the complete-case CI", and three significance-shaped uses rewritten.
  **(3) Amendment K swept across the whole class** (15 paths classified with
  reasons, in a K addendum) and made **enforceable**:
  `check_reference_conventions` fails the gate on any of 14 reference-relative
  artifacts lacking `reference_convention`. Building it immediately caught the
  4 ADD artifacts (compliant but *undeclared* — the exact state that let the
  original omission happen).
  **(4) D.4 sampling the rendered PDF caught two sites the gate PASSES**: §2 and
  §4.3 used the 0.882 bank where the 0.885 matched comparator belongs (0.882 is
  catalogued, so the gate is silent). **The gate proves a value is in the
  catalogue; it does not prove it is the right quantity for that sentence.**
  **(5) Class A locators VERIFIED against the sources** (DeepSeek Table 5:
  7B=92.8, 14B=93.9; R-KV abstract verbatim 105%/16%, model confirmed, dataset
  inferred). Source says 16%, not ~16% — our tilde removed.
  **(6) Personal email out of source** (env-var + refuse-to-send).
  **(7) BACKUP EXPOSURE — needs your decision.** Remote is current for source
  (218/218 byte-identical) but **not complete**: its git clone is pinned at
  `3e19f8b` with none of our 19 unpushed commits, and it has **no
  `results/diag`**. The committed artifacts the Reproducibility claim depends on
  exist in exactly one place.
  Honest before/after under ONE extractor definition: unsourced **278 → 0**
  (total 1577 → 1062; the drop is arXiv IDs/model scales/section refs/dates no
  longer counted as claims). Gates: lint, **219 tests**, dryrun, audit --strict
  0, PDF ×2 25pp. The 1.40422pt Overfull is **pre-existing** (identical at
  `55be556`). A **git stash** from an earlier session exists — NOT dropped.

  **★★★★ UNIT 9.4 + UNIT 10 + PART D (2026-07-28).**
  **Grader audit: CLEAN, and it found a script bug on the way.** All three
  budget-256 compressed cells re-graded: **0 disagreements over 1600 rows**, 0
  false positives, 0 false negatives, regraded accuracy identical to stored in
  every cell. The "is_equiv under-credits compressed generations" attack now has
  a measured answer. `diagnose_math_failures.py` crashed first — it predates
  token-ID sidecars and was globbing them in (`KeyError: 'decoding'`); it worked
  on the FC bank and silently acquired the crash when cells gained sidecars.
  Fixed + regression test. Artifacts `results/diag/amendment-k/grader/`.
  **Unit 10.1 NOT LAUNCHED, deliberately.** Remote is healthy (verified on a
  real `sync-apply`, not a dry run) but the injection engine is a substantial
  GPU deliverable. Built instead: `noise_control.match_gate()` — requires window
  means within 10% **and** growth shape within 25%, because either alone passes
  an obviously wrong profile (a flat curve with the right mean; a uniformly
  doubled curve). Gate carries a `contract` field saying accuracy MUST NOT be
  read if it fails. 9 tests. Read pre-registered in **Amendment M.1** with all
  three branches including GENERIC, which would weaken the paper.
  **Amendment M** also pre-registers M.2 (the missing factorial cell), M.3
  (ratio-matched TV), M.4 (content-token TV), M.5 (512/1024 confirmatory), M.6
  (HOTT config + R-KV published config) — all specified, none run.
  **Part D configs written and dry-run validated, NOT launched:**
  `grid_math500_diss{512,1024}.yaml`, `readout_math500_diss{512,1024}.yaml`,
  all four registered in the Makefile `dryrun` target.
  **§6 now states what we have NOT measured** — the missing noise control, the
  missing factorial cell, the unreproduced HOTT baseline, and the post-hoc
  status of the resolution result. Previously all four were absences a reader
  had to notice.
  **Unit 11 staged, not applied:** `docs/strategy/PROPOSED_REFRAME.md` carries
  the proposed title + ≤180-word abstract side-by-side with the current ones,
  and argues **against** the headline title while the resolution result stays
  post-hoc. Structural reframe parts already landed in the `.tex`.

  **★★★★★ UNIT 6 (2026-07-28) — THE RESOLUTION COLLAPSE IS REAL AND MEASURED;
  B.9's reframe gate PASSES.** New §4.1 `sec:resolution` + `tab:resolution`,
  PREREGISTRATION **Amendment L** (declares it post-hoc *before* the write-up
  and pre-commits the confirmatory 512/1024 prediction).

  | regime | free spread | readout spread | difference (95% CI) | ratio | rank agr |
  |---|---|---|---|---|---|
  | MATH@256 | 0.1704 | 0.1328 | +0.0376 [−0.0117, +0.0837] spans 0 | 0.78 | **6/6** |
  | MATH@2048 | 0.0283 | 0.0033 | +0.0250 [+0.0067, +0.0467] **excl 0** | 0.12 | 1/3 |
  | AIME@2048 | 0.1012 | 0.0000 | +0.1012 [+0.0357, +0.1786] **excl 0** | 0.00 | 0/3 |

  Readout keeps **full** resolution where it reports damage (ranks all 6 pairs
  right at 256) and goes blind where it certifies near-losslessness. Monotone
  ratio, both collapse cells exclude zero → gate passes. The narrow statement is
  the strong one: "readout cannot rank policies" is **false in general** and our
  own 256 cell refutes it; the claim is specific to the saturated regime.
  Statistic: mean **absolute pairwise** difference (max−min is unstable at 3–4
  presses); **the difference carries the CI, the ratio is descriptive only** (a
  ratio CI on a near-zero denominator is not honest).

  **★★★ UNIT 5.1 — the method-invariance support was inadmissible, and the
  replacement holds.** The paper set a cross-policy residual *spread* (paired)
  against marginal CI half-widths ≈0.07 (unpaired) — not comparable in either
  direction. Paired per-problem test (`paired_residual_contrast`, artifact
  `classb/residual_contrast_256.json`, 100/100 problems on all four cells):
  mean |pairwise diff| **0.038** [0.020, 0.094], and **0 of 6 pairwise contrasts
  exclude zero**. Trio diffs 0.003–0.008; the biggest gap in the paper (R-KV vs H2O,
  −0.075) still spans zero at [−0.158, 0.000]. So the claim **survives**, but as
  a non-detection at n=100, not a demonstration of equality — and the paper now
  says exactly that. Note the per-problem residual is `free − readout`, so
  **Acc_FC cancels exactly** — this contrast is immune to the Amendment K
  reference choice.
  Gates: lint, **197 tests**, audit --strict 0, PDF ×2 (0 `^!`, 1 Overfull, 22pp).

  **★★★★★ UNIT 4 (2026-07-27) — output path closed. BOTH documents at zero
  unsourced; `make audit` is the gate.**
  **(1) The last literal surface is gone.** `make_paper_figures.py` held `FC`,
  a 4-press `C` dict, `C2048`, `PHEN`, `TVWIN`, `TWOFC`, `CAIME` as hardcoded
  constants — every one stale post-K, and *unreachable* by the correction: the
  summaries were regenerated, the `.tex` was regenerated, and the figures went
  on drawing 2026-07-12 values with nothing able to notice. They now load from
  `paper_numbers.json`, **with no literal fallback** (missing key ⇒ SystemExit).
  One source, two consumers: a figure and a sentence can no longer disagree
  about the same quantity. Figures re-rendered and eyeballed.
  **(2) The two FC references are now stated in the paper.** §3.1 defines bank
  0.882 (2000 rows, quoted there only) vs matched **0.885** (400 cell
  conditions, the comparator for every Δ/readout change/reverse endpoint), and
  the five comparator sites that said 0.882 now say 0.885 (abstract,
  contribution bullet, §4.1 opener, Fig 1 caption, Table 2 caption).
  **(3) docs/results/PAPER_FINDINGS.md: a CONFLICT with the work order, resolved the repo's
  way.** §8.3 says drive its literals to zero. But §4a–4t is a *dated read-in
  log* and several entries exist precisely to show a wrong value beside its
  correction (§4c is Amendment H's old-vs-new). Flattening it would delete the
  honesty trail CLAUDE.md §4.12 requires. Resolution: added **§4-CURRENT**, a
  generated post-K headline table that is what a new session should read, and
  marked the log `<!-- AUDIT:ARCHIVAL-BEGIN/END -->` — a declaration in the
  document, not a silent exemption in code. Stale current-truth in §1b (compute,
  the 0.004 residual gap, the 0.46 share) was fixed normally.
  **(4) Four more extractor false-positive classes** (bare arXiv IDs in table
  columns, `§4.1` section refs, ISO dates, and the archival spans): both-doc
  literals 1780 → 952, and *sourced held at 543* — again, no real claim lost.
  **(5) Reproducibility paragraph rewritten** to the B.4 claim: every number
  traces to a committed artifact or a cited source, gate-enforced, figures from
  the same catalogue, intervals carrying seed + resample count + difference
  histogram, and the MC error (~0.008 across seeds) stated once.
  Gates: lint, **188 tests**, dryrun, `audit --strict` exit 0, PDF ×2 clean
  (0 `^!`, one 1.4pt Overfull, 21pp).

  **★★★★★ UNIT 3 (2026-07-27) — the `.tex` is at ZERO unsourced literals.**
  `audit_paper_numbers.py --documents <tex> --strict` exits 0: 738 literals,
  440 sourced, 281 protocol, 17 year. Both-docs unsourced 377 → 174 (the
  remainder is all `docs/results/PAPER_FINDINGS.md`, Unit 4's job). PDF rebuilds ×2 clean —
  0 `^!`, one 1.4pt Overfull at L250, 21 pages.
  **Two more gate false-positive classes found and fixed** (same "audit the
  class" rule): the bibliography body was being scanned, reporting 13 arXiv IDs
  as unsourced claims; and model-scale/identifier fragments (`1.5B`, `14B`,
  `4M`, `PG19`, `R1`) were parsed as measurements. 260 phantom literals removed,
  `sourced` unchanged at 645 — no real claim was lost. Protocol floats
  (temperature, top-p, percentile indices) are now an enumerated list like the
  integers.
  **A REAL missed correction, found by cataloguing: `fc-readout-control` was
  still on the retired convention.** Amendment K fixed the 9 decomposition
  cells and never touched this artifact, whose entire job is calibrating the
  readout channel. It differenced Acc_ro(FC)=0.9056 (392 defined rows) against
  the 2000-row bank (0.8825) → +0.023. Matched: Acc_FC on the same 392 rows is
  **0.9031**, so the pure answer-resampling effect is **+0.003** — and the
  residuals are **132× (Random) / 105× (R-KV)** that, i.e. *two* orders of
  magnitude, not one. Regenerated on the remote via an extended
  `readout_acc` harvest branch; artifact now carries
  `reference_convention: amendment-k-cell-population` and keeps the retired
  value under an explicit `_RETIRED` key.
  **Disagreement with the work order, repo wins:** StreamingLLM's readout share
  is **49.2%**, not the 49.3% the order states (artifact 49.2475). The
  threshold argument is unaffected (still below 0.5) but the margin is 0.008,
  not 0.04.
  **Also fixed:** AIME's realized ratio used a generation-only denominator
  while MATH's used prompt+generation — the same quantity computed two ways in
  one paper. Restated prompt-inclusive: **19.5% of 10,527** (AIME prompt mean
  112.07 tokens, new artifact `classb/prompt_lengths_aime.json`).
  Full claim-change list is in the Unit 3 commit message.

  **★★★★★ UNIT 2 (2026-07-27) — Class A/C sourced, and the GATE ITSELF was
  over-reporting by 65%.** Budget-2048 priority check first: **all three
  residuals still exclude zero on both layers** — Random −0.0332 CI
  [−0.065, −0.010] / layer-1 [−0.0525, −0.0325]; SLM −0.0791 CI
  [−0.125, −0.0417] / [−0.0975, −0.0775]; H2O −0.0587 CI [−0.1017, −0.0200] /
  [−0.0775, −0.0575]. B.3's first branch: restate carrying the CIs, note the
  reduced margin. No claim fails.
  **The new output-path defect (same class, found by auditing the class).**
  `Entry.displays()` emitted the zero-place form of every value, so `0.882`
  produced `"1"` and every bare 0/1 in the prose matched some accuracy or
  residual. **418 occurrences — 65% of the reported `sourced` count — were
  protocol constants credited to unrelated entries.** Two fixes, both tested:
  `displays()` no longer emits a zero-place form for `|v| < 1` (a rounding that
  destroys the value is not a display of it), and `classify()` now checks
  protocol *before* sourced as a backstop. Both land on identical counts, which
  is the cross-check. Honest basis is now **sourced 645, protocol 906,
  unsourced 377** (was "sourced 1063, protocol 488"). *Unsourced did not move* —
  the gate's verdict was always right; only its self-report was inflated.
  **Class C: the paper's compression ratios were never wrong, only unsourced.**
  With the prompt term (66.28 tokens, from the token-ID sidecars) restored, the
  published figures reproduce **exactly**: pooled trace 6725.00 → 3.807% ("3.8%
  of 6,725") and 3618.20 → 56.603% ("56.6% of 3,618"); H2O 5.785% → "5.8%".
  `build_paper_numbers.py`'s standing note that these "cannot be fully
  reproduced here" was wrong about the paper and is replaced. **Compute is a
  real claim change:** artifact gives **279.87 GPU-hours over 7,200
  generations** under an explicit inclusion rule; `.tex:134` says "~230 GPU-hours
  across ~7,300" → on the Unit 3 list.
  **Class A: 8 external entries** (HOTT 0.47/0.03/0.14/0.38 → Table 2; R-KV
  105%/16%; DeepSeek 0.928/0.939) carry citation key + locator. Four locators
  are **UNVERIFIED** (paper known, place in it not recorded) and the builder
  prints them rather than absorbing them into "sourced".
  Audit: `.tex` **161 → 154**, both-docs **386 → 377**. `make audit` now exists
  and is the output-path gate. 167 → **179 tests**.

  **★★★★★ AMENDMENT K + THE OUTPUT-PATH GATE (2026-07-27) — a real systemic
  bug found and fixed; the paper is NOT submission-ready and the
  "endgame only" bullet below is SUPERSEDED.** Read this bullet first;
  everything under it predates the correction.
  **(1) The bug (Amendment K, `f0f809e`).** Every accuracy comparison was
  computed over *mismatched populations*: readout accuracy is defined only
  where a reasoning/answer boundary exists (i.e. the trace terminated), and
  reverse-readout only where the *compressed* trace terminated — but both
  were differenced against `Acc_FC` over the **entire** bank. Models fail to
  terminate on harder problems, so the defined subset is systematically
  easier. Magnitudes: Acc_FC = 0.8825 over all 2000 rows, 0.9031 over the 392
  readout-defined rows, 0.9351 over R-KV-14B's 154 reverse-defined rows (vs
  0.880). That last one fabricated a **+0.029** reverse endpoint — the reason
  the 14B cell read "inconclusive." A second defect rode along: point
  estimates used the unmatched global mean while their own bootstrap CIs
  already restricted to the cell's problems, so estimate and interval were on
  *different populations*.
  **(2) The fix.** Three layers, applied uniformly and to every cell in one
  pass *before* any per-cell verdict was inspected: layer 1 = primary
  estimand on the full cell population with imputation bounds both ways
  (`primary_bounds`); layer 2 = complete-case sensitivity against the
  identically-restricted reference (top-level keys); the unmatched
  subset-vs-bank comparison is **retired**. Artifacts carry
  `reference_convention: amendment-k-cell-population`. Re-run + verified in
  `d01cbc9`: 24/24 cross-checks against derivations written from raw rows
  *before* the code changed; 5 regressions that fail on the pre-fix code.
  **Verdicts held — the scorecard is 3/4 on BOTH the pre-registered
  convention and the new primary estimand**, so the correction did not buy an
  upgrade (see PREREGISTRATION Amendment K for why we decline to restate it).
  Both directions moved: all four MATH-500 reverse endpoints moved *away*
  from zero (helps us), all three AIME residuals *toward* zero (hurts us) —
  that pair is the evidence the convention wasn't chosen for its result.
  **(3) The diagnosis that generalizes.** This repo's gate culture is strong
  on *inputs* (pinned revisions, token-ID sidecars, Jaccard parity, 800-row
  exact replication) and was **absent on outputs**. Nothing ever checked that
  a number in the paper matched the artifact it came from. Every defect found
  in this exercise is output-path: the population bug, 81 stale literals, 99
  never-sourced ones, and `make_paper_figures.py:48` hardcoding `FC = 0.8825`
  plus every derived constant. Four symptoms, one missing gate.
  **(4) The gate (`81c2642`).** `src/kvcache/paper_numbers.py` +
  `scripts/build_paper_numbers.py` + `scripts/audit_paper_numbers.py`.
  `.tex` alone = 817 literals: 452 sourced, 266 protocol/years, **180
  unsourced** — 81 matching a pre-K artifact and no current one (**provably
  false now**), 99 that never had a source at all (incl. `+0.185` and its CI,
  `94/100`, `44/50/6`, the TV window means, the compute figures, and the
  compression ratios). With PAPER_FINDINGS: 2006 literals, 421 unsourced.
  Gate: `.venv/bin/python scripts/audit_paper_numbers.py --strict`.
  **(5) UNIT 1 COMPLETE — Class B artifacts generated, verified, wired.**
  `windowed_tv()` (metrics/tdp.py) and `paired_policy_contrast()` (stats.py)
  + two thin runners (`scripts/report_tv_windows.py`,
  `report_policy_contrast.py`); artifacts archived at
  `results/diag/amendment-k/classb/` (remote copy at
  `~/kv-caching/results/amendk/classb/`). **All four TV ground truths
  reproduce** (pre-eviction exactly 0.0 — the hard self-control target;
  first50 0.0018169; later per-trace 0.0883564; step-pooled 0.1182457; over
  800 profiles, median onset 65) and the contrast reproduces exactly (mean
  0.185 = 0.4225 − 0.2375; 44 wins / 50 ties / 6 losses = 100; non-negative
  94). `windowed_tv` is a faithful refactor of `make_paper_figures.py`'s
  `print_windowed_tv`, so reproducing the four targets validates both paths.
  **One discrepancy, and it is now the ONLY unsourced Class B literal: the
  published CI upper bound `0.242` (×3 in the .tex, L74/L200/L205) is not
  reproducible** — a Monte-Carlo quantity whose seed was never recorded. Over
  200 seeds the upper endpoint spans [0.2375, 0.2450] (median 0.2425); our
  pinned seed=0 gives **[0.13, 0.24]**. Lower bound 0.130 matches exactly.
  Claim survives (CI excludes zero either way); on the Unit 3 list as a
  mechanical substitution. Fixed at the class level per §2.4 rather than
  patched: the artifact now records `seed`, `n_resamples`, and a
  `difference_histogram` (7 counts, a sufficient statistic for the whole
  bootstrap — anyone can recompute the interval from the artifact alone), and
  `windowed_tv` records `median_onset` (the paper quotes 65 twice; also
  unsourced before). **Audit effect: .tex unsourced 180 → 161, both-docs
  421 → 386; .tex sourced 371 → 433, protocol 232 → 189** (numbers formerly
  waved through the config allowlist now match a real artifact).
  **(5b) `make sync` RESTORED to rsync (docs/operations/remoteserver.md §35).** The §34 scp
  deviation is CLOSED. Two causes, only one previously named: contention is
  real but *intermittent* (first apply died at load ~10, a retry moved all 220
  files) — and the actual blocker was that `amendk/` sat inside the hashed
  source tree, so 39 remote-only output files made the guard's local-vs-remote
  hash comparison unsatisfiable forever. Verified file-by-file: all 206 local
  files byte-identical on the remote, zero local-only, zero content diffs.
  Fixed by `mv amendk results/amendk` (results/ is ignored by both the hash
  and rsync); nothing deleted, everything compared first. `.sync-state` now
  truthful: `sync_method=rsync`, commit 81c2642. **Standing rule added: never
  write generated output into the source tree on either side**, and never
  diagnose link health from `make sync` — the dry-run traverses without
  transferring, so it passes while the apply fails.
  **(6) NEXT.** A 14-unit work order is in play (Units 1–4 output-path
  hardening; 5–6 statistics + the resolution-collapse reframe; 7–9 claim
  calibration, structure, figures; 10 the TV-matched noise control that
  `docs/project/context.md` §1.3 calls MANDATORY and the paper does not contain; 11–12
  budgets 512/1024 + second-tier; 13 the reframe/retitle; 14 endgame).
  Target **ICLR 2027, ~late September** — not the August arXiv path below.
  **Do not arXiv or anonymize before the reframe lands.**

  **★★★★ ALL EXPERIMENTAL WORK COMPLETE (2026-07-24) — SUPERSEDED by the
  bullet above: the accuracy numbers in this bullet are pre-Amendment-K and
  several are now false. Retained for the run/gate history only.** Nightly
  queue fully
  drained, zero runs remain, ORIGINAL CAMPAIGN + AMENDMENT J both 100%
  READ.** m4-q14b-256 (14B, the last job in the 31-job queue) landed:
  grid/readout/reverse all 400/400 against fc-q14b (acc 0.880, band
  [0.82,0.96] published 0.939 → gate PASSED). StreamingLLM: free 0.240
  [0.150,0.335], readout 0.503 [0.395,0.600], residual −0.263 CI
  [−0.335,−0.170], bracket [−0.485,−0.263] — **REPLICATES cleanly.**
  R-KV: free 0.670 (again the stronger free-runner — persists across all
  three scales), readout 0.848 [0.750,0.922], residual −0.178 CI
  [−0.273,−0.085] (excludes zero) but **bracket [−0.178, 0.029] SPANS
  ZERO** — the first non-degenerate cell in the paper where it does.
  Root cause: reverse-readout accuracy 0.909 exceeds the 0.880 reference
  (+0.029 change), matching the order of magnitude of the independently
  measured +0.023 pure resampling effect (§4j) — so the span reads as
  resampling noise on an already near-lossless readout channel (0.848 vs
  0.880), not a sign-reversed trajectory effect. Not floored (trunc
  0.230), so it's not DEGENERATE either — falls under Amendment J's
  "anything else" branch: reported verbatim, not forced.
  **Scorecard: 3/4 scale-axis cells REPLICATE** (both 1.5B cells,
  SLM-14B); R-KV-14B inconclusive. Per Amendment J, ≥3/4 triggers the
  scope-upgrade sentence, now applied: "the readout-invisible trajectory
  residual holds across one model family spanning three scales
  (1.5B/7B/14B), plus an independent Llama-base check."
  Verified genuine (not an instrument abort): R-KV's self-check raises
  RuntimeError on RoPE-convention drift (`presses/rkv.py:209`) — grid log
  has zero error/traceback/abort hits, `fails:0` throughout, full 200/200
  rows landed.
  Paper edits: Scale-axis threats paragraph rewritten in full (both
  models, all 4 cells, R-KV-14B exception spelled out); Generality
  paragraph gets the scope-upgrade sentence, stale hedges removed; §4
  preamble "in progress" → "final"; conclusion's "near-term extensions"
  paragraph rewritten past-tense. Abstract/title untouched (scale axis
  was always threats+conclusion-only by design). No figure change (still
  doesn't clear the FIGURES_PLAYBOOK.md bar for a dedicated panel).
  Recorded PAPER_FINDINGS §4t; §7 status updated; RESULTS_PLAYBOOK unit
  marked fully CONSUMED. Archived results/diag/m4-q14b-256-summary.json.
  **REMAINING WORK: none experimental. Endgame only** — anonymization
  scrub (personal email in scripts/nightly_report.py; exclude docs/project/context.md/
  docs/strategy/*/complete_setup.md from the public artifact), restore
  "Anonymous Authors" for ICLR's double-blind submission, arXiv upload,
  per `docs/strategy/SUBMISSION_STRATEGY.md` Phase 11. Drop-dead
  2026-08-01 met with 8 days to spare.

  **★ SCALE-AXIS 1.5B CELL IN, BOTH PRESSES REPLICATE (2026-07-20):**
  m4-q15b-256 complete (grid/readout/reverse all 400/400) against fc-q15b
  (acc 0.800, gate passed). StreamingLLM: free 0.220 [0.130,0.315], readout
  0.495 [0.378,0.617], residual −0.275 CI [−0.362,−0.193], bracket
  [−0.418,−0.275]. R-KV: free 0.355 [0.240,0.465] (again the stronger
  free-runner), readout 0.688 [0.566,0.806], residual −0.333 CI
  [−0.439,−0.230], bracket [−0.333,−0.262]. Both CIs/brackets exclude zero,
  trunc 0.150/0.210 (≪0.5 degenerate floor) → REPLICATES per Amendment J.
  Recorded PAPER_FINDINGS §4s; new "Scale-axis check" threats paragraph
  added to the paper (after Cross-architecture paragraph); fixed a stale
  "AIME-24 is queued" sentence in the Generality paragraph (AIME was
  already read 2026-07-18, §4r) while in there. m4-q14b-256 (14B) still
  0/400 — the ≥3/4-replicate scope-upgrade sentence ("one model family
  across three scales…") is withheld until it lands; docs/results/RESULTS_PLAYBOOK.md
  marks q15b half-consumed, q14b still PENDING. No figure change (subset
  scale, threats-only per FIGURES_PLAYBOOK.md rule — revisit if q14b also
  replicates and the pair earns a small-multiples panel). Archive
  results/diag/m4-q15b-256-summary.json next sync.

  **★ GPU-UTILIZATION FIX CONFIRMED UNDER REAL LOAD (night of 07-19→20):**
  the fair-share `shard_seeds(..., total_gpus=len(gpus))` fix (written
  07-19, root cause: striding over free GPUs only starved a GPU that freed
  mid-window) held up on its first real night — `logs/nightly_autorun.log`
  shows both gpu0 and gpu1 at 34–48% util in lockstep from 01:45 through
  past 09:00, no idle gap. fc-q14b resumed 150→150 (seed 89 in flight,
  stopped cleanly by the <1h guard, rc=124 resume-safe, not a failure).
  nightly_state.json shows fails=0 on every active job. Next: q14b grid
  still needs to start (blocked on fc-q14b bank finishing all 4 seeds
  first — 3/4 done).

  **★★★ ORIGINAL CAMPAIGN 100% READ — m4-aime-2048 in (2026-07-18):** Branch A
  + two bonus findings. (1) NO >100% regime: best free Random 0.283 vs FC 0.371
  (R-KV 0.217, SLM 0.142; gained flip ≤ 0.050) — scale/protocol finding per
  Amendment E. (2) Residual REPLICATES on 2nd benchmark: −0.179/−0.246/−0.321
  (R/RKV/SLM), all CIs exclude zero. (3) Readout PRESS-INDEPENDENT at 19.7%
  dose: identical 0.462 across all 3 presses — investigated BEFORE reading
  (85/93 defined rows converge to same extracted answer via different texts; 8
  divergent rows all-wrong everywhere; NOT a bug) — the sharpest form of the
  thesis. Paper: threats >100% rewritten w/ result; related-work/audit-caption
  tense; preamble+conclusion now name only scale axis. FINDINGS §4r; archive
  diag/m4-aime-2048-summary.json. ALSO fc-q15b bank GATE PASSED (0.800, band
  [0.60,0.85]); q15b grid 39/400, q14b bank 124/200. REMAINING: 2 scale units
  (~5 nights) → endgame (venue conversion, scrub, arXiv mid-Aug).

  **★★ AC IMPROVEMENT PLAN EXECUTING (started 2026-07-17 night; log =
  logs/gpu_improvement_plan.log):** (1) SCALE AXIS QUEUED (Amendment J):
  R1-Distill-Qwen-1.5B @ad9f0ae0 + -14B @1df85071 pinned from HF API; 8 configs
  (SLM+R-KV, 50×4, own banks) dry-run validated; 8 queue jobs appended (1.5B
  chain first, 14B min_free_gb=40); harvest units m4-q15b-256/m4-q14b-256;
  playbook units w/ pre-committed REPLICATES/DEGENERATE/NON-REPLICATION branches
  + bank gates (1.5B FC∈[0.60,0.85] pub 0.839; 14B∈[0.82,0.96] pub 0.939).
  Timeline: 1.5B chain ~1-2 nights, 14B ~3-5 nights — all within budget/margin.
  (2) PAPER REFRAMED as "A Trajectory Audit of KV-Cache Compression for
  Reasoning": abstract deliverable sentence, §1 protocol positioning, NEW
  contribution bullet 1 (the audit + validity kit), §3 naming sentence, §6 audit
  paragraph FOREGROUNDED w/ validity-kit sentence before the implication.
  (3) "method-invariant" RETIRED (0 instances; existential wording everywhere;
  FINDINGS §5 rule added). (4) Bullet 2 random-beats-recency now carries the
  architecture-dependence parenthetical. (5) Tonight: readout-aime24-2048
  s[47,89] completes → READ m4-aime-2048 TOMORROW (playbook branches ready,
  FC_AIME=0.371, ratio 19.7%); fc-q15b bank starts. (6) DEFERRED: FC/HOTT
  matched-decode probe — only if GPUs idle after 14B chain; needs HOTT's decode
  config verified first. AIME >100% anchor verified intact post-reframe.

  **★ fc-readout-control read (2026-07-17) — Branch A, +0.023 measured:**
  full-cache readout of FC bank = 0.9056 (355/392; displays 0.906) vs FC
  0.8825 → pure answer-resampling effect +0.023, inside the [0.88,0.92] band;
  14.8×/12.0× below trio/R-KV residuals ("order of magnitude" exactly true).
  §4.2 calibration sentence upgraded from 2048-derived bound to direct
  measurement; PAPER_FINDINGS §4q; archive diag/fc-readout-control.json.
  ONE UNIT LEFT: m4-aime-2048 (grid 360/360 DONE, readout 180/360 → ready
  ~07-18). After it reads: queue EMPTY → endgame per CLAUDE.md (venue
  conversion SUBMISSION_STRATEGY §6 + BEST_POSSIBLE Phase 8 notation diet,
  anonymization scrub incl. email in nightly_report.py + internal docs
  excluded incl. complete_setup.md, arXiv ~mid-Aug). AIME read uses playbook
  UNIT m4-aime-2048: FC_AIME=0.371, ratio 19.7%, branches A (no gains) /
  B (>100% regime, one measured paragraph max, Amendment G stands).

  **✓ RESOLVED — server outage 2026-07-15 evening (~30 min, likely reboot):**
  went "connection refused" → timeout → back at 21:59 server time. Self-heal
  loop completed: make sync-apply clean (fresh .sync-state incl.
  complete_setup.md), crontab verified intact (3 nightly entries), autorun
  dry-run normal (correctly waiting for the 01:45 window). No action needed;
  tonight runs as planned (grid-aime24-2048 continues from 106/360).

  **★ m4-llama-256 + fc-aime24-acc read (2026-07-15) — MIXED Llama result,
  reported plainly (fits no branch):** SLM-Llama REPLICATES: free 0.235 (Qwen
  0.2375!), readout 0.508, residual −0.273 [−0.345,−0.175], bracket
  [−0.426,−0.273], revCI [−0.523,−0.327], zero excluded. Random-Llama
  DEGENERATES: free 0.000, trunc 0.805, readout 0.021, bracket [−0.021,+0.028]
  spans zero, 6/400 defined reverse. Instrument check DONE before interpreting:
  traces are '.\\n\\n' filler loops (fluency collapse, sink-eviction hypothesis,
  cite xiao2023streamingllm), SLM cell sane on identical harness → real finding.
  Random-beats-recency is ARCHITECTURE-DEPENDENT (reverses on Llama). Paper:
  threats Cross-architecture-check paragraph + scope "one primary model";
  abstract/§4 untouched. NEVER cite Llama-Random as small-residual evidence
  (both channels floored = uninformative). AIME bank: 0.371 acc / 0.221 trunc /
  10,415 mean tok (ratio 2048 = 19.7%); below soft band but cap-explained
  (§4p); FC_AIME slot = 0.371. Remaining: m4-aime-2048 (grid 106/360, ~2-3
  nights), fc-readout-control. PAPER_FINDINGS §4o+§4p; archives in diag/.

  **★ MORNING CHECK 2026-07-14 — no unit ready; Llama bank GATE PASSED:** runs
  finished overnight: grid-diss256-llama 400/400 + reverse-readout-diss256-llama
  400/400; readout at 273/400 (s47=73, s89=0) → m4-llama-256 READY tomorrow.
  fc-aime24 172/240. NOTHING read into the paper (harvest gating respected; no
  partial cells). Instrument gate run instead: fc-llama50 bank sanity — 200/200
  rows, accuracy 0.805 (161/200), trunc 0.025, mean 4,095 tok. Expected shape ✓
  (published Llama-8B MATH-500 = 0.891; our harness sits ~0.08 under published
  for Qwen too — same cap/sampling gap, not an instrument bug). Tonight per
  report: readout-llama s[47,89] + fc-aime24 s[197,263,347]. No new failures
  (FAILCHECK shows only the fixed 07-11 entries). Tomorrow: read m4-llama-256
  via playbook (Branch A replicate / Branch B non-replicate, threats-generality
  anchor "one model (DeepSeek-R1-Distill-Qwen-7B)" verified intact).

  **★ FIGURE UPGRADE PASS + FIGURES_PLAYBOOK handoff (2026-07-13, Fable final):**
  (1) Fig 1: budget-group headers now carry the realized ratios "(3.8% of mean
  trace)"/"(56.6%)" — dose structure legible from the figure alone; (2) Fig 2:
  faint dotted FC vertical (0.8825) in every panel so all rungs read against the
  reference; (3) Fig 4: new annotation "TV = 0.0000 before first eviction
  (self-control)" pointing at the flat pre-onset segment (instrument validity now
  in the figure, not just the caption). Fig 1+4 regenerated DATA-mode on remote,
  Fig 2 local DATA-mode; all viewed. (4) Paper formatting: caption package
  (small font, bold labels, 6pt skip) — builds clean. (5) NEW
  **docs/operations/FIGURES_PLAYBOOK.md**: the complete figure method for the Opus session
  (source of truth, .3f display convention, palette+validator, regen commands,
  per-figure anatomy, 9-item failure catalog, settled judgment calls);
  pointer added to RESULTS_PLAYBOOK general rules. Known matplotlib gotcha
  recorded: "---" renders literally (not an em dash).

  **★ FIGURES REFRESHED FOR 4-POLICY GRID + add-256-rkv (2026-07-12 evening):**
  user-requested figure update (supersedes Branch C figures-unchanged default).
  NEW: R-KV ADD run (report_add over existing grid rows, precedent = h2o read):
  ADD 0.459 (lowest of 4) / within 0.450 / term +0.149 / flip −0.364 → Table 3
  4th row + §4.5 sentence; split renamed recoverable-vs-concentrated in §4.5 +
  Fig 3 caption. Fig 1: 8 bars incl. R-KV 0.520 (sky #56B4E9, validated ΔE
  15.7–20.7) + H2O-2048 0.823 (tint #E5BCD3), budget-group gap, remote DATA
  histogram. Fig 2: 2×2 — top row recoverable (Random, R-KV), bottom concentrated
  (SLM, H2O), all 8 endpoints. Fig 3: 4 groups. All eyeballed; PDF clean; gates
  green (135). Archived: results/diag/add-256-rkv-summary.json. Remote harvest
  dir renamed add-256-rkv (was add-256-rkv.json — report_add --out is a DIR).

  **★ m4-rkv-256 read — BRANCH C (2026-07-12):** R-KV@256 free 0.520 [0.435,0.610]
  (BEST of 4 policies), readout 0.798, residual −0.278 [−0.335,−0.195], reverse
  0.662 (229/346; 54 undef), bracket [−0.278,−0.221], revCI [−0.355,−0.203],
  imputation −0.310/−0.175 (zero excluded always), trunc 0.135, gained flip
  0.0025. Free 0.520 = intermediate band → Branch C: row+bracket plainly, no
  framing overhaul, Fig 1/2 UNCHANGED. Key structure: forward residual nominally
  below the −0.34 trio with OVERLAPPING CIs (invariance stays scoped to trio);
  reverse side R-KV joins Random on the recoverable side (split renamed
  "recoverable vs concentrated", both separations CI-clean). Paper: Table 2 4th
  row + ‡, §4.2 R-KV paragraph, §4.3 eight endpoints, abstract clause,
  Setup/§5/threats/related-work/conclusion status flips. PAPER_FINDINGS §4m.
  Also last night: fc-llama50 bank 200/200 DONE (launcher fix worked), llama
  grid 16/400, fc-aime24 117/240. Next units: m4-llama-256 (~2 nights),
  fc-aime24-acc, then aime trio + fc-readout-control.

  **★ NIGHT OF 07-10→11: rkv-readout DONE + 3 stuck jobs root-caused & fixed
  (2026-07-11, commit 74e97a5):** readout-diss256-rkv 400/400 COMPLETE; reverse
  317/400 → m4-rkv-256 needs only s89 (83 rows), READY tomorrow. THREE jobs went
  STUCK at 0 across 3 launches; both root causes fixed with regression tests
  (135 tests): (1) build_bank.py lacked --seeds (autorun passes it to every
  runner; fc-llama50/fc-aime24 were the first bank jobs the queue reached →
  rc=2) — added, mirroring run_grid verbatim; (2) run_replay readout branch cast
  int(budget) → ValueError on readout-fc-control's budgets [full] — now passes
  "full" through (filenames fullcache-full-sN.jsonl, bank convention). Remote
  fails counters reset to 0 (backup logs/nightly_state.json.bak-20260711); all
  three launcher invocations re-verified on the REMOTE post-sync; tonight's
  plan confirmed: reverse-rkv s89 + fc-llama50. No data lost; failures were
  launch-time (rc=2 / pre-first-row crash), zero rows written by broken code.

  **★ m4-2048-h2o + replay-256-h2o read (2026-07-10):** both Branch A, both good.
  H2O@2048: free 0.8225→0.823, readout 0.8980→0.898 (≥ ref), residual −0.0755
  CI [−0.1025,−0.02] — between Random/SLM; dose response now holds for ALL 3
  policies (one clause added to §4.2 dose paragraph). replay-h2o: AUC-TV 0.073 vs
  0.086/0.083 → §4.4 "matched across all three policies"; STOP-check passed
  (before-window 0.0000 per-profile AND pooled). H2O-style now complete on every
  instrument. Paper: §4-preamble/threats/conclusion updated (H2O no longer "in
  progress"). PAPER_FINDINGS §4k+§4l; archives in results/diag/. Sync mystery
  resolved: .sync-state Jul 8 01:55 + remote queue has readout-fc-control → the
  audit commit shipped in time, no night lost; SSH auth back. R-KV: grid 400/400
  DONE, readout 208/400, reverse 71/400 → m4-rkv-256 likely ready ~07-11 (use
  playbook branches; Branch B = free ≥ 0.55 = discriminative-outlier is GOOD).
  Next pending: m4-rkv-256, then llama quartet, aime trio, fc-readout-control.

  **★ PI SUBMISSION AUDIT (2026-07-08 ~00:30, Fable, final session):** adversarial
  review + PI decision pass, commits e5b39c2 + bef350c. (1) Three defensive
  sentences added: FC 0.882 anchored to the R1 report's 0.928 MATH-500 pass@1
  (verified vs arXiv 2501.12948) in §2.2; resampling bound (+0.02 from 2048
  readout, direction shrinks residual) in §4.2 dose paragraph; truncation bound
  labeled extreme-by-construction linking to ADD §4.5. (2) NEW QUEUE JOB (23rd,
  LAST priority): readout-fc-control — full-cache readout of FC bank, 400 rows,
  directly measures the answer-resampling effect; config dry-run validated,
  Makefile dryrun line added (24 configs), harvest unit fc-readout-control (new
  readout_acc kind), playbook unit with STOP branch (expect 0.88–0.92). Decisions
  AFFIRMED (not queued): SnapKV cell, budgets 512/1024, TV-matched noise, logic
  benches, Qwen3; second grader = optional CPU-side appendix, not blocking.
  VERDICT: submit after queue drains + venue conversion (ICLR-target ~50%,
  as-today would be ~10-15% on format alone). ⚠ SYNC: bef350c sync timed out
  (link flaky); background retry loop armed — if resuming, VERIFY .sync-state is
  fresh before tonight's window or the watcher preflight may refuse.

  **★ FINAL SUBMISSION-POLISH PASS (2026-07-07 late, Fable):** independent
  reviewer-mode pass over the full manuscript; 15 external observations evaluated,
  ~7 acted on. Changes: (1) NEW Figure: protocol schematic `figures/fig0_protocol`
  (four-row diagram of FC/free-run/forward-readout/reverse-readout; fill = cache
  condition, texture = compressed, in-span text = token source) placed in §3.2 —
  compiled figure numbers shift by one, `\cref` handles it; generator
  `fig0_protocol()` in make_paper_figures.py (SUMMARY-only). (2) Abstract: dropped
  the three explicit bracket intervals (kept "excludes zero for every policy" +
  endpoint-CI clause; residual triple stays). (3) Intro ¶1: HOTT per-method numbers
  deduped (kept 0.47/0.03 hook, full audit stays §2); merged duplicate
  "contribution" sentences. (4) Thesis + conclusion opening scoped "At aggressive
  cache budgets" (licensed by the 2048 dose result). (5) §4.4 "can only add to it"
  → "adds a second divergence source" (monotonicity overclaim removed). (6) Def 6:
  dropped never-used TDP acronym. (7) §4.3 three-qualifications paragraph split
  into three paragraphs. (8) Conclusion summary gains one dose-response clause.
  NOT changed (deliberate): "certify/collapse" vocabulary (backed by the
  both-literatures-correct framing), contributions bullets (playbook anchors),
  Table 2, all §4 numbers. ALL 7 playbook anchors verified intact post-edit.
  Build 0 errors / benign 1.4pt overfull; gates green (132 tests).

  **★ m4-2048 read (2026-07-07):** dose-response decomposition IN. FC 0.8825;
  Random@2048 free 0.850, readout 0.9005 [0.82, 0.9425], residual −0.0505
  [−0.065, −0.01]; SLM@2048 free 0.8075, readout 0.9031 [0.8225, 0.9425], residual
  −0.0956 [−0.1275, −0.0425]. Readout ≥ reference at 2048 → remaining loss entirely
  trajectory-channel; residuals shrink ×3.6–6.8 vs 256, CIs still exclude zero.
  Branch A per RESULTS_PLAYBOOK (template's "order of magnitude" corrected to
  factor 3.6–6.8). Paper: new §4.2 dose paragraph + Setup/§4-preamble/§4.1/threats/
  conclusion status updates; abstract unchanged. PAPER_FINDINGS §4j. summary.json
  fetch to results/diag/ pending link (background probe armed). NOT read: m4-2048-h2o
  (readout 130/400), replay-256-h2o (21/400) — likely ready after tonight. Next
  pending units: m4-2048-h2o, replay-256-h2o, then rkv/llama/aime chains.
  ALSO (same day, full-sweep audit): display convention PINNED — every 3-decimal
  number is `.3f` of the artifact float (PAPER_FINDINGS §5). Two hand-rounded ties
  corrected everywhere: H2O free 0.248→0.247 (99/400), SLM@2048 0.808→0.807
  (323/400). Fig 1+2 regenerated to match; figure constants now store artifact
  precision. Nightly automation verified healthy (cron live, tonight resumes
  readout-diss2048-h2o s[23,47,89] + replay-diss256-h2o; disk 200G free; rkv
  smoke marker in place). All 5 figures eyeballed against captions — no other
  defect found.

  **★★★ FABLE HANDOFF COMPLETE (2026-07-06). Successor: work the morning loop in
  CLAUDE.md §Handoff-operations.** Built today: (1) `scripts/harvest_results.py` —
  ONE command turns finished runs into report artifacts + digest (unit table inside;
  tested live: correctly gated 7 pending units, harvested the 2 ready ADD units);
  (2) **docs/results/RESULTS_PLAYBOOK.md** — pre-written BRANCHED paper text + tex anchors +
  figure/docs updates for EVERY pending unit (2048 decomp, h2o-2048, replay-h2o,
  rkv-256 incl. outlier branch, llama replication incl. non-replication branch,
  aime incl. >100% branch). (3) CLAUDE.md extended with the operating rules.
  (4) report_m4 --bench flag (AIME cells would have silently read 0 rows). (5) NEW
  RESULTS IN PAPER TODAY: ADD-h2o row (0.715/0.713/+0.101/−0.636, Fig 3 now 3 groups)
  + ADD dose response @2048 (0.085/0.146 vs floor 0.068; silent-flip regime did NOT
  materialize — reported per prereg). (6) Also today: realized compression ratios in
  Setup (256 = 3.8% of mean trace; 2048 = 56.6%), all 6 bib author lists filled
  (arXiv-verified), the trajectory-audit recipe paragraph added to the conclusion.
  Remaining human endgame after queue drains: playbook units as they land → venue
  conversion (SUBMISSION_STRATEGY §6 + BEST_POSSIBLE Phase 8) → anonymize → arXiv.

  **★★★ H2O-STYLE BRACKET IN — 3-policy story COMPLETE at 256 (2026-07-06):**
  reverse-readout-diss256-h2o 400/400 -> rev 0.4194 (151/360; 40 undefined), bracket
  **[−0.463, −0.349]**, rev CI **[−0.566, −0.420]**; imputation bounds −0.505…−0.405
  (zero excluded always). STRUCTURE: reverse side splits uniform-vs-concentrated —
  H2O-style twins SLM (−0.463/−0.466, CIs overlap), both separate from Random
  (non-overlapping CIs). Paper fully updated (abstract/bullet4/§4.3 3-policy rewrite/
  threats/conclusion/captions); Fig 1 has the H2O† bar, Fig 2 is now THREE panels
  (palette #0072B2/#D55E00/#CC79A7 validated). Archived m4-h2o256-full-summary.json.
  Also in flight: readout-diss2048 539/800, grid-diss2048-h2o 289/400 (tonight).

  **★★★ H2O-STYLE FORWARD DECOMPOSITION LANDED (2026-07-06 read of last night):**
  readout-diss256-h2o 400/400 -> free 0.2475 [0.180,0.318], readout 0.597 [0.503,0.668],
  **residual −0.349 [−0.403,−0.270]**, share 45%, undef 8 (same rows). => forward
  residuals across THREE unrelated policies: −0.343/−0.347/−0.349 (spread 0.006, all
  CIs exclude zero) while readout spans 0.169 — method-invariance now a 3-policy result;
  IN THE PAPER (Table 2 row + abstract + §4.1/4.2 + threats; summary archived at
  results/diag/m4-h2o256-forward-summary.json). Multi-job tail-fill WORKED in prod
  (06:00 reverse-h2o on gpu0 while 06:10 readout-2048 started on gpu1). In flight:
  reverse-h2o 152/400, readout-2048 164/800 (both resume tonight). TODO when reverse-h2o
  completes: h2o bracket + imputation bounds into §4.3, regen Fig1 (H2O bar) + Fig2
  (third panel) in ONE batch, rebuild PDF.

  **★★★ RUN-ONLY STATE REACHED (2026-07-05, Fable): queue = 22 jobs, everything the
  paper needs is queued + code-complete.** (1) **R-KV UNLOCKED (Amendment I):** smoke
  first FAILED (Jaccard 0.805) → root cause = dataclass field shadowing (press ran
  SnapKV window/kernel 64/5, not R-KV 8/7; parity harness couldn't see it, the GATE
  caught it) → fixed (dataclass subclass + regression test + TRUE 8-query window w/
  buffered hidden states + analytic RoPE w/ per-event self-check vs framework rotation;
  scaled-RoPE models refused) → re-run **PASS Jaccard 1.0000**, marker
  `rkv-gen-smoke-PASS.json` written; press registry promotes rkv iff marker exists
  (env KVCACHE_RKV_SMOKE_MARKER or $KVCACHE_BASE_RUNS_ROOT). (2) **NEW capabilities:**
  AIME-24 dataset handle implemented + pinned (HuggingFaceH4/aime_2024 @2fe88a2, split
  train, 30 rows; loader test); R1-Distill-Llama-8B model entry pinned (@6a6f4aa —
  NOTE: llama3-scaled RoPE → rkv press refuses on it by design; llama cells use
  random/slm only); run_replay benchmark-generalized; autorun gains bench= (aime
  subtree counting), marker= (gate jobs), extra_args= (bank download flags);
  launch.sh gains rkvsmoke stage. (3) **QUEUE (22 jobs, ~4.3k rows remaining):**
  ... h2o chain → rkv-smoke(DONE) → rkv 256 grid/readout/reverse → llama FC bank
  (--allow-model-download) → llama grid/readout/reverse → fc-aime24
  (--allow-dataset-download) → aime grid+readout @2048 (random/slm/rkv). All configs
  dry-run validated; watcher dry-run verified (tonight = readout-diss256-h2o).
  131 tests; lint green; synced (docs re-sync pending final commit).

  **★★★ 2026-07-04 (Fable session; remote back up):** (1) **ROADMAP #1 DONE — bracket-
  endpoint bootstrap CIs** (report_m4 fields `unexplained_residual_ci95` /
  `reverse_residual_ci95`; problem-clustered, 2000 resamples): fwd **[−0.402,−0.258]**
  (Random) / **[−0.403,−0.268]** (SLM); rev **[−0.383,−0.213]** / **[−0.553,−0.410]** —
  ALL exclude zero; reverse endpoints separate with NON-overlapping CIs (SLM more
  trajectory-damaged on reverse estimand; attribution = estimand-dependent). Paper
  §4.3/threats/abstract/§3.1 updated; summary archived at
  `results/diag/m4-powered-diss256-summary.json`. (2) **H2O-style grid VERIFIED 400/400:
  free 0.2475, trunc 0.105** — see PAPER_FINDINGS §4h for the LABELING LANDMINE
  (ObservedAttentionPress has NO recency reservation → never claim "H2O collapses" /
  HOTT contradiction; "predictive upgrade" framing TEMPERED accordingly). readout-h2o
  runs tonight. (3) **Fig 1 DATA-mode landed** and exposed imprecision: 94/100 =
  **44 win / 50 tie / 6 loss** → paper now says "non-negative on 94/100" (never "beats
  on 94/100"). (4) **Table 3 estimator defined + reproduced**: per-trace 0.0884
  (pooled 0.1182); `make_paper_figures.py --print-windows`. (5) **Scoop-sweep cron
  INSTALLED on remote** (Mon 09:20, `scripts/scoop_sweep.py`, emails via smtp json).
  (6) make sync-apply run AFTER these changes (sync-state fresh for tonight's watcher).

  **★★ SERVER SWEEP + PRO FIGURES + QUEUE EXTENSION (2026-07-04 evening, Fable):**
  (1) **UNREAD RESULT FOUND: grid-diss2048-tok read** — Random 0.850 / SLM 0.808 vs FC
  0.882 (near-lossless pole at power; ordering persists +0.042; doubles as harness
  control). Now in paper §4.1 + Fig 1 dose bars + PAPER_FINDINGS §4i. (2) **EXACT
  REPLICATION verified**: v1 grid vs tok re-run = 800/800 rows identical (correct,
  truncated, n_tokens) → determinism claim added to paper Threats + PAPER_FINDINGS §1b.
  (3) **Figures fully restyled** (Okabe-Ito validated palette, dataviz-skill pass;
  Fig 1 gains 2048 dose bars; Fig 2 shows residual arrow + all endpoint CIs; Fig 4
  annotated staircase; all rendered + eyeballed). (4) **QUEUE EXTENDED + verified
  parse (11 jobs):** NEW reverse-readout-diss256-h2o (bracket for 3rd method; h2o grid
  HAS token sidecars) after readout-diss256-h2o; NEW replay-diss256-h2o (mechanism,
  3rd method) appended last. New configs: reverse_readout_diss256_h2o.yaml,
  replay_math500_diss256_h2o.yaml. (5) H2O-style free-run 0.248 now in paper Threats
  with the ObservedAttention-≠-H2O label (no HOTT contradiction claim).

  **★ AUTOMATION HARDENED + EMAIL REDESIGNED (2026-07-04 afternoon, Fable):**
  (1) **Watcher tail-fill fix:** previously, when the selected job's remaining seeds were
  all in flight the tick no-op'd — one long tail shard could idle the OTHER GPU for hours
  and block every later job. Now `launch_candidates()` walks the queue and launches the
  first job with needy seeds onto GPUs not occupied by ANY of our sessions (cross-job
  occupancy via one tmux parse, `live_map`); STUCK marking now falls through to the next
  candidate the same tick. 4 new tests (24 total for autorun); dry-run VERIFIED on remote
  against real state (tonight: readout-diss256-h2o shards g0=[11,47], g1=[23,89],
  MAX_HOURS 6.91 — correct). (2) **nightly_report.py redesigned:** subject
  `[kvcache] date | +N last night | %% | FLAG`; sections LAST NIGHT (per-job deltas +
  ** COMPLETED ** callouts), QUEUE (progress bars), TOTAL+ETA (nights at last night's
  rate), TONIGHT (watcher's actual plan + per-seed counts = looping-shard flag),
  DEADLINES (R-KV/AIME drop-dead D-countdown — the wiring promised on 07-03), HEALTH
  (stuck/GPU/disk). Per-job snapshot now stored for deltas (old format tolerated);
  dry-run no longer overwrites the delta baseline. Remote dry-run rendered clean.
  Synced; 124 tests, lint green.

  **★★ DECISIONS MADE 2026-07-03 (Claude, at user's request — both CONFIRMED):**
  (1) **R-KV + AIME drop-dead = 2026-08-01.** If R-KV cells (post-gen-smoke) and AIME are
  not producing rows by Aug 1, CUT both to future work; the paper ships as 3 methods x 2
  budgets with R-KV parity retained as a validity artifact (already written that way — no
  headline depends on either). Rationale: ICLR 2027 full deadline ~late Sept, arXiv stake
  target mid-Aug, AIME dataset handle still TODO, R-KV smoke still pending. TO WIRE on next
  SSH-up (either session): `nightly_report.py` shows days-to-drop-dead and flags R-KV/AIME
  as CUT after 2026-08-01.
  (2) **arXiv scoop sweep = weekly, Mondays.** Log + queries + foreclosure test:
  `notes/scoop-watch.md`. FIRST SWEEP DONE 2026-07-03: **no foreclosure** — instrument still
  ours. Two adjacent papers found and ALREADY CITED in the tex: VaSE (2606.03928, stochastic
  eviction beats deterministic selection on reasoning → corroborates Random>SLM, cited in
  §4.1) and attention-dynamics physics (2603.01426, synthetic long-input probes, cited as
  third critical strand in related work). TO WIRE on next SSH-up: `scripts/scoop_sweep.py`
  (arXiv API, last-7-days, append to notes/scoop-watch.md + Monday email) + cron `20 9 * * 1`.

  **★ FIGURES + PLANNING (2026-07-03):** all 5 paper figures generated by
  `scripts/make_paper_figures.py` (SUMMARY mode from verified constants — works anywhere;
  DATA mode adds Fig1 per-problem inset / Fig4 full TV curve / Fig5 ADD histogram when run on
  the remote with `--runs-root/--profiles-dir/--twofc-report`). PDFs in `figures/`. matplotlib
  pinned in requirements.txt. Queue reordered: **H2O-256 before readout-2048** (H2O is the
  method HOTT reports surviving → highest-value method-invariance test). **Recommended
  drop-dead 2026-08-01** for R-KV cells + AIME (confirm with user); after that, freeze generality
  and write. TODO on next SSH-up: (a) audit last night's run (both GPUs / crashes), (b) run
  figures DATA-mode on remote, (c) confirm bracket-fix + reorder were live for tonight.

  **⚠ CORRECTED 2026-07-03 (residual-bracket bug; PREREGISTRATION Amendment H):** the reverse
  bracket endpoint was computed as total − (acc_rev − acc_fc), which is the on-policy READOUT
  change, not a residual estimate. Correct endpoint = acc_rev − acc_fc itself. Corrected
  brackets: Random **[−0.343, −0.246]**, SLM **[−0.466, −0.347]** (EXACT, confirmed by the
  report re-run on remote data 2026-07-03; −0.465 was from the rounded 0.417; both FARTHER
  from 0 — headline stronger). But: bias ordering reverses for SLM (reverse −0.466 > forward
  −0.347) and reverse estimates SEPARATE the methods, so "method-invariant trajectory damage"
  is a FORWARD-estimand claim; method split still exploratory. Fixed in decomposition.py,
  report_m4.py, tests (all green); paper tex/pdf + PAPER_FINDINGS + Amendment H precision
  note updated; paper now uses −0.466 everywhere (abstract rounds to −0.47). Any bracket
  numbers below this line ([−0.34,−0.21]/[−0.35,−0.18], "brackets overlap") are STALE.

  **★★★ FIGURE 4 / TV PROFILES DONE (2026-07-02): replay-diss256 800/800.** Onset-aligned
  mechanism (800 profiles, median first-eviction step 65): mean per-step TV = 0.0000 BEFORE
  eviction, 0.0018 in the first 50 steps after, 0.0884 later -> divergence is ~zero at
  eviction and COMPOUNDS over the trajectory. Clean mechanism: trajectory damage is a slow-
  compounding drift, not an instant readout error -> explains why readout/teacher-forced eval
  misses it while free-run collapses. mean_auc_tv 0.086 (random) vs 0.083 (SLM) -> matched
  across methods, corroborates method-invariant trajectory damage. WRITABLE CORE NOW COMPLETE
  + mutually consistent: forward decomp (residual ~-0.34) + reverse bracket ([-0.35,-0.18])
  + Fig4 (compounding TV) + ADD + two-FC control. report_m4 with --replay-root gives the full
  table. Per-shard recovery + TV clamp confirmed working (no crashes, chained to grid-2048).
  Remaining = generality only: grid-diss2048-tok 526/800 (running) -> readout-2048 -> H2O ->
  R-KV -> AIME. Can START DRAFTING the core paper now.

  **⚠ FIXED 2026-06-29 (1-GPU-near-end issue):** last night ONE replay shard crashed
  (`ValueError: TV profile values must be in [0,1]` -- a TV came out 1.0000001 from float
  noise) and the watcher's old "any session running -> no-op" guard NEVER relaunched the
  dead shard (the other shard was live) -> that GPU sat idle all night (per-seed: 23=158,
  89=0 on the dead gpu0; 11=53, 47=0 on the live gpu1). TWO fixes: (1) clamp TV to [0,1] at
  the source in `replay.py` (total_variation/topk_tv_bounds/exact_tv_from_logits). (2)
  `nightly_autorun.py` now does PER-SHARD recovery: each tick launches only seeds that are
  incomplete AND not owned by a live session, onto GPUs eligible AND not running our job
  (tracks per-gpu shard assignment + per-seed counts in `logs/nightly_state.json`) -> a
  crashed/OOM-killed shard is relaunched on its freed GPU WHILE the other shard runs. Whole-
  job crash-loop guard retained (MAX_FAILS no-progress -> stuck+skip). 120 tests. Residual
  (rare): a single deterministically-crashing condition can loop its shard (other shard
  masks it in the whole-job guard) -> visible in the morning email's per-seed progress.

  **★ DAILY STATUS EMAIL (2026-06-28):** `scripts/nightly_report.py` emails 27100172@lums.edu.pk
  at 09:10 (cron, after the session ends) -- whole queue + per-job progress, what launched
  last night, total % to queue completion + paper-core status, crash/stuck alerts, GPU free.
  Transport = AUTHENTICATED Gmail SMTP (smtp.gmail.com:587) via creds in
  `~/.kvcache_smtp.json` on the remote (chmod 600, NOT in repo: {host,port,user,password,to};
  user=<personal-gmail>, to=same). Direct-to-MX was abandoned: O365 quarantines it
  (LUMS never received) and Gmail hard-rejects (sender IP has no PTR). The app-password is
  revocable in the Google account. `--dry-run` prints without sending; report logic reuses
  nightly_autorun (load_queue/count_progress). Crontab: watcher `*/10 2-8` + report `10 9`.

  **⚠ FIXED 2026-06-28 (caught from 06-28 run):** `replay-diss256` crashed
  ("between-prune cache exceeded expected bound") on the 2% of MATH-500 problems whose
  rendered prompt > budget 256 (max prompt 705 tok) -- the guard's bound was budget+interval
  but the prefill legitimately holds prompt+interval. FIXED: bound = max(prompt_len,budget)+
  interval (`run_replay.py`). The watcher then CRASH-LOOPED it 70x (no give-up), blocking the
  queue all night. FIXED: per-job no-progress failure counter in `nightly_autorun.py`
  (`logs/nightly_state.json`); after MAX_FAILS=3 launches with no progress a job is marked
  stuck and skipped so the queue advances. Both synced -> replay resumes from 16 tonight with
  the relaxed guard; any future crashing job self-skips after 3 ticks.

  **CANONICAL FC REFERENCE = `math500-fc-powered-merged` (2000 rows = 250 problems x n=8,
  accuracy 0.882). EVERY result references this one bank.** Older pins (pilot-fullcache-v2
  19/20, tokenized-v2 20/20) are RETIRED -- ignore stale mentions below. Provenance caveat
  (orphaned commits) + statement: PREREGISTRATION Amendment G.

  **PAPER SPINE (committed, Amendment G):** "compression damage to reasoning is real, large,
  and largely INVISIBLE to teacher-forced/readout-style eval -- which is why method papers
  (readout-style) report near-lossless while free-running (HOTT) reports collapse; the gap is
  the trajectory residual." >100% is DROPPED from framing (no gains observed); method-
  invariance + any >100% are upside. Generality envelope (4 methods x 2 budgets) is the
  venue-decider, queued. RSP is CUT. Two-FC control PASSED (ADD floor 0.068 << 0.56-0.75).

  **★★★ REVERSE-REPLAY BRACKET DONE — central claim SURVIVES off-policy test (2026-06-27).**
  reverse-readout-diss256 800/800. Reverse readout (compressed traj read out under FULL
  cache): Random 0.636, SLM 0.417 (FC 0.882) -> clean readout does NOT recover to FC, so a
  large chunk of loss is genuinely TRAJECTORY (reverse damage Random -0.246, SLM -0.465).
  Residual bracket: Random [-0.343,-0.214], SLM [-0.347,-0.180] -> never near 0. **The #1
  reviewer objection (off-policy bias gutting the residual) is answered: the headline holds.**
  Nuance: forward vs reverse DISAGREE on the method-level split (forward Random-more-traj,
  reverse SLM-more-traj) + brackets overlap -> method dissociation NOT identifiable; report
  central residual strongly, method split as bracketed. (PREREGISTRATION "Reverse-replay
  bracket result".) Core ~75% done: forward decomp ✅, grid ✅, bracket ✅; replay/Fig4 next.

  **★★ NIGHTLY AUTOMATION = WATCHER (2026-06-26, upgraded from once-a-night v1).** Remote
  cron (Asia/Dubai UTC+4) runs `scripts/nightly_autorun.py` **every 10 min, 02:00-08:50**
  (`*/10 2-8 * * *`). Each tick: pick next incomplete job from `configs/nightly_queue.yaml`
  whose `requires` dep is complete; **if any of OUR tmux sessions (`<run_name>-g<idx>`) are
  running -> no-op**; else check nvidia-smi free mem, SKIP any GPU below `min_free_gb` (OOM),
  shard seeds across free GPU(s) (freest first; 1→all, 2→split), `check_remote.sh` preflight,
  launch `launch.sh` in tmux with `KVCACHE_MAX_HOURS`=hours-until-09:00. Because it fires
  every 10 min and only launches when nothing of ours runs, it **FILLS THE WHOLE 02-09
  window** -- a job finishing mid-window triggers the next within ≤10 min (fixed the v1 bug
  that idled GPUs ~5h after a job completed early). Residual: ≤~20min within-job imbalance
  when one shard finishes before the other (unavoidable w/o finer sharding). Marker logic
  REMOVED (the sessions-check is the double-launch guard + auto-recovers a crashed shard).
  KILL-SWITCH: `enabled: false` in queue (or `crontab -r`). Queue (5 jobs, sequential):
  grid-diss256-tok ✅DONE → reverse-readout-diss256 (bracket) → replay-diss256 (Fig4) →
  grid-diss2048-tok → readout-diss2048. 18 tests. NOTE: after ANY code change run
  `sync_source.sh` (source-hash preflight else keeps last-synced code / aborts on drift).

  **★★ DECOMPOSITION READ — FINAL forward at 800/800 (2026-06-23).** FC=0.882.
  Random-256: free=0.422, readout=0.765 [.686,.817], readoutΔ=-0.117, residual=-0.343,
  share=0.25. StreamingLLM-256: free=0.237, readout=0.584 [.495,.647], readoutΔ=-0.298,
  residual=-0.347, share=0.46. (765/800 preliminary gave same verdict; <=0.012 drift.)
  **Verdict: COMPLICATE / qualified confirm** (see
  PREREGISTRATION Amendment F Outcome). CONFIRMED: large trajectory residual (~-0.355)
  significant for BOTH methods, readout-invisible (resolves the contradiction); methods
  dissociate on the READOUT channel (non-overlapping CIs); the ENTIRE +0.185 gap is
  readout (trajectory damage is METHOD-INVARIANT). DID NOT confirm: Random not readout-
  lossless (CI low 0.696<0.75), SLM share 0.45<0.50 -> the pilot "SLM=readout,
  Random=trajectory" framing is FALSE; reframe to "both equally trajectory-damaged,
  differ in readout fidelity." The core idea works; the headline reframes.

  **★★ REVISED PLAN (2026-06-22), priority order:** (1) finish 35 readout stragglers
  (seeds 11,23) -> forward readout 800/800 [~1h]. (2) **256 re-run WITH token IDs**
  (`grid_math500_dissociation.yaml` already has store_token_ids:true; new run name
  `grid-diss256-tok`) -> reproduces phenomenon + the token-ID compressed traces reverse
  replay needs [~2-3 nights, resumable]. (3) **Reverse readout** (`reverse_readout_
  diss256.yaml`, ref=grid-diss256-tok, press=fullcache) -> residual bracket; report via
  `report_m4 --reverse-readout-root` (wiring already built). This DEFENDS the method-
  invariant-trajectory claim against off-policy bias -- now load-bearing, not optional.
  (4) budget-2048 grid (token IDs) -> silent-flip regime. (5) replay TV profiles (Fig 4).
  (6) R-KV smoke re-run (device bug fixed) -> R-KV cells; AIME (>100%). report_m4 is now
  replay-OPTIONAL + multi-readout-root, so the decomposition reads without TV profiles.

- This is an **analysis/evaluation paper** about *why* KV-cache compression papers
  report near-lossless reasoning while the independent HOTT benchmark reports
  collapse. We are building the measurement harness, not a new compression method.
- **Roles:** the human (final-year CS undergrad, LUMS, strong ML) sets direction.
  An assistant named **Claude** (this reviewer) is the **technical decision-maker /
  reviewer**. **Codex** is the **executor** (writes agreed code, runs local
  validation, runs remote/GPU jobs). All heavy/remote work is gated.
- **Progress:** M0 ✓, M1/M2 (full-cache pilot harness) ✓ accepted, **M2.5 stack
  migration ✓ promoted** (transformers 4.52.4→**4.56.2**, kvpress 0.2.10→**0.5.3**,
  to get decoding-phase compression), **M3 compressed-generation code ✓ reviewed &
  approved**.
- **⚠ SUPERSEDED (double-BOS):** the paragraph below describes the FIRST
  compression pilot `pilot-compression-v1-r2` (51/80), which was generated with the
  **double-BOS bug** and is RETIRED to provenance. The CANONICAL single-BOS
  compression pilot is `pilot-compression-v2`, and the canonical decomposition is the
  **M4 result block further down** (free-run random-256 0.45 / SLM-256 0.20, etc.).
  Use the M4/M5 numbers, not v1-r2, for anything external.
- **Current state (M3 era, retired):** the **80-generation compression pilot completed** as
  `pilot-compression-v1-r2` and has been pulled locally. Audit: 80 schema-v2 rows,
  16 files, consistent stack/source/config hashes, 51/80 correct, 8/80 truncated,
  no logged errors, and no serialized tensor/cache artifacts. Cell results are in
  §6/§7. **M3 pilot REVIEWED & ACCEPTED by Claude (2026-06-14).** M4 design
  (`docs/design/M4_REPLAY_DESIGN.md`) is **REVIEWED & APPROVED by Claude (2026-06-14)** with
  rulings (below). **M4.0 CPU implementation REVIEWED & APPROVED by Claude
  (2026-06-14)** — verified alignment, enabled-flag compression scoping (step-counter
  gated via forward_hook early-return), onset off-by-one (correct), self-replay
  identity test, exact token-id threading + round-trip validation, readout estimator.
  86 tests pass. **GPU now AUTHORIZED for M4.1→M4.4 in order**, with a mandatory
  STOP-and-report at M4.2 (self-replay negative control) before any compressed replay.
  **M4.1 config must set `store_token_ids: true`; the new tokenized bank replaces
  v2-tf456 as the M4 reference** (v2-tf456 lacks sidecars). **Self-replay caveat:** the
  hard 1e-6 TV gate may be too tight — two CUDA forwards over identical tokens have a
  tiny nondeterminism floor (~1e-6–1e-5); make the tolerance configurable, measure the
  floor, confirm it is << τ (0.05). Minor: `compression_events` counts layer-level
  prune calls (×num_layers), not token-prune steps.

  **M4 rulings:** (1) token-ID sidecar approved (regen 20-row FC bank to capture IDs).
  (2) exact transient full-vocab TV primary, top-k bounds fallback-only. (3) one model
  / two caches, but scope compression via an explicit `press.enabled` flag toggled
  around each forward (NOT object-identity) + hard test that the cache is
  mutated-not-replaced. (4) **readout estimand defined** (accuracy-scale):
  teacher-force the reference reasoning up to the `</think>`/answer-marker boundary into
  the compressed cache, then free-generate the answer span under compression (sampled,
  temp 0.6, matched seed), score with is_equiv → `readout_accuracy`;
  `readout_change = readout_accuracy − fc_accuracy`; `residual = total_change −
  readout_change`. Per-step TV stays a SEPARATE divergence/TDP instrument — never
  subtracted from accuracy (Codex correctly blocked `accuracy − mean_TV` as
  dimensionally invalid). Disclose the on-policy bias (uses good reasoning → inflates
  residual; bracketed by reverse replay in M7). (5) replay self-control in M4;
  two-FC-bank ADD permutation control in M5. Plus: handle no-prune (high budget) →
  `onset_step=null`, TV≈0, not a bug.

  **M3 drill-down findings (Claude, per-problem, CPU):** regime separation confirmed
  (budget 2048 ≈ FC at 90–95%; budget 256 collapses to random 50% / SLM 20%). The
  **Random > StreamingLLM** gap at 256 is real and broad (Random ≥ SLM on 4/5
  problems), signed flip-mass random net −2.25 vs SLM −3.75, **no gains anywhere**
  (no >100% regime on small-budget MATH — expected). **Correction to the first
  report:** StreamingLLM-256's dominant failure is *answered-wrong* (13/20), not
  gibberish-truncation (only 3/20 truncated) — it confidently boxes wrong answers
  (trajectory derailment), the gibberish loop is secondary. Length explodes at 256
  (mean ~6.8–7.6k tokens vs FC ~1.3–6.1k) — the HOTT "low budget lengthens traces"
  effect, in-harness. Working hypothesis (to be TESTED by M4, not yet claimed): SLM
  deletes all mid-trace anchors (sinks+window) → derails; random preserves some by
  chance. Underpowered (n=4/problem); the 250-problem bank is the powered unit.

  **M4 design gotcha flagged:** replay needs the exact reference token-id sequence,
  but FC rows store only decoded text. Resolve before GPU: either store
  `trace_token_ids` (re-run the 20-row pilot FC bank; additive) or re-tokenize with a
  round-trip assertion. Lean: store token ids.

  **M4.1 correction discovered (2026-06-14):** the first tokenized-bank attempt,
  `pilot-fullcache-tokenized-v1`, stopped on its first sidecar validation because
  the rendered Qwen prompt already contained BOS while generation tokenized it with
  the default `add_special_tokens=True`, producing two BOS tokens. This affected all
  earlier M2/M3 pilot generations. Generation and prompt-length accounting now use
  `add_special_tokens=False`, with regression coverage. Earlier pilots remain
  provenance/harness diagnostics only. A fresh
  `pilot-fullcache-tokenized-v2` is the first corrected reference. It was launched
  in remote tmux on 2026-06-14 after a tokenizer-only remote assertion confirmed
  exactly one BOS and an exact prompt round trip; its first result row and token-ID
  sidecar both passed and were written successfully. **M4.1 completed and audited:
  20/20 correct, 0 truncations, mean 2,407.2 generated tokens, 20/20 sidecars
  independently validated, one BOS in every prompt, consistent source/config/version
  metadata, and no tensor artifacts.** The bank auditor was corrected to ignore the
  additive `token_ids/` namespace, with regression coverage. The authorized one-row
  M4.2 self-replay smoke (`pilot-replay-self-smoke-v1`) passed exactly: 1,004
  replay steps, min/max/mean/summed TV all 0.0, zero threshold crossings, no onset,
  and no compression events. The full 20-row M4.2 control
  (`pilot-replay-self-v1`) also **passed exactly**: 20/20 replay rows and profiles,
  48,144 total steps, global min/max/mean/summed TV all 0.0, zero nonzero steps,
  zero crossings at every preregistered threshold, no onset/TDP crossing in 20/20
  rows (operational control percentile 1.0), zero compression events, no errors,
  and no tensor artifacts. Scalar artifacts were pulled locally and the CPU-only
  report at `results/diag/pilot-replay-self-v1/` contains `summary.json`,
  `table.md`, and `self_replay_control.svg`. **Mandatory stop is active:** send
  this M4.2 report to Claude before any compressed replay. The 80-row M3
  compressed pilot must also be regenerated under this protocol before it can
  supply `total_change` for M4.4; mixing old double-BOS compressed rows with the
  corrected reference would violate exchangeability.

  **★ M4 COMPLETE + ACCEPTED by Claude (2026-06-14). Committed `67e1194`.**
  Codex hit its usage limit during the M4 report; Claude took over execution from
  here on (plans AND runs code/remote jobs). Single-BOS pilot, full-cache = 20/20.
  **THE FIRST REAL RESULT — a double dissociation at budget 256** (n=5 problems × 4
  seeds, underpowered, directional):

  | cell | free acc | readout acc | total Δ | readout Δ | residual | lost mass |
  |---|---|---|---|---|---|---|
  | random-2048 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
  | streamingllm-2048 | 0.90 | 1.00 | −0.10 | 0.00 | −0.10 | 0.10 |
  | random-256 | 0.45 | 0.90 | −0.55 | −0.10 | **−0.45** | 0.55 |
  | streamingllm-256 | 0.20 | 0.45 | −0.80 | **−0.55** | −0.25 | 0.80 |

  **Interpretation (framing-disciplined):** at 256, the two methods collapse for
  OPPOSITE reasons. StreamingLLM-256 is **readout-dominated** (−0.55 readout): its
  sinks+recent-window eviction deletes the MIDDLE of the reasoning where the
  answer-relevant work lives, so even reading off the *good* teacher-forced trace
  fails. Random-256 is **residual/trajectory-dominated** (−0.45 residual): uniform
  eviction preserves readout (0.90) but injects drift that compounds during free
  generation. This EXPLAINS the Random>StreamingLLM surprise mechanistically and is
  the decomposition's proof-of-value — accuracy alone (both "collapse at 256") cannot
  tell them apart. **Bias note:** residual is inflated by the off-policy bias
  (readout measured on good states), so SLM's readout-dominance is CONSERVATIVE
  (robust), while random's trajectory-dominance is bias-vulnerable and must be
  bounded by reverse replay (M7). No gains anywhere (no >100% on small-budget MATH).
  Length explodes at 256 (free-run mean ~8.2–8.4k tok vs FC ~2.4k); readout
  estimand short-circuits this (generates only the answer span) so it is not
  truncation-contaminated. High budget (2048) genuinely near-lossless (TV~0.002–0.004).
  Report: `results/diag/m4-pilot-v1/{summary.json,table.md,decomposition.svg}` via
  `scripts/report_m4.py`. Engine + report validated (89 tests pass).

  **★ M5 ADD pipeline BUILT + validated on pilot (2026-06-14). Committed `61760a5`.**
  `scripts/report_add.py` + test: per-problem observed ADD + permutation p, termination
  vs within-answered split, Miller-Madow entropy shift, BH across problems, mean signed
  flip-mass (net == total_change). Pilot result corroborates M4:
  streamingllm-256 ADD 0.80 (within-ans TV 0.80, 4/5 BH-significant even at n=4,
  entropy +1.22); random-256 ADD 0.55 (within-ans 0.53, 0/5 significant, entropy
  +0.98 = diffuse derailment); 2048 near-zero (random 0.00, slm 0.15). Report at
  `results/diag/m5-add-pilot-v1/`. **M5 negative control (zero-GPU smoke) PASSED**:
  FC-vs-FC (split seeds {11,47} vs {23,89}) → mean ADD 0.000, 0/5 significant — the
  ADD pipeline does not false-positive. (Weak control: pilot FC has no cross-seed
  answer variance, so the null is degenerate-but-correct; the POWERED two-FC-bank
  control comes FREE at M8 by splitting the n=8 bank — run it there, no separate GPU.)
  **M5 functionally complete for the pilot.**

  **★ EXTERNAL REVIEW + RESPONSES (2026-06-15). Committed `c502289`, `9fef4ef`;
  PREREGISTRATION Amendment D.** A sharp review praised the framing discipline, grader
  fix, double-BOS catch, and bitwise self-replay control, and flagged real risks. Actions:
  - **M4 dissociation is UNCONFIRMED, not a result.** Added problem-level bootstrap 95%
    CIs to `report_m4` — they are wide (SLM-256 readout 0.45 **CI[0.15,0.80]**;
    random-256 readout 0.90 **CI[0.80,1.00]**; the per-method readout/residual split is
    barely resolved at n=5). Pre-committed (Amendment D) to report the powered MATH-500
    result **even if it reverses** the dissociation. Do NOT let pilot directionals
    calcify; never assert the dissociation externally until 250×n=8 confirms it.
  - **Random>SLM mundane causes handled:** budget parity VERIFIED (both presses prune to
    256-after / 320-before identically at target_size=256 → not an accounting artifact);
    `press_seed(pid,seed)` adds per-(problem,seed) RNG so it can't be a lucky seed
    (wired into run_grid + run_replay; tested).
  - **One canonical FC reference pinned in code:** `report_m4` asserts all
    compression/readout rows reference `pilot-fullcache-tokenized-v2`'s (pid,seed) set.
    `pilot-fullcache-v2-tf456` (19/20, double-BOS) is RETIRED — do not use it.
  - **Readout estimand is math/span-shaped;** per-step TV is the general fallback for
    logic benchmarks (FOLIO/StrategyQA) where the answer isn't a span. Paper states which
    instrument carries the decomposition on which benchmark.
  - **R-KV port = HARD GATE (M6):** reproduce a specific published (model,benchmark,budget)
    number within noise, committed as a test, BEFORE any R-KV row enters analysis. Same
    for H2O. "If you can't reproduce their headline, you don't have R-KV."
  - **RSP demoted to bonus;** restoration-on-natural-traces is the PRIMARY mechanism
    instrument. **compression_events is layer-level** (never divide as a step rate).
    **Powered two-FC negative control MUST run at M8** (split the n=8 bank), not skipped.

  **★ M6 IN PROGRESS — H2O DONE + GPU-VERIFIED (2026-06-15). Committed in the M6 H2O
  commit.** kvpress 0.5.3 `ObservedAttentionPress` is the in-library H2O (docstring
  cites arXiv:2306.14048), a ScorerPress, requires eager attention. Implemented:
  `load_model_and_tokenizer(attn_implementation=...)`, registry `h2o`→EXECUTABLE +
  `press_needs_eager_attention`, run_grid/probe auto-load eager. Probe `h2o-probe-v1`
  passed: cache bounded 256/320 (`all_after_at_most_target`), coherent answer
  ((1,π/2) vs gold (3,π/2) — angle right, radius wrong under budget 256), no tensor
  artifacts, 810 tokens. **Three methods now executable: streamingllm, random, h2o.**
  H2O rides on kvpress (same library HOTT used), so its faithfulness is library-level;
  behavioral sanity on the pilot suffices (vs R-KV's hard published-number gate).
  **NEXT: R-KV port** (not in kvpress; our own port; the >100% method). HARD GATE per
  Amendment D: reproduce a specific published (model,benchmark,budget) number within
  noise, committed as a test, before ANY R-KV row enters analysis.

  **★ 2026-06-15 (second review actioned + scaling started):**
  - Second external review answered in full; commits include per-problem RNG +
    bootstrap CIs + pinned FC reference (`c502289`), Amendment D (`9fef4ef`), H2O
    (M6 commit), eager gate, readout-boundary rule, Amendment E. 93 tests.
  - **Eager-attention exchangeability gate PASSED:** full-cache eager 19/20 vs sdpa
    20/20, 19/20 per-problem agreement, one Bornoulli flip (927 s47), no systematic
    shift. Decision: **MIX** — sdpa for FC/StreamingLLM/Random (fast), eager for H2O;
    one pinned sdpa FC reference; verification committed as justification.
  - **POWERED MATH-500 FC BANK LAUNCHED (2026-06-15)** — the foundation. 250 problems
    × n=8 (16384 cap, sdpa, token-ID sidecars), split across both GPUs:
    `math500-fc-powered-g0` (seeds 11,23,47,89, GPU0) + `math500-fc-powered-g1`
    (seeds 131,197,263,347, GPU1), ~15-17h each. **Merge the two session dirs to the
    canonical n=8 reference when done** (different seed files, no collision).
  - **R-KV port (M6) — IMPLEMENTED + PARITY-GATED (commit `78659a6`).**
    `src/kvcache/presses/rkv.py::build_rkv_scorer` = kvpress `SnapKVPress` subclass:
    importance (observation-window attention, max-pool, window=8, kernel=7) minus
    `mix_lambda`(0.07)·cosine-redundancy; recent window force-kept; no eager needed
    (recomputes Q@K). `build_rkv_press` wraps it in DecodingPress (for the parity
    harness only). **`get_press("rkv")` REFUSES (PARITY-GATED)** so R-KV cannot enter
    any analysis until verified. Plan + exact parity procedure: `docs/design/RKV_PORT.md`.
    **PENDING (needs a free GPU — both busy with FC bank): the parity gate** — set up
    the authors' reference env, run their `r1_kv.py` (record_kept_token_indices) on
    Qwen-7B at two budgets, run our port on the same traces, require retained-set
    Jaccard≈1, commit artifacts; the one open detail to confirm is `_redundancy_score`
    vs R-KV's `cal_similarity` (threshold/retain_direction). On pass → move rkv to
    EXECUTABLE. Repo cloned at remote `~/scratch/R-KV` (HuggingFace/rkv is the ref).
  - **FC bank progress (2026-06-16): healthy ~37 tok/s** (median 61s/gen), g0 ~183/1000,
    g1 ~167/1000; ~a day to finish both halves. **Merge g0+g1 → canonical n=8 bank.**

  **★ POWERED FC REFERENCE BUILT (2026-06-17): `math500-fc-powered-merged`.** g0+g1
  merged → 2000 rows (250 problems × n=8), **full-cache acc 88.25%**, truncation 1.9%
  at 16K cap, versions consistent. 4 source_tree_hashes (from mid-run syncs) but the
  generation path is byte-identical across `ff4240b..HEAD` (verified by git diff) → rows
  exchangeable; `audit_bank --allow-mixed-provenance` accepts it. `merge_banks.py` did
  the union. This is THE canonical powered reference for all decomposition.
  **R-KV scoring parity-verified (Jaccard 1.0 vs authors' code at budgets
  {128,256,384,512}); press real-generation smoke still pending (GPU).**
  **NEXT: powered compression cells (the dissociation test at scale) — a multi-day GPU
  run because budget-256 free-runs ramble (~8k tokens). run_grid lacks --resume yet.**

  **★ TIMER VERIFIED + ADD WORKS (2026-06-19):** Live server trial of the launch.sh
  run-block (20s timer, sleep stub) → "STOPPED by timer (rc=124) after 20s"; GNU timeout
  8.32 primitives confirmed (2.5h parses, SIGTERM→124, SIGKILL-fallback→137). Metric 2
  (ADD) computed on the existing grid vs FC bank: StreamingLLM-256 mean_ADD=0.751
  (within-answered TV 0.749, termination +0.081, flip net −0.646); Random-256 mean_ADD
  =0.559 (within 0.504, termination +0.224, flip net −0.461). ADD instrument validated;
  bonus — the methods damage the answer distribution DIFFERENTLY (SLM ~all within-answered
  = readout-like; Random more termination = trajectory/rambling-like), previewing the M4
  decomposition. Silent-flip-at-flat-accuracy showcase still needs the 2048 cells. NOTE:
  report_add --out wrote a dir at /tmp/add_diss.json (minor, stdout json is correct).

  **★ DOWNTIME CODEBASE REVIEW (2026-06-20):** deep pass, lint+99 tests green. Core
  science modules verified correct (ADD, TDP, decomposition, readout/replay, Hendrycks
  is_equiv, exact-permutation + BH). FIXED: (1) `report_m4._write_svg` had a non-f-string
  gridline (`y2="{y:.1f}"` rendered literally → malformed M4-figure gridlines). BUILT (so
  we don't lose a night later): (2) **`run_grid` now stores token-ID sidecars**
  (`store_token_ids`, mirrors build_bank + resume-requires-sidecar + 3 tests); set on
  `grid_math500_dissociation.yaml`. Without this, compressed traces weren't replay-ready
  (reverse readout / TV profiles would need lossy re-tokenize). (3) **Reverse-readout
  (M7 first rung) wired into `report_m4`** (`--reverse-readout-root` → per-cell
  `residual_bracket`); the reverse RUN needs NO new engine — it's the existing readout
  mode with reference=compression-grid (token-ID-bearing) + press=fullcache. OPEN (not
  bugs): `run_mechanism`/`run_metrics` are scaffolds (`cli.run_cli`→not-implemented; real
  metrics run via report_*.py); restoration + TV-matched-noise rungs still unbuilt
  (post-confirm); `grading.py` flags a comparability caveat (confirm is_equiv matches
  HOTT/kvpress grader before cross-paper number claims); AIME/other benchmarks raise
  not-implemented (TODO dataset handles). REVERSE-READOUT RECIPE post-confirm: re-run the
  dissociation cells with store_token_ids (or use future token-ID cells), then
  `run_replay` readout mode with reference=that-grid + presses=[fullcache], then
  `report_m4 --reverse-readout-root`.

  **★ DISSOCIATION GRID DONE (2026-06-19): `grid-math500-dissociation-v1` 800/800.**
  Powered free-run result (100 problems × n=4): FC 0.885 → StreamingLLM-256 **0.237**
  [0.172,0.302], Random-256 **0.422** [0.343,0.500]. **Random − StreamingLLM = +0.185,
  CI [0.130,0.242], 94/100 problems** → the surprising ordering (random eviction beats
  "principled" StreamingLLM at low budget) SURVIVES powering decisively. VERDICT: GO.
  Honest caveat: the grid confirms only the PHENOMENON; the paper's actual IDEA (the
  readout/trajectory decomposition) is UNTESTED until the readout run lands. Readout is
  therefore the single decisive next run.

  **★ TIMED/RESUMABLE REPLAY HARNESS BUILT (2026-06-19, synced):** `run_replay.py` gains
  `--resume` (heals partial trailing line + skips done (reference_key,press,budget)
  conditions, both readout & compression_replay modes) and `--seeds A,B` (per-GPU shard
  by seed → disjoint per-seed output files, two GPUs never collide). `launch.sh` gains
  `KVCACHE_MAX_HOURS=N` → wraps python in `timeout --kill-after=120s --signal=TERM`,
  prints "STOPPED by timer (resumable)"; rerun the identical command (same
  KVCACHE_RUN_NAME + --resume) to continue exactly. `write_replay_profile(overwrite=)`
  for orphan-profile safety on resume. Tested (heal+skip unit test, 96/96 suite). Configs:
  `readout_math500_diss256.yaml` (256 readout, 800 conds, DECISIVE) + `replay_math500_
  diss256.yaml` (256 TV profiles, 800 conds). User runs nightly 1am-8am on BOTH GPUs
  (seeds 11,23→GPU0 / 47,89→GPU1), run names readout-diss256-g{0,1}/replay-diss256-g{0,1};
  then I compute total−readout=residual (the decomposition that tests the idea).

  **★ ROADMAP REORDERED per third review (2026-06-19) — READOUT IS THE ONLY THING ON
  THE CRITICAL PATH.** The review's central correction: the old roadmap buried the
  decisive cheap run (readout, was item 6) behind ~3 GPU-days of method cells (H2O, R-KV,
  2048) that ASSUME the decomposition resolved. Reordered by information value:
  (1) **READOUT @256** (`readout_math500_diss256.yaml`, 800 conds, ~2h on 2 GPUs) — the
  experiment that confirms or kills the paper's thesis; NOTHING else runs until it is read
  per Amendment F. (2) interpret + the read is pre-committed (Amendment F: confirm/
  complicate/refute). (3) IF CONFIRM → **reverse replay** (M7 first rung) on the SAME
  dissociation cells to bound off-policy bias BEFORE drafting the mechanism section, THEN
  R-KV cells (port certified, R-KV is the >100% method), THEN AIME (needs dataset handle).
  IF NOT → stop, re-scope (phenomenon + methodology may be a workshop/different framing).
  Deferred behind the readout verdict: replay TV profiles, budget-2048, H2O cells, R-KV
  cells, AIME, causal ladder. **Results-independent / do in parallel now:** draft Methods/
  Background/Related-Work/framing-discipline. **R-KV smoke BUILT (2026-06-19):**
  `scripts/rkv_smoke.py` drives real generation via `build_rkv_press` and checks (1) sane
  terminating output, (2) every compression ≤ budget, (3) finite scores, (4) recent window
  always retained + exactly budget kept, (5) live selection self-consistent with the
  parity-verified `rkv_retained_indices` (mean Jaccard ≥0.99). ~5 min; designed to chain
  into the readout tail (GPU0/small-set session, with a 6.5h wall-clock guard so it never
  overruns the window). On PASS, record it + promote R-KV to EXECUTABLE per Amendment E.

  **★ PROVENANCE FINDING (2026-06-19):** built `scripts/verify_generation_path_stable.py`
  (committed re-runnable artifact replacing the one-time eyeballed diff). It found the
  merged FC bank's 4 runmeta commits (193e12fc, 78659a6, b74f7f3, ff4240b) are NOT in the
  current git history (remote HEAD 3e19f8b) — they were orphaned when the repo was
  re-initialized during the ~/Desktop→~/dev move, so the byte-identical-generation claim
  is no longer git-verifiable (it was checked at creation). FC-bank provenance now rests
  on: identical package versions (audit_bank hard-check) + per-row source_tree_hash + FC
  has no press logic. FORWARD FIX: commit a clean state before tonight's readout so its
  runmeta.commit is reachable (currently main is dirty with uncommitted work). H2O
  CONFOUND (review): H2O is the only method on the eager kernel (others sdpa); passed
  exchangeability 19/20 vs 20/20, but keep the eager-vs-sdpa FC equivalence check handy —
  if H2O ever shows a surprising result the kernel is a confound to rule out explicitly.

  **User decisions (2026-06-14):** (1) Claude launches/manages remote GPU jobs itself
  within the guards, reports after. (2) Scope = ports-first then MATH+AIME core
  (pre-registered path). Codex is retired; Claude is reviewer+executor.

  **Remote ops reference (verified 2026-06-14):** GPU idle, no tmux jobs. Launch via
  `ssh kvcache-gpu` then export the env block (KVCACHE_REMOTE=1,
  KVCACHE_PERSONAL_FREE_GB=70, REMOTE_{DATA,CACHE,RUNS}_ROOT=/home/sk10741/
  kvcache-storage/{data,cache,runs}, ENVIRONMENT_NAME=.venv, UV/PIP/HF/TORCH cache
  dirs under .../cache/*, CUDA_VISIBLE_DEVICES=0 or 1, GPU_MODEL, GPU_COUNT=1) and run
  `bash scripts/launch_tmux.sh --i-understand <session> <bank|grid|replay|...> <config>`.
  check_remote.sh validates `.sync-state` source hash → must `make sync-apply` after any
  local source/config change before launching. New configs must be synced to remote.

  **★ NEXT: M6 — port H2O + R-KV (the "science at scale" gate).** kvpress 0.5.3
  inventory checked: **H2O ≈ `ObservedAttentionPress`** (in-library attention
  heavy-hitter; HOTT used kvpress's attention press as H2O). Needs eager attention
  (`attn_implementation="eager"`) to expose attention weights — memory-OK because
  decode attention is [1×cache_len] (only prefill is O(n²), and MATH prompts are
  short); just slower. Verify it's a `ScorerPress` (DecodingPress-compatible), label
  "H2O (ObservedAttention)", sanity-check. **R-KV** is NOT in kvpress → real port from
  the official repo (redundancy-aware, reasoning-specific), must reproduce a source
  sanity number before entering the grid. Registry: move `h2o`→EXECUTABLE via
  ObservedAttention; keep `rkv` PORT_REQUIRED until ported. Then **M8 scale**:
  MATH-500 250×n=8 + AIME (R-KV's >100% home turf), all methods × budgets; powered
  negative control falls out of the n=8 bank. Other kvpress baselines available if
  wanted: SnapKV, TOVA, Knorm, PyramidKV, AdaKV, DuoAttention, ExpectedAttention.

  **M4.2 REVIEWED & ACCEPTED by Claude (2026-06-14):** self-replay control is
  bitwise-perfect (global max/sum TV = 0.0 over 48,144 steps, 0 nonzero steps, 0
  crossings, no onset) → the replay engine is validated. Double-BOS bug accepted as
  correctly caught (by the token-ID round-trip validation — vindicates the sidecar
  design) and fixed (`tokenize_rendered_prompt`, `add_special_tokens=False`, regression
  test). Corrected single-BOS bank `pilot-fullcache-tokenized-v2` (20/20) is the
  canonical M4 reference; all double-BOS pilots (incl. v2-tf456 19/20 and
  pilot-compression-v1-r2) retire to provenance. Note: the v2-tf456 927-s11 "sign
  error" flip was partly a double-BOS artifact (single-BOS → 20/20); does not change
  the M2.5 acceptance. **Claude RULING: rerun the M3 compressed pilot single-BOS
  (`pilot-compression-v2`) — required for exchangeability.** GPU AUTHORIZED now for:
  (1) M3 single-BOS rerun [also re-measures regime separation / Random>SLM],
  (2) M4.3 forward replay (80 cond, on corrected ref), (3) M4.4 readout (80 cond),
  (4) the scalar decomposition combining them. Engine validated → no mid-gate; send one
  combined report. Still gated: M5+, H2O/R-KV, RSP, KIVI, full grid.
- **Then:** review grid results → M4 (teacher-forced replay + self-replay negative
  control) → M5 (ADD) → M6 (port H2O, then R-KV — the "science starts" gate) → M7
  (mechanism interventions) → M8 (full pre-registered grid). **Do not skip ahead.**
- **Uncommitted right now:** the verified RandomPress device fix + random probe config
  + probe tweaks (see §7). Two untracked planning docs (`publication_strategy.md`,
  `research_landscape.md`) are intentionally never committed. `docs/project/context.md` is also
  untracked and maintained as a handoff.

---

## 1. The project (science)

### 1.1 One-paragraph statement
The HOTT benchmark ("Hold Onto That Thought," arXiv:2512.12008, Dec 2025) showed
KV-cache compression methods marketed as near-lossless actually collapse on
long-generation reasoning, while method papers (R-KV 2505.24133, ThinKV 2510.01290)
report near-lossless or >100% accuracy. Nobody has explained *why*. This project
explains it. Core idea: compressing the cache **during generation** is an
intervention on the sampling process, so its effect decomposes into **readout
distortion** (how compression changes the next-token distribution holding the trace
fixed, measured by teacher-forced replay of a full-cache reference trace) and
**trajectory distortion** (the residual — how it changes what the model actually
writes, which compounds). Plus a suite of divergence instruments and a causal ladder.
Target: ICLR 2027 main (arXiv stake + workshop first); TMLR as the deadline-free
publication floor.

### 1.2 NON-NEGOTIABLE framing discipline (a reviewer kills the paper if we slip)
1. **Trajectory distortion is a DEFINITION BY CONSTRUCTION, never a measured causal
   effect.** Readout is measured on the reference trace under teacher forcing; total
   is measured under free-running compressed generation; these live on *different
   probability spaces*, so `trajectory := total − readout` is a definition, not an
   identity. Always describe the residual as *"accuracy change unexplained by
   on-reference readout distortion."* Never write "trajectory distortion causes the
   accuracy loss." Known bias: the reference trace visits *good* states, so
   on-reference readout likely *underestimates* per-step damage on the compressed
   model's own (worse) trajectory → inflates the trajectory share. Our headline is
   biased toward itself; we say so. Reverse replay + TV-matched noise control bound it.
2. **All instruments measure FIDELITY to the full-cache reference, NOT damage.** The
   field's most interesting phenomenon (>100% accuracy under compression) means
   deviation can be *improvement*. Every divergence quantity must be reported with
   its **signed** outcome consequence (gained vs lost flip-mass). Framing is
   "behavioral fidelity for deployment parity, with correctness as the quality
   arbiter." Never let divergence implicitly mean damage. (This is the
   PSNR-vs-perception blind spot.)

### 1.3 The metric suite (final post-review design)
- **ADD (Answer-Distribution Divergence):** TV distance between full-cache and
  compressed answer distributions over N seeded samples. Report **excess ADD via a
  per-problem PERMUTATION-TEST null** (pool N+N samples, permute labels, recompute TV
  → exact/MC null), NOT split-half. Coarsen answer support to {correct, top-m wrong,
  OTHER/invalid}. Entropy shift with Miller–Madow. pass@k via unbiased Codex
  (Chen 2021) estimator + paired bootstrap. **Sampled-inference only** — under greedy,
  ADD degenerates to a 0/1 flip indicator, so greedy runs report flip-rate, never
  ADD. BH-correct across the pre-registered problem family. **ADD is a LOWER BOUND on
  behavioral change** (only sees changes that reach the answer) — say so. Decompose
  ADD into **termination-mass shift** (truncation/OTHER mass moving — a length
  effect) vs **within-answered TV** (the reasoning channel), so truncation isn't
  read as reasoning change.
- **TDP + divergence profile:** per-step **total variation, NOT KL** (KL is unbounded
  and explodes on rare tokens; TV is bounded and = the probability optimally-coupled
  sampling differs, ties to speculative-decoding acceptance: `1 − acceptance ≈ TV`,
  Leviathan et al.). Threshold-free primaries — **both mean-TV (length-normalized)
  and summed-TV (total mass)** plus fraction-above-τ — with TDP (first crossing) as a
  localization aid only. τ is a **fixed interpretable level** swept over
  {0.05, 0.10, 0.15, 0.20} (NOT self-noise-calibrated; bitwise self-replay gives
  τ≈0). Onset-aligned (subtract first-compression-event step). Content-token variant
  over numerals/operators/answer-span (defuses "it's just paraphrase noise").
  Segment-aggregated by discourse boundaries.
- **Causal ladder:** evicted-anchor attribution + damage-latency; **restoration test**
  (re-insert evicted KV at a spike, recompute d_t — interventional); **branch
  recovery** with full-cache-fork AND random-fork controls; **reverse replay**
  (compressed free-run trace replayed under full cache — brackets the off-policy bias
  from the other side); **TV-matched noise control** (inject calibrated noise into a
  full-cache free-run matched to a method's TV profile — the on-policy causal test
  distinguishing structured damage from generic noise). Restoration + TV-matched
  noise are the strongest rungs; reverse replay + noise control are **mandatory** (an
  earlier deep review flagged the off-policy/covariate-shift problem as the framework's
  deepest risk).
- **Validity:** negative control (two independent full-cache banks through the whole
  suite → excess ADD ≈ 0, TDP% ≈ 1) — run as a **FIGURE**, not a sentence. If any
  instrument lights up, there's a bug. Exchangeability requires matched
  batching/kernel/transformers-version across conditions.

### 1.4 Known statistical landmines (all must be handled)
- Permutation power at N=8 is low per-problem; the powered unit is the **benchmark**
  via aggregation. Never claim per-problem significance on AIME.
- pass@k crossover hunting across ~150 cells is a multiple-comparisons trap.
  Pre-register the crossover test on R-KV-style methods at high budgets on math; use
  **paired bootstrap of the difference** (pass@1 and pass@k are correlated — naive CI
  overlap is invalid); BH the rest.
- TDP-as-minimum is an extreme-value stat with a length confound → threshold-free
  primaries are the fix; never compare raw TDP% across benchmarks of different lengths.

---

## 2. Roles, workflow, and gating

- **Human:** sets direction, makes strategic calls (e.g. stack pin changes), runs
  Codex, can run shell via `! <cmd>` in the Claude session.
- **Claude (reviewer / decision-maker):** reviews evidence, makes scientific/eng
  rulings, specifies exactly what Codex implements/runs next, authorizes gated work.
  Does NOT launch remote jobs unless explicitly asked. Has implemented CPU code
  directly when Codex was unavailable.
- **Codex (executor):** implements agreed changes, runs `make lint test dryrun`, syncs,
  runs remote/GPU jobs under the launcher, reports back.
- **Gating:** expensive/remote work requires preflight health check + `--i-understand`
  + `KVCACHE_REMOTE=1` + detached tmux. Experiment CLIs are double-guarded (refuse
  without `--i-understand` AND `KVCACHE_REMOTE=1`). The compression runner additionally
  refuses anything but the reviewed 80-row pilot.
- **Working style the human wants:** dense, rigorous, complete; pushback over
  sycophancy; verify version/library/dataset facts rather than asserting from memory
  (several past errors came from stale version assumptions).

### Sync = Policy B (rsync-first)
Edit local → `make lint test dryrun` → `make sync` (dry preview) → `make sync-apply`
→ commit after. Git is committed-history source of truth; rsync excludes
`.git data/ runs/ logs/ results/ *.safetensors *.npy` and the parallel venvs and
`.claude/`. The remote git worktree is dirty by design (rsync excludes `.git`).
`.sync-state` records a content hash of the transferred tree; remote preflight
recomputes it and refuses if files changed after sync.

---

## 3. Infrastructure (CURRENT)

### Local
- `/Users/apple/dev/kv-caching` (canonical since 2026-06-14; `~/Desktop/kv-caching`
  is a symlink to it), macOS, zsh. **Note:** `~/Desktop` is a
  TCC-protected folder; a sync that swaps the project directory inode can transiently
  break the sandboxed shell's CWD handle (EPERM "Operation not permitted"). It is
  **not** a real permission problem and self-heals; recovery levers: re-anchor the
  shell via `/tmp`, or run Bash with the sandbox bypass. A durable fix (not yet done)
  is moving the repo off `~/Desktop` and/or hardening `sync_source.sh` to update files
  in place rather than replacing the directory.

### Remote
- SSH alias `kvcache-gpu` → `sk10741@10.224.32.112` (hostname ADUAED16811WKLX).
  Project `/home/sk10741/kv-caching`. Storage under
  `/home/sk10741/kvcache-storage/{data,cache,runs}`.
- Ubuntu 22.04.5, **2× RTX 6000 Ada (~49 GB each, sm_89)**, driver 570.207 / CUDA 12.8,
  ~503 GB RAM. Env via **uv**, Python 3.11, `.venv`.
- **Storage discipline:** ~70 GB personal budget on a shared ~3.5 TB FS that runs near
  full (latest free ≈ 186 GB). Trace banks are **text JSONL only**; **KV caches are
  NEVER serialized to disk**; HF/uv caches live on storage and are pruned post-setup.
  `check_remote.sh` refuses if FS-wide free < ~30 GB.

### THE STACK (post-M2.5 promotion — canonical)
```
python 3.11, torch 2.7.1+cu126, transformers 4.56.2, tokenizers 0.22.2,
kvpress 0.5.3, datasets 2.21.0, accelerate 1.7.0, numpy 2.2.6, scipy 1.15.3, sympy 1.14.0
```
- **Why the change:** stock kvpress 0.2.10 (the original pin) compresses **prefill
  only** — it cannot enforce an absolute KV budget during long decoding, which is the
  entire regime of this project. `DecodingPress` (the decoding-phase, `target_size`
  token-budget API) first shipped in **kvpress 0.5.0**, which requires
  `transformers>=4.56,<5.3`. So decoding compression ⇒ kvpress 0.5.x ⇒ transformers
  ≥4.56. The human approved the controlled upgrade (validate-in-isolation first).
- Legacy env preserved on remote as `.venv-legacy-tf452`; legacy requirements retained.
- HF pinned revisions: R1-Distill-Qwen-7B `916b56a44061fd5cd7d6a8fb632557ed4f724f60`;
  Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c`; HuggingFaceH4/MATH-500
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

---

## 4. Experimental design (current)

- **Primary model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. **Generalization:**
  `Qwen/Qwen3-4B` — this is a **cross-training-paradigm check within the Qwen family**
  (R1-Distill-Qwen = R1 distilled onto Qwen2.5), **NOT cross-architecture**. Saying
  "two families" / "cross-architecture" is a factual error caught repeatedly. The true
  cross-architecture check is R1-Distill-Llama-8B (restored only with a 2nd GPU).
  Qwen3-4B is **deferred from the pilot**: on transformers 4.56.2 its template renders
  valid ChatML but **no explicit `<think>` prefix** even with `enable_thinking: true`,
  pending model-card/template reconciliation.
- **Sampling:** temp 0.6, top-p 0.95 (R1-Distill standard). R1-Distill needs its chat
  template applied, **NO system prompt**. Verified template (byte-identical on 4.52.4
  and 4.56.2): `<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>`.
- **Benchmarks:** MATH-500 (PRIMARY ADD benchmark — 250 problems × n=8), AIME-24/25
  (tiered deep-cell design), GSM8K, FOLIO, StrategyQA, and a synthetic Recall-Span
  Probe (RSP). Currently only **MATH-500** (and RSP scaffold) are implemented;
  AIME/StrategyQA handles are still `TODO(human)` in `configs/benchmarks.yaml`.
- **Budgets:** {128, 256, 512, 1024, 2048} + random-eviction floor. **Always report
  the compression RATIO beside the absolute budget** (256 tokens is 1.6% of a 16K AIME
  trace but 25% of a 1K GSM8K trace).
- **Greedy:** one greedy run for flip-rate (not yet implemented; `generate_full_cache`
  currently hard-codes `do_sample=True`, so a greedy branch is still needed before the
  greedy ablation — not a blocker for current milestones).

### RSP status — STRUCTURALLY INVALID, needs redesign
Current `src/kvcache/rsp.py` builds an order-invariant "sum all visible values"
task: the linked-list traversal visits all items, so the answer is just the sum of
the on-screen numbers — there is **no recall span, no late-binding, no skippable
filler**. 12/12 correct at ~356 tokens confirms it's trivial. It must be redesigned
to the preregistered spec (k≥4 step-1 quantities; d∈{512,1024,2048,4096} tokens of
**required, non-skippable** filler; final question needs exactly one step-1 quantity
at a controlled distance; interleaved-reference fallback) + a 3-criterion validity
pilot, BEFORE any RSP-based claim. RSP is **decoupled** from MATH milestones (don't
let it gate M3/M4 on MATH).

---

## 5. Repository structure (current)

```
src/kvcache/
  determinism.py     set_determinism, log_versions (τ rationale lives here)
  runmeta.py         capture_runmeta (commit, dirty, hashes, host, GPU, versions)
  io_schema.py       FROZEN JSONL schema + validate_row. SCHEMA v2: adds
                     schema_version, hit_token_cap, has_answer_marker, finish_reason.
                     Validator is backward-compatible (rows w/o schema_version = v1,
                     skip v2 checks) and enforces truncated == hit_token_cap and not
                     marker, plus finish_reason ∈ {marker,length,eos} consistency.
  canonicalize.py    boxed/option extraction + sympify canonicalizer (DEMOTED to ADD
                     bucketing / sensitivity only — NOT the correctness oracle).
  grading.py         is_equiv(gold_raw, pred_raw) = vendored Hendrycks MATH
                     normalization. THE correctness authority. GRADER_VERSION.
  datasets.py        load_benchmark(math500 + rsp). MATH gold kept RAW (is_equiv
                     grades raw vs raw); canonical form in meta["gold_canon"].
  models.py          load_model_and_tokenizer (bf16, single GPU, KVCACHE_REMOTE
                     guarded, pinned-revision guarded), render_prompt (no system role).
  generation.py      finalize_generation (is_equiv scoring; truncation/finish_reason),
                     generate_full_cache, generate_with_press, generate_compressed,
                     has_answer_marker.
  presses/__init__.py  press registry: EXECUTABLE={streamingllm,random},
                     PREFILL_ONLY={snapkv,expected_attention}, PORT_REQUIRED={h2o,rkv},
                     DEFERRED={kivi2,triattention}. build_press(...) returns a
                     DecodingPress(base_press, target_size=budget, compression_interval,
                     hidden_states_buffer_size=0) or None for fullcache. Includes a
                     DeviceSafeRandomPress subclass (see §7).
  bank.py            build_result_row (press/budget/compression_ratio params; stamps
                     grader+canonicalizer versions for re-gradeability), append_result
                     (dedup on (bench,pid,seed) within a per-cell file).
  replay.py          forward teacher-forced TV profile, bounded top-k tail (BoundedTV).
  decomposition.py   Decomposition (total, on_reference_readout, residual,
                     reverse_readout_change, flip_mass_net, residual_bracket property).
                     Honest residual semantics enforced in docstrings.
  metrics/
    add.py           coarsen_answer (+ OTHER bucket), add_tv, add_decomposition
                     (termination-mass vs within-answered), flip_mass +
                     aggregate_flip_mass (signed gained/lost/net), excess_add_permutation,
                     entropy_miller_madow, pass_at_k, truncation_rate, greedy_flip_rate.
    tdp.py           mean_tv, total_tv, fraction_above, tdp, content_token_profile,
                     segment_aggregate. Fixed τ {0.05,0.10,0.15,0.20}, TV not KL.
    reverse_replay.py  CONTRACT stub (engine = M4). reverse_replay_score.
    noise_control.py   CONTRACT stub (engine = M7). calibration_error.
    attribution.py / restoration.py / branch_recovery.py  stubs (M7).
  stats.py           bootstrap, exact_permutation_test, benjamini_hochberg,
                     negative_control_ok.
scripts/
  build_bank.py        full-cache bank runner (guarded). --probe-template flag.
  run_grid.py          compressed-generation pilot runner (guarded; refuses != 80 rows).
  diagnose_math_failures.py  CPU MATH failure/canonicalization audit (re-scores rows
                       with is_equiv vs stored, classifies, counts grader disagreements).
  audit_bank.py        schema-validates a run dir, checks dup keys / version / hash
                       consistency, reports per-benchmark accuracy + truncation_rate.
  probe_decoding.py    DecodingPress smoke probe (cache bound + memory + 1 row).
  launch_tmux.sh / launch.sh / check_remote.sh / setup_remote.sh / sync_source.sh / ...
configs/
  models.yaml, benchmarks.yaml, presses.yaml, sampling.yaml, remote.yaml
  experiments/  pilot_fullcache.yaml, pilot_fullcache_v2.yaml (MATH-only, 16384 cap),
                pilot_compression.yaml (the 80-row grid), probe_decoding*.yaml,
                grid_compression.yaml (full grid — GATED), bank_fullcache.yaml,
                replay_decomposition.yaml, add_permutation.yaml, mechanism_ladder.yaml,
                negative_control.yaml, smoke_*.yaml
docs/  METRICS.md, REMOTE_SETUP.md, KVPRESS_API_PROBE.md, STACK_MIGRATION_M2_5.md,
       M4_REPLAY_DESIGN.md (design draft; no GPU authorization)
docs/protocol/PREREGISTRATION.md  protocol authority (+ dated "Amendment A" with grading/claims/
                    off-policy/MCC/negative-control disciplines).
docs/operations/remoteserver.md     ops manual authority.
publication_strategy.md, research_landscape.md  UNTRACKED planning docs. publication_
                    strategy.md is partly SUPERSEDED (its §3c/§4d describe the retired
                    KL / self-noise-τ design; PREREGISTRATION wins). Do not commit.
```

---

## 6. Chronological narrative of this session (what happened & why)

The session-start snapshot was stale; the project was already well past "scaffold."
Milestones, in order:

1. **M0 (corrections before any run) — found already DONE & committed.** Truncated-row
   retention (retain/score-false/count/report rate), Qwen3 dropped from pilot, summed-TV
   beside mean-TV, press port/deferred marks, OTHER/invalid ADD bucket + tests.

2. **M1+M2 (full-cache vertical slice) — found already committed; accepted as
   infrastructure.** load_model_and_tokenizer, render_prompt, load_benchmark
   (MATH-500 + RSP), generate_full_cache with truncation, build_bank, schema-valid
   JSONL, audit tool.

3. **First pilots (full-cache).** A 4096-cap pilot (`pilot-fullcache-v1`) ran: 32 rows,
   MATH 13/20 (65%), 4 truncations, RSP 12/12. A smoke exposed a **canonicalization
   bug**: the sympify canonicalizer scored `(3, \dfrac{\pi}{2})` as `"2"` (false
   negative).

4. **Grader fix (the first big ruling).** The sympify canonical-string approach was
   demoted; **`grading.is_equiv` (Hendrycks)** became the correctness authority,
   applied to RAW gold vs RAW pred. A CPU **MATH audit** (`diagnose_math_failures.py`)
   on the 4096 pilot found: stored 65% → regraded **80%**, **3 grader false-negatives**
   (all `90°`≡`90` on precalculus/927), 0 false-positives, **4 token-cap truncations**,
   **0 genuine-wrong, 0 extraction failures**. Conclusion: scoring was the bug, and the
   4096 cap was the only real failure source.

5. **Schema v2 + 16384 re-baseline.** Added `hit_token_cap`/`has_answer_marker`/
   `finish_reason`/`schema_version` (additive, backward-compatible) to separate
   length/extraction/reasoning failures. Built **`pilot-fullcache-v2`** (MATH-only,
   16384 cap, transformers 4.52.4): **20/20 correct, 0 truncations, 0 grader
   disagreements.** The previously-truncated problems finished (e.g.
   intermediate_algebra/1994 up to 12,451 tokens). Cap confound resolved.

6. **kvpress decoding blocker (the second big finding).** A DecodingPress probe found
   **stock kvpress 0.2.10 is prefill-only** — `DecodingPress`/absolute decoding budget
   does not exist in it; `BasePress` skips compression once generation starts. So the
   core experiment (budget-256 *during decoding*) is impossible on 0.2.10. Verified:
   `DecodingPress` first shipped in **kvpress 0.5.0**, which needs **transformers
   ≥4.56**. HOTT itself used kvpress + a hand-rolled SnapKV-D and has no public repo.

7. **M2.5 stack migration (human chose "upgrade, validate first").** Phased:
   - Phase 0: isolated `.venv-next` with transformers **4.56.2** + kvpress **0.5.3**
     (torch/numpy/datasets/etc. held). Minimal, clean upgrade.
   - Phase 1: R1-Distill template **byte-identical** on 4.56.2. Qwen3 still no `<think>`
     → stays deferred.
   - Phase 2: re-baseline `pilot-fullcache-v2-tf456` = **19/20**. The one miss
     (precalculus/927 s11, 90→66) was verified to be a **coherent reasoning trace with
     a genuine sign error** ((2,−12,3) vs (2,−12,−3), which breaks the orthogonality
     that gives 90°), shorter trajectory (5345 vs 11374 tokens). **Ruling: Phase 2
     ACCEPTED.** The literal "20/20" gate was mis-specified — bitwise accuracy
     reproduction across a transformers numerics change is impossible by construction;
     19/20 vs 20/20 at n=20 is one Bernoulli flip (noise), and the real gates
     (template identity, 0 grader disagreements, 0 truncation regression, no systematic
     accuracy change) all passed. The flip is itself an in-vivo example of trajectory
     distortion under a numerical perturbation.
   - Phase 3: DecodingPress smoke (`decodingpress-smoke-v1`) PASS — StreamingLLM at
     `target_size=256`, post-prune max exactly 256, between-prune max 320 (=256+64),
     correct, non-truncated, **no tensor/cache artifacts written**.
   - Phase 4: **promoted** the stack to canonical; legacy preserved as
     `.venv-legacy-tf452`. `pilot-fullcache-v2-tf456` (19/20) is now the canonical
     pilot FC reference; the 4.52.4 banks retire to provenance.

8. **M3 compression code — reviewed & APPROVED.** `build_press` →
   `DecodingPress(base_press=StreamingLLMPress()/RandomPress(seed), target_size=budget,
   compression_interval=64, hidden_states_buffer_size=0)`; `generate_with_press` /
   `generate_compressed` reuse `finalize_generation` (identical is_equiv scoring);
   `run_grid.py` (double-guarded, hard 80-row check); `pilot_compression.yaml`
   (random+streamingllm × {256,2048} × 5 MATH × 4 seeds = 80). Verified both presses are
   `ScorerPress` subclasses with valid `seed`/`n_sink` signatures. `compression_ratio`
   = `min(1, budget/(prompt_tokens+gen_tokens))` (per-row realized ratio; 0.24474 =
   256/1046 in the smoke). Runner iterates `random` first → fails fast.

9. **RandomPress device fix (UNCOMMITTED).** The first authorized grid launch,
   `pilot-compression-v1`, failed before writing a row. kvpress 0.5.3's seeded
   `RandomPress` builds a CPU `torch.Generator()` then feeds it to CUDA `torch.rand` →
   device mismatch. Fixed via a lazy `DeviceSafeRandomPress` subclass that builds the
   generator on `keys.device` (preserves seeded behavior). Added
   `probe_decoding_random.yaml` and generalized the instrumented probe to exercise
   either approved press. Local gate: ruff/black clean, **67 tests**, all **13**
   aggregate dry-runs pass.

10. **RandomPress smoke PASS; corrected M3 grid launched.**
    `decodingpress-random-smoke-v1` completed one MATH generation at budget 256:
    schema-v2 row valid, correct, non-truncated, 1,176 generated tokens,
    `compression_ratio=0.2082994304312449`; 504 layer-level compression events;
    post-prune max exactly 256 and between-prune max 320; no
    `.pt/.pth/.safetensors/.npy/.npz` artifacts and no logged errors. This validates
    the device-safe adapter end-to-end. The corrected 80-row run was then launched as
    **`pilot-compression-v1-r2`** to preserve the failed first attempt for provenance.

11. **M3 compression pilot COMPLETE — strong regime separation, surprising baseline
    ordering.** The corrected run finished all 80 rows across 16 JSONL files. Corrected
    audit (after fixing `audit_bank.py`'s duplicate identity to include
    model/press/budget) reports 51/80 correct, 8/80 truncated, mean 5,007.7 generated
    tokens, consistent transformers 4.56.2 / kvpress 0.5.3 / grader / canonicalizer /
    source-tree / config hashes. No log errors and no
    `.pt/.pth/.safetensors/.npy/.npz` artifacts.
    - random-256: **10/20 (50%)**, 5/20 truncated, mean 7,636.9 tokens.
    - streamingllm-256: **4/20 (20%)**, 3/20 truncated, mean 6,789.9 tokens.
    - random-2048: **19/20 (95%)**, 0 truncated, exact correctness agreement with FC.
    - streamingllm-2048: **18/20 (90%)**, 0 truncated, one FC-correct→wrong flip.
    The canonical FC reference is 19/20. Thus high budget is near-full while budget
    256 collapses, supporting the intended regime separation. However, Random beating
    StreamingLLM at 256 contradicts the expected random-floor ordering; this is a real
    pilot observation requiring reviewer interpretation, not a result to suppress.
    A matched pair makes the low-budget failure concrete: algebra/2584 seed 89 is
    correct under FC (`14/3`, 2,219 tokens), while StreamingLLM-256 repeats fragments
    such as `ioneer`/`oupper` until 16,384 tokens, has no marker, and is retained as a
    truncated failure. This is trajectory collapse, not a grader/extraction error.

12. **M4.0 CPU implementation COMPLETE — awaiting Claude code review.** Added exact
    token-ID sidecars (`token_ids.py`) without changing the frozen result schema;
    generation now carries exact prompt/output IDs and the bank can write validated
    sidecars. Replaced the replay stub with a guarded pilot runner supporting
    self-replay, compressed replay, and accuracy-scale readout. The replay engine uses
    one model/two independent `DynamicCache` objects, tracks absolute positions
    separately from retained cache length, toggles compression explicitly via
    `press.enabled`, asserts caches are mutated-not-replaced, computes exact transient
    full-vocabulary TV, and writes scalar-only profiles. Compression onset accounts
    for kvpress pruning after the current attention output (first affected profile step
    is the following step); no-prune yields `onset_step=null`. Added teacher-force to
    `</think>`/answer-marker boundary then sampled answer continuation, scored against
    dataset gold with `is_equiv`. Added replay/readout configs, replay audit, sidecar
    validation, duplicate guards, cache-bound/pre-onset stop conditions, and
    `docs/protocol/PREREGISTRATION.md` Amendment B defining the valid accuracy-scale readout
    decomposition. Local gate: ruff/black clean, **86 tests**, **18 dry-runs**, shell
    syntax clean. No sync, model load, GPU, or remote job was performed.

---

## 7. Exact current state — HISTORICAL SNAPSHOT (M4.0 era, ~2026-06)

> **SUPERSEDED. Do not read sections 7-9 as current state.** They describe the repo
> during M4.0 development: 86 tests (311 at the time of fencing), uncommitted M4.0 work (long since
> committed), and a code review that the dated log shows was passed. Kept as the
> historical record of that handoff. **For current state read §0's top bullet.**

- **HEAD:** `0f56df4 Restore compression runner executable mode`.
- **Uncommitted (tracked) changes:** M3 fixes plus M4.0 across `Makefile`,
  `docs/protocol/PREREGISTRATION.md`, `docs/design/METRICS.md`, `docs/design/KVPRESS_API_PROBE.md`,
  `scripts/{audit_bank,build_bank,launch,probe_decoding,run_replay}.py`,
  `src/kvcache/{generation,replay}.py`,
  `src/kvcache/metrics/tdp.py`, `src/kvcache/presses/__init__.py`, and related tests.
- **Untracked:** `configs/experiments/probe_decoding_random.yaml` (new random smoke),
  M4 pilot configs, `scripts/audit_replay.py`, `src/kvcache/token_ids.py`,
  M4 tests, `docs/design/M4_REPLAY_DESIGN.md`, `publication_strategy.md`,
  `research_landscape.md` (planning docs — never commit), and this `docs/project/context.md`.
- **Local checks:** ruff clean, black clean, **86 tests pass**, all **18** aggregate
  dry-runs pass.
- **Banks/reports present (local `results/`):**
  - `diag/math-audit-v1` (4.52.4/4096: 65%→80% regrade, 4 truncations, 3 grader FNs)
  - `diag/math-audit-v2` (4.52.4/16384: 20/20)
  - `diag/math-audit-v2-tf456` (4.56.2/16384: 19/20)
  - `runs/pilot-fullcache-v2` (4.52.4 — retired)
  - `runs/pilot-fullcache-v2-tf456` (**CANONICAL FC reference**, 19/20)
  - `runs/decodingpress-smoke-v1` (Phase-3 StreamingLLM smoke, 1 row)
  - `runs/decodingpress-random-smoke-v1` (RandomPress smoke, PASS; pulled locally).
  - Remote-only: failed `pilot-compression-v1` attempt (0 rows; preserved log shows the
    upstream CPU-generator/CUDA mismatch).
  - `runs/pilot-compression-v1-r2` (**COMPLETE and pulled locally**): 80 rows / 16
    files; aggregate 63.75% accuracy, 10% truncation. Cell breakdown is in milestone
    11 above.
- **Audit tool correction (uncommitted):** `audit_bank.py` previously used
  `(bench,pid,seed)` as the duplicate identity, incorrectly rejecting valid rows from
  different press/budget cells. It now uses
  `(bench,pid,model,press,budget,seed)` with a regression test. Full local gate is
  ruff/black clean, **68 tests**.
- **kvpress DecodingPress semantics (verified, see docs/design/KVPRESS_API_PROBE.md):**
  `target_size` = exact KV tokens retained after each prune; compression every
  `compression_interval` decode steps; effective max cache ≈ `target_size +
  compression_interval`. We use `compression_interval=64`, `hidden_states_buffer_size=0`.

---

## 8. Immediate next steps (in order) — HISTORICAL (M4.0 era; all completed)

1. **Claude reviews the M4.0 code and tests.** Focus on token sidecar validation,
   enabled-flag press scoping, absolute cache positions, onset indexing, exact-TV
   scalar storage, readout boundary/sampling, and the guarded runner. No GPU action
   before this review.
2. **After approval:** sync and run M4.1 tokenized 20-row FC bank, then M4.2 one-row
   self-replay smoke and full 20-row self-replay. Stop and report unless max exact TV
   <=1e-6, no threshold crosses, and TDP percentile maps to 1.0. Only after that gate
   proceed to M4.3 replay and M4.4 readout.
3. Then M5 (ADD end-to-end) → **M6 (port H2O, then R-KV — the "science starts" gate)**
   → M7 (mechanism interventions: restoration, branch recovery, reverse replay,
   TV-matched noise) → M8 (full pre-registered grid; NOT before M0–M6 green +
   PREREGISTRATION committed).

**Still GATED / do not start without Claude's review:** the full grid
(`grid_compression.yaml`), the 250-problem MATH bank, M4+ engines, H2O/R-KV ports,
KIVI (separate quantized-cache wrapper, not a press), RSP-anything (needs redesign).

---

## 9. Open items / tech debt (non-blocking) — HISTORICAL (M4.0 era)

- **Grader comparability (before the reproduction figure, NOT before M3/M4):** verify
  which grader HOTT / kvpress-eval / MATH-500 use (`math_verify` vs Hendrycks). We use
  Hendrycks `is_equiv` for internal consistency; add `math_verify` as a parallel
  sensitivity for the "reproduce HOTT's collapse" Figure 1.
- **Greedy generation path** (`do_sample=False`, paired flip-rate) — needed for the
  greedy-vs-sampled ablation, not for M3/M4.
- **`generate_compressed` discards `prompt_len`** that `generate_with_press` computes,
  so `run_grid` re-tokenizes the prompt to recover it — harmless but redundant; thread
  it through.
- **RandomPress uses the same seed for all 5 problems** within a (budget, seed) cell —
  reproducible but mildly correlated; consider per-problem seeding before the full grid.
- **RSP redesign** to the preregistered recall-span spec + 3-criterion validity pilot.
- **`sync_source.sh` in-place hardening** so the directory-inode swap (which transiently
  broke local FS access) can't recur; or move the repo off `~/Desktop`.
- **AIME-24/25 and StrategyQA dataset handles** are still `TODO(human)` in
  `configs/benchmarks.yaml` (verify gating/licenses).

---

## 10. Landmines / gotchas for whoever continues

- **Never** call the residual a measured cause; **never** call divergence "damage"
  without signed flip-mass. (§1.2 — these two sentences are the whole load-bearing wall.)
- **Truncated rows are valid data**: retained, scored `correct=False`, counted in the
  denominator, with per-cell `truncation_rate`. Only structurally-malformed rows are
  rejected. Compression increases truncation; rejecting truncated rows would inflate
  compressed accuracy and deflate ADD.
- **Correctness uses `is_equiv` on RAW strings.** Do not route gold through the sympify
  canonicalizer (it caused a silent false-negative). `answer_canon` is bucketing only.
- **KV caches and full-vocab tensors must NEVER hit disk.** Banks are text JSONL.
- **Exchangeability:** FC and compressed rows must share the same stack
  (transformers/kvpress version, batching, kernels). That's why the whole bank was
  re-run on 4.56.2; do not mix stacks across conditions.
- **DecodingPress effective cache = target_size + compression_interval** between prunes
  — report the budget as nominal `target_size`, note the slack.
- **RandomPress on CUDA needs the device-safe generator** (kvpress 0.5.3 bug).
- **Local FS "Operation not permitted" is a transient sandbox/inode artifact**, not a
  permission you must grant — recover via `/tmp` re-anchor or sandbox bypass.
- **Qwen3 is deferred** (no `<think>` on 4.56.2); calling the generalization model
  "cross-architecture" is a factual error.

---

## 11. Quick-reference commands

```bash
# local gate
make lint test dryrun           # currently ruff+black clean, 68 tests, 13 dry-runs

# sync (Policy B)
make sync                       # dry preview
make sync-apply

# remote audit of a run dir (CPU, no GPU)
ssh kvcache-gpu 'cd /home/sk10741/kv-caching && \
  .venv/bin/python scripts/audit_bank.py --run-root <RUNS>/<session> --expected-rows N'

# MATH failure / canonicalization audit (CPU)
ssh kvcache-gpu 'cd /home/sk10741/kv-caching && KVCACHE_REMOTE=1 \
  REMOTE_DATA_ROOT=/home/sk10741/kvcache-storage/data \
  .venv/bin/python scripts/diagnose_math_failures.py \
    --rundir <RUNS>/<session> --gold-from-dataset --out <RUNS>/diagnostics/<name>'

# gated GPU job (tmux): stage ∈ {bank, grid, diagnostic}
ssh kvcache-gpu 'cd /home/sk10741/kv-caching && \
  bash scripts/launch_tmux.sh --i-understand <session> <stage> configs/experiments/<cfg>.yaml'

# pull small results back
rsync -avh kvcache-gpu:/home/sk10741/kvcache-storage/runs/<session>/ results/runs/<session>/

# completed M3 marker
ssh kvcache-gpu 'grep "Finished: pilot-compression-v1-r2" \
  /home/sk10741/kv-caching/logs/pilot-compression-v1-r2.log'
```

Reporting after remote actions follows `docs/operations/remoteserver.md` §31 (Action / Machine /
Commit / Sync / Files / Command / GPU / Log / Output / Status / Disk / Errors / Next).

---

*End of handoff. Re-read the live files before acting; this document records intent and
decisions, the repo records truth.*
