# Meeting 5 implementation review and Claude Code handoff

Reviewed 14 October 2026, Pakistan time. Repository: `whyeventry0103/kv-caching`.

**Verdict: the scientific preparation is useful, but this is not a clean acceptance of the new production paths. Fix the two launch-critical items below before the new protected-policy campaign. Finish the information and grading items before their respective experiments. Nothing here demonstrates that earlier measured results changed.**

## What was independently checked

The main repository resolves to **`6b4090018007eac7a592cf141f65c6856b14733b`**, including the protected R-KV commit `deab5f7`. I fetched the five commits since `fd77d401`, verified the Git blob hashes of all 72 changed files, and checked the package source against the current tree. The backup repository repeatedly resolved to **`76981271e7b7e8c2d7541413e78e404a95974956`** through the commit, branch and branch-ref endpoints. That is two commits behind. Claude should fetch both refs and repair the ordinary push if this still holds; I could not verify the claim that both remotes contain the latest two commits.

I ran **128 focused CPU tests successfully**. One torch-dependent test module was skipped because this review environment has no torch. I did not independently run Claude's complete 827-test suite or any model execution. Seven archival smoke-bundle tests require `git archive` of the producing commit, unavailable in the materialized review snapshot; I instead independently checked the bundle's file hashes, measurement predicates, record counts and producing-commit identity using the corresponding GitHub source files. These were consistent. This environment limitation is not a reported repository test failure.

| Claim | Review result |
|---|---|
| Protected R-KV is a separate wrapper preserving the base scorer and top-k order | Supported by code inspection. The base scoring implementation is reused; prompt retention is added separately. |
| CPU parity report: three traces, 112 layer-level prune calls per identity trace, identical K/V and TV | The committed report and log support it. All archived file and source hashes match. I did not rerun the 1.5B model. “112” counts layer-level calls, not 112 distinct decode-time eviction events. |
| Previous protected-selector smoke evidence | Hashes and checks agree; it contains 9 replay, 3 identity and 3 generation records. It is correctly distinguished from the pending 21-record rerun. |
| Cumulative P1/P2/P3 accounting | Reservation, timeout bounds, reconciliation and scheduler tests pass. Actual server ledger balances were not available to verify. |
| Prospective B=1024 interaction inference | The new problem-level test/Holm path and CLI tests pass. The B=256 application is labelled exploratory. |
| Analytic finite-channel information calculation | Identity = 2 bits / correct probability 1; uniform = 0 / 0.25; wrong permutation = 2 / 0; always OTHER = 0 / 0. These reproduce. No real-model information result exists yet. |
| Cost ledger | Recomputed exactly from the committed inventory and manifest. Full proposed package: 50.52 / 62.84 / 83.25 GPU-hours; essential pair plus smoke: about 32.4 / 40.3 / 53.4, hard caps totalling 57. These are projections using historical measured rates, not measured runtimes of the new protected R-KV campaign. |
| Problem splits | GSM8K: 24 calibration, 200 confirmation, 1,095 reserve. BBH: 25 / 200 / 25. MATH: 200 confirmation and 200 reserve outside the 100 discovery problems. The lists are disjoint within each dataset. MATH does not have a new calibration list in this manifest. |
| Live storage, deletion, GPU occupancy, server parity | Not independently verified from GitHub. The pasted terminal transcript is the evidence available here. |

The seven reusable-cell completeness statements are supported by the committed inventory, not by an independent reread of every raw server row. The full-cache bank has 2,000 rows; the relevant discovery subset has 400 keys. These are different counts, not contradictory ones.

## R1 — launch-critical: the new M5-A reader repeats earlier validation failures

Files: `src/kvcache/metrics/policy_replication.py`, especially `arm_coverage`, and `scripts/report_policy_replication.py`.

The code uses `float(bool(r["correct"]))`. Its acceptance rule ignores the foreign-key count, and the CLI always returns zero after a report. It does not use the established shared validation boundary. These are reproduced through the real CLI on all **388 strict-cohort keys per arm**, not just through an isolated utility function:

| Injected condition | Observed at 6b409001 | Required behavior |
|---|---|---|
| Every Random protected `correct` value is the string `"False"` | FINAL, exit 0, accuracy **1.0** | INVALID; reject the type before conversion |
| One `correct` value is NaN | FINAL, exit 0; it adds **1/388** to accuracy | INVALID; identify the row |
| One extra seed 999 | FINAL with `n_foreign: 1` | INVALID, except for explicitly declared and validated source supersets |
| Duplicate expected row | PARTIAL, exit 0 | INVALID, exit 2 |
| Wrong model and bogus provenance | FINAL, exit 0 | INVALID |
| Torn JSON after a successful read | Exit 1, but the old FINAL file remains | Retire old output first; write INVALID evidence |

**Repair once, through the shared boundary:** validate the complete source rows and their allowed source design before selecting the 97-problem cohort. The three declared relaxed problems and a declared larger full-cache bank are legitimate exclusions/supersets; an unexpected seed or model is not. Validate exact booleans, truncation consistency, keys, model, policy, nominal/effective budget, input identity and required controls. Missing expected conditions are PARTIAL; malformed, duplicate and foreign conditions are INVALID. Preserve the explicit legacy-provenance limitation for historical rows; do not fabricate metadata for them.

Use the existing FINAL/PARTIAL/INVALID contract with exits **0/3/2**, atomic report writes, retirement of old outputs before parsing, and a dependency fingerprint covering everything used by the read. Add real-CLI tests of the cases above. Do not maintain another independent validator that can drift away from the established one.

Also freeze the equivalence decision in the config, not just a module constant. The current zero-standard-error TOST branch reports “within margin” without a test, whereas the registered interaction helper reports NOT TESTABLE. Adopt and document one defensible prospective rule; the conservative choice is NOT TESTABLE for a degenerate sampling estimate, with the observed difference still reported descriptively. Report TOST p-values and the margin; distinguish per-comparison equivalence from a joint claim across policies.

## R2 — launch-critical: smoke evidence and row provenance are not enforced end to end

Files: `src/kvcache/presses/rkv_prompt_protect.py`, `src/kvcache/presses/protected_selectors.py`, the new grid/replay adapters, and their launcher integration.

Both production `verify_smoke` functions accept a current-identity PASS marker containing only record names and keys, **without any measurements**. Adding `raised: "injected control failure"` to one record still returns `(True, "verified")`. The validators check the top-level PASS declaration and coverage but do not reapply the record-level predicates. The archive validator does reapply them; the runtime gate must do so too.

Fix the two runtime gates to validate record schemas, strict finite values, unique selected traces, exact record coverage, family/budget/interval, and all measurement predicates. Reject missing measurements, contradictory PASS records, failed records, duplicate records and stale identities. Compute expected inputs from frozen references and check the actual token inputs used by the smoke. A self-declared PASS must never substitute for its measurements.

**The new R-KV provenance family also needs wiring into production rows and resume.** `run_grid_rkv_pp.py` currently changes the press builder and attention-backend selector, then calls the existing `run_grid.main`. That base runner attaches identity/input hashes and checks resume identity only on its onset-gated D_acc path. The R-KV wrapper explicitly rejects onset options, so registering an `rkv_pp` source list does not itself put that identity into ordinary R-KV protected rows or validate resume. The protected-selector grid wrapper follows the same legacy path.

Add an M5-specific adapter or shared new-family boundary that:

1. Computes the expected generation/config/model/input identity before model loading; refuses mismatched existing rows and sidecars before a resume.
2. Writes that identity and actual input hashes into each new row and sidecar, plus the effective budget and retention/occupancy diagnostics promised by the protocol.
3. Checks it in the reader. Deliberately changing the family implementation, config, model or reference input must refuse resume; a reader-only change must not require expensive regeneration.
4. Wires the R-KV smoke/grid/replay commands into `launch.sh`, scheduler handling and their explicitly approved campaign caps. These launch aliases are not present at the reviewed commit. Dry-run the real launch path, not only the Python adapter.

Keep the old generation paths and historical data stable. Test the new production path using CPU stubs for the model, then use the separately approved real-model smoke. Do not weaken an old identity check or relabel an old smoke as a new one.

## R3 — before information calibration: the mathematics works; the production reader is still a prototype

Files: `src/kvcache/task_information.py`, `scripts/report_information_probe.py`, `src/kvcache/information_tasks.py`.

The real information-reader CLI accepts fractional keys such as `s=0.7` and `eviction_seed=0.9` because they are converted with `int()`. It also accepts a declared greedy symbol with zero probability when another symbol has probability one. Wrong provenance and a failed-control field are ignored. A torn JSON input leaves an old FINAL file in place. Missing rows correctly suppress numbers, but return exit 0.

Before model calibration, give production rows a versioned schema and explicit gate/provenance contract. Keep analytic reference fixtures as a clearly separate mode. Validate integer types before conversion; check native greedy-token metadata against the stored distribution/top-1 evidence; require model/tokenizer/answer-alphabet/task-layout hashes, exact variant coverage and cloned-readout controls. Share the stale-output and status handling from R1. The generator's encoding checks must actually run against the pinned tokenizer before execution; toy-tokenizer tests are not evidence of model tokenization compatibility.

Implement the missing GPU runner with cache cloning, explicit readout state, no further eviction during readout, and full/full agreement controls. The runner must refuse incompatible resume and incomplete designs. Log actual retained positions and fact spans. The primary Random−H2O-style paired MI contrast from protocol §6.7 is also missing from the current report, which only produces each policy's contrast against the reference. Add it directly from paired context-level values, not by subtracting separately bootstrapped intervals.

Do not request a larger information campaign until the proposed **320-prefix, ≤4 GPU-hour calibration** has an implemented runner, dry-run manifest, CPU integration tests and separate approval. That calibration is not included in the 57-hour essential-pair recommendation.

## R4 — before new-dataset calibration: grading is not integrated and BBH admits ambiguous answers

The GSM8K normalization examples work and the historical MATH grader is unchanged. However, searching all production source and scripts finds **no call site** for `gsm8k_grade` or `bbh_grade` outside their definitions; calls currently occur in tests. “Applied when results are read” describes the intended implementation, not a completed reader integration.

The BBH scorer returns correct against gold `(C)` for both:

```text
</think> \boxed{(A)} or \boxed{(C)}
</think> \boxed{(C)}. Actually, the answer is (D).
```

This contradicts the protocol's single-final-option / ambiguous-output rule. Inspect the entire final-answer region, reject conflicting final candidates or use a prospectively frozen correction rule, and retain the parse reason. Do not let “last box” override a later conflicting final answer.

Build one dataset-aware read-time scoring path that dispatches by the frozen benchmark revision and grader version, re-scores **all** full-cache and compressed arms from stored raw text, records raw-text and grader hashes, and applies the same truncation policy. Add CLI fixtures for GSM8K numbers/units and BBH conflicting boxes/corrections. Keep historical MATH correctness untouched. Integrate the BBH loader and the selected manifests into the eventual new-dataset runner. These tasks precede M5-B, not the essential MATH accuracy pair.

## Scientific decisions to settle before outcomes

**Mutual information remains a valid candidate tool, not an established explanation.** The implemented quantity is information in a finite answer channel. Its data-processing lower-bound interpretation requires the readout to depend on the target only through the declared cache/state and fixed context. Include every persistent state actually visible to that readout. Do not describe a difference between two lower bounds as the amount of cache information lost.

Correct protocol §1: a compressed cache is a literal subset of the full cache only for an **instantaneous crop of the same cached tensors**. Sequential eviction changes later hidden states and thus later K/V vectors even with the same teacher-forced token sequence. “Same history” alone does not make forward replay or the planned decode-path M5-C run a cache subset. This correction does not invalidate the output-channel MI formula or its per-arm data-processing statement; it prevents an unjustified cross-arm inequality.

The proposed H2 predicts *later* accuracy from an earlier information measurement. The current synthetic task only implements immediate retrieval, and its greedy correctness and MI come from that same answer distribution. That cannot by itself establish a mechanism for later reasoning. Before confirmation, specify the bridge promised in the review: measure a randomized intermediate fact before a separate downstream/compositional answer, then test prediction on held-out context blocks beyond retention, age, occupancy and the existing proxies. Include the equal-budget target/nuisance intervention and task switch. If that bridge fails, report a retrieval diagnostic rather than claiming to explain the natural-dataset mismatch.

**Novelty:** I checked Garcia's primary text, *Protection Is (Nearly) All You Need*, [arXiv:2605.18053v1](https://arxiv.org/html/2605.18053v1). It already crosses structural protection with seven policies, including Random, under a matched global-cap harness. Its boundary protection and short QA generation differ from retaining the entire prompt during long reasoning. Thus M5-A is a useful replication/extension and confound check; the broad protection-factorial idea is not new. Information alone is also not automatically novel: distinguish the proposed task-specific finite channel and causal intervention from capacity-based information criteria and existing retrieval/retention studies. Treat novelty as provisional until those comparisons and results exist.

**Equivalence:** do not say a contrast is “gone” merely because it is insignificant, or because it fits within five percentage points. My recommendation is **±0.03 accuracy as the primary practical-equivalence margin**, with ±0.05 as a declared sensitivity. This is a scientific recommendation, not a previously approved requirement. Record the owner's choice before outcomes. Choose the margin because of what effect is negligible for the paper's claim, not because of what N=97 can resolve. If the study cannot establish equivalence at that margin, report inconclusive. TOST acceptance permits “equivalent within the stated margin,” not exact equality or a proven exclusive mechanism. Freeze the family of equivalence claims and its error control as well as the interaction tests.

## Recommended approvals and execution order

| Decision | Recommendation |
|---|---|
| R-KV protected smoke | Approve **1 GPU-hour maximum**, after R2 CPU acceptance |
| H2O-style protected 400 rows | Approve **28 GPU-hours maximum**, after the current 21-record P2 smoke and R1/R2 acceptance |
| R-KV protected 400 rows | Approve **28 GPU-hours maximum**, after its own real smoke and R1/R2 acceptance |
| Fresh-uniform 400-row sensitivity | Hold until the essential comparison and timing evidence justify it |
| 48 capacity-matched comparators | Hold for now; the 97-problem strict cohort is the primary read. Do not label the 100-problem comparison capacity-matched until these exist |
| BBH revision | Recommend pinning `9ee07bd481feebf959a6b59d61ea57bdcf30964d` with blob `936cd6d80db71aeacaea857c34f9af87e1c34964`, and permitting dataset preparation only; it does not approve a generation campaign |
| Information calibration | Prepare the runner now; a separate ≤4 GPU-hour request when it is concrete |

The essential recommendation totals **57 allocated GPU-hours**, including loading, retries and failures, in normal nightly windows. It is a maximum spend, not a promise that all conditions finish. The historical timing band is not a statistical upper bound for new policies. Use the first fixed, predeclared rows to update cost projections without changing the scientific design or extending the cap. Preserve P1/P2/P3's existing separate caps, shared-disk floor, VRAM thresholds and 72 GiB personal allocation. Do not infer a new out-of-window authorization.

The terminal input shown at the bottom of the pasted transcript is not proof that an approval was submitted or recorded. If the owner has since submitted “approve the essential pair and the smoke,” Claude should use that approval and not ask again. Otherwise the configurations remain held.

Execute in this order: finish the bounded R1/R2 CPU repair; complete the already approved B=1024 and smoke/pilot prerequisites; run the essential pair; perform the validated 97-problem accuracy read; then decide whether protected captures and information calibration are scientifically warranted. Build the missing CPU adapters while GPUs are busy. Protected accuracy must not be paired with old unprotected M/Δ/ω to manufacture a new mismatch.

## Paste this instruction into Claude Code

> Review commit 6b409001 against this document. Complete one bounded repair pass; do not respond with only a test count.
>
> First reproduce R1 and R2 on that commit through the real CLI/runtime gate, keeping a valid positive control. Route the new M5 readers through the established validation/output boundary. Wire generation identity, input checks and resume refusal into the new protected-policy adapters. Runtime smoke gates must validate the measurements themselves and exact coverage. Wire the R-KV commands and approved campaign budgets into the real launcher. Keep existing generation paths stable; do not fabricate provenance for historical rows.
>
> Add regressions for the listed failures and run the real CPU production paths. A failed check must leave INVALID evidence, never an old accepted FINAL. Re-read existing M5-eligible artifacts on CPU and report any numerical changes; no GPU regeneration is justified merely by a reader fix. Publish a compact acceptance table with before/after outputs, exact commands, hashes and exit codes. Run lint, tests, dry-runs and audit separately with visible exit codes.
>
> Then complete R3/R4 CPU preparation before their respective experiments: a typed information schema/runner with cache-clone controls and the registered paired contrast; dataset-aware read-time grading, including BBH ambiguity handling. Correct the protocol's cache-subset statement and specify an earlier-information/later-answer bridge before confirmation. Record the owner's equivalence margin prospectively. These unfinished later components do not need to delay the essential MATH campaign once its R1/R2 prerequisites are accepted.
>
> If I have already approved the essential pair and smoke, record caps of 1/28/28 GPU-hours and proceed after their technical prerequisites pass, without requesting the same approval again. Otherwise keep them held. Preserve normal windows and all resource guards; no extra dataset campaign, cap increase, deletion, or daytime start is authorized by this review. Keep optional fresh-uniform and capacity-comparator runs held for now.
>
> Fetch both GitHub remote refs. The reviewer saw main at 6b409001 and backup at 7698127. Push the missing commits normally if needed, and verify both fetched refs at the final repair commit. Report separately: software acceptance, real smoke acceptance, measured scientific outcomes, and anything still incomplete. Once this acceptance table passes, proceed with the agreed experiments; do not initiate another open-ended repair cycle.

## Reproduction appendix

The following CPU-only script was run against the reviewed package and real reader CLIs. All synthetic rows and outputs are temporary. It performs no generation, server access, deletion of project data, or experiment launch. Its purpose is to reproduce the observed failures before replacing them with regression assertions in the maintained tests.

Save the block as `review_meeting5_repros.py` at the repository root, then run `PYTHONPATH=src python review_meeting5_repros.py`. After repair, the invalid cases should be refused and the positive controls should still pass. The printed before-repair outcomes follow the script.

```python
"""Independent CPU-only review reproductions at 6b409001. No model or server work.

Run from the repository root: PYTHONPATH=src python review_meeting5_repros.py
All fabricated rows and reports live under TemporaryDirectory.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from kvcache import task_information as ti
from kvcache.presses import rkv_prompt_protect as rpp
from kvcache.presses import protected_selectors as ps
from kvcache.task_graders import bbh_grade

observed = {}


def cli(script, args, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT, env={**os.environ, **(env or {})}, capture_output=True, text=True,
    )


with tempfile.TemporaryDirectory(prefix="m5-review-") as td:
    tmp = Path(td)
    spec = yaml.safe_load((ROOT / "configs/meeting5/m5a_read.yaml").read_text())
    sf = json.loads((ROOT / spec["cohort_source"]).read_text())
    runs = tmp / "runs"
    by_run = {}
    for policy, sides in spec["arms"].items():
        for side, (run, press, budget) in sides.items():
            by_run.setdefault(run, []).extend(
                {"pid": pid, "press": press, "budget": budget,
                 "model": "r1_distill_qwen_7b", "bench": "math500",
                 "decoding": {"seed": seed}, "correct": False}
                for pid in sf["pids"] for seed in spec["seeds"]
            )
    rp_run, rp_press, _ = spec["arms"]["random"]["protected"]
    hp_run, hp_press, _ = spec["arms"]["h2o"]["protected"]
    out = tmp / "m5a.json"

    def run_m5a(rows):
        for run, rs in rows.items():
            p = runs / run / "rows.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(json.dumps(r) for r in rs) + "\n")
        p = cli("report_policy_replication.py", ["--cohort", "strict", "--out", out],
                {"KVCACHE_BASE_RUNS_ROOT": str(runs)})
        d = json.loads(out.read_text())
        return {"rc": p.returncode, "status": d["status"],
                "random_protected_accuracy": d.get("accuracy", {}).get("random+P", {}).get("mean"),
                "h2o_protected_coverage": d["coverage"].get("h2o+P")}

    observed["m5a_valid_minimal_no_provenance"] = run_m5a(by_run)
    bad = copy.deepcopy(by_run)
    for r in bad[rp_run]:
        if r["press"] == rp_press:
            r["correct"] = "False"
    observed["m5a_string_False"] = run_m5a(bad)
    bad = copy.deepcopy(by_run)
    next(r for r in bad[rp_run] if r["press"] == rp_press)["correct"] = float("nan")
    observed["m5a_nan"] = run_m5a(bad)
    bad = copy.deepcopy(by_run)
    bad[hp_run].append({**bad[hp_run][0], "decoding": {"seed": 999}})
    observed["m5a_foreign_seed"] = run_m5a(bad)
    bad = copy.deepcopy(by_run)
    bad[hp_run].append(copy.deepcopy(bad[hp_run][0]))
    observed["m5a_duplicate"] = run_m5a(bad)
    bad = copy.deepcopy(by_run)
    bad[hp_run][0]["model"] = "wrong_model"
    bad[hp_run][0]["provenance"] = {"implementation_id": "wrong"}
    observed["m5a_wrong_model_and_identity"] = run_m5a(bad)
    run_m5a(by_run)
    with (runs / hp_run / "rows.jsonl").open("a") as f:
        f.write('{"torn":')
    p = cli("report_policy_replication.py", ["--cohort", "strict", "--out", out],
            {"KVCACHE_BASE_RUNS_ROOT": str(runs)})
    observed["m5a_torn_json_stale"] = {"rc": p.returncode, "old_report_status": json.loads(out.read_text())["status"]}

    design = {"policies": ["full", "random"], "reference_policy": "full",
              "contexts": ["c0", "c1", "c2"], "eviction_seeds": [0], "queries": ["s"]}
    rows = [
        {"policy": pol, "query": "s", "context_id": c, "eviction_seed": 0,
         "s": r.s, "n": r.n, "symbol_logprobs": list(r.logprobs), "greedy": r.greedy}
        for pol in design["policies"] for c in design["contexts"]
        for r in ti.reference_channel("identity" if pol == "full" else "uniform")
    ]
    ip_rows, ip_design, ip_out = tmp / "info.jsonl", tmp / "design.json", tmp / "info-read.json"
    ip_design.write_text(json.dumps(design))
    args = ["--rows", ip_rows, "--design", ip_design, "--out", ip_out]

    def run_info(rs):
        ip_rows.write_text("\n".join(json.dumps(r) for r in rs) + "\n")
        p = cli("report_information_probe.py", args)
        d = json.loads(ip_out.read_text())
        return {"rc": p.returncode, "status": d["status"]}

    observed["info_valid_reference"] = run_info(rows)
    bad = copy.deepcopy(rows)
    bad[0]["s"], bad[0]["eviction_seed"] = 0.7, 0.9
    observed["info_fractional_keys"] = run_info(bad)
    bad = copy.deepcopy(rows)
    bad[0]["greedy"] = 1  # q(1)=0 while q(0)=1: impossible native argmax
    observed["info_impossible_greedy"] = run_info(bad)
    bad = copy.deepcopy(rows)
    bad[0]["provenance"] = {"implementation_id": "wrong"}
    bad[0]["controls"] = {"passed": False}
    observed["info_failed_controls"] = run_info(bad)
    observed["info_partial_exit"] = run_info(rows[1:])
    run_info(rows)
    with ip_rows.open("a") as f:
        f.write('{"torn":')
    p = cli("report_information_probe.py", args)
    observed["info_torn_json_stale"] = {"rc": p.returncode, "old_report_status": json.loads(ip_out.read_text())["status"]}

    smoke_cfg = yaml.safe_load((ROOT / "configs/experiments/rkvpp_smoke_m5.yaml").read_text())
    run_cfg = yaml.safe_load((ROOT / "configs/experiments/grid_math500_rkvpp256_m5.yaml").read_text())
    pids = ["p/a", "p/b", "p/c"]
    marker = {"passed": True, "status": "PASS", "failures": [],
              "identity": rpp.smoke_identity(smoke_cfg),
              "selected": [{"pid": p} for p in pids],
              "records": [{"kind": k, "pid": p} for p in pids
                          for k in ("identity", "replay", "generation")]}
    (tmp / rpp.SMOKE_MARKER).write_text(json.dumps(marker))
    observed["rkv_gate_no_measurements"] = rpp.verify_smoke(tmp, run_cfg)
    marker["records"][0]["raised"] = "injected control failure"
    (tmp / rpp.SMOKE_MARKER).write_text(json.dumps(marker))
    observed["rkv_gate_failed_record"] = rpp.verify_smoke(tmp, run_cfg)

    sc = yaml.safe_load((ROOT / "configs/experiments/protsel_smoke_wp4.yaml").read_text())
    pc = yaml.safe_load((ROOT / "configs/experiments/grid_math500_h2opp256_m5.yaml").read_text())
    recs = [{"kind": k, "pid": p, "press": f} for p in pids
            for f in sc["presses"] for k in ("replay", "generation")]
    recs += [{"kind": "identity", "pid": p, "press": sc["identity_family"]} for p in pids]
    marker = {"passed": True, "status": "PASS", "failures": [],
              "identity": ps.smoke_identity(sc), "families_tested": sc["presses"],
              "selected": [{"pid": p} for p in pids], "records": recs}
    (tmp / ps.SMOKE_MARKER).write_text(json.dumps(marker))
    observed["selector_gate_no_measurements"] = ps.verify_smoke(tmp, pc)
    marker["records"][0]["raised"] = "injected control failure"
    (tmp / ps.SMOKE_MARKER).write_text(json.dumps(marker))
    observed["selector_gate_failed_record"] = ps.verify_smoke(tmp, pc)

observed["bbh_two_conflicting_final_boxes"] = bbh_grade(
    r"</think> \boxed{(A)} or \boxed{(C)}", "(C)", truncated=False)
observed["bbh_correction_after_box"] = bbh_grade(
    r"</think> \boxed{(C)}. Actually, the answer is (D).", "(C)", truncated=False)

print(json.dumps(observed, indent=2, sort_keys=True))

```

Observed at the reviewed commit:

```json
{
  "bbh_correction_after_box": [
    true,
    "boxed"
  ],
  "bbh_two_conflicting_final_boxes": [
    true,
    "boxed"
  ],
  "info_failed_controls": {
    "rc": 0,
    "status": "FINAL"
  },
  "info_fractional_keys": {
    "rc": 0,
    "status": "FINAL"
  },
  "info_impossible_greedy": {
    "rc": 0,
    "status": "FINAL"
  },
  "info_partial_exit": {
    "rc": 0,
    "status": "PARTIAL"
  },
  "info_torn_json_stale": {
    "old_report_status": "FINAL",
    "rc": 1
  },
  "info_valid_reference": {
    "rc": 0,
    "status": "FINAL"
  },
  "m5a_duplicate": {
    "h2o_protected_coverage": {
      "n_covered": 388,
      "n_duplicates": 1,
      "n_expected": 388,
      "n_foreign": 0,
      "n_rows": 389
    },
    "random_protected_accuracy": null,
    "rc": 0,
    "status": "PARTIAL"
  },
  "m5a_foreign_seed": {
    "h2o_protected_coverage": {
      "n_covered": 388,
      "n_duplicates": 0,
      "n_expected": 388,
      "n_foreign": 1,
      "n_rows": 389
    },
    "random_protected_accuracy": 0.0,
    "rc": 0,
    "status": "FINAL"
  },
  "m5a_nan": {
    "h2o_protected_coverage": {
      "n_covered": 388,
      "n_duplicates": 0,
      "n_expected": 388,
      "n_foreign": 0,
      "n_rows": 388
    },
    "random_protected_accuracy": 0.002577319587628866,
    "rc": 0,
    "status": "FINAL"
  },
  "m5a_string_False": {
    "h2o_protected_coverage": {
      "n_covered": 388,
      "n_duplicates": 0,
      "n_expected": 388,
      "n_foreign": 0,
      "n_rows": 388
    },
    "random_protected_accuracy": 1.0,
    "rc": 0,
    "status": "FINAL"
  },
  "m5a_torn_json_stale": {
    "old_report_status": "FINAL",
    "rc": 1
  },
  "m5a_valid_minimal_no_provenance": {
    "h2o_protected_coverage": {
      "n_covered": 388,
      "n_duplicates": 0,
      "n_expected": 388,
      "n_foreign": 0,
      "n_rows": 388
    },
    "random_protected_accuracy": 0.0,
    "rc": 0,
    "status": "FINAL"
  },
  "m5a_wrong_model_and_identity": {
    "h2o_protected_coverage": {
      "n_covered": 388,
      "n_duplicates": 0,
      "n_expected": 388,
      "n_foreign": 0,
      "n_rows": 388
    },
    "random_protected_accuracy": 0.0,
    "rc": 0,
    "status": "FINAL"
  },
  "rkv_gate_failed_record": [
    true,
    "verified"
  ],
  "rkv_gate_no_measurements": [
    true,
    "verified"
  ],
  "selector_gate_failed_record": [
    true,
    "verified"
  ],
  "selector_gate_no_measurements": [
    true,
    "verified"
  ]
}

```
