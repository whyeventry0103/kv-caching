**Independent review of the Meeting 5 repair pass — 14 October 2026**

Reviewed commit: [`946c76a3851736e77a4e273fddfdc74fbe336fd8`](https://github.com/whyeventry0103/kv-caching/commit/946c76a3851736e77a4e273fddfdc74fbe336fd8).

**Decision: accept the fixes to the previously reproduced cases. Two reproducible execution issues still need a bounded repair before the new M5 runs rely on this path.** The existing B=1024 experiments can continue. Nothing found here invalidates an existing scientific estimate or requires regenerating historical data.

Claude's report is accurate about the named repair cases, the unchanged stored numbers, the newly recorded approvals, and the remaining absence of real M5 outcomes. It goes too far if “software acceptance passed” means that all measurements are now checked or interrupted production writes are recoverable.

**What I independently verified.**

| Check | Result |
|---|---|
| GitHub origin and backup | Both `main` refs resolve to the full reviewed SHA. The earlier backup lag is fixed. |
| Source integrity | All non-deleted files in the 63-file repair diff match their GitHub blob hashes. All 71 `src/` files used in this review also match the commit. |
| Committed acceptance harness | All 37 output entries reproduce exactly, including valid controls, refusals and exit codes. Only the harness's checkout path was adapted. The resulting JSON has the same SHA-256 as the committed after-results. |
| Focused CPU tests | 237 passed; 3 skipped because the local review environment lacks torch. This is an independent focused run, not a claim to have rerun Claude's 890-test full suite or 157 server tests. |
| Frozen M5 inputs | 100 prompt hashes and 400 prompt-plus-trace hashes. All 25 prompts and 100 traces in the older manifest are preserved exactly. |
| Re-read reports | All 6,063 pre-existing numeric leaves across the five changed reports are unchanged. The expected exploratory inference block was added to `reverse_b256`; the other changes are fingerprints/pointers. |
| Approvals/configuration | Caps 1/28/28 GPU-hours, primary equivalence margin ±0.03, sensitivity ±0.05, and the BBH pin are recorded. The new GPU queue dependencies use `requires_all`. |

The acceptance JSON SHA-256 independently reproduced is:

`ac8b325c67f03ac9c1b92d1aefacc14e1081a7eee25df4744546383beec2e051`

The five report comparisons were against files independently retrieved from the prior reviewed commit `6b4090018007eac7a592cf141f65c6856b14733b`, with their Git blob hashes checked.

| Report | Unchanged existing numeric leaves |
|---|---:|
| `protection_b256` | 3,904 |
| `origin` | 515 |
| `matched_b256` | 261 |
| `reverse_b256` | 1,230 |
| `decision_b256` | 153 |
| Total | 6,063 |

These counts include numeric metadata. They are a file comparison, not an independent regeneration of estimates from server-only raw runs. Claude's acceptance report counts all comparable leaves, including non-numeric values, so its counts differ without contradiction.

**Remaining issue 1: smoke validation is still incomplete below the top-level schema.**

Relevant code: [`smoke_validation.py`](https://github.com/whyeventry0103/kv-caching/blob/946c76a3851736e77a4e273fddfdc74fbe336fd8/src/kvcache/smoke_validation.py#L224), and each family's `smoke_failures`.

The validator checks `protect` only as a dictionary, and `post_prune_sizes` only as a list. Some predicates then compare two values supplied by the marker instead of deriving the expected value from the experiment configuration and model.

Starting from each family's accepted fixture, these mutations still return `(True, "verified")` through the actual runtime `verify_smoke` function:

| Mutation | Protected-selector gate | R-KV-pp gate | Required behavior |
|---|---|---|---|
| Set generation `protect.n_prunes` to NaN | Accepts | Accepts | Refuse: a prune count must be a finite non-negative integer, and this control must demonstrate actual pruning. |
| Change replay `expected_target`, `protect.target` and `post_prune_sizes` together to 255, although the fixture's configured effective budget is 256 | Accepts | Accepts | Refuse: derive the effective target independently. |
| Set replay `n_layers` and `protect.n_layers_pruned` together to 0 | Accepts | Accepts | Refuse: the smoke must cover the expected model layers and actual pruning. |

The NaN passes because `NaN < 1` is false. The other two examples pass because the fields agree with each other. Exact row coverage and an identity-matched marker do not repair those measurement checks.

This is a validator defect demonstrated with injected fixtures. It is **not evidence that the real October 13 smoke had these defects**.

Required bounded repair:

1. Validate the consumed nested fields and list entries before predicate evaluation: exact integer counts, no booleans as counts, finite values, sensible domains, and required fields.
2. Derive the expected effective budget from the smoke config and the independently established prompt length, using the family's existing budget rule. Compare the recorded expected target, actual target and post-prune sizes to that value.
3. Check layer coverage against the pinned model configuration; a record cannot establish its own expected number of layers. Check replay length against the configured replay prefix and the selected trace. Use model metadata, not a weight load.
4. Add the three mutations above for both gate families. Keep a genuinely consistent measured positive control. If stronger checks expose inconsistent synthetic fixture values, correct the fixture rather than weakening the checks.
5. Revalidate available real smoke measurements with the tightened validator. Do not rewrite their measurements or identities to make them pass. Execute the already-approved pending smokes when the queue reaches them.

This is the only additional gate repair requested here. It should not expand into another general validation rewrite.

**Remaining issue 2: a result saved without its sidecar cannot resume.**

Relevant code: [`run_grid_m5.py`](https://github.com/whyeventry0103/kv-caching/blob/946c76a3851736e77a4e273fddfdc74fbe336fd8/scripts/run_grid_m5.py#L215), [`_grid_done_keys`](https://github.com/whyeventry0103/kv-caching/blob/946c76a3851736e77a4e273fddfdc74fbe336fd8/scripts/run_grid.py#L67), and [`append_result`](https://github.com/whyeventry0103/kv-caching/blob/946c76a3851736e77a4e273fddfdc74fbe336fd8/src/kvcache/bank.py#L80).

The M5 runner writes the result first, then builds/writes its token sidecar. On resume, a key is complete only when both exist. However, rerunning an incomplete key calls `append_result`, which refuses to append over the existing result.

I injected an interruption at the first call to `append_token_ids` through the actual M5 runner, with only the model/generation replaced by the repository's CPU test stubs:

- Immediately after interruption: **1 result, 0 sidecars**.
- `--resume` passes the new identity checks and loads the model again.
- It fails with **`FileExistsError: result already exists`**.
- After resume: still **1 result, 0 sidecars**.

The scheduler's overnight stopping and later resuming make this a relevant execution state. This reproduction does not mean it already occurred on the server. It shows that the new path cannot recover from it.

Required bounded repair:

1. Implement recovery within the **new M5 path**, keeping historical generation paths unchanged.
2. Build and validate the row and sidecar before publishing either. Use an atomic per-condition journal/envelope containing both records, so restart can finish publishing exactly the same pair without a new generation. Publication must be idempotent and refuse conflicting records.
3. For a pre-journal orphan, preserve the original in a retirement/quarantine record before a controlled regeneration of that condition; do not silently discard evidence or manufacture token IDs from re-tokenized answer text.
4. Reconcile only the seed shards owned by the current process. A resume must not rewrite another GPU's live files.
5. Test interruption before publication, between row and sidecar publication, and after both; a second resume; an orphan sidecar; and identity-conflicting evidence. Each valid interrupted case must finish with one matching result/sidecar pair per key. Conflicts must still refuse.
6. Ensure progress/dependency completion for these M5 jobs counts complete validated pairs, so an orphan final row cannot make the scheduler call a job complete.

No scientific design, estimate or historical run needs changing for this repair.

**What is accepted from the previous review.**

The specific R1 counterexamples now refuse correctly: string/NaN correctness, unexpected seed, duplicates, wrong model/identity, missing rows and torn JSON. The positive control reaches FINAL with hand-known estimates. Full-source validation precedes selection of the 97-problem cohort. The configured equivalence margins and TOST treatment are present.

The specific R2 names-only and explicit-failure markers now refuse. Production identity is computed before model loading, stamped in new M5 rows and sidecars, and checked on resume. The missing 75 prompt inputs were added without changing the existing 25. GPU dependency lists are now enforced and tested. The additional issues above are the remaining qualifications.

The specific R3 fractional-key, inconsistent-greedy, failed-control, wrong-provenance and stale-output cases are fixed. The reader distinguishes analytic reference fixtures from production rows and reports the paired Random–H2O-style channel contrast. A backend-agnostic execution core and CPU tests exist. The executable GPU calibration workflow is still intentionally incomplete.

The specific R4 conflicting-answer cases now score incorrect. Dataset/revision-aware read-time scoring exists for GSM8K and BBH, while MATH scoring is unchanged. This does not establish that an unbuilt multi-dataset generation campaign is ready end to end.

**What remains unverified from this environment.**

I verified the two GitHub remotes, not the live GPU server checkout, current GPU utilization, cron state, current campaign balances, downloaded BBH bytes on that server, or server-only raw rows. The commit and report describe those checks; they are not independent live observations here. No GPU run, download on the server, deletion, server edit or push was performed in this review. The torch integration tests were skipped locally.

The persistent lint bug-report draft visible at the bottom of Claude's terminal is not by itself evidence that this commit failed lint. The commit records successful gates; independently, the focused tests above pass.

**Proceed in this order.**

1. **Continue the existing B=1024 work.** The committed status reports `grid-pp1024` at 800/800 and `readout-pp1024` at 345/800. Treat these as the recorded snapshot, not a live count. Complete the forward and approved reverse reads, then run the registered protection, consequence, practical-threshold and direction analyses. Preserve the predeclared inference and caveats.
2. **Close only the two execution issues above on CPU.** Preserve the original acceptance tests, validate the new counterexamples, run the required project gates with exit codes propagated, and publish one evidence-backed repair commit. No repeated approval is needed for the already-approved essential pair.
3. **Run the pending approved smokes/pilots and essential pair in queue order.** Keep P1/P2/P3 and the essential pair's cumulative caps, normal windows, storage checks and identities. A failed gate remains a failure; a consumed cap remains a consumed cap. Do not borrow from another campaign or expand an authorization to compensate.
4. **Read the four-policy protected comparison when the two 400-row arms finish.** Use the frozen 97-problem strict cohort as primary, ±0.03 equivalence and ±0.05 sensitivity. The full 100-problem matched-capacity analysis remains unavailable until its separately held 48 comparator rows and adapter exist. R-KV is descriptive in the current multiplicity plan; do not silently promote it after seeing the result.
5. **Prepare the information calibration to the point of execution before requesting its separate ≤4 GPU-hour approval.** Finish the real GPU invocation, input/design/token hashes, resume/output publication, campaign wiring, and explicit numerical calibration pass/fail criteria. The four-block calibration is instrument development, not confirmatory evidence. Keep it behind the existing higher-priority work unless the owner changes priority.
6. **Use calibration to decide whether an information explanation is viable.** The endpoint is output-channel MI through the declared reader, not total cache MI. A difference between arms' channel MIs is not a bound on lost cache information. A high-MI wrong label permutation can have zero accuracy. Report MI together with correct probability, OTHER mass, greedy accuracy and log loss.
7. **Require the later-answer bridge and equal-budget interventions before a mechanistic headline.** The updated protocol correctly adds an earlier operand readout and a separate later compositional answer, with held-out context blocks and simpler predictors as controls. Synthetic retrieval by itself does not explain the natural MATH result. If the protected-policy mismatch vanishes, report that finding rather than forcing an information-theoretic story.
8. **Keep multi-dataset generation separate and prospective.** BBH/GSM8K preparation is authorized; generation is not. Complete the loader → generation → raw-output → read-time grading path and its timing/calibration design before requesting those campaigns. The paper's current natural-data finding remains scoped to the models, dataset and matched comparisons actually measured.

**Copy-ready instruction for Claude Code.**

> The review of 946c76a independently reproduced all 37 acceptance outputs, ran 237 focused CPU tests, verified both GitHub remotes and confirmed zero changes among 6,063 pre-existing numeric leaves in the five re-read reports. The previous named failures are accepted as repaired. Do not repeat a general audit or regenerate historical experiments.
>
> Make one bounded repair for the two remaining issues documented here. First, both runtime smoke gates still accept generation `protect.n_prunes = NaN`, a coordinated wrong target of 255 instead of the configured 256, and zero reported/pruned layers. Validate nested values and derive the expected measurements from the frozen config, inputs and model metadata. Second, interrupting `run_grid_m5` between result and sidecar publication leaves 1 result/0 sidecars; `--resume` reloads and then throws `FileExistsError`. Add M5-specific atomic pair/journal publication and idempotent recovery, with shard ownership, conflict refusal and complete-pair progress counting. Preserve orphan evidence and do not reconstruct raw generated token IDs by re-tokenizing text.
>
> Reproduce these cases first using the script below. Add maintained regression tests, keep the valid controls and previous acceptance cases passing, run the required gates with individually checked exit codes, then commit/push and verify both remotes and server parity. Report software acceptance separately from real-model smoke acceptance and scientific results. Existing B=1024 work continues; the new M5 jobs use the corrected path and approved smokes within their existing 1/28/28 caps. Do not request those approvals again. The information calibration and GSM8K/BBH generation remain separately held. Prepare the information GPU workflow and numerical calibration criteria before its own approval request.

**Reproduction script.** Save the following as `review_946c76a_repros.py` outside the production source paths and run:

```bash
python review_946c76a_repros.py --repo /path/to/kv-caching
```

It uses the repository's positive fixtures but invokes the real runtime gates and real M5 runner. It only writes temporary fixture files. No model weights or GPU are used. On `946c76a`, all six mutated smoke checks return `true`, and the interrupted pair cannot resume. After repair, the mutated gates must refuse and the interruption must recover without duplication.

```python
"""CPU-only independent review reproductions for commit 946c76a.

Run: python review_946c76a_repros.py --repo /path/to/kv-caching
Only temporary fixture runs are written; no real dataset or model is loaded.
"""

import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    os.chdir(root)
    sys.path[:0] = [str(root / p) for p in ("src", "tests", "scripts")]
    import smoke_fixtures as sfx
    import test_run_grid_m5 as tm
    from kvcache.presses import protected_selectors as ps
    from kvcache.presses import rkv_prompt_protect as rp

    observations = {}
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        families = [
            ("selector", ps, "protsel_smoke_wp4.yaml", "grid_math500_h2opp256_m5.yaml"),
            ("rkv", rp, "rkvpp_smoke_m5.yaml", "grid_math500_rkvpp256_m5.yaml"),
        ]
        for family, module, smoke_path, grid_path in families:
            run = base / family
            run.mkdir()
            smoke = yaml.safe_load((root / "configs/experiments" / smoke_path).read_text())
            grid = yaml.safe_load((root / "configs/experiments" / grid_path).read_text())
            sfx.reference_bank(run, smoke)
            marker = sfx.measured_marker(run, smoke, "protsel" if family == "selector" else "rkvpp")
            observations[family + "_valid"] = module.verify_smoke(run, grid)
            for case in ("nan_prunes", "wrong_target", "zero_layers"):
                altered = copy.deepcopy(marker)
                for row in altered["records"]:
                    if case == "nan_prunes" and row["kind"] == "generation":
                        row["protect"]["n_prunes"] = float("nan")
                    if case == "wrong_target" and row["kind"] == "replay":
                        row["expected_target"] = 255
                        row["post_prune_sizes"] = [255]
                        row["protect"]["target"] = 255
                    if case == "zero_layers" and row["kind"] == "replay":
                        row["n_layers"] = 0
                        row["protect"]["n_layers_pruned"] = 0
                sfx.write(run, module.SMOKE_MARKER, altered)
                observations[family + "_" + case] = module.verify_smoke(run, grid)

        # Interrupt the real runner between saving the result and saving its
        # token sidecar. The model/generation are the repo's CPU test stubs.
        with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
            config, out, calls = tm._setup(base / "interruption", mp)
            original_append = tm.m5.append_token_ids

            def interrupted_append(*args, **kwargs):
                raise OSError("injected interruption before sidecar write")

            mp.setattr(tm.m5, "append_token_ids", interrupted_append)
            try:
                tm._run(mp, config)
            except OSError as error:
                first_error = str(error)
            before = {
                "result_rows": len(tm._rows(out)),
                "sidecars": len(tm._rows(out, "token_ids")),
            }
            mp.setattr(tm.m5, "append_token_ids", original_append)
            try:
                code = tm._run(mp, config, "--resume")
                resumed = {"rc": code}
            except Exception as error:
                resumed = {"error_type": type(error).__name__, "error": str(error)}
            observations["interrupted_pair_resume"] = {
                "first_error": first_error,
                "before_resume": before,
                "resume": resumed,
                "after_resume": {
                    "result_rows": len(tm._rows(out)),
                    "sidecars": len(tm._rows(out, "token_ids")),
                    "model_loads": calls["model_loads"],
                },
            }
    print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
```

**Observed reproduction output on the reviewed commit.** Temporary paths vary between runs.

```json
{
  "selector_valid": [
    true,
    "verified"
  ],
  "selector_nan_prunes": [
    true,
    "verified"
  ],
  "selector_wrong_target": [
    true,
    "verified"
  ],
  "selector_zero_layers": [
    true,
    "verified"
  ],
  "rkv_valid": [
    true,
    "verified"
  ],
  "rkv_nan_prunes": [
    true,
    "verified"
  ],
  "rkv_wrong_target": [
    true,
    "verified"
  ],
  "rkv_zero_layers": [
    true,
    "verified"
  ],
  "interrupted_pair_resume": {
    "first_error": "injected interruption before sidecar write",
    "before_resume": {
      "result_rows": 1,
      "sidecars": 0
    },
    "resume": {
      "error_type": "FileExistsError",
      "error": "result already exists for math500/test/precalculus/807.json seed 11: /tmp/tmppjbb5lq_/interruption/base/grid-rkvpp256-m5/math500/r1_distill_qwen_7b/rkv_pp_v1-256-s11.jsonl"
    },
    "after_resume": {
      "result_rows": 1,
      "sidecars": 0,
      "model_loads": 2
    }
  }
}

```
