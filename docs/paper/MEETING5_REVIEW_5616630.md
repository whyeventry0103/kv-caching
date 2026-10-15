# Review of e8fa7ea and 5616630

15 October 2026. Reviewed repository: whyeventry0103/kv-caching.
Reviewed head: 5616630b49d888bb35a589032021993df72d73d7.

## Decision

The committed B=1024 forward residual estimates reproduce exactly from the committed condition table. Keep these results and continue the approved P1/P2/P3 and Meeting 5 queue. No GPU rerun of the forward experiment is indicated by this review.

The information calibration remains HELD. Its new path has two software defects and one calibration-design gap that should be repaired before its separate GPU approval. The new compression-summary reader also needs to enforce its claimed validation dependency. These findings do not invalidate the independently reproduced forward residual estimates.

## What I independently checked

- Both GitHub main branches, origin and backup, resolve to 5616630.
- Inspected the two new commits, relevant readers, runner, calibration evaluator, protocol and configuration.
- Recomputed the two protected residuals and two protected-minus-unprotected residual changes with a separate standard-library problem-cluster bootstrap implementation: average eligible seeds within problem, sample 100 problems with replacement, 2,000 replicates, seed 0, registered percentile convention.
- The 800-row condition table has unique (policy, problem, seed) keys. Its SHA-256 exactly matches the report: b106e6845dfbfedc83990b563d349f43725469a338eaf743f0859afc74bbea1b.
- Ran the focused information-runner, information-core, information-metric, task-generator and compression-summary tests: 38 passed.
- Ran the additional counterexamples below through the actual execute path and reader CLI with CPU model fixtures. These are not GPU measurements.

I did not independently verify live GPU utilization, server parity, disk headroom, raw server row provenance, or the full 939-test claim. The full-cache bank contains more historical rows than the 400-condition analysis subset; the analysis subset should not be described as the entire bank's size.

## Forward result: accepted within its stated scope

| Policy | Protected residual | Change from unprotected |
|---|---:|---:|
| Random | -0.0466667 [-0.0858333, -0.0150000] | +0.0408333 [+0.0008333, +0.0808333] |
| StreamingLLM | -0.0400000 [-0.0700000, -0.0125000] | +0.1550000 [+0.1025000, +0.2175000] |

Each residual uses 392 readout-defined conditions over 100 problems per policy. The eight undefined readouts per policy are exclusions from the declared support, not missing generation records. Keep this distinction alongside the 400-condition free-run denominator.

The registered labels PERSISTS and SHRINKS follow the stored intervals. Random's shrinkage lower endpoint is only 0.0008333, or 0.0833 percentage points: retain the registered label, but do not describe this as a robust large shrinkage effect.

Interpretation: protecting the prompt does not remove the forward accuracy discrepancy at B=1024. About 4–5 percentage points remain. This is an operational residual, not an identified causal trajectory effect and not evidence of an MI mechanism.

Protected free-run means are Random 0.8375 and StreamingLLM 0.8475; their closeness is descriptive, not an equivalence or superiority conclusion. Do not carry the unprotected Random ranking into the protected comparison without its paired test.

The practical-threshold read remains SENSITIVITY. For the protected arms, its false_reassurance flag is false at the 0.03 margin for both policies. At 0.01 it is true for StreamingLLM only. Do not advertise a protected 3-point false-certification finding. The reverse analysis and combined interpretation remain pending.

## R1 — New calibration measurements are not validated before PASS

Files: src/kvcache/task_information.py, src/kvcache/information_calibration.py, scripts/report_information_probe.py.

The production reader validates the existing distribution/provenance/control fields, but not the newly consumed end-to-end and retention measurements. The evaluator skips missing end-to-end values and accepts any nonempty list. A late NaN can be ignored by Python's max aggregation. Negative retained fractions satisfy an upper-bound-only criterion.

Reproduced through the real reader CLI:

| Counterexample | Actual outcome at 5616630 |
|---|---|
| Valid execution fixture | exit 0, FINAL, calibration PASS |
| Remove 63 of 64 scheduled original-versus-clone measurements, leaving one zero | exit 0, FINAL, PASS |
| Set the final scheduled end-to-end difference to NaN | exit 0, FINAL, PASS; reported maximum 0 |
| Set every compressed-arm retained fraction to -1 | exit 0, FINAL, PASS; C5 value -1 |

Required repair:

1. Extend the shared production-row validation to every consumed calibration field before aggregation. Reject booleans/non-numeric values where numeric measurements are required, non-finite values, negative differences, retention outside [0,1], malformed fact spans and invalid prune counts.
2. Freeze the expected control schedule in the design. The present producer measures original-versus-clone agreement on the LAST query only, so require that exact per-context/policy/seed/variant set. Do not require non-null values on rows where the design intentionally does not measure it, and do not accept an arbitrary subset of required controls.
3. Verify retention measurement coverage for both declared spans in every applicable compressed row. Missing evidence cannot silently disappear from the mean.
4. Bind the selected calibration configuration/criteria to the frozen design and include their fingerprint in the report's dependencies. A reader must not silently substitute a different criteria file.
5. Keep data validity distinct from scientific calibration acceptance: malformed measurements should produce INVALID/exit 2; a complete valid channel that fails criteria may remain FINAL with calibration FAIL. Downstream acceptance must explicitly require calibration PASS, not merely exit 0.

Acceptance: the three reproduced malformed/incomplete cases must refuse through the real CLI, valid fixtures must still pass, and the valid no-eviction fixture must still receive calibration FAIL.

## R2 — The frozen design does not bind actual filler inputs

Files: scripts/run_information_probe.py, src/kvcache/information_tasks.py, src/kvcache/information_probe.py.

load_passages computes source hashes, and make_blocks chooses filler segments and aliases. However, design.json stores only the block IDs and config-derived provenance; it does not store the actual passage hashes, selected block descriptions or rendered token hashes. The nominal block IDs depend on the design seed and index, not the filler contents.

Reproduced using the real execute function: finish two blocks; replace every filler passage with different token content and different source identifiers; resume with the same config. The runner reports rows_written=0, skipped=2, and design.json is byte-identical. It neither detects the new inputs nor regenerates. If only some blocks had completed, old and changed filler could be mixed under one identity.

Required repair:

1. Before loading model weights, materialize a deterministic input manifest containing the actual selected passage hashes and lengths, the chosen block definitions (aliases, order, segments, skipped sources), and hashes of rendered instruction/body/query token IDs for every variant. Include pinned model/tokenizer identities as already done.
2. Hash this manifest into the frozen design and every production row's expected provenance. Preserve the manifest itself for reproduction.
3. Refuse a resume when any selected input changes. The reader must validate the same manifest identity. Validate block completeness before considering a stored block finished.
4. Add CPU acceptance through execute for unchanged-input resume, changed filler, and a partially completed run followed by changed filler. All changed-input resumes must refuse before model loading or new row writes.

Do not solve this by hashing only the YAML filename, reserve PID list, block IDs or code: the actual selected token inputs must be bound.

## R3 — C6 passes when every compressed policy is at the information floor

File: src/kvcache/information_calibration.py, criterion C6.

C6 computes its range over full cache plus compressed arms and checks only the compressed ceiling. Consequently, a perfect full-cache channel with all compressed channels at approximately zero bits has a two-bit range and PASSES.

Reproduced with the real execution and reader, using full cache plus four compressed stub policies at a tiny budget. All four compressed channels lose both facts, but the read is FINAL and calibration PASS. C6's reported range is approximately 2 bits.

This is a scientific calibration-design gap, not a malformed-data case. Such a task can demonstrate catastrophic loss versus full cache but cannot distinguish information preservation among the compressed policies at that operating point.

Required repair before real calibration:

- Add an explicit prospective all-compressed-at-floor rule. For example, declare a lower floor such as 0.1 bits and require that compressed channels are not all below it; justify and record the choice before real data. Keep the existing ceiling check.
- Report the full-cache-to-compressed range and the compressed-only range separately.
- Do NOT require Random to beat H2O-style or StreamingLLM to pass calibration. Equal, informative compressed channels must remain a scientifically valid null outcome. Likewise, do not tune calibration to manufacture the desired policy ranking.
- Record this criterion correction as a dated pre-data amendment. Apply the one-redesign rule to real instrument failures, not to inconvenient policy results.

## R4 — Compression summary: enforce the validation dependency and narrow labels

File: scripts/report_realized_compression.py.

The script says the S.1 readers validated its inputs, but never loads their validation results or compares fingerprints. It can publish summaries from incomplete or unprovenanced rows. The reproduction appendix includes a real-CLI case with one minimal row per policy/cell and no prior S.1 report; the script exits successfully and writes all eight summaries.

Required repair: consume the existing FINAL S.1 evidence and verify the current grid dependencies against it, or reuse the common row validator. Missing/stale/invalid evidence should produce an explicit PARTIAL/INVALID outcome. This needs no new GPU data.

The stored compression_ratio is computed as min(1, nominal budget / (prompt tokens + generated tokens)). It is a budget-to-sequence-length ratio, not measured live occupancy, actual K/V bytes, or an eviction-event count. At B=256, three prompt-protected problems use an effective budget above 256, so nominal and effective budget ratios differ there. Periodic pruning also means ratio < 1 does not by itself prove a prune occurred.

Report the current numbers as nominal-budget ratios. Label 31/400 as the count with ratio 1 unless actual prune evidence establishes the stronger never-compressed count. Add effective-budget ratios where budgets are relaxed, using recorded prompt lengths and the frozen budget rule; no generation change or rerun is needed. The protected B=1024 medians 0.434 and 0.487 are approximately twofold budget ratios, not measured hardware memory savings.

## Concrete next actions for Claude Code

1. Preserve the accepted B=1024 estimates, condition table and registered forward labels. Do not rerun generation for this review.
2. Continue the already approved queue under its existing gates, caps and windows. This review grants no new GPU time or out-of-window start.
3. Repair R1 and R2, resolve R3 prospectively, and fix R4's evidence dependency and terminology. The repro script below supplies focused counterexamples; do not spend another cycle broadly refactoring unrelated readers.
4. Run the focused CLI/execute acceptance tests and the repository's required gates, with every command's exit status checked independently. Commit and push the repairs plus compact before/after evidence; sync through the established workflow.
5. Keep the information calibration HELD until these checks pass and its separate GPU allocation is approved. Before requesting it, show that the real reserve filler exists in sufficient quantity and that the real tokenizer can render the frozen task. Those checks do not require model weights.
6. After P1 reaches complete validated coverage, run the registered reverse t tests with Holm and publish the combined forward/reverse interpretation. Keep Qwen held until its registered release conditions are met.
7. After the real selector smokes and pilots pass, run the approved protected H2O-style and R-KV grids. Protected M/Delta/omega captures remain separate future work; never substitute unprotected captures for them.

No new mutual-information mechanism has been established. The forward residual is a useful measured result; the MI experiment is still instrument preparation.

## Reproduction appendix

Save the following script at the root of a checkout of 5616630 as review_5616630_repro.py. Run with the project's CPU environment: `python review_5616630_repro.py`. It uses existing CPU fixtures, temporary output directories and the real readers; no weights, GPUs or server access are used. It prints the observed bad outcomes rather than asserting that those bad outcomes are desirable. Convert each into a regression test asserting refusal when implementing the repairs.

```python
import csv
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import pytest
import test_information_runner as tr
import test_information_probe as tip
from kvcache import information_tasks as it


def bootstrap(values):
    groups = defaultdict(list)
    for pid, value in values:
        groups[pid].append(value)
    means = [statistics.fmean(groups[p]) for p in sorted(groups)]
    rng = random.Random(0)
    draws = sorted(statistics.fmean(means[rng.randrange(len(means))] for _ in means) for _ in range(2000))
    return statistics.fmean(means), [draws[49], draws[1949]]


def forward_check():
    path = ROOT / 'results/diag/amendment-s/protection_b1024.conditions.tsv'
    rows = list(csv.DictReader(path.open(), delimiter='\t'))
    report = json.loads(path.with_name('protection_b1024.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['sufficient_statistics_sha256']
    assert len(rows) == len({(r['press'], r['pid'], r['seed']) for r in rows}) == 800
    for press in sorted({r['press'] for r in rows}):
        sub = [r for r in rows if r['press'] == press and r['readout_pp'] in ('True','False')]
        residual = [(r['pid'], int(r['free_pp']=='True')-int(r['readout_pp']=='True')) for r in sub]
        delta = [(r['pid'], int(r['free_pp']=='True')-int(r['readout_pp']=='True')-int(r['free_base']=='True')+int(r['readout_base']=='True')) for r in sub]
        for name, values in [('residual_pp',residual), ('delta_residual',delta)]:
            mean, ci = bootstrap(values)
            stored = report['per_press'][press]['primary'][name]
            assert abs(mean-stored['mean']) < 1e-14 and all(abs(a-b)<1e-14 for a,b in zip(ci,stored['ci95']))
            print(json.dumps({'forward':press, 'endpoint':name, 'n':len(values), 'mean':mean, 'ci95':ci, 'matches':True}))


def case(name, mutate=None, budget=40):
    with tempfile.TemporaryDirectory() as td, pytest.MonkeyPatch.context() as mp:
        root = Path(td)
        cfg, config = tr._cfg(root, budget=budget)
        tr._stubs(mp, budget=budget)
        if name == 'all_compressed_at_floor':
            cfg['policies'] = ['full', 'recency_a', 'recency_b', 'recency_c', 'recency_d']
            config.write_text(tr.yaml.safe_dump(cfg))
            base_factory = tr.rip.policy_factories
            def four_policies(model, cfg):
                base = base_factory(model, cfg)
                return {'full':base['full'], **{p:base['recency'] for p in cfg['policies'] if p != 'full'}}
            mp.setattr(tr.rip, 'policy_factories', four_policies)
        out = root/'run'
        tr.rip.execute(cfg, out_dir=out, base_runs=root)
        files = sorted((out/'readout').glob('*.jsonl'))
        allrows = [json.loads(line) for p in files for line in p.read_text().splitlines()]
        if mutate:
            mutate(allrows)
            for p in files:
                p.unlink()
            (out/'readout'/'all.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
        rc, read = tr._read(root, out, config)
        print(json.dumps({'case':name, 'exit':rc, 'status':read['status'], 'calibration':read.get('calibration')}))
        if name == 'valid':
            original_design = (out/'design.json').read_bytes()
            changed = [it.Passage('changed-'+p.source, tuple(reversed(p.token_ids))) for p in tip.PASSAGES]
            mp.setattr(tr.rip, 'load_passages', lambda cfg, base: changed)
            resumed = tr.rip.execute(cfg, out_dir=out, base_runs=root)
            print(json.dumps({'case':'changed_filler_resume', 'result':resumed, 'design_unchanged':original_design==(out/'design.json').read_bytes()}))


def missing_e2e(rows):
    present = [r for r in rows if r['e2e_max_abs_diff'] is not None]
    for r in present[1:]:
        r['e2e_max_abs_diff'] = None


def nan_e2e(rows):
    [r for r in rows if r['e2e_max_abs_diff'] is not None][-1]['e2e_max_abs_diff'] = float('nan')


def negative_retention(rows):
    for r in rows:
        if r['policy'] != 'full':
            r['retained_fraction'] = {'s':-1.0,'n':-1.0}


def incomplete_compression_read():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        (root/'scripts').mkdir()
        shutil.copytree(ROOT/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__'))
        script=root/'scripts/report_realized_compression.py'
        shutil.copy2(ROOT/'scripts/report_realized_compression.py',script)
        runs=root/'runs'
        for run, presses in [('grid-pp256',['random_pp','streamingllm_pp']),('grid-diss256-tok',['random','streamingllm']),('grid-pp1024',['random_pp','streamingllm_pp']),('grid-diss1024',['random','streamingllm'])]:
            path=runs/run/'math500'/'one.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps({'press':p,'compression_ratio':1.0})+'\n' for p in presses))
        proc=subprocess.run([sys.executable,str(script),'--runs-root',str(runs)],capture_output=True,text=True)
        output=root/'results/diag/amendment-s/realized_compression.json'
        d=json.loads(output.read_text()) if output.exists() else {}
        print(json.dumps({'case':'compression_one_row_per_cell_no_prior_validation','exit':proc.returncode,'output_written':output.exists(),'counts':[v['n_rows'] for b in d.get('cells',{}).values() for v in b.values()]}))


if __name__ == '__main__':
    forward_check()
    case('valid')
    case('missing_63_of_64_e2e', missing_e2e)
    case('nan_e2e', nan_e2e)
    case('negative_retention', negative_retention)
    case('all_compressed_at_floor', budget=5)
    incomplete_compression_read()
```

## Observed reproduction output (condensed)

```json
{"ci95": [-0.08583333333333334, -0.015], "endpoint": "residual_pp", "forward": "random_pp", "matches": true, "mean": -0.04666666666666667, "n": 392}
{"ci95": [0.0008333333333333332, 0.08083333333333334], "endpoint": "delta_residual", "forward": "random_pp", "matches": true, "mean": 0.04083333333333333, "n": 392}
{"ci95": [-0.07, -0.0125], "endpoint": "residual_pp", "forward": "streamingllm_pp", "matches": true, "mean": -0.04, "n": 392}
{"ci95": [0.1025, 0.2175], "endpoint": "delta_residual", "forward": "streamingllm_pp", "matches": true, "mean": 0.155, "n": 392}
{"calibration_verdict": "PASS", "case": "valid", "control_values": {"C4_end_to_end": 0.0, "C5_eviction_reaches_facts": 0.0, "C6_dynamic_range": 0.9999999999281587}, "exit": 0, "status": "FINAL"}
{"case": "changed_filler_resume", "design_unchanged": true, "result": {"blocks": 2, "rows_written": 0, "skipped": 2}}
{"calibration_verdict": "PASS", "case": "missing_63_of_64_e2e", "control_values": {"C4_end_to_end": 0.0, "C5_eviction_reaches_facts": 0.0, "C6_dynamic_range": 0.9999999999281587}, "exit": 0, "status": "FINAL"}
{"calibration_verdict": "PASS", "case": "nan_e2e", "control_values": {"C4_end_to_end": 0.0, "C5_eviction_reaches_facts": 0.0, "C6_dynamic_range": 0.9999999999281587}, "exit": 0, "status": "FINAL"}
{"calibration_verdict": "PASS", "case": "negative_retention", "control_values": {"C4_end_to_end": 0.0, "C5_eviction_reaches_facts": -1.0, "C6_dynamic_range": 0.9999999999281587}, "exit": 0, "status": "FINAL"}
{"calibration_verdict": "PASS", "case": "all_compressed_at_floor", "control_values": {"C4_end_to_end": 0.0, "C5_eviction_reaches_facts": 0.0, "C6_dynamic_range": 1.9999999998527138}, "exit": 0, "status": "FINAL"}
{"case": "compression_one_row_per_cell_no_prior_validation", "counts": [1, 1, 1, 1, 1, 1, 1, 1], "exit": 0, "output_written": true}
```
