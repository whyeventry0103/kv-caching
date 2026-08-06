# M4 Replay and Negative-Control Design

Status: approved by Claude on 2026-06-14. M4.0 CPU implementation is complete
locally and awaiting code review. No GPU execution has occurred.

Date: 2026-06-14

## 1. Scope and scientific contract

M4 measures next-token distribution divergence while teacher-forcing a fixed
reference trace. It does not measure a causal trajectory effect.

- Forward replay follows a full-cache reference trace and compares full-cache
  readout with compressed-cache readout at each reference step.
- The resulting readout quantity is paired with the already-observed free-run
  outcome change. Their residual is reported only as **accuracy change
  unexplained by on-reference readout**.
- Per-step TV is a fidelity measurement. It must travel with signed outcome
  changes and must not be called damage.
- The self-replay negative control runs before any compressed replay. If it
  fails, no compressed replay result is eligible for interpretation.

M4 does not implement reverse replay, TV-matched noise, restoration, or branch
recovery. Those remain mandatory later milestones.

## 2. Exact reference token IDs

### Decision

Store exact prompt and generated token IDs in an additive sidecar. Do not add
`trace_token_ids` to the frozen result JSONL schema, and do not rely on
re-tokenizing decoded trace text.

Decoded text is not a lossless token record:

- `skip_special_tokens=True` can remove generated EOS or control tokens.
- decode then encode is not guaranteed to reproduce the original tokenization.
- `n_tokens` can therefore differ from the re-encoded trace length.

### Sidecar format

For each generated bank row, write one record under the same run root:

```text
token_ids/{bench}/{model}/{press}-{budget}-s{seed}.jsonl
```

Each record contains:

```json
{
  "schema_version": "token-ids-v1",
  "bench": "math500",
  "pid": "test/algebra/2584.json",
  "model": "r1_distill_qwen_7b",
  "press": "fullcache",
  "budget": "full",
  "seed": 89,
  "prompt_token_ids": [1, 2, 3],
  "trace_token_ids": [4, 5, 6],
  "prompt_text_sha256": "...",
  "trace_text_sha256": "...",
  "tokenizer_revision": "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
  "source_tree_hash": "...",
  "config_hash": "..."
}
```

The sidecar writer uses the same duplicate identity as the result bank:
`(bench, pid, model, press, budget, seed)`. It refuses overwrite and validates:

1. `len(trace_token_ids) == result_row["n_tokens"]`;
2. decoding the generated IDs with the generation-time settings reproduces the
   stored trace exactly;
3. text hashes match the paired result row;
4. tokenizer, source-tree, and config identities are present.

The pilot full-cache bank must be regenerated after implementation to capture
the original output IDs. The new 20-row tokenized bank becomes the M4 reference;
the previous full-cache banks remain provenance artifacts.

## 3. Paired replay algorithm

### 3.1 Alignment

Let prompt IDs be `P` and reference trace IDs be `y[0:T]`. Profile step `t`
compares distributions predicting `y[t]` given `P, y[:t]`.

1. Prefill the prompt into two independent cache objects:
   `full_cache` and `test_cache`.
2. Compare the final-position logits from both prefill calls for step zero.
3. Consume reference token `y[t]` into both caches to obtain the logits for
   step `t + 1`.
4. Continue until all `T` reference tokens have been scored.

The loop never samples. Both conditions receive the exact same token at every
step.

### 3.2 One model, two cache states

Use one model instance and two explicit Transformers `DynamicCache` objects.
Loading two 7B model copies is unnecessary and increases memory risk.

The compressed condition uses an `EnabledDecodingPress`, derived from the
verified kvpress `DecodingPress`. Replay explicitly sets `press.enabled=False`
around each full-cache forward and `press.enabled=True` around each compressed
forward. Its interval counters therefore advance only for compressed-cache
steps. The press context remains active for the complete paired loop.

Both paths still assert that Transformers mutates and returns the caller-owned
cache object. If Transformers or kvpress replaces a cache object,
implementation stops and reports the API mismatch.

### 3.3 Exact stepwise TV

Exact full-vocabulary TV is the primary path:

```python
p = softmax(full_logits.float(), dim=-1)
q = softmax(test_logits.float(), dim=-1)
tv = 0.5 * abs(p - q).sum()
```

Only the current pair of vocabulary vectors exists in memory. After recording
the scalar TV, both probability tensors are released. Full-vocabulary vectors
are never accumulated, serialized, or written to disk.

This exact path is required for the self-replay negative control. A top-k tail
bound alone has a positive uncertainty interval even for identical
distributions because unknown tail alignment cannot prove TV equals zero.

The existing top-k-256 union-support bound remains a fallback for a backend that
cannot expose full logits. Fallback rows must store lower/estimate/upper bounds,
set `tv_method="topk_bounds"`, and are not eligible to certify the self-replay
negative control.

### 3.4 Compression onset

Instrument the cache-scoped press to record the first actual prune event. The
onset profile index is the first next-token distribution produced after that
prune. Record:

- `onset_step`;
- target budget;
- compression interval;
- maximum cache length before and after pruning;
- number of prune events.

TV before onset should be approximately zero because DecodingPress has not yet
modified the cache. A material pre-onset signal is an implementation failure.
If no prune occurs, `onset_step` is `null`, the cache-event count is zero, and
TV near zero is a legitimate high-budget result.

## 4. Stored outputs

### TV profile sidecar

Write scalar-only JSON under:

```text
profiles/{bench}/{model}/{press}-{budget}-s{seed}/{pid_hash}.json
```

Format:

```json
{
  "schema_version": "tv-profile-v1",
  "reference_key": ["math500", "pid", "model", "fullcache", "full", 89],
  "condition": {"press": "streamingllm", "budget": 256},
  "tv_method": "exact",
  "estimate": [0.0, 0.01],
  "lower": [0.0, 0.01],
  "upper": [0.0, 0.01],
  "reference_token_ids": [123, 456],
  "onset_step": 64,
  "compression_interval": 64,
  "cache_stats": {
    "events": 10,
    "max_before_prune": 320,
    "max_after_prune": 256
  }
}
```

For exact TV, lower, estimate, and upper are equal. No logits or KV tensors are
stored.

### Replay result row

The replay JSONL row copies the reference bank row's trace and outcome fields,
sets `press` and `budget` to the replay condition, and appends the frozen replay
fields:

- `replay_of`: full reference identity including run name and row key;
- `tv_profile_path`: relative path under the replay run root;
- `auc_tv`: mean TV, retained for backward compatibility;
- `frac_above`: p05, p10, p15, p20;
- `tdp`: first crossing or `null` at each threshold;
- `onset_step`.

Additional summary fields belong in the profile sidecar until the replay schema
is formally versioned. `correct`, `answer_raw`, and `truncated` in a replay row
describe the fixed reference trace, not a newly generated compressed outcome.
The free-run compressed row is joined separately for total change and signed
flip-mass.

No-crossing TDP is stored as `null`; plotting converts it to normalized endpoint
1.0. It must not be silently converted inside the raw metric.

## 5. Negative-control-first execution order

### M4.0: CPU implementation and tests

Implement without model downloads or CUDA imports at module import time:

- token-ID sidecar dataclass, validator, writer, and duplicate guard;
- generation return path carrying exact output IDs to the sidecar writer;
- paired replay state machine;
- cache-scoped press;
- exact stepwise TV and scalar profile writer;
- replay-row builder and replay-run audit;
- pilot-only replay configs and guarded CLI.

Required CPU tests:

1. token sidecar length/hash/identity validation;
2. decoded-text round-trip mismatch is rejected;
3. paired-step alignment predicts token `y[t]` from prefix `P,y[:t]`;
4. independent fake caches do not cross-contaminate;
5. cache-scoped press modifies only the selected cache;
6. exact identical logits give TV exactly zero;
7. known logits produce the expected TV;
8. no-crossing TDP remains `null` and plots at endpoint 1.0;
9. replay row validates with all frozen replay fields;
10. profile artifacts contain no logits, full distributions, or KV tensors;
11. every replay CLI dry-run exits zero and reports the bounded pilot plan.

### M4.1: tokenized full-cache pilot bank

GPU authorization required after design and code review.

Regenerate the same five MATH problems times four seeds at cap 16,384 using the
canonical stack. Produce 20 result rows plus 20 exact token-ID sidecar records.

Acceptance:

- all result and token rows validate;
- each token length equals `n_tokens`;
- decoded generated IDs equal stored trace;
- no tensor artifacts;
- no requirement that accuracy equal a previous 20-row bank exactly.

### M4.2: exact self-replay negative control

Run one short reference row first, then the complete 20-row pilot only after the
one-row smoke passes.

Condition: full cache versus independent full cache, using separate cache
objects and identical reference tokens.

Acceptance:

- maximum per-step exact TV <= `1e-6`;
- mean and summed TV <= `1e-6` times trace length;
- no threshold crossing at 0.05, 0.10, 0.15, or 0.20;
- normalized no-crossing TDP equals 1.0 in the figure layer;
- pre-onset is not applicable because there is no compression;
- repeated execution is bitwise stable or any nonzero numerical floor is
  documented before fixed thresholds are applied.

Any failure stops M4. No compressed replay is launched.

### M4.3: pilot forward replay

After self-replay acceptance, replay the 20 tokenized FC traces under:

- StreamingLLM at budgets 256 and 2048;
- Random at budgets 256 and 2048.

This is 80 replay conditions over fixed traces, not 80 generations. Begin with
one StreamingLLM-256 row and inspect onset/cache bounds before launching the
complete replay pilot.

Primary outputs per cell:

- mean TV and summed TV with trace length;
- fraction above each fixed threshold;
- onset-aligned profile;
- TDP as localization only;
- signed free-run gained/lost/net flip-mass from the completed M3 grid;
- total free-run accuracy change, on-reference readout estimate, and the
  honestly named unexplained residual.

### M4.4: accuracy-scale readout

The approved readout estimand teacher-forces the exact reference reasoning
through the `</think>` or answer-marker boundary into the compressed cache, then
freely samples the answer continuation at temperature 0.6, top-p 0.95, and the
matched reference seed. The continuation is scored with `is_equiv`.

```text
readout_change = readout_accuracy - full_cache_accuracy
total_change = compressed_free_run_accuracy - full_cache_accuracy
unexplained_residual = total_change - readout_change
```

Per-step TV remains separate and is never subtracted from accuracy. The
residual retains the name **accuracy change unexplained by on-reference
readout** and the documented good-trajectory/off-policy bias.

## 6. Safety and stop conditions

- Every non-dry replay command remains remote-only and requires
  `--i-understand`.
- Long jobs run in detached tmux with unique session and log names.
- Model and dataset downloads remain disabled; cached pinned revisions only.
- Refuse if the reference result row, token sidecar, stack versions,
  source-tree hash, or tokenizer revision disagree.
- Refuse duplicate replay rows or profile paths.
- Refuse any attempt to serialize logits, probability vectors, attention
  matrices, or KV caches.
- Stop on nonzero self-replay above tolerance, pre-onset compressed TV above
  tolerance, cache leakage between conditions, or cache length above
  `budget + compression_interval`.

## 7. Files expected in implementation

```text
src/kvcache/token_ids.py
src/kvcache/replay.py
src/kvcache/io_schema.py          # replay validation only; no base-schema change
src/kvcache/bank.py               # sidecar write integration
scripts/build_bank.py
scripts/run_replay.py
scripts/audit_replay.py
configs/experiments/pilot_replay_self.yaml
configs/experiments/pilot_replay_compression.yaml
tests/test_token_ids.py
tests/test_replay_engine.py
tests/test_replay_schema.py
tests/test_replay_dryrun.py
```

## 8. Approved rulings

1. Token-ID sidecar; frozen result schema unchanged.
2. Exact transient full-vocabulary TV primary; top-k bounds fallback-only.
3. One model/two caches with explicit `press.enabled` scoping and hard
   mutated-not-replaced assertions.
4. Accuracy-scale teacher-force-then-generate readout estimand as specified in
   M4.4; mean TV is never subtracted from accuracy.
5. Replay self-control in M4; independent-bank ADD permutation control in M5.
