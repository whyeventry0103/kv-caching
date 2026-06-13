# KVPress Decoding API Probe

Date: 2026-06-13

This note records the authoritative inspection of the installed remote package:

```text
kvpress==0.2.10
torch==2.7.1+cu126
transformers==4.52.4
```

## Available classes

The package exports:

```python
from kvpress import RandomPress, StreamingLLMPress

StreamingLLMPress(compression_ratio: float = 0.0, n_sink: int = 4)
RandomPress(compression_ratio: float = 0.0, seed: int | None = None)
```

Their concrete import paths are:

```text
kvpress.presses.streaming_llm_press.StreamingLLMPress
kvpress.presses.random_press.RandomPress
```

The normal context-manager application is:

```python
with press(model):
    ...
```

## Decoding-budget finding

Pinned `kvpress==0.2.10` does **not** expose `DecodingPress`,
`token_buffer_size`, or another absolute decoding-time token-budget API. Package
enumeration found no decoding-related submodule.

More importantly, `kvpress.presses.base_press.BasePress` explicitly documents
that compression is applied only during prefill. Its forward hook returns without
compression once generation has started:

```python
if kwargs["cache_position"][-1] > q_len:
    return output
```

`ScorerPress.compress` converts `compression_ratio` into a number retained from
the current prefill length:

```python
n_kept = int(q_len * (1 - self.compression_ratio))
```

Therefore `with press(model): model.generate(...)` compresses the prompt cache,
then permits the generated KV cache to grow. It cannot enforce an absolute
budget such as 256 throughout a long reasoning trace.

## Gate decision

The requested budget-256 compressed generation was **not run**. Producing such a
row with stock 0.2.10 would mislabel prefill-only compression as decoding-cache
compression and could not demonstrate `retained_cache <= 256`.

M3 remains blocked until one of these is reviewed:

1. Implement a decoding-time cache adapter that periodically applies
   StreamingLLM/Random selection and enforces an absolute retained-token budget.
2. Port and verify the exact decoding-enabled implementation used by HOTT
   (including SnapKV-D where applicable).
3. Change the pinned dependency only after an isolated compatibility review;
   never auto-upgrade.

No model generation, GPU time, result row, or cache artifact was produced by
this probe.

## KVPress 0.5.3 validation

The isolated migration stack added:

```text
transformers==4.56.2
tokenizers==0.22.2
kvpress==0.5.3
```

`StreamingLLMPress` and `RandomPress` are both `ScorerPress` subclasses.
`DecodingPress` has this constructor:

```python
DecodingPress(
    base_press,
    compression_interval=512,
    target_size=2048,
    hidden_states_buffer_size=256,
)
```

The verified application is:

```python
press = DecodingPress(
    base_press=StreamingLLMPress(),
    target_size=256,
    compression_interval=64,
    hidden_states_buffer_size=0,
)
with press(model):
    model.generate(...)
```

### Budget semantics

`target_size` is the exact number of KV tokens retained immediately after each
prune. Compression occurs every `compression_interval` decoding steps. After the
cache has first reached the target, the effective maximum between pruning events
is therefore approximately:

```text
target_size + compression_interval
```

The Phase 3 diagnostic measured:

```json
{
  "target_size": 256,
  "compression_interval": 64,
  "max_before_prune": 320,
  "max_after_prune": 256,
  "all_after_at_most_target": true
}
```

The schema-v2 row used:

```text
press=streamingllm
budget=256
compression_ratio=0.2447418738049713
correct=true
truncated=false
n_tokens=993
```

Here `compression_ratio` is the nominal retained ratio
`min(1, budget / (prompt_tokens + generated_tokens))`. Every result reports both
this ratio and the absolute budget. The transient between-prune allowance is
reported separately through `compression_interval`.

No `.pt`, `.pth`, `.safetensors`, `.npy`, or `.npz` artifact was written. The run
contained only one JSONL result row and `cache_stats.json`.
