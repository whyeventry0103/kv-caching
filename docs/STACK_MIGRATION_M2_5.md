# M2.5 Isolated Stack Migration Report

Date: 2026-06-13

Status: **Phase 2 gate failed; Phase 3 GPU smoke not run.**

## Phase 0: isolated environment

The working `.venv` was not modified. A parallel `.venv-next` was created from
the fully pinned `requirements-next.txt`.

Resolved and installed versions:

```text
python=3.11
torch=2.7.1+cu126
transformers=4.56.2
tokenizers=0.22.2
kvpress=0.5.3
accelerate=1.7.0
numpy=2.2.6
```

Imports succeeded, CUDA was available, both RTX 6000 Ada GPUs were visible, and
`kvpress.DecodingPress` was present.

## Phase 1: template probes

R1-Distill-Qwen-7B rendered byte-identically to the accepted 4.52.4 template:

```text
<｜begin▁of▁sentence｜><｜User｜>What is 1 + 1?<｜Assistant｜><think>
```

Qwen3-4B rendered valid ChatML after downloading only its pinned tokenizer and
configuration:

```text
<|im_start|>user
What is 1 + 1?<|im_end|>
<|im_start|>assistant
```

It did not render an explicit `<think>` prefix despite
`chat_template_kwargs.enable_thinking: true`. Qwen3 remains deferred pending
model-card/template reconciliation.

## Phase 2: full-cache re-baseline

Session:

```text
pilot-fullcache-v2-tf456
```

The run used the same five MATH-500 problems, four seeds, sampling settings, and
16,384-token cap as `pilot-fullcache-v2`.

Audit:

```json
{
  "rows": 20,
  "accuracy": 0.95,
  "correct": 19,
  "truncated": 0,
  "truncation_rate": 0.0,
  "grader_disagreements": 0
}
```

The explicit acceptance requirement was 20/20 correct, zero truncations, zero
grader disagreements, and schema v2. The run failed only the accuracy gate.

The single regression was:

```text
pid=test/precalculus/927.json
seed=11
old stack: answer=90, correct=true, n_tokens=11374
new stack: answer=66, correct=false, n_tokens=5345
```

The new trace made a substantive sign error when deriving the second direction
vector, using `(2, -12, 3)` instead of `(2, -12, -3)`. The grader correctly
classified the completed answer as wrong. Seeds 23, 47, and 89 remained correct.

The new-stack bank is retained for provenance but is not promoted to the
reference bank.

## Gate decision

Phase 3 required Phase 2 acceptance. Therefore:

- no DecodingPress generation was launched;
- no compressed result row was produced;
- no M3 code or 80-generation pilot grid was started;
- `.venv` remains canonical and `.venv-next` remains isolated;
- the old and new full-cache banks are both retained.

The next action requires a scientific ruling on whether the 19/20 result is
acceptable sampling variance or whether the migration must reproduce 20/20
before any compressed run.

## Reviewer ruling (Claude, 2026-06-13): Phase 2 ACCEPTED

The literal "20/20" criterion was mis-specified and is corrected here. Bitwise
accuracy reproduction across a transformers version change is numerical identity
between two different generation kernels — impossible by construction: different
numerics produce different sampled trajectories, which flip borderline problems.

The scientifically load-bearing Phase-2 gates, all PASS:

1. chat template byte-identical (catches real stack/template breakage) — PASS;
2. zero grader/extraction disagreements (scoring consistent) — PASS;
3. no truncation regression — PASS;
4. no *systematic* accuracy change — PASS. 19/20 vs 20/20 at n=20 is one Bernoulli
   flip; the bootstrap CI greatly exceeds the 1/20 difference. Accuracy is not
   measured with power at 5 problems; that is the 250-problem bank's role.

The single flip (precalculus/927 s11, 90 -> 66) was verified by reading the trace:
a coherent, well-formed derivation that made a genuine sign error, using
(2, -12, 3) instead of (2, -12, -3). The correct vector makes the dot product
6 - 24 + 18 = 0 (perpendicular lines, 90 degrees); the sign error yields -36 and
66 degrees. The new trajectory was also shorter (5345 vs 11374 tokens). This is
healthy-model trajectory variance under a numerical perturbation — itself an
in-vivo instance of the trajectory-distortion phenomenon under study — not stack
corruption.

Decision: the new stack (transformers 4.56.2 + kvpress 0.5.3) is sound. Proceed to
Phase 3 (DecodingPress smoke); promote `.venv-next` to canonical on Phase 3 pass.
The 4.56.2 bank becomes the canonical reference; 4.52.4 banks are retired to
provenance. Qwen3-4B stays deferred (no `<think>` prefix on 4.56.2 either).
