# Meeting #3 design spec

**Status: proposal, with corrections applied 2026-09-12.** Nothing here is
pre-registered by this file alone; the binding pre-commitments live in
`docs/protocol/PREREGISTRATION.md` **Amendment Q**, written the same day.

Written against PREREGISTRATION (A–P), PAPER_FINDINGS, METRICS, RKV_PORT,
context.md §0, and the Meeting #3 Notion page. Verified against the repo on
2026-09-12; the verification pass is recorded in §7.

---

## 0. The constraint that shapes all three designs

Meeting #2 Task 3 established that **within a fixed budget, divergence magnitude
does not predict accuracy loss** (Spearman ρ = −0.200, CI [−0.400, +0.800], n = 4
cells; "not established at this n", not refuted). The pooled ρ = −0.857 is
carried entirely by the 256-vs-2048 contrast.

So a fourth instrument producing another magnitude scalar is dead on arrival.
All three Meeting #3 tasks are fine-scale instruments; each must produce
**structure**, not magnitude, and must carry a pre-committed falsifiable
prediction against an accuracy- or answer-scale quantity it was not fitted to. A
fine-scale quantity connected to accuracy by adjacency licenses nothing.

Three storage rules bind every design. KV caches and full-vocabulary tensors
never hit disk. Replay stores scalar profiles only. Measured on the remote
2026-09-12: **68 GB used of the ~70 GB personal budget** (66 GB of it the model
cache, 2.5 GB runs), so ~1.6 GB of headroom. Any design that materialises
attention maps or hidden states to disk is not implementable and must reduce to
scalars in the same pass.

---

## 1. Task A: divergence and damage against cache size

### 1.1 Why the requested curve cannot be drawn as asked

Write `B` for the absolute budget, `t` for context length, `r` for the realised
compression ratio. Then `r = B / t`, and any two fix the third. No single slice
isolates horizon:

| Slice | Held fixed | What varies with horizon | Confound |
|---|---|---|---|
| iso-B | budget | r falls as t grows | horizon confounded with dose |
| iso-r | ratio | B grows as t grows | horizon confounded with absolute cache |
| iso-t | context length | nothing | cannot address horizon at all |

M.3 ran the iso-r slice and recorded the qualifier: holding the ratio fixed means
letting the absolute budget grow 32 → 626, so it does not show horizon is
irrelevant at fixed cache size. Nobody has run the fixed-cache-size experiment
because, under `r = B/t`, it does not exist as a one-dimensional manipulation.

### 1.2 Breaking the identity with compression onset

`r = B/t` couples ratio to context length but says nothing about **how long
compression has been running**. Introduce onset step `t0`: compression is
inactive until decode step `t0`, active with budget `B` from `t0` onward. Define
compression exposure `h = t − t0`. At a given absolute position `t`:

- realised ratio `r = B / t`, unchanged by `t0`
- exposure `h = t − t0`, controlled by `t0` alone

Two conditions can therefore sit at identical absolute position, identical
budget, and identical realised ratio while differing by an order of magnitude in
exposure. Worked pair, same reference trace, budget 256, read at `t = 5000`:

| Cell | onset `t0` | exposure `h` | ratio `r` at t=5000 |
|---|---|---|---|
| A1 | 0 | 5000 | 0.0512 |
| A2 | 4500 | 500 | 0.0512 |

Call this **onset-staggered replay**. It is the experiment M.3 pointed at and
could not reach, and it is the single most defensible new cell in this spec.

**Implementation is cheap, and this was verified rather than assumed.** The
instrumented press already carries `self.enabled` and `profile_step`
(`src/kvcache/presses/__init__.py`), and onset gating is one comparison in
`forward_hook`.

**Correction to an earlier draft of this spec:** it described `onset_step` as
unused and proposed reusing it. `onset_step` is *not* unused — it is a measured
**output**, written on the first compression to record the step at which the
prune first affected the distribution. Reusing it as the gate input would make
the gate overwrite its own measurement. The gate is therefore a new field,
`compression_start_step`, and a regression test asserts the two do not collide. **It must be built through
`build_press(..., instrumented=True)`**: a bare `DecodingPress` has no `enabled`,
so the gate would silently never fire and every arm would degenerate to the
always-on condition. That is landmine 12, the defect that invalidated
restoration v1, and it is why the regression test asserts the attribute exists.

### 1.3 The deliverable is a surface, not a curve

Run a grid over `(B, t0)` and read divergence at matched positions:

```
D(r, h)   divergence as a function of realised ratio and compression exposure
```

The supervisor's requested curve is the marginal along the deployed path
(`B` fixed, `t0 = 0`). Reads are pre-committed in Amendment Q (RATIO-ONLY /
HORIZON-PRESENT / MIXED).

### 1.4 Divergence is not damage, so plot two surfaces

`D_tv(r, h)` from per-step TV under teacher-forced replay, and `D_acc(r, h)` from
free-running accuracy against the matched reference at the same cells. Given Task
3's null, the interesting result is **where the two surfaces disagree**: a region
of high TV and low accuracy loss is divergence-then-recovery. `D_tv` is cheap and
teacher-forced; `D_acc` needs free-running generation and is gated on the `D_tv`
read.

### 1.5 Threat recorded now

Onset staggering changes what is in the cache at position `t`, not only how long
compression ran: in A2 the first 4500 entries were never subject to eviction, so
the retained set is drawn from a different history. That is what "exposure"
means. It must be written as **exposure to the compression process**, never as
"horizon" in the plain temporal sense, or the paper repeats the M.3 error.

---

## 2. Task B: attention over shared KV pairs

### 2.1 The construction

For query `q` at step `s`, let `a_full` be the full model's attention over all
`t` positions and `a_comp` the compressed model's over the retained set `S`:

```
a_full|S (j)  =  a_full(j) / Σ_{k∈S} a_full(k)    for j ∈ S
M             =  1 − Σ_{k∈S} a_full(k)             lost mass
Δ             =  TV( a_comp , a_full|S )           shape distortion
```

If compression only removed keys and the state were undamaged, `a_comp` would
equal `a_full|S` and `Δ` would be 0, with the whole effect in `M`.

### 2.2 Why this is the paper-worthy part

- **`M` is selection failure** — attention mass the full model wanted on evicted
  positions. A property of the retained set, computable from the full model plus
  the eviction decision: an **on-reference** quantity.
- **`Δ` is state drift** — the compressed model's `q_s` came from a hidden state
  already perturbed by earlier compressed steps.

This is the paper's readout-vs-trajectory split one scale down: readout ↔ `M`,
trajectory ↔ `Δ`. Agreement across scales is convergent evidence the paper does
not currently have; disagreement is more interesting still.

**Instrument self-control, free.** At the first prune under teacher-forced replay
both models have consumed identical token IDs, so `q_s` is identical and `Δ` must
be exactly `0.0000`. Gated like the pre-eviction TV self-control: if it is ever
non-zero, STOP — harness defect, not result.

### 2.3 GQA, and a finding hiding in it

Verified from the model config on the remote (2026-09-12,
`DeepSeek-R1-Distill-Qwen-7B`): `num_attention_heads` **28**,
`num_key_value_heads` **4**, `hidden_size` 3584, `num_hidden_layers` 28.

There is no pre-slice representation to work with: `W_k` and `W_v` project
directly to 4 KV heads' worth of dimension, and the 28-query-head view is a
broadcast of those 4, not a slicing of something larger. So the split cannot be
ignored — and the reason is itself worth measuring.

Eviction operates on the KV cache, which has 4 heads, so one eviction decision is
**shared across the 7 query heads in that group**. Per prune event and KV group
`g`:

```
collateral(g) = fraction of evicted positions j such that
                at least one query head in group g placed mass > τ on j
                while the group aggregate score ranked j below the cut
```

This quantifies a cost that exists under GQA and not under multi-head attention:
**eviction granularity is coarser than attention granularity.** Every published
policy scores at KV-head granularity because that is where the cache lives, and
none report what it costs. Cheap: needs only the full model's attention and the
eviction decision. Pre-committed ordering in Amendment Q: R-KV's redundancy term
spreads budget over distinct content, so it should show **lower** collateral than
a heavy-hitter policy piling onto one cluster.

**Feasibility verified:** `compress()` already receives `attentions`, and eager
attention is already supported (`press_needs_eager_attention`,
`attn_implementation`). Eager is the slow path — it is why the H2O cells cap
generation at 8192 — so Task B is budgeted above the teacher-forced TV cells.

### 2.4 Storage design — CORRECTED 2026-09-12

**The original estimate was wrong by ~12×, in the direction that mattered**, and
the schema shape is changed as a result.

Per trace the record count is `n_layers × n_kv_groups × (L / interval)`:

| trace length | events | records | scalars (4 fields) |
|---|---|---|---|
| median, measured 2652 steps | 41 | 4,592 | 18,368 |
| generation cap, 8192 steps | 128 | **14,336** | **57,344** |

The original spec said "roughly 4.6k scalars for an 8k trace." 4,592 is the
**record** count at the **median** trace length, not the scalar count at the cap.
At the cap it is 14,336 records / 57,344 scalars.

That correction matters because it is the argument the design fits in headroom.
Measured at 90 bytes per JSON record:

| encoding | per trace (cap) | 800 traces |
|---|---|---|
| per-record dicts (as originally specified) | 1.29 MB | **1.03 GB** |
| **columnar float arrays (adopted)** | 0.52 MB | **0.41 GB** |

1.03 GB against ~1.6 GB of headroom is over half the remaining budget for one
cell, on a `runs/` tree that is only 2.5 GB in total. **`attn-profile-v1` is
therefore specified columnar**: four flat arrays plus an index header, not a list
of per-record objects. Same information, 0.41 GB, and the repeated key names —
which are most of the 90 bytes — disappear.

Schema `attn-profile-v1` is defined separately from `tv-profile-v1` so the Task 4
lesson (stored profiles lacked fields a later question needed) is not repeated.
The field set is fixed in Amendment Q **before** any cell runs, because a re-run
to add a field costs a full pass.

---

## 3. Task C: trajectory direction by SVD/PCA

### 3.1 Scope

PREREGISTRATION P.4 states, verbatim, "Latent-state divergence is out of scope."
This task therefore requires an explicit dated widening — **Amendment Q does
that** — and every result from it is exploratory by construction. The paper must
not present it as the mechanism.

Naively comparing hidden-state vectors is not well posed: the geometry is
non-metric and distances are dominated by a few high-norm dimensions.

### 3.2 On-manifold vs off-manifold displacement

Do not compare states; compare **where the displacement points**.

1. Fit a principal subspace from **full-cache** trajectories only, over many
   problems, on whitened hidden states at the layers where eviction bites. Top-`k`
   basis `U_k`.
2. For each compressed trajectory under teacher forcing, `δ_s = h_comp(s) − h_full(s)`.
3. Split:

```
δ∥ = U_k U_kᵀ δ      inside the full-cache reasoning subspace
δ⊥ = δ − δ∥          orthogonal to it
ω  = ‖δ⊥‖ / ‖δ∥‖     off-manifold fraction
```

`δ∥` means the compressed model moved to a state the full model could plausibly
have occupied. `δ⊥` means it moved somewhere the full model never goes —
degenerate, off-distribution, the regime of non-termination and repetition.

### 3.3 The falsifiable prediction, and the estimand fixed in advance

Verified against the artifacts 2026-09-12 (`add-256-*.json`, and truncation from
the m4 summaries):

| Policy | within-answered TV | termination shift | truncation |
|---|---|---|---|
| Random | 0.504 | 0.224 | 0.233 |
| StreamingLLM | 0.749 | 0.081 | 0.070 |
| H2O-style | 0.713 | 0.101 | 0.105 |
| R-KV | 0.450 | 0.149 | 0.135 |

Random fails by rambling and not terminating; StreamingLLM by answering
confidently and wrongly.

**Estimand, fixed now and not after seeing the numbers.** Four policies is four
points and a rank correlation over them is not a test. The pre-committed estimand
is therefore **a per-policy ordering claim with per-trace clustered CIs**, not a
correlation: specifically the pairwise contrast `ω(Random) − ω(StreamingLLM)`,
with a problem-clustered bootstrap CI, plus the full four-policy ordering reported
descriptively. The correlation form is **not** licensed at this n and is not to be
substituted later if the ordering looks favourable. Full statement in Amendment Q.

### 3.4 Storage and fitting

Hidden states are 28 × 3584 × L and cannot be written. Fit `U_k` by randomised
SVD over a subsample in a streaming pass, persist only the basis
(28 × 3584 × k floats, a few MB at k = 64), then project on the fly and store only
`{‖δ∥‖, ‖δ⊥‖, ω}` per step per layer, columnar. **Whitening statistics are fitted
on the full-cache condition only and frozen before any compressed trace is
projected**, or the comparison is circular.

---

## 4. The three questions

### 4.1 Q1: is the budget a strict constraint? — and the control, REDESIGNED

**The budget is a sawtooth, not a ceiling.** `DecodingPress` prunes every
`compression_interval` steps and the cache grows between prunes. Verified in
PREREGISTRATION: at `target_size=256` both presses prune to
`max_after_prune=256` and `max_before_prune=320` identically. Mean cache over a
cycle is `B + I/2 = 288`, 12.5% above nominal. Report the nominal `target_size`
and note the slack, which the paper does.

**The real threat.** P.2 branch A reports divergence peaking at phase 0, the prune
step, read as eviction-locking. The original control proposed holding mean cache
at 288 and varying amplitude:

| Arm | `target_size` | `interval` | mean | amplitude |
|---|---|---|---|---|
| S1 | 284 | 8 | 288 | 8 |
| S2 | 256 | 64 | 288 | 64 |
| S3 | 160 | 256 | 288 | 256 |

**That control is confounded, and it is confounded in exactly the way §1.1
diagnoses for Task A.** Mean cache is `B + I/2`, so fixing the mean at 288 forces
`B = 288 − I/2`: amplitude and floor are locked together by construction. S1
never falls below 284; S3 floors at 160. A collapse of phase excess at S1 is
therefore equally consistent with "the discontinuity was too small to lock to"
(the hypothesis) and with "the cache never fell far enough to do damage" (a
confound). The arms cannot separate them.

There is no third knob inside the eviction schedule: the parameterisation is
genuinely two-dimensional in `(B, I)` and mean is determined. So the identity is
broken the other way — **by dropping the mean-matching constraint and replacing
it with a one-sided dose argument.**

**Redesigned control: fixed floor, swept amplitude.**

| Arm | `target_size` B | `interval` I | floor | amplitude | mean cache |
|---|---|---|---|---|---|
| T1 | 256 | 16 | 256 | 16 | 264 |
| T2 | 256 | 64 | 256 | 64 | **288 — the deployed config, = P.2's cell** |
| T3 | 256 | 256 | 256 | 256 | 384 |

The floor is **identical at 256 in every arm**, matching the published cell, so
"the cache never fell far" is no longer available as an explanation of any
collapse. Mean cache now varies — and the direction closes the confound rather
than opening it: **T1 has the lowest mean cache (264) and is therefore the
harshest arm overall.** If phase excess collapses at T1, it cannot be because T1
was gentler; T1 compresses more on average than the deployed configuration. The
only remaining explanation is per-event amplitude. That is a one-sided argument
and it is why the sweep is run at fixed floor rather than fixed mean.

**Negative control on the instrument, added.** Split the prune into the two
operations it normally fuses: **selection** (tokens become unattendable, at step
`s`) and **removal** (the tensor actually shrinks, at step `s + d`, `d = 32`).
Because RoPE is applied at cache-write time and softmax over 320 keys with 64
masked is arithmetically identical to softmax over the remaining 256, the removal
step is a functional no-op by construction. **Pre-commit: phase excess anchored on
the removal step must be indistinguishable from zero.** If it is not, the phase
instrument is responding to something with no functional content and every phase
result including branch A is suspect — STOP and investigate before any paper edit.

**Analysis constraint, from the code.** `phase_amplitude_excess` raises
`ValueError("profiles mix compression intervals")`, so each arm is analysed
separately with its own sham at its own bucket count. Raw amplitude scales with
bucket count, so **the cross-arm read is on whether each arm's excess excludes
zero, not on the ordering of excess magnitudes across arms.** Raw and sham
amplitudes are reported per arm so the bucket-count effect stays visible. Fixed
in Amendment Q.

**What the redesigned control licenses, and what it does not.** It licenses a
statement about whether phase-0 locking survives when per-event eviction
amplitude is small, at a floor matched to the published cell. It does **not**
license any claim about horizon, about behaviour at fixed mean cache (mean varies
by construction), or about the delicensed staircase. It does not relicense
anything M.3 removed. This cell is aimed at a result already written into the
paper: **if T1's excess does not exclude zero while T3's does, P.2 branch A is
not refuted but must be qualified in the paper** — from "divergence is locked to
the eviction event" to "locked to eviction events above an amplitude threshold,
untested below 64" — and §4.4 plus the threats section must say so. That outcome
is written down here before the cell runs precisely because it is the
uncomfortable one.

### 4.2 Q2 and Q3: why reinsertion does not help

Facts from `restoration-diss256-v2`, re-verified 2026-09-12: matched arm
**−0.0013** against baseline **0.0383** (~3% of baseline, indistinguishable from
no help); random arm **−0.2485**; instrument gate 800/800, every reinserted entry
bitwise identical, zero anchor disagreements.

**The specificity control costs 6.49× baseline for Random and 5.49× baseline for
StreamingLLM.** Both ratios are stated explicitly and neither stands in for the
other. This is landmine 11: "Five to six times the baseline" once passed
`audit --strict` while being false, because the gate checks literals and not
English. A single "roughly six times" is the same error in a smaller dose.

**Positional information is not the problem.** RoPE is applied to K at cache-write
time, so a reinserted key carries its original rotary phase, and the gate confirms
bitwise identity.

**Proposed explanation: the damage is in the query stream, not the key store.**
Attention is `q·K`; reinsertion repairs `K`. By the trigger, `q_s` was computed
from a hidden state already perturbed by every preceding compressed step.
Restoring what the query can attend to does not restore the query. Under the Task
B decomposition this is exactly: reinsertion can repair `M` and cannot touch `Δ`.
Cross-instrument prediction — **if `Δ` is large relative to `M` at the trigger,
the measured null is explained** — so Task B and this question validate each other.

**Design: latency-swept restoration.** Task 2 reinserted at a single trigger with
a 200-step read window, measuring only the far regime. Sweep the lag
`ℓ ∈ {1, 4, 16, 64, 256}` decode steps between eviction and reinsertion, keeping
the four arms (matched, random, relaxed, none). Reads pre-committed in Amendment
Q (DECAY / FLAT-NULL).

The random-arm harm has a consistent reading worth stating: matched-budget
reinsertion must **evict live entries to make room**, so random reinsertion both
adds stale keys and destroys currently load-bearing ones. The cache behaves like
a working set, not a bag of tokens.

---

## 5. Build order

Ordered by defensibility per GPU-hour, not by the order the tasks were given.

| # | Cell | Depends on | Cost | Why here |
|---|---|---|---|---|
| 1 | Sawtooth control, redesigned (T1/T2/T3 + removal-lag negative control) | none, teacher-forced | low | Threatens an existing published result. Run before anything is built on branch A. |
| 2 | Onset-staggered replay, `D_tv` | onset gating + test | low–med | The experiment M.3 pointed at. Highest single payoff. |
| 3 | Attention channels `M`, `Δ`, collateral | `attn-profile-v1` | med–high (eager) | Self-control is free; GQA collateral is novel alone. Eager attention is the slow path. |
| 4 | Latency-swept restoration | Task B for the quantitative version | med | Converts a null into a timescale. |
| 5 | `D_acc` surface | cell 2 read first | high, free-running | Only worth the GPU time if `D_tv` shows structure. |
| 6 | On-manifold split | P.4 widening; basis-fitting pass | high | Most novel, least defensible, weakest n. Last. |

Schema work landed before cells 3–6 run: `attn-profile-v1` field set and the
latent projection record format. A missing field costs a full re-run.

---

## 6. Open items requiring a human decision

1. **The sawtooth control can go against us.** If T1's excess does not exclude
   zero, branch A needs qualifying in a paper already written. It is queued first
   anyway, and the qualifying language is drafted in Amendment Q before the read.
2. **Task C's n is weak and the estimand is now fixed** (§3.3): per-policy
   ordering with clustered CIs, not a four-point correlation. Not to be swapped
   later.
3. **None of this lands in the current paper.** ICLR 2027 abstracts are due
   2026-09-18 and full papers 2026-09-25 (verified against the ICLR 2027 author
   guidelines). This is a second paper, or at most an appendix pointer.

---

## 7. Verification pass, 2026-09-12

Everything checkable in the original draft was checked against the repo and the
remote. Correct as written: the Task 3 null and pooled ρ; **Q** as the next free
amendment letter (A–P all used); P.4's "Latent-state divergence is out of scope"
verbatim; the restoration numbers; all eight ADD values to 3 dp; the truncation
column including the exact tie `0.2325 → 0.233` under the repo's own formatter;
the 256/320 sawtooth; the GQA head counts and the no-pre-slice-representation
argument; and the ~1.6 GB headroom.

Corrected here: the Task B storage arithmetic (§2.4) and the single-ratio
"roughly six times" (§4.2). Redesigned here: the Q1 control (§4.1). Added from
the repo rather than the draft: the `instrumented=True` requirement (§1.2), the
eager-attention cost (§2.3), and the mixed-interval analysis constraint (§4.1).
