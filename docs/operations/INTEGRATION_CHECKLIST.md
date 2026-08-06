# Integration checklist — what each queued run touches, and what to write

**Purpose.** When a queued run finishes, this file plus `make integrate UNIT=<u>`
should be enough to land it without a fresh reasoning pass. `make integrate`
gets the numbers in and prints what moved; this file says what the numbers
*mean*, because that was pre-registered and must not be re-derived at harvest
time — re-deriving a read after seeing the data is how a pre-registration turns
into a rationalisation.

**Order for every unit:**

1. `make parity` — confirm the remote state you are harvesting from matches.
2. `make integrate UNIT=<unit>` — harvest, recatalogue, gate, figures, PDF, and
   a printed before/after diff of every number that moved.
3. Open this file at the unit. Take the branch the numbers select. **Apply the
   wording as written.**
4. `make lint test dryrun audit parity`, each bare.
5. Dated note in `docs/results/PAPER_FINDINGS.md` §4-LOG (never in §4-CURRENT — that is
   generated), top bullet in `docs/project/context.md`, commit naming the unit, push to
   `backup`.

**Standing constraints for every entry below.** Truncations stay in every
denominator, scored incorrect. Every absolute budget is reported with its
realized compression ratio. Layer-1 primary bounds are deterministic imputation
ranges carrying no sampling uncertainty and are never described as significance.
Divergence is not damage. If a result fits no branch, report it plainly and add
a dated note — do not force it into one.

---

## 1. `grid-diss512` / `readout-diss512` / `grid-diss1024` / `readout-diss1024`

**The confirmatory test of the resolution collapse.** Amendment L pre-commits the
prediction; Amendment M.5 restates it. This is the highest-value run in the queue
because it converts §4.1 from post-hoc pattern-spotting into a prediction tested
on data that did not exist when it was made.

**Reads needed after both arms of a budget land:**

```
ssh kvcache-gpu 'cd ~/kv-caching && R=/home/sk10741/kvcache-storage/runs && \
  .venv/bin/python scripts/report_resolution.py --runs-root $R \
    --cell random:grid-diss512:readout-diss512 \
    --cell streamingllm:grid-diss512:readout-diss512 \
    --budget 512 --label math500-512 --out /tmp/res_512.json'
```
Copy to `results/diag/amendment-k/classb/resolution_math500_512.json`, add the
`res512` block to `build_paper_numbers.py` beside `res256`/`res2048`/`resaime`,
rebuild.

**Sites touched:** `\label{sec:resolution}` §4.1 prose and `tab:resolution`
(add a row per budget); `PROPOSED_REFRAME.md` §1 (the title recommendation is
explicitly conditional on this test); PREREGISTRATION Amendment L's result table.

### Branch A — CONFIRMS

*Selected when:* the readout/free-run spread ratio at both 512 and 1024 falls
strictly inside (0.12, 0.78), and the ordering is monotone in budget
(256 > 512 > 1024 > 2048).

Add the rows to `tab:resolution`. Then in §4.1, replace the closing paragraph's
post-hoc caveat with:

> We flagged this pattern as post-hoc when we first measured it, and
> pre-registered the prediction it makes: that the readout-to-free-run spread
> ratio at budgets 512 and 1024 would fall strictly between the budget-256 and
> budget-2048 values, monotone in budget (Amendment L). Both cells were then run.
> The prediction holds: the ratio falls from {R256} at 256 through {R512} and
> {R1024} to {R2048} at 2048. The resolution loss is therefore not an artifact of
> the two regimes we happened to look at first — it is a graded property of the
> protocol, tested on data that did not exist when the prediction was made.

Downgrade the "exploratory" label to "confirmed on a pre-registered prediction;
the original observation was post-hoc and is recorded as such in Amendment L".
**Do not delete the post-hoc disclosure** — that it was confirmed does not change
how it was found. In `PROPOSED_REFRAME.md`, the title recommendation flips: the
headline title becomes defensible, because the finding is no longer post-hoc.

### Branch B — REFUTES

*Selected when:* the ratio at 512 or 1024 falls outside [0.12, 0.78], **or** the
ordering is non-monotone in budget.

**This is the branch that matters, and it is written now so it cannot be softened
later.** Add the rows to `tab:resolution` exactly as measured. Then in §4.1:

> We pre-registered the prediction this reading makes — that the spread ratio at
> budgets 512 and 1024 would fall strictly between the 256 and 2048 values,
> monotone in budget (Amendment L) — and ran both cells to test it. **It does not
> hold:** {describe, e.g. "the ratio at 512 is {R512}, outside the predicted
> interval" or "the ordering is non-monotone: {values}"}. The resolution loss is
> therefore not the graded property we proposed. What survives is the endpoint
> observation: readout separates the four presses at budget 256 and cannot
> separate three of them on AIME-24, which is measured and stands. The mechanism
> we inferred connecting those endpoints does not.

Then:
- **§4.1 stops being a primary contribution.** Move it after the decomposition
  and retitle to "Held-trace resolution at two budgets", stating the endpoints
  without the graded claim.
- **`PROPOSED_REFRAME.md`: withdraw the title recommendation.** Both proposed
  titles are built on the graded reading; neither survives. Keep the current
  title.
- **Contribution reorder is withdrawn** — resolution collapse cannot be
  contribution 1.
- Record the refutation in Amendment L with the date, and in
  `docs/results/PAPER_FINDINGS.md` §4-LOG.

**Do not** reinterpret a non-monotone result as "noisy but consistent". The
prediction had an interval and an ordering; failing either is a refutation.

### Branch C — UNDERPOWERED

*Selected when:* the spread-difference CI at 512 or 1024 spans zero so widely
that the ratio is not usefully bounded.

Report the point estimates with their intervals and say the test was
inconclusive at this sample size. Do **not** count an inconclusive cell as
confirmation. §4.1 keeps its post-hoc label unchanged.

---

## 2. `grid-hottcap`

**Amendment N.1.** Cap 2048 (HOTT's, verified from their text), everything else
ours. **Greedy decoding is NOT matched** — `generation.py` hardcodes
`do_sample=True` — so this isolates the cap alone.

**Pre-written caveat sentence. It must appear wherever this result is stated, and
it is written here so it cannot be forgotten at integration time:**

> This cell matches HOTT's 2048-token generation cap and not their greedy
> decoding, which our generation path does not support; sampling remains
> $T{=}0.6$, top-$p$ $0.95$. It therefore measures how much of the baseline gap
> the cap alone accounts for, and any residual gap is not attributable to the
> cap.

**Sites touched:** §2's HOTT paragraph (the "different decode-length and
sampling" parenthetical); §5's *"we do not reproduce HOTT's baseline"* item in
the What-we-have-not-measured paragraph; the validity kit if the result is clean.

**Read:** compare `Acc_FC` under the cap against our 0.882 and HOTT's 0.47, and
report the truncation rate alongside — it is the mechanism.

- **EXPLAINED** (within ~0.05 of 0.47): replace §2's parenthetical with the
  measurement — *"the gap is a decode-length artifact: under HOTT's 2048-token
  cap our own full-cache accuracy falls to {X} with {T} truncation"* — plus the
  greedy caveat. Remove the corresponding item from §5's not-measured list, since
  it is now measured.
- **PARTLY EXPLAINED** (moves substantially, stops short): state both numbers and
  call the remainder unexplained. **Do not attribute the remainder to greedy
  decoding** — that variable was not tested.
- **NOT EXPLAINED** (stays near 0.882): §2's parenthetical becomes a claim we
  tested and failed to support. Rewrite to *"we cannot account for the
  difference in baselines; it is not explained by decode length"* and keep the
  §5 item, strengthened.

---

## 3. `fc-aime24-llama8b` then `grid-aime24-rkv-published`

**Amendment N.2/N.3.** R-KV in the model and benchmark of its published headline.

**Bank gate first (N.3):** accuracy must land in [0.30, 0.70]. Outside that band,
**stop** and investigate the bank as an instrument problem before reading any
cell against it.

**Sites touched:** §2's R-KV sentence (*"tested in an adjacent configuration"*);
§5's *"No >100% regime observed"* paragraph; §5's not-measured list; `tab:audit`
if the audited claim changes.

- **REPLICATES** (r ≥ 1.0, difference CI includes or exceeds zero): §5 gains
  *"In R-KV's own published configuration — R1-Distill-Llama-8B on AIME-24 at
  {ratio}% cache — our certified port reaches {r}× full-cache accuracy,
  reproducing the published >100% result. Our own no->100%-regime finding is
  therefore specific to 7B on MATH-500, exactly as Amendment E anticipated."*
  §2's "adjacent configuration" hedge is replaced by the direct test. **This
  strengthens their claim, and we report it plainly.**
- **DOES NOT REPLICATE** (r < 1.0, difference CI excluding zero): state it with
  maximum care — *"our parity-certified port (retained-set Jaccard 1.0 against
  the authors' code) reaches {r}× full-cache accuracy in the published
  configuration, and we could not reproduce the reported 105%."* Report our
  accuracy, theirs, the CI, and the port certification **together**. Say "could
  not reproduce", never "the claim is false".
- **INCONCLUSIVE** (CI spans; n=30 problems): say so, and keep §2's "adjacent
  configuration" wording. AIME-24 is 30 problems and may simply lack power.

---

## 4. Not queued — `10.1` noise control

Reserved for supervised work. Its instrument gate
(`noise_control.match_gate`) must pass **before** any accuracy is read, and per
Amendment M.1 a failed gate means the experiment did not run. If it ever lands
and the GENERIC branch is selected, that outcome goes in the abstract — it
weakens the mechanism reading and the pre-registration says so.
