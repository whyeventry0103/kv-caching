# Qwen3 bank: secondary scoring sensitivity and gate resolution memo

**Dated 2026-10-05.** This was written AFTER the 31 bank errors were inspected (S.4 note of 2026-10-04) and BEFORE any dependent Qwen3 cell was read. The Qwen3 grid is 138/400 and its readout and reverse cells have 0 rows. The secondary score is therefore a post-inspection sensitivity analysis, not a prospectively registered grader.

**What it does not change.**
- The frozen primary grader. The primary bank score stays 169/200 = 0.845.
- The S.4 gate outcome, which stays **FAIL** against [0.85, 0.99].

## 1. Code and artifacts

| Piece | Path |
|---|---|
| Rules | `src/kvcache/grading_sensitivity.py` (`SECONDARY_VERSION = "format-aware-v2"` since the review of fbf869f; v1 described in §2, changes in §5) |
| Tests | `tests/test_grading_sensitivity.py`: a positive and an adversarial negative per rule, both transition directions |
| Audit runner | `scripts/audit_qwen_scoring.py --runs-root <runs> --bank {qwen3_4b,r1_qwen7b}` |
| Outputs | `results/diag/qwen_scoring/bank_audit_{bank}.json` and `.conditions.tsv` (v2); `bank_audit_{bank}_v1.*` (v1) |

Each TSV row has:
- the stored and recomputed primary score;
- the boundary status;
- both extracted answers;
- the secondary verdict, rule and transition;
- the trace sha256;
- the last 400 characters of the trace, enough to check the score without the server.

## 2. Rules, v1 (fixed list, applied identically to every row of every compared arm; superseded in two places by §5)

Three questions are kept separate.

**Boundary.** The categories are:
- `closed_with_answer`
- `closed_no_answer`
- `answer_inside_unclosed_thinking`
- `truncated_no_answer`
- `no_answer`

An answer that appears only inside unclosed `<think>` is never treated as final. Such a row is **AMBIGUOUS**, whichever way the primary scored it.

**Extraction.** The answer comes from the text after the last `</think>`.
- Multiple boxes are joined only when they form one terminal block. That means:
  - only separators lie between them (commas, "and", `\quad`, math delimiters);
  - only closing delimiters follow the last box.
- Boxes separated by prose are never joined; the last box wins.
- Boxes inside the reasoning are never used.
- No rule looks at the gold answer to decide how to extract.

**Equivalence.**
1. `is_equiv` (the frozen grader) first.
2. Then the rules below, in order. Each one is recorded by its ID.

| Rule ID | What it does | Guard against false positives |
|---|---|---|
| `TEXT_WRAPPER` | Strips `\text{}`, `\textbf{}`, `\mathrm{}`, `\mbox{}` and `\operatorname{}` wrappers on both sides, keeping their content. Also accepts a case-insensitive match of a purely alphabetic answer. | Content must still be equal under `is_equiv`; nothing is dropped except the wrapper |
| `UNORDERED_LIST` | Order-insensitive, multiset element matching (`is_equiv` per element) | Only when both sides are bare comma lists (no enclosing brackets, so tuples and intervals keep their order) AND the PROBLEM text asks for an unordered list ("separated by commas", "all solutions/values/roots…", "enter all") AND does not ask for an order ("in order", "ordered", "increasing", "decreasing") |
| `BASE_SUBSCRIPT` | Gold `N_b` and prediction `N` → **AMBIGUOUS**, never correct | Whether the bare numeral answers "in base b" is a semantic judgement this rule list does not make |

**Ambiguous rows.**
- In the point estimate they keep their primary score.
- They are reported as an ambiguity range: all ambiguous incorrect / all ambiguous correct. This is not a confidence interval.

**Not added, on purpose.**
- Symbolic or numeric equivalence beyond `is_equiv`: `crosscheck_equiv` exists but has documented false negatives on structured answers.
- Per-problem exceptions.
- Model-as-judge adjudication.

## 3. Results (FINAL; `reproduce` lines are inside each JSON)

| Bank | Primary (frozen) | Recompute mismatches | Secondary point (v1; v2 identical, §5) | Ambiguity range (not a CI) | Ambiguous | Correct→incorrect |
|---|---|---|---|---|---|---|
| Qwen3-4B, 50 problems × 4 seeds | 169/200 = **0.845, gate FAIL** | 0 | 181/200 = 0.905 | [0.900, 0.960] | 12 (8 unclosed thinking, 4 base subscript) | 0 |
| Qwen-7B R1-Distill anchor, grid100 × 4 seeds | 354/400 = 0.885 (in band) | 0 | 368/400 = 0.920 | [0.920, 0.930] | 4 (2 + 2) | 0 |

**The S.4 error categories reproduce exactly** from the mechanical mapping: 4 text wrapper, 2 set order, 6 multi-box, 4 base subscript, 11 unclosed thinking, 4 wrong value.

**Qwen3 transitions:**
- 12 incorrect→correct, all checked by reading the trace tails:
  - 4 × `\text{Evelyn}` (algebra/1349);
  - 4 × the {1, −2} set (intermediate_algebra/1388): two given in another order, two boxed separately;
  - 4 × "3, 5, 7" boxed separately (intermediate_algebra/428).
- 12 ambiguous:
  - 4 × "52" for `52_8` (number_theory/864);
  - 8 answers inside unclosed thinking, one of which the primary scored correct.

**Who checked the rescued rows.** Claude checked every one, from the 400-character tails. The check was not blind: the S.4 note had already been read. A check by the researcher or another person is still required (plan §8.3).

**The format effect is not Qwen3-specific.** The same rules move the anchor model by +0.035 (Qwen3: +0.060). The same problems appear in both lists (algebra/1349, intermediate_algebra/1388 and /428). So any cross-model comparison must score both arms with the same grader, primary against primary and secondary against secondary.

## 4. Gate resolution memo (decision needed: owner and supervisor)

The registered rule was a sanity band, and the bank misses its lower edge by one generation (169 against 170). Under the frozen grader the gate FAILS, and that stays recorded. The investigation found three things:

- (i) the shortfall is explained by documented grader-format mismatches;
- (ii) those mismatches also affect the anchor model;
- (iii) 4 rows are genuine model errors and 11 are boundary failures (4 truncated, 7 answered inside unclosed thinking).

**Options:**

- **(a) Qualified exploratory, primary score.** Read the Qwen3 cells under the frozen grader as *exploratory*. Every Qwen3 sentence names the FAIL and the 0.845. The secondary-score result is reported alongside as a sensitivity.
- **(b) Hold primary; read under a distinct secondary protocol.** Keep primary-score Qwen3 analyses held. Read the cells only under this secondary protocol, labelled as a post-inspection protocol, with both arms rescored.
- **(c) Hold all.** Hold every Qwen3 read and report the bank investigation only.

**Recommendation: (a).** Reasons:
- The FAIL is a one-generation miss with a documented, model-independent cause.
- Option (a) keeps the frozen grader as the only primary, so it cannot be read as an override.
- Option (b) would make a post-inspection grader the reading basis for a whole model, which the plan advises against.

Either way:
- No Qwen3 claim may be described as confirmatory.
- The anchor comparison uses the same grader on both sides.

Until the owner chooses, the Qwen3 cells may keep generating (already authorized). Nothing dependent on them is read.

## 5. Version 2 (2026-10-05, after the independent review of fbf869f)

v2 is still a post-inspection sensitivity. The primary grader, 169/200 and the gate FAIL are unchanged.

**Two holes in v1, found by the review:**
1. *Commas were read as unordered.* v1 treated "separated by commas" as evidence that order does not matter. But x and y, coordinates, and numerator/denominator are ordered roles listed with commas.
2. *Later corrections were ignored.* v1's multi-box guard read only up to the first blank line, so a later paragraph that corrected the boxes was missed.

**v2 rules:**
- `UNORDERED_LIST` needs POSITIVE unordered wording: "all solutions/values/roots/zeros/numbers/integers/answers", "enter all", "list all".
- Ordered-role wording makes a reordered list incorrect (`ORDER_REQUIRED`): in order, ordered, increasing, decreasing, coordinate, numerator, denominator, respectively, in that order, in the form.
- With neither kind of wording, the row is ambiguous (`ORDER_UNCERTAIN`).
- A multi-box block counts as the answer only if nothing but closing delimiters follows it in the WHOLE final region. Otherwise the row is ambiguous (`MULTIBOX_NOT_TERMINAL`), never a confident fallback.

**Tests.** `tests/test_grading_sensitivity.py` covers the reviewer's cases verbatim (x/y, coordinates, numerator/denominator, the later "Actually … 11" correction). On the v1 code 5 tests fail; on v2 all 19 pass.

**Rescore (both banks, on the host, `--previous` = the v1 table).**

| Bank | Frozen grader | v2 point | Ambiguity range (not a CI) | Rows changed vs v1 |
|---|---|---|---|---|
| Qwen3-4B | 169/200 = 0.845, FAIL | 181/200 = 0.905 | [0.900, 0.960] | **0** |
| Qwen-7B anchor | 354/400 = 0.885 | 368/400 = 0.920 | [0.920, 0.930] | **0** |

Neither hole touched these banks. They mattered for future outputs.

**Artifacts.** The v1 artifacts are kept as `bank_audit_*_v1.*`. v2 is `bank_audit_*.json`, which carries `changes_vs_previous`.

**Human review sheet.** `results/diag/qwen_scoring/human_review_v2.tsv` lists every rescued or ambiguous row of both banks: 42 rows (Qwen3: 12 rescued, 12 ambiguous; anchor: 14 rescued, 4 ambiguous). Each row has its gold, both answers, the rule, the trace hash and the trace tail, plus blank reviewer columns. Secondary scores support substantive claims only after a person fills it in.

**Gate disposition.** Unchanged and still the owner's decision (§4). The independent review also recommends option (a), recorded as a documented exception, not a passed gate.

## 6. Decision recorded (2026-10-06)

**The owner chose option (a)** in session. It is recorded in PREREGISTRATION ("Owner decisions and registrations, 2026-10-06").

- Qwen3 cells are read under the frozen grader as EXPLORATORY.
- The gate FAIL (0.845) is stated beside every Qwen3 result.
- The v2 secondary score appears only after a person completes `human_review_v2.tsv`.

Qwen3 generation is held behind B=1024 (queue `blocked:`).
