# results/diag/ — archived analysis artifacts

Every number in the paper resolves to a file here. `paper_numbers.json` is built
from these by `scripts/build_paper_numbers.py`; `make audit` refuses any
result-precision literal in the paper that does not trace back to one.

## Naming conventions

A status tag in the filename, introduced by an underscore, marks an artifact
that is retained but is **not** the one to cite:

| suffix | meaning |
|---|---|
| `_INVALID_<reason>` | produced by an instrument later shown to be defective; invalidated wholesale, never patched, kept so the disclosure is checkable |
| `_PRE-AMENDMENT-K` | measured against the unrestricted full-cache bank, before Amendment K's population discipline; the canonical re-derivation is the same basename under `amendment-k/` |
| `_UNRESTRICTED` / `_RESTRICTED` | whole-trace vs above-floor window (Amendments M.3, P.2-v2) |

An artifact with no status tag is current. Identity-bearing basenames are unique
across this tree at any depth — asserted by
`tests/test_paper_numbers_properties.py`. The three exceptions are *role* names
that describe a file's position inside its own unit directory and are never
cited bare: `summary.json`, `records.json`, `table.md`.

## Retained artifacts that no document references

Unreferenced is not the same as worthless. This project deliberately keeps
invalidated cells, withdrawn statistics and superseded values, because the
honesty trail is checkable only if the superseded thing still exists. Two files
here are referenced by no document and are kept on that basis:

- **`amendment-p/shape_ext_diss256.json`** — profile-shape statistics over the
  extended replay pass with **both budgets pooled**. Superseded by the
  budget-split pair (`shape_ext_b256.json`, `shape_ext_b2048.json`) once
  Amendment P.3 established that pooling budgets conflates a budget effect with
  the quantity of interest. Kept as the pooled variant that motivated the split.

- **`amendment-k/classb/gate_extended_replay.json`** — the pass receipt for the
  v1-vs-v2 TV-emitter equivalence gate run before the extended replay pass
  (Amendment P.4). A gate that passed and was therefore never written up. Kept
  because a gate with no surviving receipt is indistinguishable from a gate that
  was never run.
