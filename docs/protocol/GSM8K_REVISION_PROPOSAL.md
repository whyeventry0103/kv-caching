# GSM8K revision proposal (for a human pin)

**Prepared 2026-10-05.** Status: PROPOSAL. `configs/benchmarks.yaml` is NOT pinned: the TODO(human) rule reserves dataset revisions for a person, and `kvcache.datasets.load_dataset_guarded` refuses to load an unpinned revision. A test checks that the refusal holds.

## Candidate

| Field | Value | Source |
|---|---|---|
| Handle | `openai/gsm8k`, config `main`, split `test` | `configs/benchmarks.yaml` |
| Revision (main) | `740312add88f781978c0658806c59bc2815b9866` | `https://huggingface.co/api/datasets/openai/gsm8k/revision/main`, queried 2026-10-05 (metadata only, nothing downloaded) |
| Last modified | 2026-03-23T10:18:13Z | same API |
| Files | `main/test-00000-of-00001.parquet` (plus train and socratic) | same API |
| License | MIT | dataset tags |

## What a person does to pin it

1. Check the sha on the dataset page, and check that the test split has 1,319 rows as published. That row count is from the dataset card and was not checked here.
2. Add `revision: 740312add88f781978c0658806c59bc2815b9866` under `gsm8k` in `configs/benchmarks.yaml`, and commit it with your name in the message.
3. Authorize the download separately. It needs `--allow-dataset-download` on the remote, and the dataset is small (two parquet files).

## Prepared on CPU (no download, no pin)

- `kvcache.datasets._gsm8k_problems` and `gsm8k_gold`:
  - the gold is the number after `####`, with thousands separators removed;
  - a missing marker or a non-numeric gold stops the load;
  - pids are `gsm8k/test/<index>`;
  - `subset` takes `{count, seed}` for the frozen seeded sample, or an integer for the engineering pilot.
- Tests in `tests/test_datasets.py`.

## Still to decide before any GSM8K outcome exists

- **Frozen sample.** Size from `results/diag/paper_plan/sample_size.json` for the chosen primary endpoint; seed recorded in the registration.
- **Pilot.** Fixed 8-problem pilot, excluded from claims, run first.
- **Budgets.** GSM8K traces are short, so a 1024 budget may never evict. Check the realized compression ratio on the pilot before freezing budgets.
