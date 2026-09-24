# Storage footprint — measured 2026-09-25 00:20 Dubai (before any Meeting #3 GPU run)

`du -sk` on the GPU host. **Combined** uses `du -sck` over both trees, which
counts each hardlinked inode once. Budget: 70 GiB, the user-confirmed personal
allocation. Margin: 1 GiB held back by the scheduler.

| Tree | Size | Notes |
|---|---|---|
| `~/kvcache-storage` | **68.858 GiB** | cache 65.543 GiB (model weights, uv cache), runs 3.314 GiB, data 0.001 GiB |
| `~/kv-caching` (source + `.venv`) | 5.851 GiB counted alone | `.venv` 5.635 GiB, mostly hardlinks into the uv cache under `~/kvcache-storage` |
| **Both, hardlinks counted once** | **69.190 GiB** | the source tree adds only 0.332 GiB not already counted |

**Correction.** R.8.5 and `MEETING3_RESULTS.md` said `~/kv-caching` is
"11.2 GiB, mostly `.venv`". That number came from `du --count-links`, which
counts hardlinks more than once. The combined 69.19 GiB, and hence the
0.81 GiB headroom under the stricter scope, were measured correctly then and
now.

## Headroom under each scope

| Scope | Used | Headroom | Usable after the 1 GiB margin |
|---|---|---|---|
| Storage only (what the guard measures today, R.8.5) | 68.858 GiB | 1.142 GiB | **0.142 GiB** |
| Storage + source (not resolved by the owner) | 69.190 GiB | 0.810 GiB | **−0.190 GiB**: every new cell BLOCKED |

## Queued outputs (not yet written)

| Job | Declared `est_output_gb` | Expected from measurements |
|---|---|---|
| `q6-basis-fit` | 0.03 | ~27 MB (float32 basis, mean, scale). Transient peak is the same: the npz is written to a temp file and renamed, so only one copy exists. |
| `capture-b256`, `capture-b2048` | 0.03 each | ~19 MB each (~20 KB row + ~28 KB npz × 400) |
| `dacc-t2048`, `-t512`, `-t0` | 0.03 each | ~17 MB each (from grid-diss256/2048 rows) |
| `exact-lag-r5` | 0.01 | < 1 MB (200 rows) |
| `fork-control-random`, `-slm` | 0.01 each | ~5 MB each (75 rows incl. TV profiles) |
| `t1-tv512`, `t1-tv1024` | 0.01 each | ~6 MB each (rows plus ~3 KB summary profiles) |
| **Total** | **0.23 GiB declared** | **~0.137 GiB expected** |

Under the storage-only scope, the expected total (~0.137 GiB) fits inside the
0.142 GiB usable, with about 5 MB to spare. The declared total (0.23 GiB)
does not.

The scheduler admits cells one at a time. At each admission it compares:
- the MEASURED current usage, plus
- the declared remainder of every running cell, plus
- the candidate's declared remainder.

So the last cells (`t1-tv512` and `t1-tv1024`) may be reported
`BLOCKED <name>: storage deficit … (<bytes>)` if the earlier cells write more
than expected. The scheduler logs that deficit exactly and keeps running the
cells that fit. Nothing is deleted to make room. The environment, caches and
allocation are unchanged.

**Open owner decision.** Whether the 70 GiB covers `~/kv-caching`. If it does,
nothing new can run until 0.19 GiB plus each cell's output is freed, which
needs the owner's instruction.
