# M4 Decomposition Pilot

Residual is **accuracy change unexplained by on-reference readout**. It is not a measured cause.

| Press | Budget | Free acc | Readout acc | Total Δ | Readout Δ | Residual | Lost mass | Trunc | Mean AUC-TV | Frac TV>0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 2048 | 1.00 | 1.00 | +0.00 | +0.00 | +0.00 | 0.00 | 0.00 | 0.002 | 0.02 |
| random | 256 | 0.45 | 0.90 | -0.55 | -0.10 | -0.45 | 0.55 | 0.20 | 0.085 | 0.31 |
| streamingllm | 2048 | 0.90 | 1.00 | -0.10 | +0.00 | -0.10 | 0.10 | 0.05 | 0.004 | 0.02 |
| streamingllm | 256 | 0.20 | 0.45 | -0.80 | -0.55 | -0.25 | 0.80 | 0.15 | 0.078 | 0.28 |
