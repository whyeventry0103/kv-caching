# configs/paper — proposed design manifests (execution plan §7.2)

These are PROPOSED and HELD. No runner reads them and no scheduler queue
references them; nothing here is authorized GPU work. A manifest becomes
runnable only after (1) its gate list passes, (2) its cost sits inside an
existing authorization or receives one approval, and (3) a dated
PREREGISTRATION entry freezes it before any outcome row exists.

- `protected_selector_q7_math_b256_v1.yaml` — WP5 discovery extension.
- `confirm_qwen3_4b_math500_v1.yaml` — WP6 model transfer.
- `confirm_q7_gsm8k_v1.yaml` — WP6 task transfer.
- `cost_inputs.yaml` — measured per-row costs used by
  `scripts/plan_paper_sample_size.py cost`.
