"""Reproduce the external reviewer's CPU findings against the CURRENT code.

Pre-fix output (commit 3cc44da) is kept verbatim in probe_findings_3cc44da.txt;
running this now shows which findings are fixed.

Run: .venv/bin/python results/meeting3/probe_findings.py
Pure-Python probes only (torch-dependent restoration probes run on the remote).
"""

import inspect
import sys

sys.path.insert(0, "src")

from kvcache.metrics.attn_channels import collateral_fraction  # noqa: E402
from kvcache.metrics.latent_split import (  # noqa: E402
    FrozenBasis,
    omega,
    split_displacement,
)
from kvcache.metrics.profile_shape import (  # noqa: E402
    phase_amplitude,
    phase_amplitude_excess,
)
from kvcache.presses import _stash_candidates_by_age  # noqa: E402

print("== F1 phase amplitude is invariant to circular rotation ==")
prof = [1.0] + [0.1] * 63
shifted = prof[-32:] + prof[:-32]
print(
    "  selection-anchored amplitude:",
    round(phase_amplitude(prof), 6),
    "peak at",
    prof.index(max(prof)),
)
print(
    "  32-shifted amplitude        :",
    round(phase_amplitude(shifted), 6),
    "peak at",
    shifted.index(max(shifted)),
)

print("== F2 legitimate zero excess reported as None ==")
const = [([0.5] * 640, 0, 64) for _ in range(4)]
r = phase_amplitude_excess(const, n_resamples=50, seed=0)
print(
    "  constant profiles: amplitude=",
    r["phase_amplitude"],
    "sham=",
    r["phase_amplitude_sham"],
    "excess=",
    r["excess"],
    "CI=",
    r["excess_ci95"],
)

print("== F3 FrozenBasis accepts a non-orthonormal basis ==")
try:
    b = FrozenBasis(
        [(2.0, 0.0)],
        mean=[0.0, 0.0],
        scale=[1.0, 1.0],
        condition="fullcache",
        fitted_on_n=1,
    )
    par, perp = split_displacement([1.0, 0.0], b)
    print("  STILL PRESENT: (parallel, perp) =", (par, perp), "(true projector: (1,0))")
except ValueError as exc:
    print("  FIXED (Amendment R.6): construction refused ->", exc)
print(
    "  omega(0,0) =",
    omega(0.0, 0.0),
    "| omega(0,3) =",
    omega(0.0, 3.0),
    "(legacy API; omega_with_reason now distinguishes them)",
)

print("== F4 collateral has no rank/cut condition ==")
print("  signature:", inspect.signature(collateral_fraction))

print("== F5 requested lag vs selected age (_stash_candidates_by_age) ==")


class _K:  # minimal stand-in for a stash tensor
    def __init__(self, n):
        self.shape = (1, 1, n, 1)


class _B:
    pass


steps = [0] * 64 + [64] * 64 + [128] * 64 + [192] * 64
for lag in (1, 4, 16, 64, 256):
    base = _B()
    base.stash_keys = _K(len(steps))
    base.stash_steps = steps
    base.n_reinsert = 64
    base.reinsert_lag = lag
    cand = _stash_candidates_by_age(base, 192)
    # matched arm takes the most recent n_reinsert of the candidate list
    chosen = sorted(cand)[-64:]
    ages = sorted({192 - steps[i] for i in chosen})
    print(f"  requested lag {lag:>3} -> selected ages {ages}")
