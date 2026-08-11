"""Per-step divergence measures on the next-token distribution (Amendment P.4).

**Softmax only.** Raw logits are excluded on purpose: they are uncalibrated
across positions, arbitrary in scale, and a temperature difference alone would
make two identical policies look different. Latent/hidden-state divergence is
out of scope; if it is ever added it is descriptive and can never sit adjacent
to an accuracy number.

Four measures, each answering something the others cannot:

* **TV** -- retained for continuity with the published profile, and it has a
  behavioural reading: under speculative decoding, 1 - acceptance ~= TV.
* **JSD** -- bounded and symmetric. Unlike KL it does not blow up exactly when
  the compressed model commits confidently to a token the full model gave
  near-zero mass, which is the case we most need to measure. Reported in bits
  (log base 2), so it lies in [0, 1].
* **top-1 agreement** -- did the argmax change. Binary, and it captures the
  behaviourally consequential part of a shift that TV smears across the vocab.
* **rank displacement** -- where the full-cache argmax lands under compression.
  0 means still top-1. Sharper than top-1 for localising onset, because it
  registers the token sliding from rank 0 to rank 3 before it ever leaves the
  top spot.

KL is computed in **both** directions or not at all; a single direction invites
reporting whichever is finite. Non-finite values are emitted as ``None`` rather
than silently clamped, so a divergence is visible as one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

MEASURES = ("tv", "jsd", "top1_agree", "rank_displacement", "kl_fwd", "kl_rev")


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def measures_from_probs(full: Sequence[float], test: Sequence[float]) -> dict[str, Any]:
    """Reference implementation over two probability vectors.

    Pure Python so the CPU-only test gate can exercise the actual formulas
    rather than a mock. The torch path below must agree with this.
    """
    if len(full) != len(test):
        raise ValueError("distributions must have the same support")
    tv = 0.5 * sum(abs(p - q) for p, q in zip(full, test, strict=True))
    jsd = 0.0
    kl_fwd = 0.0
    kl_rev = 0.0
    for p, q in zip(full, test, strict=True):
        m = 0.5 * (p + q)
        if p > 0 and m > 0:
            jsd += 0.5 * p * math.log2(p / m)
        if q > 0 and m > 0:
            jsd += 0.5 * q * math.log2(q / m)
        if p > 0:
            kl_fwd = kl_fwd + p * math.log2(p / q) if q > 0 else math.inf
        if q > 0:
            kl_rev = kl_rev + q * math.log2(q / p) if p > 0 else math.inf
    argmax_full = max(range(len(full)), key=lambda i: full[i])
    argmax_test = max(range(len(test)), key=lambda i: test[i])
    threshold = test[argmax_full]
    rank = sum(1 for q in test if q > threshold)
    return {
        "tv": min(1.0, tv),
        "jsd": min(1.0, max(0.0, jsd)),
        "top1_agree": argmax_full == argmax_test,
        "rank_displacement": rank,
        "kl_fwd": _finite(kl_fwd),
        "kl_rev": _finite(kl_rev),
    }


def measures_from_logits(full_logits: Any, test_logits: Any) -> dict[str, Any]:
    """The live path: one softmax pair, all four measures, no vector retained.

    Kept in torch throughout -- materialising a 150k-entry vocabulary into
    Python lists once per step would dominate the replay.
    """
    import torch

    full = torch.softmax(full_logits.float().flatten(), dim=-1)
    test = torch.softmax(test_logits.float().flatten(), dim=-1)
    mean = 0.5 * (full + test)

    tv = float(0.5 * torch.abs(full - test).sum().item())

    def _kl(p: Any, q: Any) -> float:
        mask = p > 0
        if bool((q[mask] <= 0).any()):
            return math.inf
        return float((p[mask] * torch.log2(p[mask] / q[mask])).sum().item())

    jsd = 0.5 * _kl(full, mean) + 0.5 * _kl(test, mean)
    argmax_full = int(torch.argmax(full).item())
    argmax_test = int(torch.argmax(test).item())
    rank = int((test > test[argmax_full]).sum().item())
    return {
        "tv": min(1.0, tv),
        "jsd": min(1.0, max(0.0, jsd)),
        "top1_agree": argmax_full == argmax_test,
        "rank_displacement": rank,
        "kl_fwd": _finite(_kl(full, test)),
        "kl_rev": _finite(_kl(test, full)),
    }
