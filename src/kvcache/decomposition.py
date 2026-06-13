"""Honest, direction-aware accuracy decomposition.

``unexplained_residual = total_change - on_reference_readout_change`` is reported
as "accuracy change unexplained by on-reference readout." It is **not** a measured
cause and must not be relabeled as causal trajectory distortion.

The residual is a *biased* estimand and the bias direction is known: on-reference
readout is measured on the full-cache reference trajectory, which visits good
states, so it plausibly *underestimates* per-step damage along the compressed
model's own (worse) trajectory — which inflates the residual ("trajectory")
share. This is the imitation-learning covariate-shift problem (Ross & Bagnell,
DAgger lineage) in a compression costume.

Two things bound the bias and must be carried in the data model, not discovered in
rebuttal:

* **Reverse replay** (``reverse_readout_change``): the compressed free-run trace
  replayed under the full cache. Forward and reverse readout *bracket* the
  on-policy truth from both sides; :attr:`Decomposition.residual_bracket` reports
  the interval rather than a single point estimate.
* **Signed outcome** (``flip_mass_net``): every divergence-derived quantity must
  travel with its gained-vs-lost correct-mass sign, because deviation from the
  reference can be improvement (the >100% regime), not damage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decomposition:
    total_change: float
    on_reference_readout_change: float
    unexplained_residual: float
    reverse_readout_change: float | None = None
    flip_mass_net: float | None = None

    @property
    def residual_bracket(self) -> tuple[float, float] | None:
        """Interval bracketing the on-policy residual via forward+reverse readout.

        ``None`` until a reverse-replay estimate is supplied. With both readout
        estimates, the on-policy residual is bounded by the two ``total - readout``
        endpoints; we never collapse the bracket to a single causal number.
        """
        if self.reverse_readout_change is None:
            return None
        forward = self.total_change - self.on_reference_readout_change
        reverse = self.total_change - self.reverse_readout_change
        return (min(forward, reverse), max(forward, reverse))


def decompose(
    total_change: float,
    readout_change: float,
    *,
    reverse_readout_change: float | None = None,
    flip_mass_net: float | None = None,
) -> Decomposition:
    """Build the honest residual decomposition.

    ``readout_change`` is the on-reference (forward teacher-forced) readout change.
    Supply ``reverse_readout_change`` once a reverse-replay run exists to expose
    the residual bracket, and ``flip_mass_net`` to carry the signed outcome.
    """
    return Decomposition(
        total_change=total_change,
        on_reference_readout_change=readout_change,
        unexplained_residual=total_change - readout_change,
        reverse_readout_change=reverse_readout_change,
        flip_mass_net=flip_mass_net,
    )
