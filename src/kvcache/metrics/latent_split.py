"""On-manifold / off-manifold displacement split (Amendment Q.6).

Admitted only by the Q.0 scope widening (2026-09-12), which replaced Amendment
P.4's "Latent-state divergence is out of scope" with a dated, limited admission.
Those limits travel with every function here: latent quantities support
**structural** claims only, never an accuracy-scale claim by adjacency, and every
result is exploratory by construction.

**Why not compare hidden states directly.** Naively asking "how different are
these two vectors" is not well posed: the geometry is non-metric and the
distances are dominated by a handful of high-norm dimensions, so the resulting
scalar predicts nothing. The refinement is to stop comparing states and compare
**where the displacement points** relative to the subspace the full-cache model
actually moves in.

``delta_parallel`` means the compressed model moved to a state the full model
could plausibly have occupied -- a different reasoning step, but a structurally
valid one. ``delta_perp`` means it moved somewhere the full model never goes:
off-distribution, the regime of repetition loops and non-termination. ``omega``
is the ratio, and it is the quantity Amendment Q.6 pre-commits a prediction on.

**Circularity is a stop condition, not a caveat.** The whitening statistics and
the basis are fitted on the FULL-CACHE condition only and frozen before any
compressed trace is projected. A basis fitted on both conditions has already seen
the displacement it is being asked to characterise, and the split would be
meaningless. ``FrozenBasis`` refuses to be modified after construction so this
cannot happen by accident partway through a run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["FrozenBasis", "whiten", "split_displacement", "omega"]


class FrozenBasis:
    """An immutable principal basis plus the whitening stats it was fitted with.

    Immutable on purpose. The failure this guards against is not malice but
    ordinary drift: refitting or rescaling midway through a run would silently
    change what every subsequent projection means, and nothing downstream would
    look wrong.
    """

    __slots__ = ("_vectors", "_mean", "_scale", "_condition", "_fitted_on_n")

    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        mean: Sequence[float],
        scale: Sequence[float],
        condition: str,
        fitted_on_n: int,
    ) -> None:
        if condition != "fullcache":
            raise ValueError(
                f"basis must be fitted on the full-cache condition only, got "
                f"{condition!r}; fitting on compressed traces makes the "
                "on-manifold split circular (Amendment Q.0, limit 3)"
            )
        dim = len(mean)
        if len(scale) != dim:
            raise ValueError("mean and scale must have the same dimension")
        if any(s <= 0 for s in scale):
            raise ValueError("whitening scale entries must be positive")
        for v in vectors:
            if len(v) != dim:
                raise ValueError("every basis vector must match the whitening dim")
        object.__setattr__(
            self, "_vectors", tuple(tuple(float(x) for x in v) for v in vectors)
        )
        object.__setattr__(self, "_mean", tuple(float(x) for x in mean))
        object.__setattr__(self, "_scale", tuple(float(x) for x in scale))
        object.__setattr__(self, "_condition", condition)
        object.__setattr__(self, "_fitted_on_n", int(fitted_on_n))

    def __setattr__(self, *_: object) -> None:
        raise AttributeError(
            "FrozenBasis is immutable: refitting or rescaling partway through a "
            "run would silently redefine every projection taken before it"
        )

    @property
    def vectors(self) -> tuple[tuple[float, ...], ...]:
        return self._vectors

    @property
    def mean(self) -> tuple[float, ...]:
        return self._mean

    @property
    def scale(self) -> tuple[float, ...]:
        return self._scale

    @property
    def k(self) -> int:
        return len(self._vectors)

    def provenance(self) -> dict[str, object]:
        """What must be recorded beside any number this basis produced."""
        return {
            "condition": self._condition,
            "fitted_on_n": self._fitted_on_n,
            "k": self.k,
            "dim": len(self._mean),
        }


def whiten(vector: Sequence[float], basis: FrozenBasis) -> list[float]:
    """Centre and scale with the FROZEN statistics, never per-trace ones."""
    if len(vector) != len(basis.mean):
        raise ValueError("vector dimension does not match the frozen basis")
    return [
        (float(v) - m) / s
        for v, m, s in zip(vector, basis.mean, basis.scale, strict=True)
    ]


def split_displacement(
    displacement: Sequence[float], basis: FrozenBasis
) -> tuple[float, float]:
    """Return ``(norm_parallel, norm_perp)`` for one whitened displacement.

    The displacement is projected onto the frozen basis; the parallel part is the
    reconstruction and the perpendicular part is what the basis cannot express.
    The basis is treated as orthonormal, which is what a randomised SVD returns.
    """
    d = list(float(x) for x in displacement)
    if len(d) != len(basis.mean):
        raise ValueError("displacement dimension does not match the frozen basis")
    parallel = [0.0] * len(d)
    for vec in basis.vectors:
        coeff = sum(a * b for a, b in zip(d, vec, strict=True))
        for i, b in enumerate(vec):
            parallel[i] += coeff * b
    perp = [a - b for a, b in zip(d, parallel, strict=True)]
    return (
        math.sqrt(sum(x * x for x in parallel)),
        math.sqrt(sum(x * x for x in perp)),
    )


def omega(norm_parallel: float, norm_perp: float) -> float | None:
    """Off-manifold fraction ``||delta_perp|| / ||delta_parallel||``.

    ``None`` when the parallel norm is zero: the ratio is undefined, and
    substituting a large sentinel would read as maximal off-manifold drift when
    the truth is that there was no displacement to classify.
    """
    if norm_parallel <= 0.0:
        return None
    return float(norm_perp) / float(norm_parallel)
