"""Hand-computed checks for the Amendment Q.6 on-manifold split."""

from __future__ import annotations

import math

import pytest

from kvcache.metrics.latent_split import (
    FrozenBasis,
    omega,
    split_displacement,
    whiten,
)


def _basis(vectors=((1.0, 0.0, 0.0),), dim=3):
    return FrozenBasis(
        vectors,
        mean=[0.0] * dim,
        scale=[1.0] * dim,
        condition="fullcache",
        fitted_on_n=100,
    )


def test_basis_refuses_a_non_fullcache_condition():
    """Q.0 limit 3: a basis fitted on compressed traces makes the split circular."""
    with pytest.raises(ValueError, match="circular"):
        FrozenBasis(
            [(1.0, 0.0)],
            mean=[0.0, 0.0],
            scale=[1.0, 1.0],
            condition="compressed",
            fitted_on_n=10,
        )


def test_basis_is_immutable_after_construction():
    b = _basis()
    with pytest.raises(AttributeError, match="immutable"):
        b._mean = (1.0, 1.0, 1.0)


def test_basis_rejects_nonpositive_whitening_scale():
    with pytest.raises(ValueError, match="positive"):
        FrozenBasis(
            [(1.0, 0.0)],
            mean=[0.0, 0.0],
            scale=[1.0, 0.0],
            condition="fullcache",
            fitted_on_n=10,
        )


def test_whiten_uses_the_frozen_statistics():
    b = FrozenBasis(
        [(1.0, 0.0)],
        mean=[1.0, 2.0],
        scale=[2.0, 4.0],
        condition="fullcache",
        fitted_on_n=10,
    )
    assert whiten([3.0, 10.0], b) == pytest.approx([1.0, 2.0])


def test_displacement_entirely_inside_the_subspace_has_zero_perp():
    par, perp = split_displacement([3.0, 0.0, 0.0], _basis())
    assert par == pytest.approx(3.0)
    assert perp == pytest.approx(0.0)
    assert omega(par, perp) == pytest.approx(0.0)


def test_displacement_entirely_outside_the_subspace_has_zero_parallel():
    par, perp = split_displacement([0.0, 4.0, 0.0], _basis())
    assert par == pytest.approx(0.0)
    assert perp == pytest.approx(4.0)
    # parallel is zero -> the ratio is undefined, not huge
    assert omega(par, perp) is None


def test_split_is_pythagorean_hand_computed():
    # delta = (3,4,0); basis spans e0 -> parallel 3, perp 4, omega 4/3
    par, perp = split_displacement([3.0, 4.0, 0.0], _basis())
    assert par == pytest.approx(3.0)
    assert perp == pytest.approx(4.0)
    assert math.hypot(par, perp) == pytest.approx(5.0)
    assert omega(par, perp) == pytest.approx(4.0 / 3.0)


def test_two_dimensional_basis_absorbs_more_of_the_displacement():
    b = _basis(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    par, perp = split_displacement([3.0, 4.0, 12.0], b)
    assert par == pytest.approx(5.0)
    assert perp == pytest.approx(12.0)


def test_dimension_mismatch_is_refused():
    with pytest.raises(ValueError, match="does not match the frozen basis"):
        split_displacement([1.0, 2.0], _basis())


def test_provenance_records_what_must_travel_with_any_number():
    p = _basis().provenance()
    assert p["condition"] == "fullcache"
    assert p["k"] == 1 and p["dim"] == 3 and p["fitted_on_n"] == 100


# ---------------------------------------------------------------------------
# Amendment R.6: the basis must actually be a projector basis.


def test_non_orthonormal_basis_is_refused_the_reviewer_counterexample():
    """[[2,0]] gave (parallel, perp) = (4, 3) for [1,0]; it must not construct."""
    with pytest.raises(ValueError, match="norm"):
        FrozenBasis(
            [(2.0, 0.0)],
            mean=[0.0, 0.0],
            scale=[1.0, 1.0],
            condition="fullcache",
            fitted_on_n=1,
        )


def test_non_orthogonal_basis_is_refused():
    s = 2**-0.5
    with pytest.raises(ValueError, match="not orthogonal"):
        FrozenBasis(
            [(1.0, 0.0), (s, s)],
            mean=[0.0, 0.0],
            scale=[1.0, 1.0],
            condition="fullcache",
            fitted_on_n=1,
        )


def test_non_finite_inputs_and_empty_fit_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        FrozenBasis(
            [(1.0, 0.0)],
            mean=[float("nan"), 0.0],
            scale=[1.0, 1.0],
            condition="fullcache",
            fitted_on_n=1,
        )
    with pytest.raises(ValueError, match="fitted_on_n"):
        FrozenBasis(
            [(1.0, 0.0)],
            mean=[0.0, 0.0],
            scale=[1.0, 1.0],
            condition="fullcache",
            fitted_on_n=0,
        )


def test_a_valid_basis_gives_a_true_projection():
    b = FrozenBasis(
        [(1.0, 0.0)],
        mean=[0.0, 0.0],
        scale=[1.0, 1.0],
        condition="fullcache",
        fitted_on_n=1,
    )
    assert split_displacement([1.0, 0.0], b) == pytest.approx((1.0, 0.0))


def test_omega_reasons_distinguish_the_two_missing_cases():
    from kvcache.metrics.latent_split import omega_with_reason

    assert omega_with_reason(0.0, 0.0) == (None, "undefined_direction")
    assert omega_with_reason(0.0, 3.0) == (None, "unbounded")
    val, why = omega_with_reason(3.0, 4.0)
    assert why == "finite" and val == pytest.approx(4.0 / 3.0)
    # the legacy API is unchanged
    assert omega(0.0, 0.0) is None and omega(0.0, 3.0) is None
