import pytest

from kvcache.decomposition import decompose


def test_residual_is_arithmetic_and_honestly_named():
    result = decompose(total_change=-0.30, readout_change=-0.10)
    assert result.unexplained_residual == pytest.approx(-0.20)
    assert not hasattr(result, "causal_trajectory_effect")
    assert result.residual_bracket is None


def test_reverse_replay_brackets_the_residual_and_sign_is_carried():
    result = decompose(
        total_change=-0.30,
        readout_change=-0.10,
        reverse_readout_change=-0.25,
        flip_mass_net=-0.20,
    )
    assert result.unexplained_residual == pytest.approx(-0.20)
    low, high = result.residual_bracket
    assert low == pytest.approx(-0.20)
    assert high == pytest.approx(-0.05)
    assert result.flip_mass_net == pytest.approx(-0.20)
