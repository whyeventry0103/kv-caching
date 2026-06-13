import pytest

from kvcache.decomposition import decompose


def test_residual_is_arithmetic_and_honestly_named():
    result = decompose(total_change=-0.30, readout_change=-0.10)
    assert result.unexplained_residual == pytest.approx(-0.20)
    assert not hasattr(result, "causal_trajectory_effect")
