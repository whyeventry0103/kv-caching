from kvcache.stats import negative_control_ok


def test_negative_control_expectations():
    assert negative_control_ok(excess_add=0.01, tdp_percentile=0.99)
    assert not negative_control_ok(excess_add=0.2, tdp_percentile=0.99)
    assert not negative_control_ok(excess_add=0.0, tdp_percentile=0.5)
