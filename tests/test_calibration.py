"""Contract tests for the distribution-free calibration threshold."""
import pytest
import torch
from scipy.stats import binom

from vrse._algorithm import _tolerance_limit_tau


def _expected_order_statistic(u: torch.Tensor, p0: float, confidence: float) -> float:
    rank = int(binom.ppf(confidence, len(u), p0)) + 1
    if rank > len(u):
        raise ValueError("insufficient calibration sample")
    return float(u.sort().values[rank - 1].item())


def test_tau_matches_wilks_order_statistic():
    torch.manual_seed(0)
    u = torch.rand(200).double() * 2.0
    actual = _tolerance_limit_tau(u, p0=0.95, confidence=0.95)
    expected = _expected_order_statistic(u, p0=0.95, confidence=0.95)
    assert actual == expected


def test_tau_matches_recorded_calibration_size():
    """The reference lifecycle uses 2,000 independent calibration samples."""
    torch.manual_seed(2)
    u = torch.rand(2000).double() * 2.0
    actual = _tolerance_limit_tau(u, p0=0.95, confidence=0.95)
    expected = _expected_order_statistic(u, p0=0.95, confidence=0.95)
    assert actual == expected


def test_tau_calibration_fails_closed():
    """n=58 must raise; n=59 is the smallest n where Wilks' rule does not
    (k == n == 59, i.e. tau degenerates to max(u))."""
    torch.manual_seed(1)
    u_58 = torch.rand(58).double()
    with pytest.raises(ValueError):
        _tolerance_limit_tau(u_58, p0=0.95, confidence=0.95)
    u_59 = torch.rand(59).double()
    actual = _tolerance_limit_tau(u_59, p0=0.95, confidence=0.95)
    assert actual == float(u_59.sort().values[-1].item())
