"""Differential tests for vrse._algorithm._tolerance_limit_tau against the
reference Stage-3B implementation (src/calibration.py). Per Phase2B_Plan.md
W1/§6, this compares the port to the reference on the same input rather than
re-deriving the formula in the test -- re-deriving only writes the same
possibly-wrong understanding twice.
"""
import pytest
import torch

from src.calibration import tolerance_limit_tau as _reference_tau
from vrse._algorithm import _tolerance_limit_tau


def test_tau_matches_reference_impl():
    torch.manual_seed(0)
    u = torch.rand(200).double() * 2.0
    ported = _tolerance_limit_tau(u, p0=0.95, confidence=0.95)
    reference = _reference_tau(u, p0=0.95, confidence=0.95)
    assert ported == reference


def test_tau_matches_reference_impl_on_recorded_calib_size():
    """n=2000 is the protocol-recorded calibration size (src/config.py
    id_calib_n, Stage-3B amendment) -- not a boundary value."""
    torch.manual_seed(2)
    u = torch.rand(2000).double() * 2.0
    ported = _tolerance_limit_tau(u, p0=0.95, confidence=0.95)
    reference = _reference_tau(u, p0=0.95, confidence=0.95)
    assert ported == reference


def test_tau_calibration_fails_closed():
    """n=58 must raise; n=59 is the smallest n where Wilks' rule does not
    (k == n == 59, i.e. tau degenerates to max(u)). Both implementations
    must agree on the boundary."""
    torch.manual_seed(1)
    u_58 = torch.rand(58).double()
    with pytest.raises(ValueError):
        _tolerance_limit_tau(u_58, p0=0.95, confidence=0.95)
    with pytest.raises(ValueError):
        _reference_tau(u_58, p0=0.95, confidence=0.95)

    u_59 = torch.rand(59).double()
    ported = _tolerance_limit_tau(u_59, p0=0.95, confidence=0.95)
    reference = _reference_tau(u_59, p0=0.95, confidence=0.95)
    assert ported == reference == float(u_59.sort().values[-1].item())
