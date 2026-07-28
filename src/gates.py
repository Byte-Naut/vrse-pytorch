"""Gating mechanisms for epistemic-uncertainty-gated residual updates (Plan.md §2, §7).

All gates share the interface `gate.compute(m, v, h) -> a` where:
  m: (batch, dim) posterior mean of the candidate residual
  v: (batch, dim) posterior variance (always used with stopgrad, per §6)
  h: (batch, in_dim) the deterministic hidden state (only used by the "learned" gate)

a_l is broadcast (elementwise or scalar-per-sample) against m when computing
h_{l+1} = h_l + a_l * m_l.
"""

import torch
import torch.nn as nn


def _stopgrad(v: torch.Tensor) -> torch.Tensor:
    return v.detach()


class ScalarSNRGate(nn.Module):
    """First-version gate from §2: collapse to one scalar per sample.

    u = mean(v) / (mean(m^2) + eps)
    a = 1 / (1 + lambda * u)
    """

    def __init__(self, dim: int, lambda_init: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(lambda_init)))
        self.eps = eps

    def compute(self, m: torch.Tensor, v: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        v = _stopgrad(v)
        lam = torch.exp(self.log_lambda)
        u = v.mean(dim=-1, keepdim=True) / (m.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        a = 1.0 / (1.0 + lam * u)
        return a  # (batch, 1), broadcasts over dim


class VectorSNRGate(nn.Module):
    """Elementwise SNR gate from §2: a = m^2 / (m^2 + lambda*v + eps)."""

    def __init__(self, dim: int, lambda_init: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(lambda_init)) * torch.ones(dim))
        self.eps = eps

    def compute(self, m: torch.Tensor, v: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        v = _stopgrad(v)
        lam = torch.exp(self.log_lambda)
        a = m.pow(2) / (m.pow(2) + lam * v + self.eps)
        return a  # (batch, dim)


class LearnedGate(nn.Module):
    """Ordinary learned gate a = sigmoid(G(h)) — ignores v entirely.

    Matched in parameter count to the SNR gates for a fair ablation (Plan §7.2):
    the real variance gate must beat this to support the mechanism's claim.
    """

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, dim)

    def compute(self, m: torch.Tensor, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(h))


class ShuffledVarianceGate(nn.Module):
    """SNR gate but v is shuffled across the batch dimension before use (Plan §7.2).

    If performance is roughly unchanged vs. the real SNR gate, the gain is coming
    from generic regularization/gating capacity rather than epistemic uncertainty.
    """

    def __init__(self, dim: int, lambda_init: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(lambda_init)) * torch.ones(dim))
        self.eps = eps

    def compute(self, m: torch.Tensor, v: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        v = _stopgrad(v)
        perm = torch.randperm(v.shape[0], device=v.device)
        v_shuffled = v[perm]
        lam = torch.exp(self.log_lambda)
        a = m.pow(2) / (m.pow(2) + lam * v_shuffled + self.eps)
        return a


class OracleGate(nn.Module):
    """Gate constructed from the true prediction error (upper-bound reference, Plan §7.1).

    Not learnable — this is a diagnostic ceiling, not a deployable mechanism. It
    requires the ground-truth residual target to be supplied at compute time via
    `set_target`, so it can only be used in an eval loop that has access to labels.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self._target = None

    def set_target(self, target_residual: torch.Tensor):
        """target_residual: (batch, dim), the true residual this layer should have produced."""
        self._target = target_residual

    def compute(self, m: torch.Tensor, v: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        if self._target is None:
            raise RuntimeError("OracleGate requires set_target() before compute().")
        error_sq = (m - self._target).pow(2)
        # Gate should be high when m is close to the true target (low error), low otherwise.
        a = m.pow(2) / (m.pow(2) + error_sq + self.eps)
        return a


class NoGate(nn.Module):
    """a = 1 always: plain deterministic-mean residual (posterior-mean-only baseline)."""

    def compute(self, m: torch.Tensor, v: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        return torch.ones_like(m)


def build_gate(gate_type: str, in_dim: int, dim: int, lambda_init: float = 1.0, eps: float = 1e-6) -> nn.Module:
    if gate_type == "snr_scalar":
        return ScalarSNRGate(dim, lambda_init, eps)
    if gate_type == "snr_vector":
        return VectorSNRGate(dim, lambda_init, eps)
    if gate_type == "learned":
        return LearnedGate(in_dim, dim)
    if gate_type == "shuffled_variance":
        return ShuffledVarianceGate(dim, lambda_init, eps)
    if gate_type == "oracle":
        return OracleGate(dim, eps)
    if gate_type == "none":
        return NoGate()
    raise ValueError(f"Unknown gate_type: {gate_type}")
