"""Output-level safe residual service + SNGP gate (Plan3.md §II.2, §4, STAGE3_PROTOCOL.md).

    a(x)     = 1[u(x) <= tau]
    y_hat(x) = B(x) + a(x) * mu_Delta(x)

Because `a` is a hard {0,1} mask multiplying `mu_Delta` before the add, `a=0`
gives `y_hat == B(x)` to exact floating-point precision -- not merely
approximately -- which is what lets the infrastructure precondition
`max|y_hat - B(x)| < 1e-6` on rejected samples hold regardless of how the
decoder/GP were fit (Plan3.md's correction over the Stage-1/2 architecture,
where closing the gate did not guarantee exact fallback).
"""

import torch

from src.calibration import tolerance_limit_tau
from src.dataset import backbone
from src.sngp_gp import GPHead
from src.sngp_feature import PhiSN


def residual_target(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """r = y - B(x), the regression target the GP head is fit/updated on."""
    return y - torch.as_tensor(backbone(x))


class SafeResidualService:
    """Wraps a frozen phi_SN + one GP head (deployment OR shadow) + a fixed tau."""

    def __init__(self, phi_sn: PhiSN, gp_head: GPHead, tau: float):
        self.phi_sn = phi_sn
        self.gp_head = gp_head
        self.tau = tau

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """x: (batch, 1). Returns (y_hat, a, u), each (batch, 1)."""
        z = self.phi_sn(x)
        mu, u = self.gp_head.predict(z)  # (batch,), (batch,)
        a = (u <= self.tau).to(mu.dtype)
        b = backbone(x).squeeze(-1)
        y_hat = b + a * mu
        return y_hat.unsqueeze(-1), a.unsqueeze(-1), u.unsqueeze(-1)


@torch.no_grad()
def compute_tau(phi_sn: PhiSN, gp_head: GPHead, x_calib: torch.Tensor, percentile: float = 95.0, confidence: float = 0.95) -> float:
    """Stage-3B: one-sided distribution-free tolerance limit on latent variance u(x),
    computed on an INDEPENDENT ID calibration set (src/calibration.py).

    Superseded Stage-3A method (plain empirical percentile) systematically
    under-covered fresh ID draws at n=200 -- see results/STAGE3A_RESULT.md.
    This guarantees, with `confidence` probability over the calibration draw,
    that a fresh ID draw's true acceptance rate is >= percentile/100.

    Computed once, from train/ID data only, at freeze time. Never touched by
    any online stream afterward (protocol requirement).
    """
    z = phi_sn(x_calib)
    _, u = gp_head.predict(z)
    return tolerance_limit_tau(u, p0=percentile / 100.0, confidence=confidence)
