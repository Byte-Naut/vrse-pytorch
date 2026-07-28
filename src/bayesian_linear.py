"""Bayesian linear side-branch with diagonal Gaussian weight posterior (Plan.md §5).

Given a deterministic input h_l, the layer proposes a stochastic residual
    R_l = W_l h_l + b_l,   W_l ~ N(M_l, S_l^2),  b_l ~ N(mu_b, v_b)
and since h_l is deterministic, the posterior moments of R_l are analytic:
    m_l = M_l h_l + mu_b
    v_l = S_l^2 h_l^2 + v_b        (elementwise square, matmul over input dim)

No sampling is needed for the mean/variance used by the gate — this is the
"layer-local epistemic gating" simplification from §5 (no covariance
propagation across layers, no nonlinear moment closure).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, prior_std: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.prior_std = prior_std

        # Weight posterior mean, and rho parameterizing std via softplus (always positive).
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        self.rho_W = nn.Parameter(torch.empty(out_dim, in_dim))
        # Bias posterior mean and rho.
        self.mu_b = nn.Parameter(torch.empty(out_dim))
        self.rho_b = nn.Parameter(torch.empty(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        nn.init.constant_(self.rho_W, -4.0)  # softplus(-4) ~= 0.018, small initial variance
        fan_in = self.in_dim
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.mu_b, -bound, bound)
        nn.init.constant_(self.rho_b, -4.0)

    @property
    def S(self) -> torch.Tensor:
        """Weight posterior std, shape (out_dim, in_dim)."""
        return F.softplus(self.rho_W)

    @property
    def b_std(self) -> torch.Tensor:
        """Bias posterior std, shape (out_dim,)."""
        return F.softplus(self.rho_b)

    def analytic_moments(self, h: torch.Tensor):
        """Compute (m, v) for R = W h + b given deterministic h.

        h: (batch, in_dim)
        returns m, v: (batch, out_dim)
        """
        m = F.linear(h, self.M, self.mu_b)
        v = F.linear(h.pow(2), self.S.pow(2)) + self.b_std.pow(2)
        return m, v

    def sample_forward(self, h: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Monte-Carlo samples of R = W h + b via the reparameterization trick.

        h: (batch, in_dim)
        returns: (n_samples, batch, out_dim)
        """
        batch = h.shape[0]
        eps_W = torch.randn(n_samples, self.out_dim, self.in_dim, device=h.device)
        eps_b = torch.randn(n_samples, self.out_dim, device=h.device)
        W_samples = self.M.unsqueeze(0) + self.S.unsqueeze(0) * eps_W  # (n_samples, out, in)
        b_samples = self.mu_b.unsqueeze(0) + self.b_std.unsqueeze(0) * eps_b  # (n_samples, out)

        # (n_samples, batch, out) = einsum over in_dim
        R = torch.einsum("soi,bi->sbo", W_samples, h) + b_samples.unsqueeze(1)
        return R

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(W,b) || N(0, prior_std^2)) summed over all weight and bias dims."""
        prior_var = self.prior_std**2

        def kl_gaussian(mu, var):
            return 0.5 * (var / prior_var + mu.pow(2) / prior_var - 1.0 + math.log(prior_var) - torch.log(var))

        kl_W = kl_gaussian(self.M, self.S.pow(2)).sum()
        kl_b = kl_gaussian(self.mu_b, self.b_std.pow(2)).sum()
        return kl_W + kl_b
