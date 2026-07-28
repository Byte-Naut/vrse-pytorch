"""Spectral-normalized residual feature extractor phi_SN (Plan3.md §4, STAGE3_PROTOCOL.md).

phi_SN is trained ONCE offline (plain SGD regression on y - B(x)) together
with a throwaway linear head, then frozen and shared, read-only, by both the
deployment and shadow GP heads (src/sngp_gp.py). Only phi_SN survives after
training -- the head is discarded.

Architecture (locked):
    h_0 = encode(x)                         plain linear lift, NOT spectral-normalized
    for l in 0..n_blocks-1:
        h_{l+1} = h_l + relu(SNLinear_l(h_l))
    z = h_{n_blocks}

Each SNLinear enforces its weight matrix's spectral norm to be exactly
`sn_multiplier` (0.95): normalize by the estimated top singular value (torch's
built-in power-iteration spectral_norm parametrization), then scale by the
fixed multiplier. Bias is left unscaled -- Lipschitz constant of an affine map
depends only on the weight's operator norm. No LayerNorm, no L2 unit
normalization anywhere in phi_SN (protocol requirement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SNGPConfig


class SNLinear(nn.Module):
    """Linear layer whose weight spectral norm is fixed to `coeff` (not merely bounded)."""

    def __init__(self, in_dim: int, out_dim: int, coeff: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.utils.parametrizations.spectral_norm(self.linear, name="weight")
        self.coeff = coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.linear.weight has spectral norm ~1 via the parametrization's
        # power iteration; scale to exactly `coeff`, leave bias unscaled.
        return F.linear(x, self.linear.weight * self.coeff, self.linear.bias)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, coeff: float):
        super().__init__()
        self.sn_linear = SNLinear(dim, dim, coeff)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + F.relu(self.sn_linear(h))


class PhiSN(nn.Module):
    def __init__(self, cfg: SNGPConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.input_dim == cfg.hidden_dim:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(cfg.hidden_dim, cfg.sn_multiplier) for _ in range(cfg.n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for block in self.blocks:
            h = block(h)
        return h

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)


def train_feature_extractor(
    cfg: SNGPConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 1000,
    lr: float = 1e-3,
) -> PhiSN:
    """Fit phi_SN + a throwaway linear head on residual targets r = y - B(x) = y - x.

    Returns the frozen phi_SN (requires_grad=False, eval mode). The head is
    discarded -- only phi_SN is kept and shared by the deployment/shadow GPs.
    """
    phi = PhiSN(cfg)
    head = nn.Linear(cfg.hidden_dim, 1)
    params = list(phi.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    r_train = y_train - x_train  # B(x) = x

    phi.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = phi(x_train)
        pred = head(z)
        loss = F.mse_loss(pred, r_train)
        loss.backward()
        optimizer.step()

    phi.freeze()
    return phi
