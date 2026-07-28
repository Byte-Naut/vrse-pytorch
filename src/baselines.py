"""Deterministic (non-Bayesian) counterparts used as baselines in Experiment 1 (Plan.md §7.1).

- BackboneOnly: y_hat = B(x) = x directly, zero learned parameters. The trusted safe
  baseline every other variant is compared against.
- PlainResidualMLP: ordinary deterministic residual net, h_{l+1} = h_l + Linear_l(h_l),
  point-estimate weights (no posterior, no KL, no gate) — matches the Bayesian model's
  architecture (encoder/backbone-identity/decoder) but with plain nn.Linear branches.
"""

import torch
import torch.nn as nn

from src.config import ModelConfig


class BackboneOnly(nn.Module):
    """B(x) = x. No parameters, no training needed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PlainResidualMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.input_dim == cfg.hidden_dim:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.branches = nn.ModuleList([nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_layers)])
        self.decoder = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        h = self.encoder(x)
        for branch in self.branches:
            h = h + branch(h)
        y_hat = self.decoder(h)
        if return_diagnostics:
            return y_hat, {"a": [torch.ones_like(h) for _ in self.branches]}
        return y_hat
