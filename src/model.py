"""Bayesian residual MLP with layer-local epistemic gating (Plan.md §5).

Architecture:
    h_0 = encode(x)                     deterministic lift into hidden_dim (identity if dims match)
    for l in 0..N-1:
        m_l, v_l = BayesianLinear_l.analytic_moments(h_l)
        a_l      = gate_l(m_l, v_l, h_l)
        h_{l+1}  = B_l(h_l) + a_l * m_l      with B_l(h) = h (minimal fallback, §5)
    y_hat = C h_N + c                   deterministic output head

Only the residual chain (BayesianLinear + gate + backbone identity) is the
mechanism under test. The encoder/decoder are plain deterministic linear maps
needed only to lift a 1D input into a working hidden width; Plan §5's
"h_0 = x" is the input_dim == hidden_dim special case.

No variance is propagated across layers (v_l is computed fresh from the
deterministic h_l each layer) — this is the "layer-local epistemic gating"
simplification, not full-network ADF (§5).
"""

import torch
import torch.nn as nn

from src.bayesian_linear import BayesianLinear
from src.config import ModelConfig
from src.gates import build_gate


class BayesianResidualMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.input_dim == cfg.hidden_dim:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        self.branches = nn.ModuleList(
            [BayesianLinear(cfg.hidden_dim, cfg.hidden_dim, prior_std=cfg.prior_std) for _ in range(cfg.n_layers)]
        )
        self.gates = nn.ModuleList(
            [
                build_gate(cfg.gate_type, in_dim=cfg.hidden_dim, dim=cfg.hidden_dim, lambda_init=cfg.lambda_init, eps=cfg.eps)
                for _ in range(cfg.n_layers)
            ]
        )
        self.decoder = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        h = self.encoder(x)

        ms, vs, a_s = [], [], []
        for branch, gate in zip(self.branches, self.gates):
            m, v = branch.analytic_moments(h)
            a = gate.compute(m, v, h)
            h = h + a * m
            if return_diagnostics:
                ms.append(m)
                vs.append(v)
                a_s.append(a)

        y_hat = self.decoder(h)

        if return_diagnostics:
            return y_hat, {"m": ms, "v": vs, "a": a_s, "h_final": h}
        return y_hat

    def kl_divergence(self) -> torch.Tensor:
        return sum(branch.kl_divergence() for branch in self.branches)

    def mc_forward(self, x: torch.Tensor, n_samples: int, use_gate: bool = True) -> torch.Tensor:
        """Monte-Carlo forward pass sampling actual weights at each layer (for MC variance checks).

        Each of the S particles carries its own independently-sampled weights and its own
        hidden-state trajectory, since gating on a noisy m/v can make particles diverge
        layer over layer. Gate values are recomputed per particle from that particle's own
        analytic (m, v) — the gate itself is never sampled (Plan §5 treats a_l as a
        deterministic function of the analytic moments, not a random variable).

        Returns (n_samples, batch, output_dim).
        """
        batch = x.shape[0]
        h = self.encoder(x).unsqueeze(0).expand(n_samples, -1, -1).clone()  # (S, batch, hidden)

        for branch, gate in zip(self.branches, self.gates):
            flat_h = h.reshape(n_samples * batch, -1)
            m_analytic, v_analytic = branch.analytic_moments(flat_h)

            R_samples = torch.stack(
                [branch.sample_forward(h[s], n_samples=1).squeeze(0) for s in range(n_samples)], dim=0
            ).reshape(n_samples * batch, -1)

            if use_gate:
                a = gate.compute(m_analytic, v_analytic, flat_h)
            else:
                a = torch.ones_like(R_samples)

            h = (flat_h + a * R_samples).reshape(n_samples, batch, -1)

        y_hat = self.decoder(h)
        return y_hat
