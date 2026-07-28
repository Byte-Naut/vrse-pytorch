"""Online learning harness for Stage-2 (Plan2.md, results/STAGE2_PROTOCOL.md).

Predict-then-update loop over the frozen log-variance-gated Bayesian residual
MLP. All parameter freezing, gate computation, and gradient tracking needed by
every method in Task #18 live here so the four method implementations only
differ in which forward path serves predictions and which forward path
receives gradients.

Per the locked protocol:
  - Common start: results/frozen_mean_only_model.pt for every seed/method/stream.
  - Common gate calibration: (c_l, tau_l) from results/diagnostic3_results.pkl,
    K=2, train-only. Never recalibrated online.
  - Only trainable parameters: each residual branch's M and mu_b.
  - Frozen: encoder, decoder, rho_W, rho_b (gate has no learnable params here —
    it is the externally-calibrated log-variance gate, not a ScalarSNRGate).
  - Gradient norm := global L2 norm over all trainable (M, mu_b) grads,
    measured BEFORE the update step (i.e. the norm of the gradient that
    produces that step).
"""

import copy
import pickle

import torch

from src.model import BayesianResidualMLP

RESULTS_DIR = "results"


def load_frozen_model_and_calib():
    """Load the common starting checkpoint and its train-only log-var gate calibration.

    Returns a fresh BayesianResidualMLP each call (caller mutates freely) plus
    the calib list of (c_l, tau_l) tuples, one per branch.
    """
    ckpt = torch.load(f"{RESULTS_DIR}/frozen_mean_only_model.pt", weights_only=False)
    model = BayesianResidualMLP(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])

    with open(f"{RESULTS_DIR}/diagnostic3_results.pkl", "rb") as f:
        d3 = pickle.load(f)
    calib = d3["calib"]  # list of (c, tau) per layer, train-only, K=2

    return model, calib


def fresh_model_for_seed():
    """Return a fresh copy of the common starting model (identical weights every call)."""
    model, calib = load_frozen_model_and_calib()
    return model, calib


def set_online_trainable(model: BayesianResidualMLP):
    """Freeze everything except each branch's M and mu_b (Plan2 §4)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for branch in model.branches:
        branch.M.requires_grad_(True)
        branch.mu_b.requires_grad_(True)


def trainable_params(model: BayesianResidualMLP):
    params = []
    for branch in model.branches:
        params.append(branch.M)
        params.append(branch.mu_b)
    return params


def logvar_gate_values(branch, calib_l, h: torch.Tensor):
    """a = sigmoid(-(log(mean(v)) - c) / tau), broadcasting over hidden dim.

    v = S_l^2 * h_l^2 + v_b depends on h_l, which depends on EARLIER layers'
    trainable M, mu_b (through the residual chain), even though this layer's
    own rho_W/rho_b are frozen. Without stopgrad, loss gradient could flow
    back through the gate arm (v -> h_l -> earlier M's) and shape earlier
    layers' updates via "making v bigger/smaller to open/close the gate"
    rather than via the intended residual-correction arm. We stopgrad v
    before computing a, matching the Plan.md §6 principle and gates.py's
    _stopgrad(v) convention: the gate is a fixed novelty detector, not a
    trainable/gameable signal. Gradient flows to trainable params ONLY
    through m (i.e. through a_detached * m).
    """
    c, tau = calib_l
    m, v = branch.analytic_moments(h)
    v_detached = v.detach()
    log_v = torch.log(v_detached.mean(dim=-1, keepdim=True) + 1e-8)
    a = torch.sigmoid(-(log_v - c) / tau)
    return m, v, a


def forward_gated(model: BayesianResidualMLP, calib: list, x: torch.Tensor):
    """Log-var-gated forward pass. Returns y_hat, list of a per layer."""
    h = model.encoder(x)
    a_values = []
    for branch, calib_l in zip(model.branches, calib):
        m, v, a = logvar_gate_values(branch, calib_l, h)
        a_values.append(a)
        h = h + a * m
    y_hat = model.decoder(h)
    return y_hat, a_values


def forward_ungated(model: BayesianResidualMLP, x: torch.Tensor):
    """a=1 at every layer (max-plasticity forward path)."""
    h = model.encoder(x)
    for branch in model.branches:
        m, _ = branch.analytic_moments(h)
        h = h + m
    y_hat = model.decoder(h)
    return y_hat


def global_grad_norm(params) -> float:
    """L2 norm over all provided parameters' current .grad tensors (0 for None grads)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total**0.5


def clone_model_state(model: BayesianResidualMLP):
    return copy.deepcopy(model.state_dict())
