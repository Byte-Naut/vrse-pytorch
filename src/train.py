"""Variational training loop for the Bayesian residual MLP (Plan.md §6).

Two-stage training (default, `cfg.two_stage=True`):
  Stage 1 — train the Bayesian side-branch posteriors q(W_l) via a variational
            objective (Gaussian NLL + KL-to-prior) with the gate bypassed
            (a_l = 1, i.e. plain deterministic-mean residual). This is meant
            to produce a *trustworthy* q(W) before any gating is introduced.
  Stage 2 — switch on the real gate. Freeze (or restrict) the variance
            parameters (rho_W, rho_b) so the task loss cannot manipulate
            v_l, and continue training the mean path (M, mu_b, encoder,
            decoder) plus the gate's own parameters (e.g. lambda_l).

Single-stage mode (`cfg.two_stage=False`) trains everything jointly from the
start, relying on gate.compute()'s internal stopgrad(v) (see src/gates.py)
to prevent the task loss from inflating/shrinking variance to game the gate.

Rationale for freezing/stopgrad on v during gating (Plan §6): if the task
loss can shape variance freely through an open gate, the model could learn to
inflate variance to shut off inconvenient branches, or shrink it to
rubber-stamp favorable branches — degenerating the mechanism into an ordinary
hidden attention logit.
"""

from dataclasses import dataclass

import torch

from src.config import TrainConfig
from src.model import BayesianResidualMLP


def _gaussian_nll(y_hat: torch.Tensor, y: torch.Tensor, noise_std: float) -> torch.Tensor:
    noise_var = noise_std**2
    return (0.5 * (y_hat - y).pow(2) / noise_var).mean()


def _set_variance_params_trainable(model: BayesianResidualMLP, trainable: bool):
    for branch in model.branches:
        branch.rho_W.requires_grad_(trainable)
        branch.rho_b.requires_grad_(trainable)


def _set_gate_bypass(model: BayesianResidualMLP, bypass: bool):
    """When bypass=True, monkeypatch gates to always return a=1 (plain mean residual)."""
    for gate in model.gates:
        gate._bypass = bypass


def _install_bypass_hook(model: BayesianResidualMLP):
    for gate in model.gates:
        gate._bypass = False
        gate._orig_compute = gate.compute
        gate.compute = (lambda g: (lambda m, v, h=None: g._orig_compute(m, v, h) if not g._bypass else torch.ones_like(m)))(gate)


@dataclass
class TrainResult:
    train_losses: list
    val_losses: list


def train(
    model: BayesianResidualMLP,
    train_data,
    val_data,
    cfg: TrainConfig,
    noise_std: float,
) -> TrainResult:
    x_train, y_train = train_data
    x_val, y_val = val_data
    n_train = x_train.shape[0]

    _install_bypass_hook(model)

    train_losses, val_losses = [], []

    def run_epochs(n_epochs, optimizer, gate_bypassed: bool, start_epoch: int):
        _set_gate_bypass(model, gate_bypassed)
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            for i in range(0, n_train, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                xb, yb = x_train[idx], y_train[idx]

                optimizer.zero_grad()
                y_hat = model(xb)
                nll = _gaussian_nll(y_hat, yb, noise_std)
                kl = model.kl_divergence() / n_train
                loss = nll + cfg.kl_weight * kl
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]

            epoch_loss /= n_train
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                y_val_hat = model(x_val)
                val_loss = _gaussian_nll(y_val_hat, y_val, noise_std).item()
            val_losses.append(val_loss)

    if cfg.two_stage:
        # Stage 1: train posteriors with gate bypassed (plain mean residual).
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        run_epochs(cfg.stage1_epochs, optimizer, gate_bypassed=True, start_epoch=0)

        # Stage 2: freeze variance params, open the real gate, keep training mean path + gate.
        _set_variance_params_trainable(model, trainable=False)
        stage2_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(stage2_params, lr=cfg.lr)
        remaining_epochs = cfg.epochs - cfg.stage1_epochs
        run_epochs(remaining_epochs, optimizer, gate_bypassed=False, start_epoch=cfg.stage1_epochs)
        _set_variance_params_trainable(model, trainable=True)  # restore for potential reuse
    else:
        # Single-stage: gate is always on, relying on stopgrad(v) inside each gate.
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        run_epochs(cfg.epochs, optimizer, gate_bypassed=False, start_epoch=0)

    return TrainResult(train_losses=train_losses, val_losses=val_losses)
