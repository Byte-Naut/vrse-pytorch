"""Stage-2 online metrics — exactly four, per results/STAGE2_PROTOCOL.md §5.

Given a run's recorded per-step predictions and diagnostics (as produced by
running a Stream's `online` segment through one of the four methods in
src/online_methods.py), compute:

  1. abrupt_loss    -- prequential RMSE over the first 32 points of the new region.
  2. adaptation_loss -- RMSE over the last 64 points of the new region.
  3. cumulative_risk -- cumulative MSE over all 256 new-region points.
  4. forgetting      -- ID-test RMSE increment, (after online segment) minus
                         (before online segment), evaluated on the SAME fixed
                         ID test set (id_pre vs id_return).

Diagnostics (explanation-only, never part of the verdict):
  - mean acceptance rate per layer, over the online segment
  - mean gradient norm over the online segment
"""

import numpy as np
import torch


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((y_hat - y) ** 2).item()


def compute_run_metrics(online_y_hats: list, online_y_true: list, id_pre_rmse: float, id_return_rmse: float) -> dict:
    """online_y_hats/online_y_true: lists of per-step (batch=1) tensors, length 256,
    in the exact order they were predicted (prequential order).
    """
    y_hat_all = torch.cat(online_y_hats, dim=0)  # (256, 1)
    y_true_all = torch.cat(online_y_true, dim=0)  # (256, 1)

    abrupt_loss = _rmse(y_hat_all[:32], y_true_all[:32])
    adaptation_loss = _rmse(y_hat_all[-64:], y_true_all[-64:])
    cumulative_risk = _mse(y_hat_all, y_true_all)
    forgetting = id_return_rmse - id_pre_rmse

    return {
        "abrupt_loss": abrupt_loss,
        "adaptation_loss": adaptation_loss,
        "cumulative_risk": cumulative_risk,
        "forgetting": forgetting,
    }


def compute_diagnostics(online_a_values: list, online_grad_norms: list) -> dict:
    """online_a_values: list (length 256) of [a_layer0, a_layer1, a_layer2] or None (Frozen-logvar
    excluded since it never updates but still has a_values -- included for acceptance stats).
    online_grad_norms: list (length 256) of floats.
    """
    diag = {"mean_grad_norm": float(np.mean(online_grad_norms)) if online_grad_norms else 0.0}

    valid_a = [a for a in online_a_values if a is not None]
    if valid_a:
        n_layers = len(valid_a[0])
        for l in range(n_layers):
            layer_vals = [step[l].mean().item() for step in valid_a]
            diag[f"mean_acceptance_layer{l}"] = float(np.mean(layer_vals))
    return diag


def prequential_mse(online_y_hats: list, online_y_true: list) -> float:
    """Cumulative MSE over the whole online segment, used for LR selection (protocol §4)."""
    y_hat_all = torch.cat(online_y_hats, dim=0)
    y_true_all = torch.cat(online_y_true, dim=0)
    return _mse(y_hat_all, y_true_all)
