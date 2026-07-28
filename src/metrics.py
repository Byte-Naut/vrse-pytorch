"""Evaluation metrics for the Bayesian residual MLP experiments (Plan.md §8).

Metrics implemented:
  - ID/OOD RMSE
  - Max error and 95th-percentile absolute error
  - Gaussian NLL and a regression calibration error (based on predictive quantile coverage)
  - Spearman correlation between predictive variance and absolute error
  - Per-layer acceptance rate E[a_l]
  - Relative inference overhead vs. a plain deterministic ANN of matched width/depth
"""

import time

import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


@torch.no_grad()
def max_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.max(torch.abs(y_pred - y_true)).item()


@torch.no_grad()
def quantile_error(y_pred: torch.Tensor, y_true: torch.Tensor, q: float = 0.95) -> float:
    abs_err = torch.abs(y_pred - y_true).flatten()
    return torch.quantile(abs_err, q).item()


@torch.no_grad()
def predictive_mean_and_var(model, x: torch.Tensor, n_samples: int = 200):
    """Predictive mean/variance at the output via MC sampling of the full model.

    Returns (mean, var), each (batch, output_dim).
    """
    model.eval()
    samples = model.mc_forward(x, n_samples=n_samples, use_gate=True)  # (S, batch, out)
    mean = samples.mean(dim=0)
    var = samples.var(dim=0, unbiased=True)
    return mean, var


@torch.no_grad()
def gaussian_nll(y_pred_mean: torch.Tensor, y_pred_var: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> float:
    var = y_pred_var.clamp_min(eps)
    nll = 0.5 * (torch.log(2 * np.pi * var) + (y_true - y_pred_mean) ** 2 / var)
    return nll.mean().item()


@torch.no_grad()
def calibration_error(y_pred_mean: torch.Tensor, y_pred_std: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10) -> float:
    """Regression calibration error: for nominal confidence levels p, compare empirical
    coverage of the predictive interval to p, average |empirical - nominal| over bins.
    """
    from scipy.stats import norm

    nominal_levels = np.linspace(0.05, 0.95, n_bins)
    errors = []
    z = (y_true - y_pred_mean) / y_pred_std.clamp_min(1e-6)
    z_np = z.flatten().cpu().numpy()
    for p in nominal_levels:
        bound = norm.ppf(0.5 + p / 2)
        empirical = np.mean(np.abs(z_np) <= bound)
        errors.append(abs(empirical - p))
    return float(np.mean(errors))


@torch.no_grad()
def variance_error_spearman(y_pred_var: torch.Tensor, y_pred_mean: torch.Tensor, y_true: torch.Tensor) -> float:
    abs_err = torch.abs(y_pred_mean - y_true).flatten().cpu().numpy()
    var_np = y_pred_var.flatten().cpu().numpy()
    if np.std(abs_err) < 1e-12 or np.std(var_np) < 1e-12:
        return float("nan")
    corr, _ = spearmanr(var_np, abs_err)
    return float(corr)


@torch.no_grad()
def per_layer_acceptance_rate(diagnostics: dict) -> list:
    """E[a_l] for each layer, averaged over batch and (if vector gate) feature dims."""
    return [a.mean().item() for a in diagnostics["a"]]


def inference_overhead(model, plain_model, x: torch.Tensor, n_repeats: int = 50) -> float:
    """Relative wall-clock overhead of `model` vs `plain_model` on the same input.

    Returns (model_time / plain_model_time). >1 means the Bayesian model is slower.
    """
    model.eval()
    plain_model.eval()

    with torch.no_grad():
        for _ in range(5):
            model(x)
            plain_model(x)

        start = time.perf_counter()
        for _ in range(n_repeats):
            model(x)
        model_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(n_repeats):
            plain_model(x)
        plain_time = time.perf_counter() - start

    return model_time / max(plain_time, 1e-9)


def evaluate_split(model, x: torch.Tensor, y: torch.Tensor, n_mc_samples: int = 200) -> dict:
    """Compute the full metric suite on one data split."""
    model.eval()
    with torch.no_grad():
        y_hat_point, diagnostics = model(x, return_diagnostics=True)

    mean, var = predictive_mean_and_var(model, x, n_samples=n_mc_samples)
    std = torch.sqrt(var.clamp_min(1e-12))

    return {
        "rmse": rmse(y_hat_point, y),
        "max_error": max_error(y_hat_point, y),
        "q95_error": quantile_error(y_hat_point, y, q=0.95),
        "nll": gaussian_nll(mean, var, y),
        "calibration_error": calibration_error(mean, std, y),
        "variance_error_spearman": variance_error_spearman(var, mean, y),
        "acceptance_rate_per_layer": per_layer_acceptance_rate(diagnostics),
    }
