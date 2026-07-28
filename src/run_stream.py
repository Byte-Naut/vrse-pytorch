"""Glue: run one (stream, method, seed, lr) combination end-to-end.

Strict predict-then-update, one point at a time (batch size 1), matching the
protocol's online-learning framing. Returns everything online_metrics.py needs.
"""

import torch

from src.online import fresh_model_for_seed, set_online_trainable, trainable_params
from src.online_metrics import compute_run_metrics, compute_diagnostics


def _predict_only_rmse(method_fn, model, calib, x: torch.Tensor, y: torch.Tensor, optimizer) -> float:
    """Run a whole batch through the method with update=False, return RMSE.

    Used for the id_pre / id_return segments (protocol: predict-only, no update,
    evaluated as a single batch rather than point-by-point since there is no
    online adaptation happening here).
    """
    out = method_fn(model, calib, x, y, optimizer, update=False)
    y_hat = out["y_hat"]
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def run_stream(method_fn, stream, lr: float, seed: int):
    """Run one full stream (id_pre -> online -> id_return) with the given method.

    method_fn: one of the functions in src/online_methods.py METHODS.
    stream: a src.streams.Stream instance.
    lr: learning rate for the SGD optimizer over trainable (M, mu_b) params.
    seed: only used to select which frozen starting point... but per protocol
          ALL seeds share the same frozen_mean_only_model.pt + calib. seed is
          accepted for bookkeeping/logging only, not used to vary the model init.

    Returns: dict with "metrics" (4-metric dict), "diagnostics" (explanation-only),
             and raw per-step traces for potential downstream plotting.
    """
    model, calib = fresh_model_for_seed()
    set_online_trainable(model)
    params = trainable_params(model)
    optimizer = torch.optim.SGD(params, lr=lr)

    # --- ID pre-test: predict only, no update ---
    x_pre, y_pre = stream.id_pre
    id_pre_rmse = _predict_only_rmse(method_fn, model, calib, x_pre, y_pre, optimizer)

    # --- Online segment: strict predict-then-update, one point at a time ---
    x_on, y_on = stream.online
    n_online = x_on.shape[0]

    online_y_hats, online_y_true, online_a_values, online_grad_norms = [], [], [], []

    for i in range(n_online):
        xi = x_on[i : i + 1]
        yi = y_on[i : i + 1]
        out = method_fn(model, calib, xi, yi, optimizer, update=True)
        online_y_hats.append(out["y_hat"])
        online_y_true.append(yi)
        online_a_values.append(out["a_values"])
        online_grad_norms.append(out["grad_norm"])

    # --- ID return-test: predict only, no update ---
    x_ret, y_ret = stream.id_return
    id_return_rmse = _predict_only_rmse(method_fn, model, calib, x_ret, y_ret, optimizer)

    metrics = compute_run_metrics(online_y_hats, online_y_true, id_pre_rmse, id_return_rmse)
    diagnostics = compute_diagnostics(online_a_values, online_grad_norms)

    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "id_pre_rmse": id_pre_rmse,
        "id_return_rmse": id_return_rmse,
        "online_y_hats": online_y_hats,
        "online_y_true": online_y_true,
        "online_a_values": online_a_values,
        "online_grad_norms": online_grad_norms,
    }
