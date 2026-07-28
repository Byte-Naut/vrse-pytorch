"""Stage-3 metrics -- exactly the six quantities + promotion outcome (Plan3.md §8,
results/STAGE3_PROTOCOL.md).

Given one src/methods3.py run dict, compute:
  1. initial_rejection        -- OOD rejection rate over the first 32 service points.
  2. fallback_fidelity         -- max|y_hat - B(x)| on REJECTED service points (a=0).
  3. abrupt_rmse                -- RMSE over the first 32 service points.
  4. post_decision_rmse         -- RMSE over the last 64 service points (post-decision phase).
  5. cumulative_service_mse     -- MSE over all 256 service points.
  6. id_forgetting              -- id_return_rmse - id_pre_rmse.
  7. promoted / promotion_step  -- from the method's promotion_info.

Diagnostics (part of the §9 GO conditions, but never independently a
verdict on their own): post_promotion_new_region_acceptance (mean a over the
post-decision phase) and old_id_acceptance (mean a on the id_return probe).
"""

import torch

from src.dataset import backbone
from src.streams3 import N_POST_DECISION

WINDOW_ABRUPT = 32


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((y_hat - y) ** 2).item()


def compute_run_metrics(run: dict) -> dict:
    y_hat = run["service_y_hat"]
    y_true = run["service_y_true"]
    a = run["service_a"]
    x = run["service_x"]

    b = torch.as_tensor(backbone(x))

    initial_rejection = 1.0 - a[:WINDOW_ABRUPT].mean().item()

    rejected_mask = (a < 0.5).squeeze(-1)
    if rejected_mask.any():
        fallback_fidelity = (y_hat[rejected_mask] - b[rejected_mask]).abs().max().item()
    else:
        fallback_fidelity = float("nan")

    abrupt_rmse = _rmse(y_hat[:WINDOW_ABRUPT], y_true[:WINDOW_ABRUPT])
    post_decision_rmse = _rmse(y_hat[-N_POST_DECISION:], y_true[-N_POST_DECISION:])
    cumulative_service_mse = _mse(y_hat, y_true)
    id_forgetting = run["id_return_rmse"] - run["id_pre_rmse"]

    promotion_info = run["promotion_info"]

    post_promotion_new_region_acceptance = a[-N_POST_DECISION:].mean().item()
    old_id_acceptance = run["id_return_a_mean"]

    return {
        "initial_rejection": initial_rejection,
        "fallback_fidelity": fallback_fidelity,
        "abrupt_rmse": abrupt_rmse,
        "post_decision_rmse": post_decision_rmse,
        "cumulative_service_mse": cumulative_service_mse,
        "id_forgetting": id_forgetting,
        "promoted": promotion_info["promoted"],
        "promotion_step": promotion_info["promotion_step"],
        "post_promotion_new_region_acceptance": post_promotion_new_region_acceptance,
        "old_id_acceptance": old_id_acceptance,
    }
