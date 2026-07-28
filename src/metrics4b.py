"""Metrics for Stage-4B regional-expert runs."""

import torch

from src.dataset import backbone


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _fallback_fidelity(trace: dict) -> float:
    rejected = trace["route"].reshape(-1) == 0
    if not bool(rejected.any()):
        return float("nan")
    b = torch.as_tensor(backbone(trace["x"]), dtype=trace["y_hat"].dtype)
    return (trace["y_hat"][rejected] - b[rejected]).abs().max().item()


def compute_stage4b_metrics(run: dict) -> dict:
    phases = run["phases"]
    first = phases["shadow_train"]
    post = phases["post_decision"]
    second = phases["second_unknown"]

    second_b = torch.as_tensor(backbone(second["x"]), dtype=second["y_hat"].dtype)
    audit = run.get("id_routing_audit", {})
    promoted = run["promotion_info"]["promoted"]

    return {
        "initial_rejection": 1.0 - (first["route"][:32].reshape(-1) != 0).float().mean().item(),
        "initial_fallback_fidelity": _fallback_fidelity(first),
        "post_decision_rmse": _rmse(post["y_hat"], post["y"]),
        "promoted_route_frac": (post["route"].reshape(-1) == 2).float().mean().item(),
        "second_unknown_rmse": _rmse(second["y_hat"], second["y"]),
        "second_unknown_fallback_fidelity": (second["y_hat"] - second_b).abs().max().item(),
        "second_unknown_promoted_route_frac": (
            second["route"].reshape(-1) == 2
        ).float().mean().item(),
        "id_pre_rmse": run["id_pre"]["rmse"],
        "id_return_rmse": run["id_return"]["rmse"],
        "id_prediction_max_diff_vs_frozen": audit.get("prediction_max_diff", float("nan")),
        "id_route_change_frac_vs_frozen": audit.get("route_change_frac", float("nan")),
        "promoted": promoted,
    }
