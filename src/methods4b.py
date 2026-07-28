"""Paired Stage-4B scenario runner.

One stream/seed bundle trains the shadow posterior once, makes one promotion
decision, and evaluates three service policies on exactly the same artifacts:

* Hard-original: globally replace with the shadow posterior, retain hard gate.
* Promotion-aware: register the shadow only inside the validated region.
* Ungated-128: same 128-label shadow posterior served globally without a gate;
  diagnostic performance upper bound, not a safe policy.
"""

from typing import Callable

import torch

from src.config import Stage4BConfig
from src.dataset import backbone
from src.sngp_feature import PhiSN
from src.sngp_gp import GPHead
from src.sngp_service import SafeResidualService, residual_target
from src.stage4b import (
    PromotedRegion,
    RegionalExpertService,
    evaluate_region_promotion,
    ungated_gp_predict,
)
from src.streams4b import Stream4B


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _batch_from_predictor(predictor: Callable, batch: tuple) -> dict:
    x, y = batch
    y_hat, route, diagnostics = predictor(x)
    return {
        "x": x,
        "y": y,
        "y_hat": y_hat,
        "route": route,
        "diagnostics": diagnostics,
    }


def _safe_predictor(service: SafeResidualService):
    def predict(x):
        y_hat, a, u = service.predict(x)
        return y_hat, a.to(torch.int64), {"u_deploy": u}

    return predict


def _ungated_predictor(phi_sn: PhiSN, head: GPHead):
    def predict(x):
        y_hat, u = ungated_gp_predict(phi_sn, head, x)
        return y_hat, torch.full_like(y_hat, 2, dtype=torch.int64), {"u_shadow": u}

    return predict


def _prequential_ungated_shadow_train(phi_sn: PhiSN, head: GPHead, batch: tuple) -> dict:
    x, y = batch
    y_hats, routes, us = [], [], []
    for i in range(x.shape[0]):
        xi, yi = x[i : i + 1], y[i : i + 1]
        y_hat, u = ungated_gp_predict(phi_sn, head, xi)
        y_hats.append(y_hat)
        routes.append(torch.full_like(y_hat, 2, dtype=torch.int64))
        us.append(u)
        with torch.no_grad():
            z = phi_sn(xi)
        head.update_incremental(z.squeeze(0), residual_target(xi, yi).reshape(-1)[0])
    return {
        "x": x,
        "y": y,
        "y_hat": torch.cat(y_hats, dim=0),
        "route": torch.cat(routes, dim=0),
        "diagnostics": {"u_shadow": torch.cat(us, dim=0)},
    }


def _train_shadow(phi_sn: PhiSN, initial_head: GPHead, batch: tuple) -> GPHead:
    head = initial_head.clone()
    x, y = batch
    for i in range(x.shape[0]):
        xi, yi = x[i : i + 1], y[i : i + 1]
        with torch.no_grad():
            z = phi_sn(xi)
        head.update_incremental(z.squeeze(0), residual_target(xi, yi).reshape(-1)[0])
    return head


def _id_summary(predictor: Callable, batch: tuple) -> dict:
    trace = _batch_from_predictor(predictor, batch)
    return {
        "rmse": _rmse(trace["y_hat"], trace["y"]),
        "y_hat": trace["y_hat"],
        "route": trace["route"],
    }


def run_stage4b_bundle(
    phi_sn: PhiSN,
    initial_gp_head: GPHead,
    tau: float,
    stream: Stream4B,
    cfg: Stage4BConfig,
) -> dict:
    """Run one paired stream/seed bundle and return all three method records."""
    deploy_service = SafeResidualService(phi_sn, initial_gp_head, tau)
    deploy_predict = _safe_predictor(deploy_service)

    # Both safe policies serve the frozen deployment model while the shadow
    # learns invisibly.  The upper-bound method is prequential on the same 128
    # labels and is then frozen; promotion-validation labels are not updates.
    safe_shadow_trace = _batch_from_predictor(deploy_predict, stream.shadow_train)
    shadow_head = _train_shadow(phi_sn, initial_gp_head, stream.shadow_train)

    ungated_head = initial_gp_head.clone()
    ungated_id_pre = _id_summary(_ungated_predictor(phi_sn, initial_gp_head), stream.id_pre)
    ungated_shadow_trace = _prequential_ungated_shadow_train(phi_sn, ungated_head, stream.shadow_train)

    evaluation = evaluate_region_promotion(
        phi_sn=phi_sn,
        deploy_head=initial_gp_head,
        shadow_head=shadow_head,
        tau_deploy=tau,
        x_shadow_train=stream.shadow_train[0],
        x_promotion_val=stream.promotion_val[0],
        y_promotion_val=stream.promotion_val[1],
        x_id_guard=stream.id_anchor[0],
        protected_id_ranges=stream.protected_id_ranges,
        cfg=cfg,
    )
    promoted = bool(evaluation["passed"])
    region = None
    if promoted:
        region = PromotedRegion(**evaluation["region_result"]["region"])

    hard_service = SafeResidualService(phi_sn, shadow_head if promoted else initial_gp_head, tau)
    hard_predict = _safe_predictor(hard_service)

    regional_service = RegionalExpertService(
        phi_sn,
        initial_gp_head,
        tau,
        shadow_head=shadow_head if promoted else None,
        region=region,
    )
    regional_predict = regional_service.predict
    ungated_predict = _ungated_predictor(phi_sn, ungated_head)

    methods = {}
    for method_name, post_predict, shadow_trace in (
        ("Hard-original", hard_predict, safe_shadow_trace),
        ("Promotion-aware", regional_predict, safe_shadow_trace),
        ("Ungated-128", ungated_predict, ungated_shadow_trace),
    ):
        pre_predict = ungated_predict if method_name == "Ungated-128" else deploy_predict
        phases = {
            "shadow_train": shadow_trace,
            "promotion_val": _batch_from_predictor(pre_predict, stream.promotion_val),
            "post_decision": _batch_from_predictor(post_predict, stream.post_decision),
            "second_unknown": _batch_from_predictor(post_predict, stream.second_unknown),
        }
        id_pre = ungated_id_pre if method_name == "Ungated-128" else _id_summary(pre_predict, stream.id_pre)
        methods[method_name] = {
            "method": method_name,
            "stream": stream.name,
            "phases": phases,
            "id_pre": id_pre,
            "id_return": _id_summary(post_predict, stream.id_return),
            "promotion_info": {
                "promoted": promoted if method_name != "Ungated-128" else None,
                "evaluation": evaluation if method_name != "Ungated-128" else None,
                "region": evaluation["region_result"]["region"] if promoted else None,
            },
            "second_unknown_range": stream.second_unknown_range,
            "second_unknown_seed": stream.second_unknown_seed,
        }

    # Structural audit for the target policy: compare it against the frozen
    # deploy service on the exact same ID-return points.
    target_id = methods["Promotion-aware"]["id_return"]
    deploy_id = _id_summary(deploy_predict, stream.id_return)
    methods["Promotion-aware"]["id_routing_audit"] = {
        "prediction_max_diff": (target_id["y_hat"] - deploy_id["y_hat"]).abs().max().item(),
        "route_change_frac": (
            target_id["route"].reshape(-1) != deploy_id["route"].reshape(-1)
        ).float().mean().item(),
    }
    return methods
