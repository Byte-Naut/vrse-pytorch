"""Stage-4C paired bundle runner.

Identical budgets and service baselines to Stage-4B; the only changed call is
the observed-span-first promotion evaluator.
"""

import torch

from src.config import Stage4CConfig
from src.methods4b import (
    _batch_from_predictor,
    _id_summary,
    _prequential_ungated_shadow_train,
    _safe_predictor,
    _train_shadow,
    _ungated_predictor,
)
from src.sngp_feature import PhiSN
from src.sngp_gp import GPHead
from src.sngp_service import SafeResidualService
from src.stage4b import PromotedRegion, RegionalExpertService
from src.stage4c import evaluate_observed_span_promotion
from src.streams4b import Stream4B


def run_stage4c_bundle(
    phi_sn: PhiSN,
    initial_gp_head: GPHead,
    tau: float,
    stream: Stream4B,
    cfg: Stage4CConfig,
) -> dict:
    deploy_service = SafeResidualService(phi_sn, initial_gp_head, tau)
    deploy_predict = _safe_predictor(deploy_service)

    safe_shadow_trace = _batch_from_predictor(deploy_predict, stream.shadow_train)
    shadow_head = _train_shadow(phi_sn, initial_gp_head, stream.shadow_train)

    ungated_head = initial_gp_head.clone()
    ungated_id_pre = _id_summary(_ungated_predictor(phi_sn, initial_gp_head), stream.id_pre)
    ungated_shadow_trace = _prequential_ungated_shadow_train(phi_sn, ungated_head, stream.shadow_train)

    evaluation = evaluate_observed_span_promotion(
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
    region = PromotedRegion(**evaluation["region_result"]["region"]) if promoted else None

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
        id_pre = ungated_id_pre if method_name == "Ungated-128" else _id_summary(pre_predict, stream.id_pre)
        methods[method_name] = {
            "method": method_name,
            "stream": stream.name,
            "phases": {
                "shadow_train": shadow_trace,
                "promotion_val": _batch_from_predictor(pre_predict, stream.promotion_val),
                "post_decision": _batch_from_predictor(post_predict, stream.post_decision),
                "second_unknown": _batch_from_predictor(post_predict, stream.second_unknown),
            },
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

    target_id = methods["Promotion-aware"]["id_return"]
    deploy_id = _id_summary(deploy_predict, stream.id_return)
    methods["Promotion-aware"]["id_routing_audit"] = {
        "prediction_max_diff": (target_id["y_hat"] - deploy_id["y_hat"]).abs().max().item(),
        "route_change_frac": (
            target_id["route"].reshape(-1) != deploy_id["route"].reshape(-1)
        ).float().mean().item(),
    }
    return methods
