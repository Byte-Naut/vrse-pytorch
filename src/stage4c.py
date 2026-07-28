"""Stage-4C observed-span-first promotion.

This module changes exactly one Stage-4B mechanism: a promoted region is the
continuous min/max hull of shadow-train and promotion-validation inputs.  A
dense uncertainty scan audits that hull, but scan-grid coordinates never
replace its continuous endpoints.  This removes Stage-4B's grid endpoint
coverage failure while keeping the frozen-deploy regional routing unchanged.
"""

from dataclasses import asdict

import torch

from src.config import Stage4CConfig
from src.sngp_feature import PhiSN
from src.sngp_gp import GPHead
from src.sngp_service import SafeResidualService
from src.stage4b import PromotedRegion, RegionalExpertService, ungated_gp_predict


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _q95_abs_error(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.quantile(torch.abs(y_hat - y), 0.95).item()


def _intervals_overlap(lo: float, hi: float, protected_ranges: tuple) -> bool:
    """Closed-interval overlap: touching a protected boundary is a failure."""
    return any(lo <= float(id_hi) and hi >= float(id_lo) for id_lo, id_hi in protected_ranges)


def build_observed_span_region(
    phi_sn: PhiSN,
    shadow_head: GPHead,
    tau_deploy: float,
    x_shadow_train: torch.Tensor,
    x_promotion_val: torch.Tensor,
    protected_id_ranges: tuple,
    cfg: Stage4CConfig,
) -> dict:
    """Return the observed input hull after structural and uncertainty audits.

    The region endpoints come only from inputs available before promotion.
    The scan is used to audit uncertainty inside that fixed interval, never to
    expand it into ID or an adjacent unknown region.
    """
    observed = torch.cat([x_shadow_train.reshape(-1), x_promotion_val.reshape(-1)])
    if observed.numel() == 0 or not bool(torch.isfinite(observed).all()):
        return {"passed": False, "reason": "observed_inputs_empty_or_nonfinite", "region": None}

    observed_lo = float(observed.min().item())
    observed_hi = float(observed.max().item())
    if observed_hi - observed_lo < cfg.observed_span_min_width:
        return {"passed": False, "reason": "observed_span_too_narrow", "region": None}
    if observed_lo <= float(cfg.scan_domain[0]) or observed_hi >= float(cfg.scan_domain[1]):
        return {"passed": False, "reason": "observed_span_touches_scan_boundary", "region": None}
    if _intervals_overlap(observed_lo, observed_hi, protected_id_ranges):
        return {"passed": False, "reason": "observed_span_overlaps_protected_id", "region": None}

    scan = torch.linspace(
        observed_lo,
        observed_hi,
        int(cfg.scan_points),
        dtype=x_shadow_train.dtype,
        device=x_shadow_train.device,
    ).unsqueeze(-1)
    observed_x = observed.unsqueeze(-1)
    with torch.no_grad():
        _, u_scan = shadow_head.predict(phi_sn(scan))
        _, u_observed = shadow_head.predict(phi_sn(observed_x))

    if not bool(torch.isfinite(u_scan).all()) or not bool(torch.isfinite(u_observed).all()):
        return {"passed": False, "reason": "shadow_uncertainty_nonfinite", "region": None}

    max_u_scan = float(u_scan.max().item())
    max_u_observed = float(u_observed.max().item())
    tau_region = max(float(tau_deploy), max_u_scan, max_u_observed)
    region = PromotedRegion(observed_lo, observed_hi, tau_region)
    return {
        "passed": True,
        "reason": "ok",
        "observed_lo": observed_lo,
        "observed_hi": observed_hi,
        "max_u_scan": max_u_scan,
        "max_u_observed": max_u_observed,
        "region": asdict(region),
    }


def evaluate_observed_span_promotion(
    phi_sn: PhiSN,
    deploy_head: GPHead,
    shadow_head: GPHead,
    tau_deploy: float,
    x_shadow_train: torch.Tensor,
    x_promotion_val: torch.Tensor,
    y_promotion_val: torch.Tensor,
    x_id_guard: torch.Tensor,
    protected_id_ranges: tuple,
    cfg: Stage4CConfig,
) -> dict:
    """Stage-4C competence gates plus structural routing isolation."""
    deploy_service = SafeResidualService(phi_sn, deploy_head, tau_deploy)
    y_deploy_new, _, _ = deploy_service.predict(x_promotion_val)
    y_shadow_new, _ = ungated_gp_predict(phi_sn, shadow_head, x_promotion_val)

    deploy_rmse = _rmse(y_deploy_new, y_promotion_val)
    shadow_rmse = _rmse(y_shadow_new, y_promotion_val)
    deploy_q95 = _q95_abs_error(y_deploy_new, y_promotion_val)
    shadow_q95 = _q95_abs_error(y_shadow_new, y_promotion_val)
    cond1 = shadow_rmse <= cfg.promotion_rmse_ratio * deploy_rmse
    cond2 = shadow_q95 <= cfg.promotion_q95_ratio * deploy_q95

    region_result = build_observed_span_region(
        phi_sn,
        shadow_head,
        tau_deploy,
        x_shadow_train,
        x_promotion_val,
        protected_id_ranges,
        cfg,
    )
    cond3 = bool(region_result["passed"])

    protected_range_overlap = True
    id_overlap_frac = 1.0
    id_prediction_max_diff = float("inf")
    id_route_change_frac = 1.0
    if cond3:
        region = PromotedRegion(**region_result["region"])
        protected_range_overlap = _intervals_overlap(region.x_lo, region.x_hi, protected_id_ranges)
        promoted_service = RegionalExpertService(phi_sn, deploy_head, tau_deploy, shadow_head, region)
        y_before, a_before, _ = deploy_service.predict(x_id_guard)
        y_after, route_after, _ = promoted_service.predict(x_id_guard)
        id_overlap_frac = region.contains(x_id_guard).float().mean().item()
        id_prediction_max_diff = (y_after - y_before).abs().max().item()
        id_route_change_frac = (
            route_after.reshape(-1) != a_before.reshape(-1).to(torch.int64)
        ).float().mean().item()

    cond4 = (
        not protected_range_overlap
        and id_overlap_frac <= cfg.max_id_region_overlap_frac
        and id_prediction_max_diff < cfg.exact_fidelity_tol
        and id_route_change_frac == 0.0
    )
    return {
        "cond1_new_region_rmse": bool(cond1),
        "cond2_new_region_q95": bool(cond2),
        "cond3_region_constructible": bool(cond3),
        "cond4_id_routing_invariant": bool(cond4),
        "passed": bool(cond1 and cond2 and cond3 and cond4),
        "deploy_new_rmse": deploy_rmse,
        "shadow_new_rmse": shadow_rmse,
        "deploy_new_q95": deploy_q95,
        "shadow_new_q95": shadow_q95,
        "id_overlap_frac": id_overlap_frac,
        "protected_range_overlap": protected_range_overlap,
        "id_prediction_max_diff": id_prediction_max_diff,
        "id_route_change_frac": id_route_change_frac,
        "region_result": region_result,
    }
