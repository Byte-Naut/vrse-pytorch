"""Stage-4B regional-expert promotion and routing.

The frozen deployment GP remains authoritative everywhere except inside a
validated promoted region.  Promotion registers a shadow GP as a regional
expert; it never replaces the deployment posterior globally.

Route codes returned by :class:`RegionalExpertService`:
    0 -- frozen deployment gate rejected; exact backbone fallback
    1 -- frozen deployment GP accepted
    2 -- promoted shadow expert, served ungated inside its validated region
"""

from dataclasses import asdict, dataclass
from typing import Optional

import torch

from src.config import Stage4BConfig
from src.dataset import backbone
from src.sngp_feature import PhiSN
from src.sngp_gp import GPHead
from src.sngp_service import SafeResidualService


@dataclass(frozen=True)
class PromotedRegion:
    x_lo: float
    x_hi: float
    tau_region: float

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        return (flat >= self.x_lo) & (flat <= self.x_hi)


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _q95_abs_error(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.quantile(torch.abs(y_hat - y), 0.95).item()


def ungated_gp_predict(phi_sn: PhiSN, gp_head: GPHead, x: torch.Tensor):
    """Return B(x)+mu(x), plus the latent variance, without a gate mask."""
    with torch.no_grad():
        z = phi_sn(x)
        mu, u = gp_head.predict(z)
        b = torch.as_tensor(backbone(x), dtype=mu.dtype, device=mu.device).reshape(-1)
        return (b + mu).unsqueeze(-1), u.unsqueeze(-1)


def build_promoted_region(
    phi_sn: PhiSN,
    shadow_head: GPHead,
    tau_deploy: float,
    x_shadow_train: torch.Tensor,
    x_promotion_val: torch.Tensor,
    protected_id_ranges: tuple,
    cfg: Stage4BConfig,
) -> dict:
    """Build the unique finite low-variance component covering observed support.

    Only shadow-train inputs, promotion-validation inputs, the frozen deploy
    threshold, and shadow uncertainty are used.  No true interval label or
    post-decision input is visible to this function.
    """
    observed = torch.cat([x_shadow_train.reshape(-1), x_promotion_val.reshape(-1)])
    observed_lo = float(observed.min().item())
    observed_hi = float(observed.max().item())

    scan = torch.linspace(
        float(cfg.scan_domain[0]),
        float(cfg.scan_domain[1]),
        int(cfg.scan_points),
        dtype=x_shadow_train.dtype,
        device=x_shadow_train.device,
    ).unsqueeze(-1)
    with torch.no_grad():
        _, u_scan = shadow_head.predict(phi_sn(scan))

    observed_mask = (scan.reshape(-1) >= observed_lo) & (scan.reshape(-1) <= observed_hi)
    if not bool(observed_mask.any()):
        return {"passed": False, "reason": "observed_span_has_no_scan_points", "region": None}

    tau_region = max(float(tau_deploy), float(u_scan[observed_mask].max().item()))
    supported_tensor = u_scan <= tau_region
    scan_flat = scan.reshape(-1)
    for id_lo, id_hi in protected_id_ranges:
        protected = (scan_flat >= float(id_lo)) & (scan_flat <= float(id_hi))
        supported_tensor = supported_tensor & ~protected
    supported = supported_tensor.tolist()

    segments = []
    start = None
    for idx, flag in enumerate(supported):
        if flag and start is None:
            start = idx
        if start is not None and (not flag or idx == len(supported) - 1):
            end = idx if flag and idx == len(supported) - 1 else idx - 1
            segments.append((start, end))
            start = None

    covering = []
    for start_idx, end_idx in segments:
        lo = float(scan[start_idx].item())
        hi = float(scan[end_idx].item())
        if lo <= observed_lo and hi >= observed_hi:
            covering.append((start_idx, end_idx, lo, hi))

    if len(covering) != 1:
        return {
            "passed": False,
            "reason": "covering_component_not_unique",
            "n_covering_components": len(covering),
            "region": None,
        }

    start_idx, end_idx, lo, hi = covering[0]
    if start_idx == 0 or end_idx == len(supported) - 1:
        return {"passed": False, "reason": "component_touches_scan_boundary", "region": None}

    region = PromotedRegion(lo, hi, tau_region)
    return {
        "passed": True,
        "reason": "ok",
        "observed_lo": observed_lo,
        "observed_hi": observed_hi,
        "region": asdict(region),
    }


class RegionalExpertService:
    """Serve a shadow expert only inside a promoted region.

    Outside the region the exact frozen deployment posterior and its original
    hard gate are used.  The shadow posterior cannot affect old-ID predictions
    unless the region builder incorrectly overlaps old ID, which is checked by
    the promotion rule.
    """

    def __init__(
        self,
        phi_sn: PhiSN,
        deploy_head: GPHead,
        tau_deploy: float,
        shadow_head: Optional[GPHead] = None,
        region: Optional[PromotedRegion] = None,
    ):
        if (shadow_head is None) != (region is None):
            raise ValueError("shadow_head and region must either both be set or both be None")
        self.phi_sn = phi_sn
        self.deploy_head = deploy_head
        self.tau_deploy = tau_deploy
        self.shadow_head = shadow_head
        self.region = region

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        z = self.phi_sn(x)
        mu_deploy, u_deploy = self.deploy_head.predict(z)
        a_deploy = u_deploy <= self.tau_deploy

        b = torch.as_tensor(backbone(x), dtype=mu_deploy.dtype, device=mu_deploy.device).reshape(-1)
        residual = a_deploy.to(mu_deploy.dtype) * mu_deploy
        route = a_deploy.to(torch.int64)
        u_shadow = torch.full_like(u_deploy, float("nan"))

        if self.region is not None:
            mu_shadow, u_shadow = self.shadow_head.predict(z)
            in_region = self.region.contains(x)
            residual = torch.where(in_region, mu_shadow, residual)
            route = torch.where(in_region, torch.full_like(route, 2), route)

        y_hat = (b + residual).unsqueeze(-1)
        diagnostics = {
            "u_deploy": u_deploy.unsqueeze(-1),
            "u_shadow": u_shadow.unsqueeze(-1),
            "in_promoted_region": (route == 2).unsqueeze(-1),
        }
        return y_hat, route.unsqueeze(-1), diagnostics


def evaluate_region_promotion(
    phi_sn: PhiSN,
    deploy_head: GPHead,
    shadow_head: GPHead,
    tau_deploy: float,
    x_shadow_train: torch.Tensor,
    x_promotion_val: torch.Tensor,
    y_promotion_val: torch.Tensor,
    x_id_guard: torch.Tensor,
    protected_id_ranges: tuple,
    cfg: Stage4BConfig,
) -> dict:
    """Evaluate competence, constructibility, and routing isolation.

    ID labels are intentionally absent: the shadow expert is not responsible
    for ID.  The ID guard checks the property that promotion leaves actual ID
    service predictions and routes unchanged.
    """
    deploy_service = SafeResidualService(phi_sn, deploy_head, tau_deploy)
    y_deploy_new, _, _ = deploy_service.predict(x_promotion_val)
    y_shadow_new, _ = ungated_gp_predict(phi_sn, shadow_head, x_promotion_val)

    deploy_rmse = _rmse(y_deploy_new, y_promotion_val)
    shadow_rmse = _rmse(y_shadow_new, y_promotion_val)
    deploy_q95 = _q95_abs_error(y_deploy_new, y_promotion_val)
    shadow_q95 = _q95_abs_error(y_shadow_new, y_promotion_val)

    cond1 = shadow_rmse <= cfg.promotion_rmse_ratio * deploy_rmse
    cond2 = shadow_q95 <= cfg.promotion_q95_ratio * deploy_q95

    region_result = build_promoted_region(
        phi_sn,
        shadow_head,
        tau_deploy,
        x_shadow_train,
        x_promotion_val,
        protected_id_ranges,
        cfg,
    )
    cond3 = bool(region_result["passed"])

    id_overlap_frac = 1.0
    protected_range_overlap = True
    id_prediction_max_diff = float("inf")
    id_route_change_frac = 1.0
    if cond3:
        region = PromotedRegion(**region_result["region"])
        protected_range_overlap = any(
            region.x_lo <= float(id_hi) and region.x_hi >= float(id_lo)
            for id_lo, id_hi in protected_id_ranges
        )
        promoted_service = RegionalExpertService(phi_sn, deploy_head, tau_deploy, shadow_head, region)
        y_before, a_before, _ = deploy_service.predict(x_id_guard)
        y_after, route_after, _ = promoted_service.predict(x_id_guard)
        in_region = region.contains(x_id_guard)
        id_overlap_frac = in_region.float().mean().item()
        id_prediction_max_diff = (y_after - y_before).abs().max().item()
        id_route_change_frac = (route_after.reshape(-1) != a_before.reshape(-1).to(torch.int64)).float().mean().item()

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
