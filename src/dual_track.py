"""Dual-track deployment/shadow posteriors + the single atomic promotion rule
(Plan3.md §II.3, §7, STAGE3_PROTOCOL.md).

DualTrack holds:
  - `deploy_head`: frozen GP posterior, served via SafeResidualService. NEVER
    touched by online data before an atomic promotion.
  - `shadow_head`: initialized as an exact copy of `deploy_head`, updated ONLY
    via `GPHead.update_incremental` (no SGD). Invisible to the served path.

`evaluate_promotion` implements the single four-condition rule (protocol
§"Promotion rule"), evaluated on 64 promotion-validation samples using the
candidate predictions that WOULD be served if promoted (i.e. the shadow
posterior run through the SAME SafeResidualService pipeline, gate included)
plus an independent ID anchor set. All four must pass; promotion is an atomic
one-shot replacement, never gradual mixing.
"""

from dataclasses import dataclass, field

import torch

from src.config import Stage3Config, SNGPConfig
from src.sngp_feature import PhiSN
from src.sngp_gp import GPHead
from src.sngp_service import SafeResidualService


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _q95_abs_err(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.quantile(torch.abs(y_hat - y), 0.95).item()


class DualTrack:
    def __init__(self, phi_sn: PhiSN, deploy_head: GPHead, tau: float):
        self.phi_sn = phi_sn
        self.deploy_head = deploy_head
        self.shadow_head = deploy_head.clone()
        self.tau = tau
        self.promoted = False
        self.promotion_step = None
        self.support_ranges = []

    def deployment_service(self) -> SafeResidualService:
        return SafeResidualService(self.phi_sn, self.deploy_head, self.tau)

    def candidate_service(self) -> SafeResidualService:
        """The service that WOULD be live if the shadow posterior were promoted now."""
        return SafeResidualService(self.phi_sn, self.shadow_head, self.tau)

    def shadow_update(self, x: torch.Tensor, y: torch.Tensor):
        """x, y: (1, 1) single-sample update. Shadow only -- never touches deploy_head."""
        from src.sngp_service import residual_target

        z = self.phi_sn(x)
        r = residual_target(x, y)
        self.shadow_head.update_incremental(z.squeeze(0), r.reshape(-1)[0])

    def _atomic_promote(self, support_range_label=None, step: int = None):
        self.deploy_head = self.shadow_head.clone()
        self.promoted = True
        self.promotion_step = step
        if support_range_label is not None:
            self.support_ranges.append(support_range_label)

    def force_promote(self, support_range_label=None, step: int = None):
        """Unconditional promotion -- used ONLY by the Shadow-count ablation (no competence check)."""
        self._atomic_promote(support_range_label, step)

    def promote_if_passed(self, promotion_result: dict, support_range_label=None, step: int = None) -> bool:
        if promotion_result["passed"]:
            self._atomic_promote(support_range_label, step)
        return promotion_result["passed"]


def evaluate_promotion(
    dual_track: DualTrack,
    x_new_val: torch.Tensor,
    y_new_val: torch.Tensor,
    x_id_anchor: torch.Tensor,
    y_id_anchor: torch.Tensor,
    cfg: Stage3Config,
) -> dict:
    """The single §7 promotion rule. Returns per-condition bools + raw numbers, never mutates state."""
    deploy_service = dual_track.deployment_service()
    candidate_service = dual_track.candidate_service()

    y_hat_deploy_new, _, _ = deploy_service.predict(x_new_val)
    y_hat_candidate_new, _, u_candidate_new = candidate_service.predict(x_new_val)

    y_hat_deploy_id, _, _ = deploy_service.predict(x_id_anchor)
    y_hat_candidate_id, _, _ = candidate_service.predict(x_id_anchor)

    rmse_deploy_new = _rmse(y_hat_deploy_new, y_new_val)
    rmse_candidate_new = _rmse(y_hat_candidate_new, y_new_val)

    q95_deploy_new = _q95_abs_err(y_hat_deploy_new, y_new_val)
    q95_candidate_new = _q95_abs_err(y_hat_candidate_new, y_new_val)

    rmse_deploy_id = _rmse(y_hat_deploy_id, y_id_anchor)
    rmse_candidate_id = _rmse(y_hat_candidate_id, y_id_anchor)

    support_frac = (u_candidate_new <= dual_track.tau).float().mean().item()

    cond1 = rmse_candidate_new <= cfg.promotion_rmse_ratio * rmse_deploy_new
    cond2 = q95_candidate_new <= q95_deploy_new
    cond3 = rmse_candidate_id <= cfg.promotion_id_anchor_ratio * rmse_deploy_id
    cond4 = support_frac >= cfg.promotion_support_frac

    return {
        "cond1_rmse": cond1,
        "cond2_q95": cond2,
        "cond3_id_anchor": cond3,
        "cond4_support": cond4,
        "passed": bool(cond1 and cond2 and cond3 and cond4),
        "rmse_deploy_new": rmse_deploy_new,
        "rmse_candidate_new": rmse_candidate_new,
        "q95_deploy_new": q95_deploy_new,
        "q95_candidate_new": q95_candidate_new,
        "rmse_deploy_id": rmse_deploy_id,
        "rmse_candidate_id": rmse_candidate_id,
        "support_frac": support_frac,
    }
