"""The four Stage-3 online methods (Plan3.md §6, results/STAGE3_PROTOCOL.md).

Each method function takes (phi_sn, initial_gp_head, tau, stream, cfg) and
walks the SAME five-phase stream (src/streams3.py), strict predict-then-update
one point at a time where the method actually learns online. Returns a
uniform result dict so src/metrics3.py can score all four identically:

    {
      "method": str,
      "id_pre_rmse": float, "id_return_rmse": float,
      "id_pre_a_mean": float, "id_return_a_mean": float,
      "service_y_hat": (256, 1) tensor, "service_y_true": (256, 1) tensor,
      "service_a": (256, 1) tensor, "service_u": (256, 1) tensor, "service_x": (256, 1) tensor,
      "promotion_info": {"promoted": bool|None, "promotion_step": int|None, "evaluation": dict|None},
    }

The 256 "service" points are shadow-train(128) + promotion-validation(64) +
post-decision(64), in that prequential order. The ID-anchor phase (64 points)
is NEVER part of this 256-point sequence -- it exists solely to feed the
promotion rule's ID-anchor condition (Shadow-validated only).

`initial_gp_head` is never mutated in place by any of these functions:
GPHead/GPPosterior always reassign `self.Lambda`/`self.q` rather than
mutating tensors in place, and every method either never updates it
(Static-safe), routes updates through a `.clone()` (Online-ungated), or
routes updates through `DualTrack`'s own `shadow_head.clone()` (Shadow-count,
Shadow-validated) -- so the same frozen checkpoint object can be reused
across methods/streams/seeds without cross-contamination.
"""

import torch

from src.config import Stage3Config
from src.dataset import backbone
from src.dual_track import DualTrack, evaluate_promotion
from src.sngp_feature import PhiSN
from src.sngp_gp import GPHead
from src.sngp_service import SafeResidualService, residual_target
from src.streams3 import Stream3


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def _predict_only(service: SafeResidualService, x: torch.Tensor, y: torch.Tensor):
    """Returns (rmse, mean_acceptance) for a predict-only batch (ID pre/return probes)."""
    y_hat, a, _ = service.predict(x)
    return _rmse(y_hat, y), a.mean().item()


def _ungated_predict(phi_sn: PhiSN, gp_head: GPHead, x: torch.Tensor):
    """y_hat = B(x) + mu_Delta(x), unconditionally (a=1). u still computed, diagnostic-only."""
    with torch.no_grad():
        z = phi_sn(x)
        mu, u = gp_head.predict(z)
        b = backbone(x).squeeze(-1)
        y_hat = (b + mu).unsqueeze(-1)
    return y_hat, torch.ones_like(y_hat), u.unsqueeze(-1)


def _stack_phases(*phase_lists):
    y_hat = torch.cat([t for lst in phase_lists for t in lst[0]], dim=0)
    y_true = torch.cat([t for lst in phase_lists for t in lst[1]], dim=0)
    a = torch.cat([t for lst in phase_lists for t in lst[2]], dim=0)
    u = torch.cat([t for lst in phase_lists for t in lst[3]], dim=0)
    x = torch.cat([t for lst in phase_lists for t in lst[4]], dim=0)
    return y_hat, y_true, a, u, x


def static_safe(phi_sn: PhiSN, initial_gp_head: GPHead, tau: float, stream: Stream3, cfg: Stage3Config) -> dict:
    service = SafeResidualService(phi_sn, initial_gp_head, tau)

    id_pre_rmse, id_pre_a_mean = _predict_only(service, *stream.id_pre)

    phases = []
    for x, y in (stream.shadow_train, stream.promotion_val, stream.post_decision):
        y_hats, y_trues, a_s, u_s, x_s = [], [], [], [], []
        for i in range(x.shape[0]):
            xi, yi = x[i : i + 1], y[i : i + 1]
            y_hat, a, u = service.predict(xi)
            y_hats.append(y_hat)
            y_trues.append(yi)
            a_s.append(a)
            u_s.append(u)
            x_s.append(xi)
        phases.append((y_hats, y_trues, a_s, u_s, x_s))

    service_y_hat, service_y_true, service_a, service_u, service_x = _stack_phases(*phases)

    id_return_rmse, id_return_a_mean = _predict_only(service, *stream.id_return)

    return {
        "method": "Static-safe",
        "id_pre_rmse": id_pre_rmse,
        "id_return_rmse": id_return_rmse,
        "id_pre_a_mean": id_pre_a_mean,
        "id_return_a_mean": id_return_a_mean,
        "service_y_hat": service_y_hat,
        "service_y_true": service_y_true,
        "service_a": service_a,
        "service_u": service_u,
        "service_x": service_x,
        "promotion_info": {"promoted": False, "promotion_step": None, "evaluation": None},
    }


def online_ungated(phi_sn: PhiSN, initial_gp_head: GPHead, tau: float, stream: Stream3, cfg: Stage3Config) -> dict:
    online_head = initial_gp_head.clone()  # never mutate the shared frozen checkpoint

    y_hat0, _, _ = _ungated_predict(phi_sn, online_head, stream.id_pre[0])
    id_pre_rmse = _rmse(y_hat0, stream.id_pre[1])

    phases = []
    for x, y in (stream.shadow_train, stream.promotion_val, stream.post_decision):
        y_hats, y_trues, a_s, u_s, x_s = [], [], [], [], []
        for i in range(x.shape[0]):
            xi, yi = x[i : i + 1], y[i : i + 1]
            y_hat, a, u = _ungated_predict(phi_sn, online_head, xi)
            y_hats.append(y_hat)
            y_trues.append(yi)
            a_s.append(a)
            u_s.append(u)
            x_s.append(xi)
            r = residual_target(xi, yi)
            with torch.no_grad():
                z = phi_sn(xi)
            online_head.update_incremental(z.squeeze(0), r.reshape(-1)[0])
        phases.append((y_hats, y_trues, a_s, u_s, x_s))

    service_y_hat, service_y_true, service_a, service_u, service_x = _stack_phases(*phases)

    y_hat_ret, _, _ = _ungated_predict(phi_sn, online_head, stream.id_return[0])
    id_return_rmse = _rmse(y_hat_ret, stream.id_return[1])

    return {
        "method": "Online-ungated",
        "id_pre_rmse": id_pre_rmse,
        "id_return_rmse": id_return_rmse,
        "id_pre_a_mean": 1.0,
        "id_return_a_mean": 1.0,
        "service_y_hat": service_y_hat,
        "service_y_true": service_y_true,
        "service_a": service_a,
        "service_u": service_u,
        "service_x": service_x,
        "promotion_info": {"promoted": None, "promotion_step": None, "evaluation": None},
    }


def _shadow_based_run(
    method_name: str, force_unconditional: bool, phi_sn: PhiSN, initial_gp_head: GPHead, tau: float, stream: Stream3, cfg: Stage3Config
) -> dict:
    dual_track = DualTrack(phi_sn, initial_gp_head, tau)

    id_pre_rmse, id_pre_a_mean = _predict_only(dual_track.deployment_service(), *stream.id_pre)

    # --- shadow-train: deploy serves fallback (frozen); labels update shadow only ---
    shadow_train_trace = []
    x_st, y_st = stream.shadow_train
    for i in range(x_st.shape[0]):
        xi, yi = x_st[i : i + 1], y_st[i : i + 1]
        y_hat, a, u = dual_track.deployment_service().predict(xi)
        shadow_train_trace.append((y_hat, yi, a, u, xi))
        dual_track.shadow_update(xi, yi)

    # --- ID anchor: predict only, deploy still frozen, shadow still frozen from here on ---
    id_anchor_x, id_anchor_y = stream.id_anchor

    # --- promotion-validation: shadow frozen, deploy still serves fallback ---
    promotion_val_trace = []
    x_pv, y_pv = stream.promotion_val
    for i in range(x_pv.shape[0]):
        xi, yi = x_pv[i : i + 1], y_pv[i : i + 1]
        y_hat, a, u = dual_track.deployment_service().predict(xi)
        promotion_val_trace.append((y_hat, yi, a, u, xi))

    # --- promotion decision, exactly at point 192 (= end of promotion-validation) ---
    if force_unconditional:
        dual_track.force_promote(support_range_label=stream.name, step=x_st.shape[0] + x_pv.shape[0])
        evaluation = None
    else:
        evaluation = evaluate_promotion(dual_track, x_pv, y_pv, id_anchor_x, id_anchor_y, cfg)
        dual_track.promote_if_passed(evaluation, support_range_label=stream.name, step=x_st.shape[0] + x_pv.shape[0])

    # --- post-decision: serve promoted model if promoted, else continue fallback ---
    post_decision_trace = []
    x_pd, y_pd = stream.post_decision
    for i in range(x_pd.shape[0]):
        xi, yi = x_pd[i : i + 1], y_pd[i : i + 1]
        y_hat, a, u = dual_track.deployment_service().predict(xi)
        post_decision_trace.append((y_hat, yi, a, u, xi))

    def _phase_from_trace(trace):
        y_hats = [t[0] for t in trace]
        y_trues = [t[1] for t in trace]
        a_s = [t[2] for t in trace]
        u_s = [t[3] for t in trace]
        x_s = [t[4] for t in trace]
        return y_hats, y_trues, a_s, u_s, x_s

    service_y_hat, service_y_true, service_a, service_u, service_x = _stack_phases(
        _phase_from_trace(shadow_train_trace),
        _phase_from_trace(promotion_val_trace),
        _phase_from_trace(post_decision_trace),
    )

    id_return_rmse, id_return_a_mean = _predict_only(dual_track.deployment_service(), *stream.id_return)

    return {
        "method": method_name,
        "id_pre_rmse": id_pre_rmse,
        "id_return_rmse": id_return_rmse,
        "id_pre_a_mean": id_pre_a_mean,
        "id_return_a_mean": id_return_a_mean,
        "service_y_hat": service_y_hat,
        "service_y_true": service_y_true,
        "service_a": service_a,
        "service_u": service_u,
        "service_x": service_x,
        "promotion_info": {
            "promoted": dual_track.promoted,
            "promotion_step": dual_track.promotion_step,
            "evaluation": evaluation,
        },
    }


def shadow_count(phi_sn: PhiSN, initial_gp_head: GPHead, tau: float, stream: Stream3, cfg: Stage3Config) -> dict:
    return _shadow_based_run("Shadow-count", True, phi_sn, initial_gp_head, tau, stream, cfg)


def shadow_validated(phi_sn: PhiSN, initial_gp_head: GPHead, tau: float, stream: Stream3, cfg: Stage3Config) -> dict:
    return _shadow_based_run("Shadow-validated", False, phi_sn, initial_gp_head, tau, stream, cfg)


METHODS = {
    "Static-safe": static_safe,
    "Online-ungated": online_ungated,
    "Shadow-count": shadow_count,
    "Shadow-validated": shadow_validated,
}
