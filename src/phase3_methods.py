"""Shared Phase-3 model construction and five-method experiment bundle."""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.phase3_cmapss import Phase3Batch, Phase3Split
from vrse import VRSEConfig, VRSEModel
from vrse._algorithm import _ShadowLearner, spectral_norm_audit


METHODS = ("Frozen", "Online-ungated", "Static-reject", "Shadow-global", "VRSE-KNN")
STREAMS = ("stable_condition", "reversed_condition")
EXACT_TOL = 1e-6


class RULBaseline(nn.Module):
    def __init__(self, input_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_baseline(split: Phase3Split) -> RULBaseline:
    set_seed(split.seed)
    model = RULBaseline(split.id_fit.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(split.seed)
    n = split.id_fit.x.shape[0]
    model.train()
    for _ in range(100):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, 256):
            idx = permutation[start:start + 256]
            optimizer.zero_grad()
            loss = F.mse_loss(model(split.id_fit.x[idx]), split.id_fit.y[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def build_seed_template(split: Phase3Split) -> Tuple[VRSEModel, dict]:
    """Train the shared baseline/PhiSN/fit-time GP exactly once per seed."""
    baseline = train_baseline(split)
    with torch.no_grad():
        residual = split.id_fit.y - baseline(split.id_fit.x)
    noise_std = max(1e-3, 0.10 * float(residual.std(unbiased=False).item()))
    config = VRSEConfig(
        preset="regional_regression_highdim",
        noise_std=noise_std,
        random_seed=split.seed,
        version="0.1.0-phase3",
    )
    model = VRSEModel.wrap(baseline=baseline, config=config)
    started = time.perf_counter()
    model.fit(split.id_fit.x, split.id_fit.y, split.id_calibration.x)
    fit_seconds = time.perf_counter() - started
    norms = spectral_norm_audit(model._phi_sn)
    return model, {
        "seed": split.seed,
        "fit_seconds": fit_seconds,
        "noise_std": noise_std,
        "spectral_norms": norms,
        "config": asdict(config),
    }


def _rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_hat - y) ** 2)).item())


def _mae(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y_hat - y)).item())


def _q95(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.quantile(torch.abs(y_hat - y), 0.95).item())


def _nasa_score(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    d = (y_hat - y).reshape(-1).double()
    early = torch.exp(torch.clamp(-d / 13.0, max=50.0)) - 1.0
    late = torch.exp(torch.clamp(d / 10.0, max=50.0)) - 1.0
    return float(torch.where(d < 0, early, late).sum().item())


def _domain_metrics(y_hat: torch.Tensor, y: torch.Tensor, route: torch.Tensor,
                    baseline_hat: torch.Tensor) -> dict:
    route = route.reshape(-1).bool()
    return {
        "rmse": _rmse(y_hat, y),
        "mae": _mae(y_hat, y),
        "q95": _q95(y_hat, y),
        "nasa_score": _nasa_score(y_hat, y),
        "route_frac": float(route.float().mean().item()),
        "prediction_max_diff_vs_frozen": float((y_hat - baseline_hat).abs().max().item()),
    }


@torch.no_grad()
def _baseline_predict(model: VRSEModel, x: torch.Tensor) -> torch.Tensor:
    return model._baseline(x)


@torch.no_grad()
def _gp_predict(model: VRSEModel, head, x: torch.Tensor,
                gate_tau: float | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    b = model._baseline(x)
    z = model._phi_sn(x)
    mu, u = head.predict(z)
    if gate_tau is None:
        route = torch.ones(x.shape[0], dtype=torch.bool)
    else:
        route = u <= gate_tau
    residual = torch.where(route, mu, torch.zeros_like(mu))
    return b + residual.unsqueeze(-1), route


def _prequential_online_trace(template: VRSEModel, batch: Phase3Batch) -> Tuple[list, object]:
    """Serve the live GP before each ordered chunk, then update in exact stream order."""
    head = template._pretrain_deploy_head.clone()
    learner = _ShadowLearner(head, template._phi_sn, template._baseline)
    trace = []
    for checkpoint, start in enumerate(range(0, batch.x.shape[0], 256)):
        stop = min(start + 256, batch.x.shape[0])
        y_hat, _ = _gp_predict(template, head, batch.x[start:stop], gate_tau=None)
        trace.append({
            "checkpoint": checkpoint,
            "samples_seen_before": start,
            "samples": stop - start,
            "rmse": _rmse(y_hat, batch.y[start:stop]),
        })
        learner.update(batch.x[start:stop], batch.y[start:stop])
    return trace, head


def _routed_subset_metrics(y_hat: torch.Tensor, y: torch.Tensor,
                           baseline_hat: torch.Tensor, route: torch.Tensor) -> dict:
    mask = route.reshape(-1).bool()
    if not bool(mask.any()):
        return {"count": 0, "rmse": float("nan"), "frozen_rmse": float("nan")}
    return {
        "count": int(mask.sum().item()),
        "rmse": _rmse(y_hat[mask], y[mask]),
        "frozen_rmse": _rmse(baseline_hat[mask], y[mask]),
    }


def run_phase3_bundle(template: VRSEModel, split: Phase3Split, stream: str) -> Dict[str, dict]:
    if stream not in STREAMS:
        raise ValueError(f"Unknown stream {stream!r}")
    y_val, y_post_new = split.stream_targets(stream)

    # One canonical isolated candidate for every method comparison.
    vrse_model = copy.deepcopy(template)
    probe_x = split.promotion_validation.x
    probe_baseline = vrse_model._baseline(probe_x).detach()
    isolation_diff = 0.0
    observe_seconds = 0.0
    for start in range(0, split.shadow_observe.x.shape[0], 256):
        stop = min(start + 256, split.shadow_observe.x.shape[0])
        before = vrse_model(probe_x).detach()
        isolation_diff = max(
            isolation_diff, float((before - probe_baseline).abs().max().item())
        )
        observe_started = time.perf_counter()
        vrse_model.observe(
            split.shadow_observe.x[start:stop], split.shadow_observe.y[start:stop]
        )
        observe_seconds += time.perf_counter() - observe_started
        after = vrse_model(probe_x).detach()
        isolation_diff = max(
            isolation_diff, float((after - probe_baseline).abs().max().item())
        )
    candidate_head = vrse_model._shadow_head.clone()

    proposal = vrse_model.evaluate(
        split.promotion_validation.x,
        y_val,
        guard_x=split.id_guard.x,
    )
    capability_passed = bool(
        proposal.validation_result.get("cond1_rmse", False)
        and proposal.validation_result.get("cond2_q95", False)
        and proposal.validation_result.get("cond_min_shadow_updates", False)
    )
    vrse_promoted = vrse_model.promote(proposal)

    online_trace, online_head = _prequential_online_trace(template, split.shadow_observe)
    baseline_mean = float(split.id_fit.y.mean().item())

    with torch.no_grad():
        _, initial_u = template._pretrain_deploy_head.predict(
            template._phi_sn(split.promotion_validation.x)
        )
    initial_rejection = float((initial_u > template._tau).float().mean().item())

    domains = {
        "id_guard": (split.id_guard.x, split.id_guard.y),
        "post_new": (split.post_new.x, y_post_new),
        "post_unknown": (split.post_unknown.x, split.post_unknown.y),
    }
    outputs: Dict[str, dict] = {}
    for method in METHODS:
        method_domains = {}
        for domain_name, (x, y) in domains.items():
            b = _baseline_predict(template, x)
            if method == "Frozen":
                y_hat = b
                route = torch.zeros(x.shape[0], dtype=torch.bool)
            elif method == "Online-ungated":
                y_hat, route = _gp_predict(template, online_head, x, gate_tau=None)
            elif method == "Static-reject":
                y_hat, route = _gp_predict(
                    template, template._pretrain_deploy_head, x, gate_tau=template._tau,
                )
            elif method == "Shadow-global":
                if capability_passed:
                    y_hat, route = _gp_predict(template, candidate_head, x, gate_tau=None)
                else:
                    y_hat = b
                    route = torch.zeros(x.shape[0], dtype=torch.bool)
            else:
                y_hat = vrse_model(x)
                route = vrse_model.route_mask(x)
            metrics = _domain_metrics(y_hat, y, route, b)
            if domain_name == "post_new":
                metrics["routed_subset"] = _routed_subset_metrics(y_hat, y, b, route)
            method_domains[domain_name] = metrics

        outputs[method] = {
            "method": method,
            "stream": stream,
            "seed": split.seed,
            "promoted": (
                bool(vrse_promoted) if method == "VRSE-KNN"
                else bool(capability_passed) if method == "Shadow-global"
                else None
            ),
            "domains": method_domains,
            "initial_rejection": initial_rejection,
            "isolation_max_diff": isolation_diff if method == "VRSE-KNN" else None,
            "observe_seconds": observe_seconds,
            "online_trace": online_trace if method == "Online-ungated" else [],
            "promotion": proposal.validation_result if method == "VRSE-KNN" else None,
            "capability_passed": capability_passed,
            "trivial_id_rmse": float(torch.sqrt(torch.mean(
                (split.id_guard.y - baseline_mean) ** 2
            )).item()),
        }
    return outputs


def seed_preconditions(template: VRSEModel, split: Phase3Split) -> dict:
    with torch.no_grad():
        baseline_id = template(split.id_guard.x)
        constant = torch.full_like(split.id_guard.y, float(split.id_fit.y.mean().item()))
        _, u_new = template._pretrain_deploy_head.predict(
            template._phi_sn(split.promotion_validation.x)
        )
    p1_ratio = _rmse(baseline_id, split.id_guard.y) / max(_rmse(constant, split.id_guard.y), 1e-12)
    p2_rejection = float((u_new > template._tau).float().mean().item())

    candidate = copy.deepcopy(template)
    candidate.observe(split.shadow_observe.x, split.shadow_observe.y)
    proposal = candidate.evaluate(
        split.promotion_validation.x,
        split.promotion_validation.y,
        guard_x=split.id_guard.x,
    )
    norms = spectral_norm_audit(template._phi_sn)
    norm_limit = template.config.sn_multiplier + 1e-3
    return {
        "seed": split.seed,
        "p1_baseline_vs_constant_ratio": p1_ratio,
        "p1_pass": bool(p1_ratio <= 0.80),
        "p2_initial_rejection": p2_rejection,
        "p2_pass": bool(p2_rejection >= 0.90),
        "p3_cond1": bool(proposal.validation_result.get("cond1_rmse", False)),
        "p3_cond2": bool(proposal.validation_result.get("cond2_q95", False)),
        "p3_pass": bool(
            proposal.validation_result.get("cond1_rmse", False)
            and proposal.validation_result.get("cond2_q95", False)
        ),
        "spectral_norms": norms,
        "spectral_pass": bool(norms and all(value <= norm_limit for value in norms.values())),
        "validation": proposal.validation_result,
    }
