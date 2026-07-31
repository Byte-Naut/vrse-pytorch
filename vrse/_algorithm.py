"""Internal algorithm components: shadow learning, validation, region construction."""
import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vrse.config import VRSEConfig


# ---------------------------------------------------------------------------
# Minimal spectral-normalized feature extractor (phi_SN)
# ---------------------------------------------------------------------------

class _SNLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, coeff: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.utils.parametrizations.spectral_norm(self.linear, name="weight")
        self.coeff = coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.coeff, self.linear.bias)


class _PhiSN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_blocks: int,
                 sn_multiplier: float, spectral_input: bool = False):
        super().__init__()
        if input_dim == hidden_dim:
            self.encoder = nn.Identity()
        elif spectral_input:
            self.encoder = _SNLinear(input_dim, hidden_dim, sn_multiplier)
        else:
            # Preserve the frozen one-dimensional reference path exactly.
            self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(_SNLinear(hidden_dim, hidden_dim, sn_multiplier), nn.ReLU())
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for block in self.blocks:
            h = h + block(h)
        return h

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)


_SPECTRAL_NORM_SETTLE_ITERS = 500


def train_phi_sn(cfg: VRSEConfig, x_train: torch.Tensor, y_train: torch.Tensor,
                 baseline: nn.Module, epochs: Optional[int] = None,
                 lr: Optional[float] = None) -> _PhiSN:
    input_dim = x_train.shape[-1]
    epochs = cfg.phi_epochs if epochs is None else epochs
    lr = cfg.phi_lr if lr is None else lr
    phi = _PhiSN(
        input_dim, cfg.hidden_dim, cfg.n_blocks, cfg.sn_multiplier,
        spectral_input=cfg.spectral_input,
    ).to(x_train.device)
    head = nn.Linear(cfg.hidden_dim, 1).to(x_train.device)
    opt = torch.optim.Adam(list(phi.parameters()) + list(head.parameters()), lr=lr)
    with torch.no_grad():
        r_train = y_train - baseline(x_train)
    phi.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(head(phi(x_train)), r_train)
        loss.backward()
        opt.step()
    # spectral_norm's power iteration only advances one step per forward call
    # and lags behind a weight matrix that moves every optimizer step, so the
    # sigma estimate it ends training with is stale relative to the now-fixed
    # weights. Run settle-only forward passes (no optimizer step) so u/v
    # converge onto the frozen weights before the Lipschitz bound is audited.
    with torch.no_grad():
        for _ in range(_SPECTRAL_NORM_SETTLE_ITERS):
            phi(x_train)
    phi.freeze()
    return phi


# ---------------------------------------------------------------------------
# GP posterior (RFF + closed-form Bayesian linear regression)
# ---------------------------------------------------------------------------

class _RFFMap:
    def __init__(self, in_dim: int, rff_dim: int, length_scale: float, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.W = torch.randn(rff_dim, in_dim, generator=gen, dtype=torch.float64) / length_scale
        self.b = torch.rand(rff_dim, generator=gen, dtype=torch.float64) * 2 * math.pi
        self.rff_dim = rff_dim

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        proj = z.double() @ self.W.t() + self.b
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(proj)


class GPPosterior:
    def __init__(self, rff_dim: int, noise_var: float, prior_precision: float, jitter: float = 1e-6):
        self.rff_dim = rff_dim
        self.noise_var = noise_var
        self.prior_precision = prior_precision
        self.jitter = jitter
        self.reset()

    def reset(self):
        self.Lambda = self.prior_precision * torch.eye(self.rff_dim, dtype=torch.float64)
        self.q = torch.zeros(self.rff_dim, dtype=torch.float64)

    def fit_batch(self, Phi: torch.Tensor, r: torch.Tensor):
        r = r.reshape(-1).double()
        self.Lambda = self.prior_precision * torch.eye(self.rff_dim, dtype=torch.float64) + Phi.t() @ Phi / self.noise_var
        self.q = Phi.t() @ r / self.noise_var

    def update_incremental(self, phi_x: torch.Tensor, r_x: float):
        phi_x = phi_x.reshape(-1)
        self.Lambda = self.Lambda + torch.outer(phi_x, phi_x) / self.noise_var
        self.q = self.q + phi_x * r_x / self.noise_var

    def predict(self, Phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L = torch.linalg.cholesky(self.Lambda + self.jitter * torch.eye(self.rff_dim, dtype=torch.float64))
        alpha = torch.cholesky_solve(self.q.reshape(-1, 1), L).reshape(-1)
        mu = (Phi @ alpha).float()
        X = torch.cholesky_solve(Phi.t(), L)
        u = (Phi * X.t()).sum(dim=-1).float()
        return mu, u

    def clone(self) -> "GPPosterior":
        new = GPPosterior(self.rff_dim, self.noise_var, self.prior_precision, self.jitter)
        new.Lambda = self.Lambda.clone()
        new.q = self.q.clone()
        return new


class GPHead:
    def __init__(self, rff_map: _RFFMap, noise_var: float, prior_precision: float, jitter: float = 1e-6):
        self.rff_map = rff_map
        self.posterior = GPPosterior(rff_map.rff_dim, noise_var, prior_precision, jitter)

    def fit_batch(self, z: torch.Tensor, r: torch.Tensor):
        self.posterior.fit_batch(self.rff_map(z), r)

    def update_incremental(self, z_x: torch.Tensor, r_x: float):
        phi_x = self.rff_map(z_x.reshape(1, -1)).squeeze(0)
        self.posterior.update_incremental(phi_x, r_x)

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.posterior.predict(self.rff_map(z))

    def clone(self) -> "GPHead":
        new = GPHead(self.rff_map, self.posterior.noise_var, self.posterior.prior_precision, self.posterior.jitter)
        new.posterior = self.posterior.clone()
        return new


def _median_length_scale(z: torch.Tensor, max_points: int = 2048) -> float:
    with torch.no_grad():
        if z.shape[0] < 2:
            raise ValueError("At least two embeddings are required to estimate length scale.")
        if z.shape[0] > max_points:
            # Deterministic coverage of the already-deterministic input order;
            # avoids an O(N^2) allocation on real streams.
            idx = torch.linspace(0, z.shape[0] - 1, max_points, device=z.device).long()
            z = z.index_select(0, idx)
        dists = torch.cdist(z, z)
        n = dists.shape[0]
        iu = torch.triu_indices(n, n, offset=1)
        value = dists[iu[0], iu[1]].median().item()
        return max(float(value), 1e-6)


def build_gp_head(cfg: VRSEConfig, z_train: torch.Tensor, r_train: torch.Tensor) -> GPHead:
    ls = _median_length_scale(z_train, max_points=cfg.length_scale_max_points)
    rff_map = _RFFMap(z_train.shape[-1], cfg.rff_dim, ls, seed=cfg.random_seed)
    prior_precision = 1.0 / (cfg.prior_std ** 2)
    head = GPHead(rff_map, cfg.noise_std ** 2, prior_precision)
    head.fit_batch(z_train, r_train)
    return head


# ---------------------------------------------------------------------------
# Tolerance-limit tau (distribution-free one-sided)
# ---------------------------------------------------------------------------

def _tolerance_limit_tau(u: torch.Tensor, p0: float = 0.95, confidence: float = 0.95) -> float:
    """Wilks' one-sided distribution-free tolerance limit.

    Raises when the calibration set is too small to support ``(p0,
    confidence)`` instead of silently degrading to a weaker threshold. The
    boundary behavior is covered by deterministic order-statistic tests.
    """
    from scipy.stats import binom
    n = u.numel()
    k = int(binom.ppf(confidence, n, p0)) + 1
    if k > n:
        raise ValueError(
            f"Tolerance limit requires the {k}-th of {n} order statistics, which exceeds the "
            f"calibration sample size -- increase the calibration set size."
        )
    sorted_u, _ = torch.sort(u.reshape(-1))
    return sorted_u[k - 1].item()


# ---------------------------------------------------------------------------
# Authorized region
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DeploymentSnapshot:
    """The single unit moved by promote()/revoke(). Binds exactly what was
    validated -- deploying anything else (e.g. the live shadow head, which
    keeps changing via observe()) would let an unvalidated candidate serve.
    Depth is 1 by design: only the most recently replaced snapshot is kept,
    so revoke() is not a repeatable-rollback stack; a second consecutive
    revoke() must error."""
    deploy_head: "GPHead"
    authorized_region: Any
    tau: float
    config_version: str


@dataclass(frozen=True)
class AuthorizedRegion:
    x_lo: float
    x_hi: float
    tau_region: float

    def contains(self, x: torch.Tensor, phi_sn: Optional[_PhiSN] = None,
                 head: Optional[GPHead] = None) -> torch.Tensor:
        flat = x.reshape(-1)
        return (flat >= self.x_lo) & (flat <= self.x_hi)


@dataclass(frozen=True)
class KNNFeatureRegion:
    """Fixed high-dimensional support set used by the C-MAPSS benchmark.

    The region is a sublevel set of fifth-neighbour distance in the frozen
    PhiSN space, intersected with a frozen candidate-uncertainty sublevel
    set.  All tensors are copied into the deployment snapshot at evaluate()
    time; subsequent shadow updates therefore cannot move the served region.
    """

    prototypes: torch.Tensor
    k: int
    radius: float
    tau_region: float

    def contains(self, x: torch.Tensor, phi_sn: Optional[_PhiSN] = None,
                 head: Optional[GPHead] = None) -> torch.Tensor:
        if phi_sn is None or head is None:
            raise ValueError("KNNFeatureRegion.contains requires phi_sn and GP head.")
        with torch.no_grad():
            z = phi_sn(x)
            prototypes = self.prototypes.to(device=z.device, dtype=z.dtype)
            d = torch.cdist(z, prototypes)
            d_k = torch.topk(d, k=self.k, largest=False, dim=1).values[:, -1]
            _, u = head.predict(z)
        return (d_k <= self.radius) & (u.to(d_k.device) <= self.tau_region)


def spectral_norm_audit(phi_sn: _PhiSN) -> dict:
    """Return observed operator norms for every constrained linear layer."""
    audit = {}
    for name, module in phi_sn.named_modules():
        if isinstance(module, _SNLinear):
            with torch.no_grad():
                weight = module.linear.weight.detach() * module.coeff
                audit[name] = float(torch.linalg.matrix_norm(weight, ord=2).item())
    return audit


# ---------------------------------------------------------------------------
# _ShadowLearner
# ---------------------------------------------------------------------------

class _ShadowLearner:
    """Wraps a GPHead shadow and accumulates incremental updates."""

    def __init__(self, gp_head: GPHead, phi_sn: _PhiSN, baseline: nn.Module):
        self.gp_head = gp_head
        self.phi_sn = phi_sn
        self.baseline = baseline
        self._x_seen: list = []

    def update(self, x: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            r = y - self.baseline(x)
        z = self.phi_sn(x)
        for i in range(x.shape[0]):
            self.gp_head.update_incremental(z[i], float(r[i].reshape(-1)[0]))
            self._x_seen.append(x[i].detach())

    def observed_x(self) -> Optional[torch.Tensor]:
        if not self._x_seen:
            return None
        return torch.stack(self._x_seen)


# ---------------------------------------------------------------------------
# _SupportBuilder (observed-span-first reference)
# ---------------------------------------------------------------------------

_SCAN_POINTS = 1025
_SCAN_DOMAIN = (-8.0, 7.0)   # frozen one-dimensional reference domain
_MIN_SPAN_WIDTH = 1e-6
_EXACT_FIDELITY_TOL = 1e-6


def _intervals_overlap(lo: float, hi: float, protected_ranges: tuple) -> bool:
    """Closed-interval overlap: touching a protected boundary counts as overlap."""
    return any(lo <= float(hi_p) and hi >= float(lo_p) for lo_p, hi_p in protected_ranges)


def build_observed_span_region(
    phi_sn: _PhiSN,
    shadow_head: GPHead,
    tau_deploy: float,
    x_shadow_train: torch.Tensor,
    x_promotion_val: torch.Tensor,
    protected_id_ranges: tuple = (),
    scan_domain: tuple = _SCAN_DOMAIN,
) -> Optional[AuthorizedRegion]:
    """Return the observed-span authorized region after structural and uncertainty audits.

    `protected_id_ranges`: closed intervals that must not overlap the candidate region
    (typically the ID training ranges from the data config). Touching a boundary fails.
    `scan_domain`: the outer boundary; an observed span that reaches its edges fails
    (the scan would have no margin to audit beyond the observations).
    """
    observed = torch.cat([x_shadow_train.reshape(-1), x_promotion_val.reshape(-1)])
    if observed.numel() == 0 or not bool(torch.isfinite(observed).all()):
        return None
    lo = float(observed.min().item())
    hi = float(observed.max().item())
    if hi - lo < _MIN_SPAN_WIDTH:
        return None
    # Retain the frozen reference implementation's defensive checks.
    if lo <= float(scan_domain[0]) or hi >= float(scan_domain[1]):
        return None  # observed_span_touches_scan_boundary
    if protected_id_ranges and _intervals_overlap(lo, hi, protected_id_ranges):
        return None  # observed_span_overlaps_protected_id
    scan = torch.linspace(
        lo, hi, _SCAN_POINTS,
        dtype=x_shadow_train.dtype,
        device=x_shadow_train.device,
    ).unsqueeze(-1)
    with torch.no_grad():
        _, u_scan = shadow_head.predict(phi_sn(scan))
        _, u_obs = shadow_head.predict(phi_sn(observed.unsqueeze(-1)))
    if not (torch.isfinite(u_scan).all() and torch.isfinite(u_obs).all()):
        return None
    tau_region = max(tau_deploy, float(u_scan.max()), float(u_obs.max()))
    return AuthorizedRegion(lo, hi, tau_region)


def _farthest_point_prototypes(z: torch.Tensor, max_prototypes: int) -> torch.Tensor:
    """Deterministic farthest-point coreset, seeded at the sample nearest the mean."""
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError("z must be a non-empty 2-D embedding tensor.")
    m = min(int(max_prototypes), z.shape[0])
    mean = z.mean(dim=0, keepdim=True)
    first = int(torch.argmin(torch.sum((z - mean) ** 2, dim=1)).item())
    selected = [first]
    min_d2 = torch.sum((z - z[first]) ** 2, dim=1)
    min_d2[first] = -1.0
    for _ in range(1, m):
        idx = int(torch.argmax(min_d2).item())
        selected.append(idx)
        d2 = torch.sum((z - z[idx]) ** 2, dim=1)
        min_d2 = torch.minimum(min_d2, d2)
        min_d2[selected] = -1.0
    return z[torch.as_tensor(selected, device=z.device)].detach().clone()


def build_knn_feature_region(
    phi_sn: _PhiSN,
    shadow_head: GPHead,
    x_shadow_train: torch.Tensor,
    x_promotion_val: torch.Tensor,
    cfg: VRSEConfig,
) -> Optional[KNNFeatureRegion]:
    """Build the fixed high-dimensional authorization support."""
    if x_shadow_train.ndim != 2 or x_promotion_val.ndim != 2:
        return None
    if x_shadow_train.shape[1] != x_promotion_val.shape[1]:
        return None
    if x_shadow_train.shape[0] < cfg.knn_k or x_promotion_val.shape[0] == 0:
        return None
    with torch.no_grad():
        z_train = phi_sn(x_shadow_train)
        z_val = phi_sn(x_promotion_val)
        if not (torch.isfinite(z_train).all() and torch.isfinite(z_val).all()):
            return None
        prototypes = _farthest_point_prototypes(z_train, cfg.max_support_prototypes)
        if prototypes.shape[0] < cfg.knn_k:
            return None
        d = torch.cdist(z_val, prototypes)
        d_k = torch.topk(d, k=cfg.knn_k, largest=False, dim=1).values[:, -1]
        _, u_val = shadow_head.predict(z_val)
    if not (torch.isfinite(d_k).all() and torch.isfinite(u_val).all()):
        return None
    try:
        radius = _tolerance_limit_tau(
            d_k, p0=cfg.tau_percentile / 100.0, confidence=cfg.tau_confidence,
        )
        tau_region = _tolerance_limit_tau(
            u_val, p0=cfg.tau_percentile / 100.0, confidence=cfg.tau_confidence,
        )
    except ValueError:
        return None
    return KNNFeatureRegion(
        prototypes=prototypes,
        k=cfg.knn_k,
        radius=float(radius),
        tau_region=float(tau_region),
    )


# ---------------------------------------------------------------------------
# _Validator
# ---------------------------------------------------------------------------

def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def _q95(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.quantile(torch.abs(a - b), 0.95).item()


def validate_promotion(
    phi_sn: _PhiSN,
    deploy_head: GPHead,
    shadow_head: GPHead,
    tau_deploy: float,
    x_shadow_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_id_guard: torch.Tensor,
    baseline: nn.Module,
    cfg: VRSEConfig,
    protected_id_ranges: tuple = (),
    scan_domain: tuple = _SCAN_DOMAIN,
) -> Tuple[dict, Optional[AuthorizedRegion]]:
    support_kind = cfg.support_kind
    if support_kind == "auto":
        support_kind = "observed_span" if x_val.shape[-1] == 1 else "knn_feature"
    if support_kind == "knn_feature":
        return validate_highdim_promotion(
            phi_sn=phi_sn,
            deploy_head=deploy_head,
            shadow_head=shadow_head,
            tau_deploy=tau_deploy,
            x_shadow_train=x_shadow_train,
            x_val=x_val,
            y_val=y_val,
            x_id_guard=x_id_guard,
            baseline=baseline,
            cfg=cfg,
        )
    with torch.no_grad():
        b_val = baseline(x_val)
        z_val = phi_sn(x_val)
        mu_deploy, u_deploy = deploy_head.predict(z_val)
        a_deploy = (u_deploy <= tau_deploy).to(mu_deploy.dtype)
        y_deploy = b_val.squeeze(-1) + a_deploy * mu_deploy

        mu_shadow, _ = shadow_head.predict(z_val)
        y_shadow = b_val.squeeze(-1) + mu_shadow

    rmse_deploy = _rmse(y_deploy.unsqueeze(-1), y_val)
    rmse_shadow = _rmse(y_shadow.unsqueeze(-1), y_val)
    q95_deploy = _q95(y_deploy.unsqueeze(-1), y_val)
    q95_shadow = _q95(y_shadow.unsqueeze(-1), y_val)

    cond1 = rmse_shadow <= cfg.promotion_rmse_ratio * rmse_deploy
    cond2 = q95_shadow <= cfg.promotion_q95_ratio * q95_deploy

    region = build_observed_span_region(
        phi_sn, shadow_head, tau_deploy, x_shadow_train, x_val,
        protected_id_ranges=protected_id_ranges,
        scan_domain=scan_domain,
    )
    cond3 = region is not None

    cond4 = False
    id_max_diff = float("inf")
    id_overlap_frac = 1.0
    id_route_change_frac = 1.0
    if cond3:
        with torch.no_grad():
            b_guard = baseline(x_id_guard)
            z_guard = phi_sn(x_id_guard)
            mu_d, u_d = deploy_head.predict(z_guard)
            a_d = (u_d <= tau_deploy).to(mu_d.dtype)
            y_before = b_guard.squeeze(-1) + a_d * mu_d

            in_region = region.contains(x_id_guard)
            mu_s, _ = shadow_head.predict(z_guard)
            # After promotion: inside region shadow ungated, outside deploy-gated (vrse
            # routing is more conservative than RegionalExpertService -- region-outside
            # falls back to pure baseline, not deploy-gated; but for the ID invariant
            # check retained for reference-semantics equivalence).
            a_after = torch.where(in_region, torch.ones_like(a_d), a_d)
            mu_after = torch.where(in_region, mu_s, mu_d)
            y_after = b_guard.squeeze(-1) + a_after * mu_after

            route_before = a_d.to(torch.int64)
            route_after = torch.where(in_region, torch.full_like(route_before, 2), route_before)

        id_max_diff = float((y_after - y_before).abs().max().item())
        id_overlap_frac = in_region.float().mean().item()
        id_route_change_frac = (route_after != route_before).float().mean().item()
        # Retain the full four-term protected-region condition:
        # protected_range_overlap is handled upstream in build_observed_span_region,
        # so here we only need the three guard-sample based checks.
        cond4 = (
            id_overlap_frac <= 0.0
            and id_max_diff < _EXACT_FIDELITY_TOL
            and id_route_change_frac == 0.0
        )

    passed = bool(cond1 and cond2 and cond3 and cond4)
    result = {
        "passed": passed,
        "cond1_rmse": bool(cond1),
        "cond2_q95": bool(cond2),
        "cond3_region": bool(cond3),
        "cond4_id_invariant": bool(cond4),
        "rmse_deploy": rmse_deploy,
        "rmse_shadow": rmse_shadow,
        "id_max_diff": id_max_diff,
        "id_overlap_frac": id_overlap_frac,
        "id_route_change_frac": id_route_change_frac,
    }
    return result, region if passed else None


def validate_highdim_promotion(
    phi_sn: _PhiSN,
    deploy_head: GPHead,
    shadow_head: GPHead,
    tau_deploy: float,
    x_shadow_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_id_guard: torch.Tensor,
    baseline: nn.Module,
    cfg: VRSEConfig,
) -> Tuple[dict, Optional[KNNFeatureRegion]]:
    """High-dimensional promotion validator.

    Before authorization the actual VRSE service is the frozen baseline.
    Candidate competence is therefore assessed directly against that service;
    support geometry is audited independently afterward.
    """
    with torch.no_grad():
        b_val = baseline(x_val)
        z_val = phi_sn(x_val)
        mu_shadow, _ = shadow_head.predict(z_val)
        y_shadow = b_val.squeeze(-1) + mu_shadow

    rmse_deploy = _rmse(b_val, y_val)
    rmse_shadow = _rmse(y_shadow.unsqueeze(-1), y_val)
    q95_deploy = _q95(b_val, y_val)
    q95_shadow = _q95(y_shadow.unsqueeze(-1), y_val)
    cond1 = rmse_shadow <= cfg.promotion_rmse_ratio * rmse_deploy
    cond2 = q95_shadow <= cfg.promotion_q95_ratio * q95_deploy

    region = build_knn_feature_region(
        phi_sn=phi_sn,
        shadow_head=shadow_head,
        x_shadow_train=x_shadow_train,
        x_promotion_val=x_val,
        cfg=cfg,
    )
    cond3 = region is not None

    cond4 = False
    id_max_diff = float("inf")
    id_overlap_frac = 1.0
    id_route_change_frac = 1.0
    if cond3:
        with torch.no_grad():
            b_guard = baseline(x_id_guard)
            in_region = region.contains(x_id_guard, phi_sn=phi_sn, head=shadow_head)
            z_guard = phi_sn(x_id_guard)
            mu_s, _ = shadow_head.predict(z_guard)
            y_after = b_guard.squeeze(-1) + torch.where(
                in_region, mu_s, torch.zeros_like(mu_s)
            )
        id_max_diff = float((y_after - b_guard.squeeze(-1)).abs().max().item())
        id_overlap_frac = float(in_region.float().mean().item())
        id_route_change_frac = id_overlap_frac
        cond4 = (
            id_overlap_frac == 0.0
            and id_max_diff < _EXACT_FIDELITY_TOL
            and id_route_change_frac == 0.0
        )

    passed = bool(cond1 and cond2 and cond3 and cond4)
    result = {
        "passed": passed,
        "support_kind": "knn_feature",
        "cond1_rmse": bool(cond1),
        "cond2_q95": bool(cond2),
        "cond3_region": bool(cond3),
        "cond4_id_invariant": bool(cond4),
        "rmse_deploy": rmse_deploy,
        "rmse_shadow": rmse_shadow,
        "q95_deploy": q95_deploy,
        "q95_shadow": q95_shadow,
        "id_max_diff": id_max_diff,
        "id_overlap_frac": id_overlap_frac,
        "id_route_change_frac": id_route_change_frac,
        "region_radius": None if region is None else region.radius,
        "region_tau": None if region is None else region.tau_region,
        "region_prototypes": 0 if region is None else int(region.prototypes.shape[0]),
    }
    return result, region if passed else None
