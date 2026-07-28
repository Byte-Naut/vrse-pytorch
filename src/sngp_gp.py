"""RFF + closed-form Bayesian linear regression GP head (Plan3.md §4, STAGE3_PROTOCOL.md).

Locked design (exact RBF-GP excluded this round):
    phi(z) = sqrt(2/rff_dim) * cos(W z + b),   W ~ N(0, 1/l^2), b ~ U[0, 2*pi]
    Lambda = lambda * I + Phi^T Phi / sigma^2
    q      = Phi^T r / sigma^2
    mu_Delta(x) = phi(x)^T Lambda^-1 q
    u(x)        = phi(x)^T Lambda^-1 phi(x)        (latent variance, excludes sigma^2)

`W, b` are drawn ONCE from a fixed seed and shared, frozen, by every method,
stream, and experiment seed (never redrawn). The RBF length scale `l` is the
median-distance heuristic computed on `phi_SN(x_train)` only.

GPPosterior holds the sufficient statistics (Lambda, q) directly so the
deployment/shadow split (src/dual_track.py) can freeze one and incrementally
accumulate the other with no SGD, and later replace the former with the
latter atomically (a plain tensor copy).
"""

import math

import torch

from src.config import SNGPConfig


def median_length_scale(z: torch.Tensor) -> float:
    """Median pairwise Euclidean distance heuristic, computed on ID features only."""
    with torch.no_grad():
        dists = torch.cdist(z, z)
        n = dists.shape[0]
        iu = torch.triu_indices(n, n, offset=1)
        vals = dists[iu[0], iu[1]]
        return vals.median().item()


class RFFMap:
    """Fixed (non-trainable) random Fourier feature map for the RBF kernel.

    Internally float64: with rff_dim=128 fit from ~400 samples and
    1/sigma^2=400, Lambda's condition number reaches ~1e5, which leaves only
    ~2 accurate digits in float32 after the Cholesky solve (verified via the
    batch-vs-incremental equivalence check). float64 is purely an internal
    precision choice -- it changes no locked hyperparameter (sigma^2, rff_dim,
    length scale, lambda are unaffected) and outputs are cast back to
    float32 in GPPosterior.predict so the rest of the codebase (float32
    x/y tensors) needs no changes.
    """

    def __init__(self, in_dim: int, rff_dim: int, length_scale: float, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.W = torch.randn(rff_dim, in_dim, generator=gen, dtype=torch.float64) / length_scale
        self.b = torch.rand(rff_dim, generator=gen, dtype=torch.float64) * 2 * math.pi
        self.rff_dim = rff_dim
        self.length_scale = length_scale

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        proj = z.double() @ self.W.t() + self.b
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(proj)


class GPPosterior:
    """Sufficient statistics (Lambda, q) for the closed-form Bayesian linear regression.

    Held in float64 (see RFFMap docstring); `predict` casts mu/u back to float32.
    """

    def __init__(self, rff_dim: int, noise_var: float, prior_precision: float, jitter: float):
        self.rff_dim = rff_dim
        self.noise_var = noise_var
        self.prior_precision = prior_precision
        self.jitter = jitter
        self.reset_to_prior()

    def reset_to_prior(self):
        self.Lambda = self.prior_precision * torch.eye(self.rff_dim, dtype=torch.float64)
        self.q = torch.zeros(self.rff_dim, dtype=torch.float64)

    def fit_batch(self, Phi: torch.Tensor, r: torch.Tensor):
        """Overwrite the posterior with prior + this ENTIRE batch (used once, at freeze time)."""
        r = r.reshape(-1).double()
        self.Lambda = self.prior_precision * torch.eye(self.rff_dim, dtype=torch.float64) + (Phi.t() @ Phi) / self.noise_var
        self.q = (Phi.t() @ r) / self.noise_var

    def update_incremental(self, phi_x: torch.Tensor, r_x):
        """Accumulate ONE new (feature, residual) pair. No SGD -- pure sufficient-statistic update."""
        phi_x = phi_x.reshape(-1)
        r_val = float(r_x)
        self.Lambda = self.Lambda + torch.outer(phi_x, phi_x) / self.noise_var
        self.q = self.q + phi_x * r_val / self.noise_var

    def _cholesky(self) -> torch.Tensor:
        return torch.linalg.cholesky(self.Lambda + self.jitter * torch.eye(self.rff_dim, dtype=torch.float64))

    def predict(self, Phi: torch.Tensor):
        """Phi: (batch, rff_dim). Returns (mu_Delta, u), each (batch,), cast to float32."""
        L = self._cholesky()
        alpha = torch.cholesky_solve(self.q.reshape(-1, 1), L).reshape(-1)  # Lambda^-1 q
        mu = Phi @ alpha

        X = torch.cholesky_solve(Phi.t(), L)  # Lambda^-1 Phi^T, (rff_dim, batch)
        u = (Phi * X.t()).sum(dim=-1)  # phi^T Lambda^-1 phi per row
        return mu.float(), u.float()

    def clone(self) -> "GPPosterior":
        new = GPPosterior(self.rff_dim, self.noise_var, self.prior_precision, self.jitter)
        new.Lambda = self.Lambda.clone()
        new.q = self.q.clone()
        return new


class GPHead:
    """Combines the fixed RFF map with a GPPosterior. z -> (mu_Delta, u)."""

    def __init__(self, cfg: SNGPConfig, rff_map: RFFMap):
        self.cfg = cfg
        self.rff_map = rff_map
        self.posterior = GPPosterior(cfg.rff_dim, cfg.noise_var, cfg.prior_precision, cfg.jitter)

    def fit_batch(self, z: torch.Tensor, r: torch.Tensor):
        Phi = self.rff_map(z)
        self.posterior.fit_batch(Phi, r)

    def update_incremental(self, z_x: torch.Tensor, r_x):
        phi_x = self.rff_map(z_x.reshape(1, -1))
        self.posterior.update_incremental(phi_x, r_x)

    def predict(self, z: torch.Tensor):
        Phi = self.rff_map(z)
        return self.posterior.predict(Phi)

    def clone(self) -> "GPHead":
        new = GPHead(self.cfg, self.rff_map)  # RFF map is fixed/shared, not cloned
        new.posterior = self.posterior.clone()
        return new


def build_gp_head(cfg: SNGPConfig, z_train: torch.Tensor, r_train: torch.Tensor) -> GPHead:
    """Factory: compute length scale from z_train, build the fixed RFF map, fit the posterior."""
    length_scale = median_length_scale(z_train)
    rff_map = RFFMap(in_dim=z_train.shape[-1], rff_dim=cfg.rff_dim, length_scale=length_scale, seed=cfg.rff_seed)
    head = GPHead(cfg, rff_map)
    head.fit_batch(z_train, r_train)
    return head


def check_batch_incremental_equivalence(cfg: SNGPConfig, z: torch.Tensor, r: torch.Tensor, tol: float = 1e-6) -> dict:
    """Infrastructure-only diagnostic (STAGE3_PROTOCOL.md): batch fit vs. per-sample
    incremental accumulation on the SAME data must agree on both mu_Delta and u to
    within `tol` relative error. Not a method, not part of any verdict.
    """
    length_scale = median_length_scale(z)
    rff_map = RFFMap(in_dim=z.shape[-1], rff_dim=cfg.rff_dim, length_scale=length_scale, seed=cfg.rff_seed)

    batch_head = GPHead(cfg, rff_map)
    batch_head.fit_batch(z, r)

    incr_head = GPHead(cfg, rff_map)
    for i in range(z.shape[0]):
        incr_head.update_incremental(z[i], r[i])

    mu_batch, u_batch = batch_head.predict(z)
    mu_incr, u_incr = incr_head.predict(z)

    mu_rel_err = ((mu_batch - mu_incr).abs() / mu_batch.abs().clamp_min(1e-8)).max().item()
    u_rel_err = ((u_batch - u_incr).abs() / u_batch.abs().clamp_min(1e-8)).max().item()

    return {
        "mu_rel_err": mu_rel_err,
        "u_rel_err": u_rel_err,
        "passed": mu_rel_err < tol and u_rel_err < tol,
    }
