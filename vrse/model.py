import copy
import hashlib
import io
import secrets
from dataclasses import asdict
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn

from vrse.config import VRSEConfig
from vrse.proposal import PromotionProposal
from vrse._algorithm import (
    AuthorizedRegion,
    GPHead,
    _DeploymentSnapshot,
    _PhiSN,
    _SCAN_DOMAIN,
    _ShadowLearner,
    build_gp_head,
    build_observed_span_region,
    train_phi_sn,
    validate_promotion,
    _tolerance_limit_tau,
)


class VRSEState(Enum):
    QUARANTINE = auto()
    PENDING_EVAL = auto()
    AUTHORIZED = auto()
    REVOKED = auto()  # audit-only value; revoke() never leaves the model resting here


class VRSEStateError(RuntimeError):
    pass


def _fingerprint(model: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def _fingerprint_gp(head: GPHead) -> str:
    buf = io.BytesIO()
    torch.save({"Lambda": head.posterior.Lambda, "q": head.posterior.q}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def _fingerprint_config(config: VRSEConfig) -> str:
    # Full field set, not just .version -- a hand-edited config field
    # (e.g. promotion_rmse_ratio) between evaluate() and promote() must be
    # caught even if the caller forgot to bump version.
    payload = repr(sorted(asdict(config).items())).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _validate_xy(x: torch.Tensor, y: torch.Tensor) -> None:
    """Input validation at the observe/fit boundary."""
    if x.shape[0] == 0:
        raise ValueError("x must not be empty (got shape %s)." % (tuple(x.shape),))
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must have the same number of samples "
            "(got x=%s, y=%s)." % (tuple(x.shape), tuple(y.shape))
        )
    if x.ndim != 2:
        raise ValueError(f"x must have shape (n, d), got {tuple(x.shape)}")
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"y must have shape (n, 1), got {tuple(y.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("x contains non-finite values.")
    if not torch.isfinite(y).all():
        raise ValueError("y contains non-finite values.")


class VRSEModel(nn.Module):
    """Lifecycle: QUARANTINE -> PENDING_EVAL -> AUTHORIZED (-> REVOKED)."""

    def __init__(self, baseline: nn.Module, config: VRSEConfig,
                 protected_id_ranges: tuple = (),
                 scan_domain: tuple = _SCAN_DOMAIN):
        super().__init__()
        self.config = config
        self._protected_id_ranges = protected_id_ranges
        self._scan_domain = scan_domain
        # Internal immutable copy (W4): decouple from the caller's object so
        # a caller mutating their own reference after wrap() can't move the
        # baseline this model serves out from under it. Frozen immediately:
        # eval() fixes BatchNorm/Dropout, requires_grad_(False) blocks
        # gradients. train() is overridden below to keep it frozen even if
        # this VRSEModel (a submodule owner of _baseline) is recursively
        # flipped back to training mode.
        self._baseline = copy.deepcopy(baseline)
        self._baseline.eval()
        for p in self._baseline.parameters():
            p.requires_grad_(False)
        self._state = VRSEState.QUARANTINE
        self._shadow_update_count = 0
        self._authorized_region: Optional[AuthorizedRegion] = None
        self._phi_sn: Optional[_PhiSN] = None
        self._deploy_head: Optional[GPHead] = None
        # The GPHead fit() built from ID data, before any promotion ever
        # happens. _deploy_head gets reassigned by _atomic_promote(); this
        # reference stays fixed so revoke()'s "restore to pure baseline"
        # path has something real to restore _deploy_head to (see revoke()).
        self._pretrain_deploy_head: Optional[GPHead] = None
        self._shadow_head: Optional[GPHead] = None
        self._tau: Optional[float] = None
        self._learner: Optional[_ShadowLearner] = None
        self._x_shadow_seen: list = []
        self._initialized = False
        # Deployment snapshot bookkeeping (W2): _deploy_snapshot is the
        # currently-served _DeploymentSnapshot (None before any promotion).
        # _prev_snapshot is the one restore point revoke() can roll back to
        # (may itself be None, meaning "restore to pure baseline"). Depth is
        # 1 by design: _revoke_available is consumed by revoke() and only
        # re-armed by the next successful promote().
        self._deploy_snapshot: Optional[_DeploymentSnapshot] = None
        self._prev_snapshot: Optional[_DeploymentSnapshot] = None
        self._revoke_available = False
        # W3: single-use issuance token. Each evaluate() call mints a fresh
        # token and remembers only the latest one; promote() must be handed
        # that exact token and consumes it on use (success or failure). This
        # closes the forged-proposal attack: fingerprint checks alone can't
        # detect a PromotionProposal an external caller hand-built with
        # passed=True and copied-in fingerprints, since a forged proposal
        # has no way to have obtained a token this model actually issued.
        self._pending_issue_token: Optional[str] = None

    @classmethod
    def wrap(cls, baseline: nn.Module, config: VRSEConfig,
             protected_id_ranges: tuple = (),
             scan_domain: tuple = _SCAN_DOMAIN) -> "VRSEModel":
        return cls(baseline, config, protected_id_ranges=protected_id_ranges, scan_domain=scan_domain)

    def train(self, mode: bool = True) -> "VRSEModel":
        """Override nn.Module.train(): calling model.train() must not
        recursively flip the frozen baseline or phi_SN back into training
        mode (BatchNorm running stats and spectral-norm's internal power
        iteration state would otherwise start drifting again, silently
        invalidating the frozen safety reference)."""
        self.training = mode
        # No other submodules currently accept gradient updates in this
        # lifecycle, so there is nothing else to recurse into -- unlike the
        # base implementation, this deliberately does NOT call
        # module.train(mode) on children.
        self._baseline.eval()
        if self._phi_sn is not None:
            self._phi_sn.eval()
        return self

    # ------------------------------------------------------------------
    # Public lifecycle methods
    # ------------------------------------------------------------------

    def fit(self, x_id: torch.Tensor, y_id: torch.Tensor, x_id_calib: torch.Tensor) -> None:
        """Freeze phi_SN and the deploy GP on `(x_id, y_id)`, then calibrate
        tau on the independent `x_id_calib` batch. Must be called exactly
        once, before any observe()/evaluate() call, and both data roles
        must come from the frozen ID domain -- never from the new region
        that observe() will later train the shadow on.

        `x_id_calib` must be an independent batch from `x_id`. Too few points raises ValueError
        rather than silently returning a degenerate threshold.
        """
        if self._initialized:
            raise VRSEStateError("fit() may only be called once.")
        if self._state != VRSEState.QUARANTINE:
            raise VRSEStateError(f"fit() not allowed in state {self._state}.")
        _validate_xy(x_id, y_id)
        if x_id_calib.shape[0] == 0:
            raise ValueError("x_id_calib must not be empty.")
        if not torch.isfinite(x_id_calib).all():
            raise ValueError("x_id_calib contains non-finite values.")
        cfg = self.config
        self._phi_sn = train_phi_sn(cfg, x_id, y_id, self._baseline)
        with torch.no_grad():
            z_id = self._phi_sn(x_id)
            r_id = (y_id - self._baseline(x_id)).detach()
        self._deploy_head = build_gp_head(cfg, z_id, r_id)
        self._pretrain_deploy_head = self._deploy_head
        self._shadow_head = self._deploy_head.clone()
        self._learner = _ShadowLearner(self._shadow_head, self._phi_sn, self._baseline)
        with torch.no_grad():
            z_calib = self._phi_sn(x_id_calib)
            _, u_calib = self._deploy_head.predict(z_calib)
        self._tau = _tolerance_limit_tau(
            u_calib, p0=cfg.tau_percentile / 100.0, confidence=cfg.tau_confidence
        )
        self._initialized = True

    def observe(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not self._initialized:
            raise VRSEStateError("Call fit() before observe().")
        if self._state == VRSEState.REVOKED:
            raise VRSEStateError("Cannot observe after revocation.")
        _validate_xy(x, y)
        self._shadow_update(x, y)
        # Count samples, not calls: batch size must not change promotion semantics.
        self._shadow_update_count += x.shape[0]

    def evaluate(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        guard_x: torch.Tensor,
    ) -> PromotionProposal:
        if not self._initialized:
            raise VRSEStateError("Call fit() before evaluate().")
        if self._state not in (VRSEState.QUARANTINE, VRSEState.PENDING_EVAL, VRSEState.AUTHORIZED):
            raise VRSEStateError(f"evaluate() not allowed in state {self._state}.")
        # Freeze the *real* trained candidate (the live shadow head's
        # sufficient statistics), not an inert, never-trained baseline copy.
        candidate_head = self._shadow_head.clone()
        candidate_fp = _fingerprint_gp(candidate_head)
        result, region = self._validate(x_val, y_val, guard_x)
        cond_min_updates = self._shadow_update_count >= self.config.min_shadow_updates
        result = {**result, "cond_min_shadow_updates": cond_min_updates}
        passed = result.get("passed", False) and cond_min_updates
        result["passed"] = passed
        if not cond_min_updates:
            region = None
        snapshot = None
        if passed:
            snapshot = _DeploymentSnapshot(
                deploy_head=candidate_head,
                authorized_region=region,
                tau=self._tau,
                config_version=self.config.version,
            )
        # Mint a fresh single-use token for this proposal. Only the latest
        # minted token is remembered -- an older proposal from an earlier
        # evaluate() becomes unpromotable the moment a new evaluate() runs,
        # in addition to going stale via the fingerprint recompute below.
        token = secrets.token_hex(16)
        self._pending_issue_token = token
        proposal = PromotionProposal(
            baseline_fingerprint=_fingerprint(self._baseline),
            deployment_snapshot=snapshot,
            candidate_fingerprint=candidate_fp,
            config_version=self.config.version,
            config_fingerprint=_fingerprint_config(self.config),
            authorized_region=region,
            validation_result=result,
            passed=passed,
            shadow_update_count=self._shadow_update_count,
            issue_token=token,
        )
        # evaluate() must never stop a currently-served candidate: only
        # advance QUARANTINE -> PENDING_EVAL (first-ever evaluation).
        # Re-evaluating while AUTHORIZED or PENDING_EVAL leaves state as-is.
        if self._state == VRSEState.QUARANTINE:
            self._state = VRSEState.PENDING_EVAL
        return proposal

    def promote(self, proposal: PromotionProposal) -> bool:
        if self._state not in (VRSEState.PENDING_EVAL, VRSEState.AUTHORIZED):
            raise VRSEStateError(f"promote() not allowed in state {self._state}.")
        # Forged-proposal check first: a hand-built PromotionProposal with
        # copied-in fingerprints has no way to present a token this model
        # actually minted. Consumed unconditionally below (success or
        # failure) so a rejected proposal can't be retried unmodified.
        token = self._pending_issue_token
        self._pending_issue_token = None
        if token is None or proposal.issue_token != token:
            raise VRSEStateError("Unrecognized or already-consumed issuance token.")
        # Stale-proposal checks: recompute every fingerprint from *live*
        # state and compare, rather than trusting values embedded in the
        # proposal (which came from evaluate()-time state that observe()
        # or a config edit may have since moved past).
        if proposal.baseline_fingerprint != _fingerprint(self._baseline):
            raise VRSEStateError("Baseline has changed since proposal was created.")
        if proposal.config_fingerprint != _fingerprint_config(self.config):
            raise VRSEStateError("Config has changed since proposal was created.")
        if proposal.deployment_snapshot is not None:
            live_candidate_fp = _fingerprint_gp(self._shadow_head)
            if proposal.candidate_fingerprint != live_candidate_fp:
                raise VRSEStateError(
                    "Candidate is stale -- shadow head changed via observe() since evaluate()."
                )
        if not proposal.passed:
            # Rejecting a new candidate must not un-serve an already
            # AUTHORIZED deployment; only fall back to QUARANTINE if there
            # was never a promoted snapshot to begin with.
            self._state = VRSEState.AUTHORIZED if self._deploy_snapshot is not None else VRSEState.QUARANTINE
            return False
        self._prev_snapshot = self._deploy_snapshot
        self._atomic_promote(proposal)
        self._revoke_available = True
        self._state = VRSEState.AUTHORIZED
        return True

    def revoke(self) -> None:
        if self._state != VRSEState.AUTHORIZED:
            raise VRSEStateError(f"revoke() not allowed in state {self._state}.")
        if not self._revoke_available:
            raise VRSEStateError(
                "revoke() has no further rollback available -- depth is 1, "
                "and the single restore point was already consumed."
            )
        restore = self._prev_snapshot
        self._deploy_snapshot = restore
        if restore is None:
            # "Restore to pure baseline" means restore the fit()-time GP
            # head, not leave _deploy_head as None -- evaluate() calls
            # validate_promotion() unconditionally regardless of state, and
            # that always needs a real deploy_head to compare the shadow
            # against, even pre-promotion.
            self._deploy_head = self._pretrain_deploy_head
            self._authorized_region = None
            self._state = VRSEState.QUARANTINE
        else:
            self._deploy_head = restore.deploy_head
            self._authorized_region = restore.authorized_region
            self._state = VRSEState.AUTHORIZED
        self._prev_snapshot = None
        self._revoke_available = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._route(x)

    @torch.no_grad()
    def review_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Flag inputs unfamiliar to the baseline and not currently served.

        Baseline familiarity uses the frozen fit-time feature map, residual
        head, and calibration threshold. Current coverage uses route_mask(), so
        promotion removes served inputs from review and revoke restores them.
        A True value means "send for review", not universal OOD proof.
        """
        if not self._initialized:
            raise VRSEStateError("Call fit() before review_mask().")
        if x.ndim != 2:
            raise ValueError(f"x must have shape (n, d), got {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("x must not be empty.")
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values.")
        z = self._phi_sn(x)
        _, uncertainty = self._pretrain_deploy_head.predict(z)
        baseline_unfamiliar = uncertainty.to(x.device) > self._tau
        return baseline_unfamiliar & ~self.route_mask(x)

    @torch.no_grad()
    def route_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the internal authorization mask for audit/experiments.

        This is diagnostic state, not a new replaceable routing protocol.
        """
        if self._state != VRSEState.AUTHORIZED or self._authorized_region is None:
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        return self._authorized_region.contains(
            x, phi_sn=self._phi_sn, head=self._deploy_head
        )

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _shadow_update(self, x: torch.Tensor, y: torch.Tensor):
        self._learner.update(x, y)
        self._x_shadow_seen.append(x.detach())

    def _validate(self, x_val: torch.Tensor, y_val: torch.Tensor, guard_x: torch.Tensor):
        x_shadow = torch.cat(self._x_shadow_seen, dim=0) if self._x_shadow_seen else x_val
        return validate_promotion(
            phi_sn=self._phi_sn,
            deploy_head=self._deploy_head,
            shadow_head=self._shadow_head,
            tau_deploy=self._tau,
            x_shadow_train=x_shadow,
            x_val=x_val,
            y_val=y_val,
            x_id_guard=guard_x,
            baseline=self._baseline,
            cfg=self.config,
            protected_id_ranges=self._protected_id_ranges,
            scan_domain=self._scan_domain,
        )

    def _atomic_promote(self, proposal: PromotionProposal):
        snapshot = proposal.deployment_snapshot
        self._deploy_head = snapshot.deploy_head
        self._authorized_region = snapshot.authorized_region
        self._deploy_snapshot = snapshot

    @torch.no_grad()
    def _route(self, x: torch.Tensor) -> torch.Tensor:
        b = self._baseline(x)
        # Serve GP residuals only in AUTHORIZED state with a valid region.
        # Outside that state, return exact baseline to guarantee the no-serve invariant.
        if self._state != VRSEState.AUTHORIZED or self._authorized_region is None:
            return b
        z = self._phi_sn(x)
        in_region = self.route_mask(x)  # (n,) bool
        # Serve the frozen promoted snapshot (_deploy_head), NOT the live
        # _shadow_head -- the shadow keeps learning via observe() even in
        # AUTHORIZED state, and must never silently change what's served
        # without going through evaluate()/promote() again.
        mu_d, _ = self._deploy_head.predict(z)             # (n,)
        # Outside region: add zero residual → exact baseline; inside: add deploy mean
        residual = torch.where(in_region, mu_d, torch.zeros_like(mu_d))
        return b + residual.unsqueeze(-1)  # (n,1)
