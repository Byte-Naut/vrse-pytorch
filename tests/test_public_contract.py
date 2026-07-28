"""Phase-2B 1-D regression tests for the VRSEModel public contract.

They remain the frozen Stage-4C reference path after Phase 3 adds a separate
high-dimensional support geometry in tests/test_phase3_highdim.py.
"""
import copy
import torch
import torch.nn as nn
import pytest

from vrse import VRSEConfig, VRSEModel, VRSEStateError
from vrse.model import VRSEState
from vrse._algorithm import build_observed_span_region, _PhiSN, GPHead, _RFFMap


class _TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


_ID_CALIB_N = 2000  # protocol-recorded value, src/config.py id_calib_n (Stage-3B amendment)


def _make_model():
    baseline = _TinyBaseline()
    config = VRSEConfig(preset="regional_regression")
    return VRSEModel.wrap(baseline, config), baseline


def _true_fn(x: torch.Tensor) -> torch.Tensor:
    return x + 2.0 * torch.sin(3.0 * x)


def _make_fitted_model():
    """fit() on the frozen ID domain [-1, 0], disjoint from both the new
    region [2, 3] used by observe()/evaluate() and the ID guard domain
    [-3, -2]. Role 1 (ID train) and role 2 (ID calib) are independent
    batches, per Phase2B_Plan.md W1."""
    model, baseline = _make_model()
    x_id_train = torch.empty(50, 1).uniform_(-1.0, 0.0)
    y_id_train = _true_fn(x_id_train)
    x_id_calib = torch.empty(_ID_CALIB_N, 1).uniform_(-1.0, 0.0)
    model.fit(x_id_train, y_id_train, x_id_calib)
    return model, baseline


def _make_promotable_model():
    """Build a model whose shadow head is deliberately much better fit than
    deploy on a validation set drawn from the same new-region domain, with
    a guard set drawn from a disjoint ID domain so cond4 (zero ID-region
    overlap) holds. Roles 3/4/5 (new-region shadow train, new-region val,
    ID guard) are independent batches, per Phase2B_Plan.md W1."""
    torch.manual_seed(0)
    model, baseline = _make_fitted_model()

    for _ in range(40):
        x_batch = torch.empty(5, 1).uniform_(2.0, 3.0)
        y_batch = _true_fn(x_batch)
        model.observe(x_batch, y_batch)

    x_val = torch.empty(20, 1).uniform_(2.0, 3.0)
    y_val = _true_fn(x_val)
    x_guard = torch.empty(10, 1).uniform_(-3.0, -2.0)  # disjoint domain

    return model, baseline, x_val, y_val, x_guard


def test_shadow_updates_do_not_change_deployment():
    model, baseline = _make_fitted_model()
    params_before = {k: v.clone() for k, v in baseline.state_dict().items()}
    x = torch.empty(5, 1).uniform_(2.0, 3.0)
    y = _true_fn(x)
    for _ in range(10):
        model.observe(x, y)
    for k, v in baseline.state_dict().items():
        assert torch.equal(v, params_before[k]), f"Baseline param {k} changed after observe()"


def test_no_candidate_served_before_promotion():
    model, baseline = _make_model()
    x = torch.randn(3, 1)
    y_baseline = baseline(x).detach()
    y_model = model(x)
    assert torch.equal(y_model, y_baseline), "Model output differs from baseline before any promotion"


def test_output_outside_authorized_region_equals_baseline():
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    promoted = model.promote(proposal)
    assert promoted is True
    assert model._authorized_region is not None

    # Points outside authorized region ([2, 3]) must match baseline exactly
    # (residual is literally zeros_like -- exact fallback, not merely close).
    x_outside = torch.linspace(50.0, 60.0, 20).unsqueeze(-1)
    y_hat = model(x_outside)
    y_base = baseline(x_outside).detach()
    assert torch.equal(y_hat, y_base), "Output outside authorized region does not equal baseline exactly"

    # Sanity: before promotion, output must have equalled baseline everywhere too.
    x_probe = torch.randn(5, 1)
    fresh_model, fresh_baseline = _make_model()
    assert torch.equal(fresh_model(x_probe), fresh_baseline(x_probe).detach())


def test_revoke_restores_previous_service():
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    x = torch.randn(4, 1)
    y_before_promote = model(x).detach()

    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    model.promote(proposal)
    model.revoke()

    y_after_revoke = model(x)
    assert torch.equal(y_after_revoke, y_before_promote), "Output after revoke differs from pre-promotion output"


def test_post_promotion_observe_does_not_change_served_output():
    """The deploy snapshot served after promotion must be frozen: observe()
    keeps training the live shadow head even in AUTHORIZED state, but that
    must never silently change what's served without a new evaluate()/promote()."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    model.promote(proposal)

    x_inside = torch.tensor([[2.5]])
    y1 = model(x_inside).detach().clone()

    for _ in range(10):
        x_batch = torch.empty(5, 1).uniform_(2.0, 3.0)
        y_batch = _true_fn(x_batch)
        model.observe(x_batch, y_batch)

    y2 = model(x_inside).detach()
    assert torch.equal(y1, y2), "Served output changed after post-promotion observe() without a new promotion"


def test_parameter_isolation():
    """The candidate snapshot bound into a PromotionProposal at evaluate()
    time must be a real, independent copy of the shadow head: further
    observe() calls keep training the live shadow, but must not mutate an
    already-issued proposal's snapshot underneath it."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    snapshot_head = proposal.deployment_snapshot.deploy_head
    lambda_before = snapshot_head.posterior.Lambda.clone()
    q_before = snapshot_head.posterior.q.clone()

    for _ in range(10):
        x_batch = torch.empty(5, 1).uniform_(2.0, 3.0)
        y_batch = _true_fn(x_batch)
        model.observe(x_batch, y_batch)

    assert torch.equal(snapshot_head.posterior.Lambda, lambda_before), \
        "Proposal snapshot's Lambda mutated by observe() after evaluate()"
    assert torch.equal(snapshot_head.posterior.q, q_before), \
        "Proposal snapshot's q mutated by observe() after evaluate()"


def test_invalid_state_transitions():
    model, _ = _make_model()
    with pytest.raises(VRSEStateError):
        model.promote(object())  # not in PENDING_EVAL
    with pytest.raises(VRSEStateError):
        model.revoke()  # not in AUTHORIZED


def test_reevaluate_while_authorized_does_not_stop_serving():
    """evaluate() must never un-serve an already AUTHORIZED deployment,
    even when called again (e.g. to check a fresher shadow) before any
    new promote()."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    model.promote(proposal)
    assert model._state == VRSEState.AUTHORIZED

    x_inside = torch.tensor([[2.5]])
    y_before = model(x_inside).detach().clone()

    proposal2 = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert model._state == VRSEState.AUTHORIZED, \
        "Re-evaluating while AUTHORIZED must not leave the AUTHORIZED state"
    y_during = model(x_inside).detach()
    assert torch.equal(y_before, y_during), "Re-evaluating while AUTHORIZED changed served output"


def test_revoke_restores_full_prior_snapshot():
    """When a second promotion replaces an already-AUTHORIZED deployment,
    revoke() must restore the exact prior snapshot (deploy_head AND
    authorized_region) and land back in AUTHORIZED -- not QUARANTINE,
    since a real prior deployment existed (Phase2B_Plan.md W5)."""
    torch.manual_seed(0)
    model, baseline = _make_fitted_model()
    for _ in range(40):
        x_batch = torch.empty(5, 1).uniform_(2.0, 3.0)
        y_batch = _true_fn(x_batch)
        model.observe(x_batch, y_batch)
    x_val = torch.empty(20, 1).uniform_(2.0, 3.0)
    y_val = _true_fn(x_val)
    x_guard = torch.empty(10, 1).uniform_(-3.0, -2.0)

    proposal1 = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal1.passed, f"First proposal did not pass: {proposal1.validation_result}"
    model.promote(proposal1)
    region_after_first = model._authorized_region
    x_probe = torch.tensor([[2.5]])
    y_after_first = model(x_probe).detach().clone()

    # Widen the shadow's training domain past the first deployment's region
    # so a second, legitimately different promotion is achievable.
    for _ in range(60):
        x_batch = torch.empty(5, 1).uniform_(2.0, 4.0)
        y_batch = _true_fn(x_batch)
        model.observe(x_batch, y_batch)
    x_val2 = torch.empty(20, 1).uniform_(2.0, 4.0)
    y_val2 = _true_fn(x_val2)
    proposal2 = model.evaluate(x_val2, y_val2, guard_x=x_guard)
    assert proposal2.passed, f"Second proposal did not pass: {proposal2.validation_result}"
    model.promote(proposal2)
    assert model._authorized_region != region_after_first, \
        "Second promotion did not actually change the served region -- test setup invalid"

    model.revoke()
    assert model._state == VRSEState.AUTHORIZED, \
        "Revoking a second promotion with a real prior snapshot must land in AUTHORIZED"
    assert model._authorized_region == region_after_first
    y_after_revoke = model(x_probe).detach()
    assert torch.equal(y_after_first, y_after_revoke), \
        "Output after revoke does not match the restored prior snapshot's output"


def test_revoked_can_resume_and_repromote():
    """Whichever state revoke() lands in (QUARANTINE or AUTHORIZED), the
    model must remain usable: observe()/evaluate()/promote() keep working
    afterward (Phase2B_Plan.md W5 -- REVOKED is audit-only, never a resting
    state that blocks further lifecycle)."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    model.promote(proposal)
    model.revoke()
    assert model._state == VRSEState.QUARANTINE

    for _ in range(40):
        x_batch = torch.empty(5, 1).uniform_(2.0, 3.0)
        y_batch = _true_fn(x_batch)
        model.observe(x_batch, y_batch)
    x_val2 = torch.empty(20, 1).uniform_(2.0, 3.0)
    y_val2 = _true_fn(x_val2)
    proposal2 = model.evaluate(x_val2, y_val2, guard_x=x_guard)
    assert proposal2.passed, f"Re-promotion after revoke did not pass: {proposal2.validation_result}"
    assert model.promote(proposal2) is True
    assert model._state == VRSEState.AUTHORIZED


def test_failed_reeval_preserves_authorization():
    """A second evaluate()/promote() cycle that fails validation must
    leave an already-AUTHORIZED deployment serving exactly as before --
    not reverted to QUARANTINE (Phase2B_Plan.md W5)."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal1 = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal1.passed, f"First proposal did not pass: {proposal1.validation_result}"
    model.promote(proposal1)
    x_probe = torch.tensor([[2.5]])
    y_before = model(x_probe).detach().clone()

    # A validation set drawn from the ID guard domain (disjoint from the
    # trained region) makes the shadow compare poorly against deploy on
    # this val set, and additionally violates cond4 (ID overlap) --
    # engineered to fail, not a real candidate.
    x_val_bad = torch.empty(20, 1).uniform_(-3.0, -2.0)
    y_val_bad = _true_fn(x_val_bad)
    proposal2 = model.evaluate(x_val_bad, y_val_bad, guard_x=x_guard)
    assert not proposal2.passed, "Test setup expected this second proposal to fail validation"
    promoted = model.promote(proposal2)
    assert promoted is False
    assert model._state == VRSEState.AUTHORIZED, \
        "A failed re-evaluation/promotion must not revert an AUTHORIZED deployment"

    y_after = model(x_probe).detach()
    assert torch.equal(y_before, y_after), \
        "Served output changed after a failed re-evaluation/promotion"


def test_revoke_depth_is_one():
    """revoke() has exactly one restore point; a second consecutive
    revoke() must raise rather than keep rolling back further (Stage-4C
    never validated multi-level rollback)."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()

    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    model.promote(proposal)

    model.revoke()
    with pytest.raises(VRSEStateError):
        model.revoke()  # single restore point already consumed


def test_evaluate_after_revoke_to_quarantine_does_not_crash():
    """After revoking the very first promotion (fallback to QUARANTINE),
    _deploy_head must still be the fit()-time pretrain head, not None --
    evaluate() calls validate_promotion() unconditionally regardless of
    state, and needs a real deploy_head to compare the shadow against."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    model.promote(proposal)
    model.revoke()
    assert model._state == VRSEState.QUARANTINE

    for _ in range(5):
        x_batch = torch.empty(5, 1).uniform_(2.0, 3.0)
        y_batch = _true_fn(x_batch)
        model.observe(x_batch, y_batch)
    x_val2 = torch.empty(20, 1).uniform_(2.0, 3.0)
    y_val2 = _true_fn(x_val2)
    proposal2 = model.evaluate(x_val2, y_val2, guard_x=x_guard)
    assert proposal2.passed, f"Re-evaluation after revoke-to-baseline did not pass: {proposal2.validation_result}"


def test_promote_rejects_stale_candidate():
    """A proposal must go stale if observe() runs between evaluate() and
    promote() -- promote() recomputes the candidate fingerprint from the
    live shadow head rather than trusting the fingerprint baked into the
    proposal at evaluate() time."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"

    x_batch = torch.empty(5, 1).uniform_(2.0, 3.0)
    y_batch = _true_fn(x_batch)
    model.observe(x_batch, y_batch)  # mutates the live shadow head in place

    with pytest.raises(VRSEStateError):
        model.promote(proposal)


def test_promote_rejects_forged_proposal():
    """An externally hand-built PromotionProposal with passed=True and
    copied-in fingerprints must still be rejected: promote() requires the
    exact single-use token minted by this model's own evaluate() call."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    real_proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert real_proposal.passed, f"Promotion proposal did not pass: {real_proposal.validation_result}"

    from vrse.proposal import PromotionProposal
    forged = PromotionProposal(
        baseline_fingerprint=real_proposal.baseline_fingerprint,
        deployment_snapshot=real_proposal.deployment_snapshot,
        candidate_fingerprint=real_proposal.candidate_fingerprint,
        config_version=real_proposal.config_version,
        config_fingerprint=real_proposal.config_fingerprint,
        authorized_region=real_proposal.authorized_region,
        validation_result=real_proposal.validation_result,
        passed=True,
        shadow_update_count=real_proposal.shadow_update_count,
        issue_token="forged-token-not-minted-by-this-model",
    )
    with pytest.raises(VRSEStateError):
        model.promote(forged)


def test_promote_token_is_single_use():
    """Once a token is consumed by one promote() call (success or
    failure), the same proposal object cannot be replayed through
    promote() a second time."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"

    assert model.promote(proposal) is True
    with pytest.raises(VRSEStateError):
        model.promote(proposal)  # token already consumed by the call above


def test_baseline_internal_copy_is_decoupled_from_caller():
    """wrap() must deep-copy the baseline internally: mutating the
    caller's own reference afterward must not move what the model serves."""
    baseline = _TinyBaseline()
    config = VRSEConfig(preset="regional_regression")
    model = VRSEModel.wrap(baseline, config)

    x = torch.randn(3, 1)
    y_before = model(x).detach().clone()

    with torch.no_grad():
        baseline.linear.weight.add_(100.0)
        baseline.linear.bias.add_(100.0)

    y_after = model(x).detach()
    assert torch.equal(y_before, y_after), \
        "Mutating the caller's baseline object after wrap() changed model output"


def test_baseline_stays_eval_after_train():
    """model.train() must not recursively flip the frozen baseline (or
    phi_SN, once fit() has built it) back into training mode -- otherwise
    BatchNorm running stats / spectral-norm internal state would start
    drifting again, silently invalidating the frozen safety reference."""
    model, baseline = _make_fitted_model()
    assert model._baseline.training is False
    assert model._phi_sn.training is False

    model.train()
    assert model._baseline.training is False, "baseline flipped to training mode by model.train()"
    assert model._phi_sn.training is False, "phi_SN flipped to training mode by model.train()"
    for p in model._baseline.parameters():
        assert not p.requires_grad, "baseline parameter regained requires_grad"


def test_revoke_first_promotion_falls_back_to_quarantine():
    """Revoking the very first promotion (no prior snapshot existed) must
    land in QUARANTINE, serving pure baseline -- not a resting REVOKED
    state and not a second AUTHORIZED snapshot that doesn't exist."""
    model, baseline, x_val, y_val, x_guard = _make_promotable_model()
    proposal = model.evaluate(x_val, y_val, guard_x=x_guard)
    assert proposal.passed, f"Promotion proposal did not pass: {proposal.validation_result}"
    model.promote(proposal)
    model.revoke()
    assert model._state == VRSEState.QUARANTINE

    x_probe = torch.randn(5, 1)
    y_hat = model(x_probe)
    y_base = baseline(x_probe).detach()
    assert torch.equal(y_hat, y_base), "After revoking the first promotion, output must equal baseline exactly"


# ---------------------------------------------------------------------------
# Phase 2B-2 spine tests
# ---------------------------------------------------------------------------

def _make_tiny_gp(x: torch.Tensor) -> tuple:
    """Return a minimal (phi_sn, shadow_head) pair fitted on x for region tests."""
    from vrse._algorithm import _SCAN_POINTS
    phi = _PhiSN(input_dim=1, hidden_dim=32, n_blocks=2, sn_multiplier=0.95)
    phi.freeze()
    import math
    ls = 1.0
    rff = _RFFMap(in_dim=32, rff_dim=128, length_scale=ls, seed=0)
    head = GPHead(rff, noise_var=0.05**2, prior_precision=1.0)
    with torch.no_grad():
        z = phi(x)
        r = torch.zeros(x.shape[0])
        head.fit_batch(z, r)
    return phi, head


def test_region_rejects_protected_overlap():
    """build_observed_span_region must return None when the observed span
    overlaps a protected ID range (closed-interval: touching the boundary
    counts as overlap, per Stage-4C src/stage4c.py _intervals_overlap)."""
    # Observed span [2.0, 3.0]; protected range [2.5, 4.0] -- overlaps.
    x_train = torch.linspace(2.0, 3.0, 30).unsqueeze(-1)
    x_val = torch.linspace(2.1, 2.9, 10).unsqueeze(-1)
    phi, head = _make_tiny_gp(x_train)

    region = build_observed_span_region(
        phi, head, tau_deploy=1.0,
        x_shadow_train=x_train,
        x_promotion_val=x_val,
        protected_id_ranges=((2.5, 4.0),),
        scan_domain=(-8.0, 7.0),
    )
    assert region is None, (
        "Region should be None when observed span overlaps a protected ID range"
    )

    # Touching the boundary also fails (closed-interval rule).
    region_touch = build_observed_span_region(
        phi, head, tau_deploy=1.0,
        x_shadow_train=x_train,
        x_promotion_val=x_val,
        protected_id_ranges=((3.0, 4.0),),  # lo of protected == hi of observed
        scan_domain=(-8.0, 7.0),
    )
    assert region_touch is None, (
        "Region should be None when observed span touches (closed) a protected ID boundary"
    )

    # Non-overlapping protected range must not block construction.
    region_ok = build_observed_span_region(
        phi, head, tau_deploy=1.0,
        x_shadow_train=x_train,
        x_promotion_val=x_val,
        protected_id_ranges=((-3.0, -1.0),),
        scan_domain=(-8.0, 7.0),
    )
    assert region_ok is not None, "Non-overlapping protected range must not block region construction"


def test_region_rejects_scan_boundary_touch():
    """build_observed_span_region must return None when the observed span
    reaches or exceeds the scan_domain boundary (Stage-4C check
    observed_span_touches_scan_boundary)."""
    # Observed span [2.0, 6.5]; scan_domain=(-8.0, 7.0) -> hi < 7.0 required.
    x_train = torch.linspace(2.0, 6.5, 50).unsqueeze(-1)
    x_val = torch.linspace(2.1, 6.4, 10).unsqueeze(-1)
    phi, head = _make_tiny_gp(x_train)

    region_ok = build_observed_span_region(
        phi, head, tau_deploy=1.0,
        x_shadow_train=x_train,
        x_promotion_val=x_val,
        protected_id_ranges=(),
        scan_domain=(-8.0, 7.0),
    )
    assert region_ok is not None, "Span well inside scan_domain should succeed"

    # Now push hi to exactly scan_domain[1] -- must fail.
    x_train_edge = torch.cat([x_train, torch.tensor([[7.0]])])
    region_edge = build_observed_span_region(
        phi, head, tau_deploy=1.0,
        x_shadow_train=x_train_edge,
        x_promotion_val=x_val,
        protected_id_ranges=(),
        scan_domain=(-8.0, 7.0),
    )
    assert region_edge is None, (
        "Observed span touching scan_domain upper boundary must be rejected"
    )
