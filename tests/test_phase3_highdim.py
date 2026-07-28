"""Phase-3 high-dimensional spine tests.

These tests are delivered but intentionally not executed by the implementation
turn.  They use synthetic 24-D tensors and never download C-MAPSS.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from src.phase3_cmapss import (
    CMapssFD002,
    ROLE_SIZES,
    discovery_normalization,
    normalization_audit,
)
from vrse import VRSEConfig, VRSEModel
from vrse._algorithm import (
    GPHead,
    KNNFeatureRegion,
    _PhiSN,
    _RFFMap,
    build_knn_feature_region,
)
from vrse.model import VRSEState


class _Zero24(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)


def _identity_feature_gp(x_train: torch.Tensor):
    phi = _PhiSN(24, 24, n_blocks=0, sn_multiplier=0.95, spectral_input=True)
    phi.freeze()
    rff = _RFFMap(24, 32, length_scale=2.0, seed=0)
    head = GPHead(rff, noise_var=0.1**2, prior_precision=1.0)
    with torch.no_grad():
        head.fit_batch(phi(x_train), torch.zeros(x_train.shape[0]))
    return phi, head


def test_phase3_role_sizes_consume_non_discovery_units():
    assert sum(ROLE_SIZES.values()) == 240


def _normalization_fixture(outside_shift: bool) -> CMapssFD002:
    features = np.zeros((4, 24), dtype=np.float64)
    features[:, 0] = [0.0, 2.0, 1.0, 1.0]
    if outside_shift:
        features[2:, 1] = 1.0
    return CMapssFD002(
        unit=np.asarray([1, 2, 21, 22]),
        cycle=np.ones(4, dtype=np.int64),
        settings=features[:, :3],
        features=features,
        target=np.zeros((4, 1), dtype=np.float64),
    )


def test_phase3b_normalization_uses_only_frozen_discovery_units():
    data = _normalization_fixture(outside_shift=False)
    mean, std = discovery_normalization(data)
    assert mean[0] == 1.0
    assert std[0] == 1.0


def test_phase3b_normalization_fails_when_floor_hides_outside_shift():
    audit = normalization_audit(_normalization_fixture(outside_shift=True))
    assert audit["floor_features_with_outside_variation"] == [1]
    assert audit["passed"] is False


def test_knn_region_separates_id_new_and_unknown_clusters():
    torch.manual_seed(0)
    x_train = 3.0 + 0.05 * torch.randn(120, 24)
    x_val = 3.0 + 0.05 * torch.randn(100, 24)
    x_id = 0.05 * torch.randn(100, 24)
    x_unknown = -3.0 + 0.05 * torch.randn(100, 24)
    phi, head = _identity_feature_gp(x_train)
    cfg = VRSEConfig(
        preset="regional_regression_highdim",
        rff_dim=32,
        hidden_dim=24,
        max_support_prototypes=32,
        knn_k=5,
    )
    region = build_knn_feature_region(phi, head, x_train, x_val, cfg)
    assert region is not None
    assert region.contains(x_val, phi, head).float().mean() >= 0.90
    assert region.contains(x_id, phi, head).sum() == 0
    assert region.contains(x_unknown, phi, head).sum() == 0


def test_highdim_model_lifecycle_smoke():
    torch.manual_seed(1)
    cfg = VRSEConfig(
        preset="regional_regression_highdim",
        hidden_dim=24,
        n_blocks=0,
        rff_dim=16,
        max_support_prototypes=16,
        length_scale_max_points=64,
        phi_epochs=1,
        min_shadow_updates=1,
    )
    model = VRSEModel.wrap(_Zero24(), cfg)
    x_id = 0.05 * torch.randn(80, 24)
    y_id = torch.zeros(80, 1)
    x_calib = 0.05 * torch.randn(100, 24)
    model.fit(x_id, y_id, x_calib)
    x_new = 3.0 + 0.05 * torch.randn(80, 24)
    y_new = torch.ones(80, 1)
    model.observe(x_new, y_new)
    proposal = model.evaluate(
        3.0 + 0.05 * torch.randn(100, 24),
        torch.ones(100, 1),
        guard_x=0.05 * torch.randn(100, 24),
    )
    assert proposal.validation_result["support_kind"] == "knn_feature"
    assert model(torch.randn(7, 24)).shape == (7, 1)


def test_highdim_region_outside_is_exact_fallback():
    torch.manual_seed(2)
    model = VRSEModel.wrap(
        _Zero24(),
        VRSEConfig(
            preset="regional_regression_highdim",
            hidden_dim=24,
            n_blocks=0,
            rff_dim=16,
            phi_epochs=1,
        ),
    )
    x_id = 0.05 * torch.randn(80, 24)
    model.fit(x_id, torch.zeros(80, 1), 0.05 * torch.randn(100, 24))
    prototypes = 3.0 + 0.01 * torch.randn(8, 24)
    model._authorized_region = KNNFeatureRegion(
        prototypes=prototypes,
        k=5,
        radius=0.5,
        tau_region=float("inf"),
    )
    model._state = VRSEState.AUTHORIZED
    x_outside = -3.0 + 0.01 * torch.randn(20, 24)
    assert torch.equal(model(x_outside), model._baseline(x_outside))


def test_highdim_region_snapshot_does_not_alias_live_prototypes():
    prototypes = torch.randn(8, 24)
    live = KNNFeatureRegion(prototypes.clone(), 5, 1.0, 2.0)
    snapshot = copy.deepcopy(live)
    live.prototypes.add_(100.0)
    assert not torch.equal(snapshot.prototypes, live.prototypes)
