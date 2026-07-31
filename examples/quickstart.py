"""Deterministic VRSE walkthrough with no external data.

The frozen baseline knows a simple linear rule. A stable +2.5 residual appears
only in a disjoint new interval. VRSE learns that residual in quarantine,
examines it on held-out points, and grants permission only around the observed
new interval. Old and adjacent-unknown inputs remain exact baseline fallbacks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vrse import VRSEConfig, VRSEModel


class FrozenRule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x


def target(x: torch.Tensor) -> torch.Tensor:
    regional_offset = ((x >= 3.0) & (x <= 4.0)).to(x.dtype) * 2.5
    return 0.5 * x + regional_offset


def rmse(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - truth) ** 2)).item())


def harmful_update(x: torch.Tensor) -> torch.Tensor:
    """A deliberately wrong regional update used to exercise rejection."""
    return 0.5 * x - 2.5


def make_model(baseline: nn.Module, seed: int) -> VRSEModel:
    return VRSEModel.wrap(
        baseline=baseline,
        config=VRSEConfig(
            preset="regional_regression",
            phi_epochs=120,
            min_shadow_updates=64,
            random_seed=seed,
        ),
    )


def main() -> None:
    torch.manual_seed(7)
    baseline = FrozenRule()
    model = make_model(baseline, seed=7)

    # Four disjoint roles: known fit, known calibration, shadow observation,
    # and held-out promotion validation/guard.
    x_id = torch.linspace(-1.0, 1.0, 128).unsqueeze(1)
    x_calib = torch.linspace(-0.997, 0.997, 2000).unsqueeze(1)
    model.fit(x_id, target(x_id), x_calib)

    x_new_observe = torch.linspace(3.0, 4.0, 160).unsqueeze(1)
    x_new_test = torch.linspace(3.02, 3.98, 80).unsqueeze(1)
    before_observe = model(x_new_test).clone()
    model.observe(x_new_observe, target(x_new_observe))
    after_observe = model(x_new_test).clone()

    # Learning in quarantine must not change the served output.
    isolation_diff = float((after_observe - before_observe).abs().max().item())

    x_validation = torch.linspace(3.01, 3.99, 120).unsqueeze(1)
    x_id_guard = torch.linspace(-0.95, 0.95, 120).unsqueeze(1)
    proposal = model.evaluate(
        x_validation,
        target(x_validation),
        guard_x=x_id_guard,
    )
    promoted = model.promote(proposal)
    if not promoted:
        raise RuntimeError(f"The deterministic candidate was rejected: {proposal.validation_result}")

    x_adjacent_unknown = torch.linspace(5.0, 5.8, 80).unsqueeze(1)
    old_diff = float((model(x_id_guard) - baseline(x_id_guard)).abs().max().item())
    unknown_diff = float(
        (model(x_adjacent_unknown) - baseline(x_adjacent_unknown)).abs().max().item()
    )
    route_fraction = float(model.route_mask(x_new_test).float().mean().item())
    before_rmse = rmse(before_observe, target(x_new_test))
    after_rmse = rmse(model(x_new_test), target(x_new_test))

    # A separately trained, deliberately wrong candidate must fail the same
    # held-out exam. This keeps the rejection demonstration deterministic.
    harmful_baseline = FrozenRule()
    harmful_model = make_model(harmful_baseline, seed=11)
    harmful_model.fit(x_id, target(x_id), x_calib)
    harmful_model.observe(x_new_observe, harmful_update(x_new_observe))
    harmful_proposal = harmful_model.evaluate(
        x_validation,
        target(x_validation),
        guard_x=x_id_guard,
    )
    harmful_promoted = harmful_model.promote(harmful_proposal)

    # The first promotion has one restore point: the frozen baseline service.
    model.revoke()
    rollback_diff = float((model(x_new_test) - baseline(x_new_test)).abs().max().item())

    yes_no = lambda condition: "yes" if condition else "no"
    print("VRSE lifecycle check")
    print(f"  Candidate learned in isolation       {yes_no(after_rmse < before_rmse)}")
    print(f"  Served model changed before review   {yes_no(isolation_diff > 0.0)}")
    print(f"  Useful candidate promoted            {yes_no(promoted)}")
    print(f"  Supported-region RMSE improved       {yes_no(after_rmse < before_rmse)} "
          f"({before_rmse:.3f} -> {after_rmse:.3f})")
    print(f"  Harmful candidate promoted           {yes_no(harmful_promoted)}")
    print(f"  Old behavior changed                 {yes_no(old_diff > 0.0)}")
    print(f"  Unknown inputs changed               {yes_no(unknown_diff > 0.0)}")
    print(f"  Revoke restored previous snapshot    {yes_no(rollback_diff == 0.0)}")
    print(f"  Supported inputs routed to candidate {route_fraction:.1%}")


if __name__ == "__main__":
    main()
