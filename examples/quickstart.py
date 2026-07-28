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


def main() -> None:
    torch.manual_seed(7)
    baseline = FrozenRule()
    model = VRSEModel.wrap(
        baseline=baseline,
        config=VRSEConfig(
            preset="regional_regression",
            phi_epochs=120,
            min_shadow_updates=64,
            random_seed=7,
        ),
    )

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

    print("VRSE quickstart")
    print(f"  isolated learning max output change : {isolation_diff:.3e}")
    print(f"  promotion passed                    : {promoted}")
    print(f"  new-region route fraction           : {route_fraction:.3f}")
    print(f"  new-region RMSE before -> after      : "
          f"{rmse(before_observe, target(x_new_test)):.3f} -> "
          f"{rmse(model(x_new_test), target(x_new_test)):.3f}")
    print(f"  old-region max fallback difference  : {old_diff:.3e}")
    print(f"  unknown max fallback difference     : {unknown_diff:.3e}")


if __name__ == "__main__":
    main()
