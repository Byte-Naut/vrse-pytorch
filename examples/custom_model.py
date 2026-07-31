"""Wrap an existing scalar-regression ``torch.nn.Module`` with VRSE.

The baseline can be any module that maps ``(batch, features)`` to ``(batch, 1)``.
VRSE copies and freezes it; the caller remains responsible for training and
validating that baseline before wrapping it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vrse import VRSEConfig, VRSEModel


class ExistingRegressor(nn.Module):
    """Stand-in for a model that was trained before VRSE is introduced."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.25 * x + 1.0


def current_target(x: torch.Tensor) -> torch.Tensor:
    new_condition = ((x >= 2.0) & (x <= 3.0)).to(x.dtype)
    return 0.25 * x + 1.0 + 1.5 * new_condition


def main() -> None:
    baseline = ExistingRegressor()
    model = VRSEModel.wrap(
        baseline=baseline,
        config=VRSEConfig(
            preset="regional_regression",
            min_shadow_updates=64,
            phi_epochs=120,
            random_seed=19,
        ),
    )

    # Keep fit, calibration, observation, validation and guard roles disjoint.
    x_fit = torch.linspace(-1.0, 1.0, 128).unsqueeze(1)
    x_calibration = torch.linspace(-0.997, 0.997, 2000).unsqueeze(1)
    model.fit(x_fit, current_target(x_fit), x_calibration)

    x_observe = torch.linspace(2.0, 3.0, 160).unsqueeze(1)
    model.observe(x_observe, current_target(x_observe))

    x_validation = torch.linspace(2.01, 2.99, 120).unsqueeze(1)
    x_guard = torch.linspace(-0.95, 0.95, 120).unsqueeze(1)
    proposal = model.evaluate(
        x_validation,
        current_target(x_validation),
        guard_x=x_guard,
    )
    print(f"promotion_passed={model.promote(proposal)}")
    print(f"authorized_fraction={model.route_mask(x_validation).float().mean().item():.3f}")


if __name__ == "__main__":
    main()
