"""A small stream showing that observation and serving are separate events."""
from __future__ import annotations

import torch

from examples.quickstart import FrozenRule, make_model, target


def main() -> None:
    baseline = FrozenRule()
    model = make_model(baseline, seed=23)

    x_fit = torch.linspace(-1.0, 1.0, 128).unsqueeze(1)
    x_calibration = torch.linspace(-0.997, 0.997, 2000).unsqueeze(1)
    model.fit(x_fit, target(x_fit), x_calibration)

    probe = torch.tensor([[3.5]])
    baseline_value = float(model(probe).item())
    stream = torch.linspace(3.0, 4.0, 160).unsqueeze(1)

    print("event                 samples  served@3.5  changed")
    for stop in range(20, len(stream) + 1, 20):
        chunk = stream[stop - 20:stop]
        model.observe(chunk, target(chunk))
        served = float(model(probe).item())
        print(f"observe               {stop:7d}  {served:10.3f}  {served != baseline_value}")

    x_validation = torch.linspace(3.01, 3.99, 120).unsqueeze(1)
    x_guard = torch.linspace(-0.95, 0.95, 120).unsqueeze(1)
    proposal = model.evaluate(
        x_validation,
        target(x_validation),
        guard_x=x_guard,
    )
    promoted = model.promote(proposal)
    served = float(model(probe).item())
    print(f"promote={promoted!s:<11} {len(stream):7d}  {served:10.3f}  {served != baseline_value}")


if __name__ == "__main__":
    main()
