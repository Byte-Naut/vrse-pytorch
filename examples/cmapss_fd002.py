"""One-seed C-MAPSS FD002 lifecycle example.

Run cmapss_prepare_data.py and cmapss_preconditions.py first. This example
loads the frozen seed-4300 checkpoint and performs only the VRSE stable stream.
"""
from __future__ import annotations

import copy

import torch

from experiments.cmapss_common import CHECKPOINTS, PREPARED
from benchmarks.cmapss_fd002 import PROTOCOL_REVISION, load_prepared, make_cmapss_split


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    data, definition = load_prepared(PREPARED)
    split = make_cmapss_split(data, definition, seed=4300)
    checkpoint = _load_checkpoint(CHECKPOINTS / "seed_4300.pt")
    if checkpoint.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError("Checkpoint belongs to an older C-MAPSS protocol.")
    model = copy.deepcopy(checkpoint["model"])

    model.observe(split.shadow_observe.x, split.shadow_observe.y)
    proposal = model.evaluate(
        split.promotion_validation.x,
        split.promotion_validation.y,
        guard_x=split.id_guard.x,
    )
    promoted = model.promote(proposal)
    y_hat = model(split.post_new.x)
    route = model.route_mask(split.post_new.x)
    rmse = torch.sqrt(torch.mean((y_hat - split.post_new.y) ** 2)).item()

    print(f"promoted={promoted}")
    print(f"post_new_route_frac={route.float().mean().item():.6f}")
    print(f"post_new_rmse={rmse:.6f}")
    print(f"validation={proposal.validation_result}")


if __name__ == "__main__":
    main()
