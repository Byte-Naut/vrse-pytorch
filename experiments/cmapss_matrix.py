"""Execute the frozen C-MAPSS 2 x 5 x 5 experiment matrix."""
from __future__ import annotations

import pickle

import torch

from experiments.cmapss_common import (
    CHECKPOINTS,
    MATRIX_JSON,
    MATRIX_PKL,
    PRECONDITIONS_JSON,
    PREPARED,
    load_json,
    write_json,
)
from benchmarks.cmapss_fd002 import PROTOCOL_REVISION, SEEDS, load_prepared, make_cmapss_split
from benchmarks.cmapss_methods import STREAMS, run_cmapss_bundle


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    pre = load_json(PRECONDITIONS_JSON)
    if pre.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError(
            "Precondition artifacts predate the frozen normalization revision; rerun them."
        )
    if pre["verdict"] != "READY_FOR_MATRIX":
        raise RuntimeError(
            f"Precondition verdict is {pre['verdict']}; protocol forbids running the matrix."
        )
    data, definition = load_prepared(PREPARED)
    matrix = {stream: {} for stream in STREAMS}
    for seed in SEEDS:
        split = make_cmapss_split(data, definition, seed)
        checkpoint = _load_checkpoint(CHECKPOINTS / f"seed_{seed}.pt")
        if checkpoint.get("protocol_revision") != PROTOCOL_REVISION:
            raise RuntimeError(f"Seed {seed} checkpoint belongs to an older C-MAPSS protocol.")
        template = checkpoint["model"]
        for stream in STREAMS:
            bundle = run_cmapss_bundle(template, split, stream)
            matrix[stream][seed] = bundle
            print(f"[{stream}][seed={seed}] complete")
    payload = {
        "protocol_revision": PROTOCOL_REVISION,
        "seeds": SEEDS,
        "streams": STREAMS,
        "matrix": matrix,
    }
    with MATRIX_PKL.open("wb") as handle:
        pickle.dump(payload, handle)
    write_json(MATRIX_JSON, payload)
    print(MATRIX_PKL)


if __name__ == "__main__":
    main()
