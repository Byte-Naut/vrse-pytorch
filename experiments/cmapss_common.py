"""Paths and serialization helpers shared by the C-MAPSS entry points."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("VRSE_RESULTS_DIR", ROOT / "results"))
PREPARED = RESULTS / "cmapss_fd002_prepared.npz"
MANIFEST = RESULTS / "cmapss_fd002_data_manifest.json"
PRECONDITIONS_JSON = RESULTS / "cmapss_fd002_preconditions.json"
PRECONDITIONS_MD = RESULTS / "CMAPSS_FD002_PRECONDITIONS.md"
CHECKPOINTS = RESULTS / "cmapss_fd002_checkpoints"
MATRIX_PKL = RESULTS / "cmapss_fd002_matrix.pkl"
MATRIX_JSON = RESULTS / "cmapss_fd002_matrix.json"
VERDICT_JSON = RESULTS / "cmapss_fd002_verdict.json"
RESULT_MD = RESULTS / "CMAPSS_FD002_RESULT.md"
METRICS_MD = RESULTS / "cmapss_fd002_metrics.md"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
