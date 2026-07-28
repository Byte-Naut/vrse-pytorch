"""Paths and serialization helpers shared by Phase-3 entry points."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREPARED = RESULTS / "phase3_prepared_fd002.npz"
MANIFEST = RESULTS / "phase3_data_manifest.json"
PRECONDITIONS_JSON = RESULTS / "phase3_preconditions.json"
PRECONDITIONS_MD = RESULTS / "PHASE3_PRECONDITIONS.md"
CHECKPOINTS = RESULTS / "phase3_checkpoints"
MATRIX_PKL = RESULTS / "phase3_matrix.pkl"
MATRIX_JSON = RESULTS / "phase3_matrix.json"
VERDICT_JSON = RESULTS / "phase3_verdict.json"
RESULT_MD = RESULTS / "PHASE3_RESULT.md"
METRICS_MD = RESULTS / "phase3_metrics_table.md"


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
