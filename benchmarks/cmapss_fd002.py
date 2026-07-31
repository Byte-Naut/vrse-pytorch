"""C-MAPSS FD002 data contract for the frozen VRSE benchmark.

No model-selection logic lives here.  Regime discovery uses only the three
operational settings from units 1--20, and every supervised role is split by
engine before rows are selected.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans


NASA_LANDING_PAGE = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
NASA_RESOURCE_PAGE = (
    "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/"
    "resource/5224bcd1-ad61-490b-93b9-2817288accb8"
)
FD002_TRAIN_MEMBER = "train_FD002.txt"
DISCOVERY_UNITS = tuple(range(1, 21))
# Historical identifier embedded in the frozen artifacts. Keep it unchanged so
# independent reproductions can reject incompatible prepared data.
PROTOCOL_REVISION = "phase3b-discovery-global-normalization-v1"
NORMALIZATION_STD_FLOOR = 1e-6
SEEDS = (4300, 4301, 4302, 4303, 4304)
ROLE_SIZES = {
    "id_fit": 60,
    "id_calibration": 30,
    "id_guard": 30,
    "shadow_observe": 50,
    "promotion_validation": 35,
    "post_decision": 35,
}


@dataclass(frozen=True)
class CMapssFD002:
    unit: np.ndarray
    cycle: np.ndarray
    settings: np.ndarray
    features: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class RegimeDefinition:
    mean: np.ndarray
    std: np.ndarray
    centers_standardized: np.ndarray
    centers_original: np.ndarray
    id_regime: int
    new_regime: int
    unknown_regime: int

    def assign(self, settings: np.ndarray) -> np.ndarray:
        z = (settings - self.mean) / self.std
        d2 = ((z[:, None, :] - self.centers_standardized[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1).astype(np.int64)


@dataclass(frozen=True)
class CMapssBatch:
    x: torch.Tensor
    y: torch.Tensor
    unit: torch.Tensor
    cycle: torch.Tensor

    def with_target(self, y: torch.Tensor) -> "CMapssBatch":
        return CMapssBatch(self.x, y, self.unit, self.cycle)


@dataclass(frozen=True)
class CMapssSplit:
    seed: int
    roles: Dict[str, Tuple[int, ...]]
    normalization_mean: torch.Tensor
    normalization_std: torch.Tensor
    normalization_source: str
    id_fit: CMapssBatch
    id_calibration: CMapssBatch
    id_guard: CMapssBatch
    shadow_observe: CMapssBatch
    promotion_validation: CMapssBatch
    post_new: CMapssBatch
    post_unknown: CMapssBatch

    def stream_targets(self, stream: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if stream == "stable_condition":
            return self.promotion_validation.y, self.post_new.y
        if stream == "reversed_condition":
            return 125.0 - self.promotion_validation.y, 125.0 - self.post_new.y
        raise ValueError(f"Unknown C-MAPSS stream: {stream}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_zip(url: str, destination: Path) -> None:
    """Download only when the user explicitly supplies a direct ZIP URL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vrse-cmapss/0.1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise ValueError("The supplied URL did not return a valid ZIP resource.")


def _resolve_fd002_file(source: Path) -> Path:
    candidates = [path for path in source.rglob(FD002_TRAIN_MEMBER) if path.is_file()]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {FD002_TRAIN_MEMBER!r} below {source}, found {candidates}."
        )
    return candidates[0]


def _read_fd002_matrix(source: Path) -> np.ndarray:
    if source.is_dir():
        matrix = np.loadtxt(_resolve_fd002_file(source), dtype=np.float64)
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            candidates = [name for name in archive.namelist() if name.endswith(FD002_TRAIN_MEMBER)]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one {FD002_TRAIN_MEMBER!r} in {source}, found {candidates}."
                )
            raw = archive.read(candidates[0])
        matrix = np.loadtxt(io.BytesIO(raw), dtype=np.float64)
    else:
        raise ValueError(f"C-MAPSS source must be an extracted directory or ZIP: {source}")
    if matrix.ndim != 2 or matrix.shape[1] != 26:
        raise ValueError(f"FD002 train matrix must have 26 columns, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("FD002 contains non-finite numeric values.")
    return matrix


def load_fd002(source: Path) -> CMapssFD002:
    matrix = _read_fd002_matrix(source)
    unit = matrix[:, 0].astype(np.int64)
    cycle = matrix[:, 1].astype(np.int64)
    settings = matrix[:, 2:5]
    features = matrix[:, 2:26]
    unique_units = np.unique(unit)
    if unique_units.size != 260:
        raise ValueError(f"FD002 train must contain 260 units, found {unique_units.size}.")
    max_cycle = {int(u): int(cycle[unit == u].max()) for u in unique_units}
    rul = np.asarray([max_cycle[int(u)] - int(c) for u, c in zip(unit, cycle)], dtype=np.float64)
    target = np.minimum(125.0, rul).reshape(-1, 1)
    return CMapssFD002(unit, cycle, settings, features, target)


def discover_regimes(data: CMapssFD002) -> RegimeDefinition:
    discovery = np.isin(data.unit, np.asarray(DISCOVERY_UNITS))
    settings = data.settings[discovery]
    mean = settings.mean(axis=0)
    std = settings.std(axis=0)
    std = np.maximum(std, 1e-6)
    z = (settings - mean) / std
    km = KMeans(n_clusters=6, random_state=31415, n_init=20)
    raw_labels = km.fit_predict(z)
    centers_z = km.cluster_centers_
    centers_original = centers_z * std + mean

    # Canonical label order: lexicographic in original operational units.
    order = np.lexsort(tuple(centers_original[:, j] for j in reversed(range(3))))
    old_to_new = np.empty(6, dtype=np.int64)
    old_to_new[order] = np.arange(6)
    labels = old_to_new[raw_labels]
    centers_z = centers_z[order]
    centers_original = centers_original[order]

    counts = np.bincount(labels, minlength=6)
    id_regime = int(np.flatnonzero(counts == counts.max())[0])
    distances = np.linalg.norm(centers_z - centers_z[id_regime], axis=1)
    ranked = sorted(
        (i for i in range(6) if i != id_regime),
        key=lambda i: (-float(distances[i]), i),
    )
    return RegimeDefinition(
        mean=mean,
        std=std,
        centers_standardized=centers_z,
        centers_original=centers_original,
        id_regime=id_regime,
        new_regime=int(ranked[0]),
        unknown_regime=int(ranked[1]),
    )


def _role_units(seed: int) -> Dict[str, Tuple[int, ...]]:
    units = np.arange(21, 261, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    roles = {}
    start = 0
    for name, size in ROLE_SIZES.items():
        roles[name] = tuple(int(v) for v in units[start:start + size])
        start += size
    if start != 240:
        raise AssertionError("C-MAPSS role sizes must consume exactly 240 non-discovery units.")
    return roles


def _select_rows(data: CMapssFD002, regimes: np.ndarray,
                 units: Iterable[int], regime: int) -> np.ndarray:
    mask = np.isin(data.unit, np.asarray(tuple(units))) & (regimes == regime)
    idx = np.flatnonzero(mask)
    # Deterministic fleet stream: cycle first, unit id second.
    return idx[np.lexsort((data.unit[idx], data.cycle[idx]))]


def _batch(data: CMapssFD002, idx: np.ndarray,
           mean: np.ndarray, std: np.ndarray) -> CMapssBatch:
    if idx.size == 0:
        raise ValueError("A C-MAPSS role/regime selection produced an empty batch.")
    x = ((data.features[idx] - mean) / std).astype(np.float32)
    y = data.target[idx].astype(np.float32)
    return CMapssBatch(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        unit=torch.from_numpy(data.unit[idx].copy()),
        cycle=torch.from_numpy(data.cycle[idx].copy()),
    )


def discovery_normalization(data: CMapssFD002) -> Tuple[np.ndarray, np.ndarray]:
    """Freeze one unlabeled, cross-regime scale before any supervised split.

    Units 1--20 are already isolated for regime discovery and never enter a
    train/calibration/validation/test role.  Using all of their 24-D inputs
    preserves between-regime displacement without consulting online data.
    """
    discovery = np.isin(data.unit, np.asarray(DISCOVERY_UNITS))
    features = data.features[discovery]
    if features.size == 0:
        raise ValueError("The frozen regime-discovery partition is empty.")
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), NORMALIZATION_STD_FLOOR)
    return mean, std


def normalization_audit(data: CMapssFD002) -> dict:
    """Detect a scale floor that conceals variation outside discovery data.

    This is a structural audit, not a performance-tuned z-score cutoff: any
    feature put on the numerical floor must also be constant in the complete
    raw FD002 input, otherwise normalization fails closed.
    """
    discovery = np.isin(data.unit, np.asarray(DISCOVERY_UNITS))
    raw_std = data.features[discovery].std(axis=0)
    floor_indices = np.flatnonzero(raw_std < NORMALIZATION_STD_FLOOR)
    full_span = np.ptp(data.features, axis=0)
    hidden_shift_indices = np.asarray(
        [j for j in floor_indices if full_span[j] >= NORMALIZATION_STD_FLOOR],
        dtype=np.int64,
    )
    mean, std = discovery_normalization(data)
    normalized = (data.features - mean) / std
    finite = bool(np.isfinite(normalized).all())
    return {
        "source": "unlabeled regime_discovery units 1-20, all regimes",
        "std_floor": NORMALIZATION_STD_FLOOR,
        "floor_feature_indices": floor_indices.astype(int).tolist(),
        "floor_features_with_outside_variation": hidden_shift_indices.astype(int).tolist(),
        "max_abs_z_all_rows": float(np.max(np.abs(normalized))),
        "all_finite": finite,
        "passed": bool(finite and hidden_shift_indices.size == 0),
    }


def make_cmapss_split(data: CMapssFD002, definition: RegimeDefinition,
                      seed: int) -> CMapssSplit:
    if seed not in SEEDS:
        raise ValueError(f"Seed {seed} is not pre-registered; expected one of {SEEDS}.")
    regimes = definition.assign(data.settings)
    roles = _role_units(seed)
    id_fit_idx = _select_rows(data, regimes, roles["id_fit"], definition.id_regime)
    mean, std = discovery_normalization(data)

    return CMapssSplit(
        seed=seed,
        roles=roles,
        normalization_mean=torch.from_numpy(mean.astype(np.float32)),
        normalization_std=torch.from_numpy(std.astype(np.float32)),
        normalization_source="unlabeled regime_discovery units 1-20, all regimes",
        id_fit=_batch(data, id_fit_idx, mean, std),
        id_calibration=_batch(
            data, _select_rows(data, regimes, roles["id_calibration"], definition.id_regime),
            mean, std,
        ),
        id_guard=_batch(
            data, _select_rows(data, regimes, roles["id_guard"], definition.id_regime),
            mean, std,
        ),
        shadow_observe=_batch(
            data, _select_rows(data, regimes, roles["shadow_observe"], definition.new_regime),
            mean, std,
        ),
        promotion_validation=_batch(
            data,
            _select_rows(data, regimes, roles["promotion_validation"], definition.new_regime),
            mean, std,
        ),
        post_new=_batch(
            data, _select_rows(data, regimes, roles["post_decision"], definition.new_regime),
            mean, std,
        ),
        post_unknown=_batch(
            data, _select_rows(data, regimes, roles["post_decision"], definition.unknown_regime),
            mean, std,
        ),
    )


def _source_hashes(source: Path) -> dict:
    if source.is_dir():
        return {
            str(path.relative_to(source)).replace("\\", "/"): sha256_file(path)
            for path in sorted(source.rglob("*")) if path.is_file()
        }
    return {source.name: sha256_file(source)}


def save_prepared(source: Path, npz_path: Path, manifest_path: Path,
                  source_origin: str = "unknown") -> dict:
    data = load_fd002(source)
    definition = discover_regimes(data)
    regimes = definition.assign(data.settings)
    normalization_mean, normalization_std = discovery_normalization(data)
    normalization_check = normalization_audit(data)
    if not normalization_check["passed"]:
        raise ValueError(f"Discovery normalization audit failed: {normalization_check}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        unit=data.unit,
        cycle=data.cycle,
        settings=data.settings,
        features=data.features,
        target=data.target,
        regimes=regimes,
        regime_mean=definition.mean,
        regime_std=definition.std,
        centers_standardized=definition.centers_standardized,
        centers_original=definition.centers_original,
        id_regime=definition.id_regime,
        new_regime=definition.new_regime,
        unknown_regime=definition.unknown_regime,
        protocol_revision=np.asarray(PROTOCOL_REVISION),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )
    manifest = {
        "protocol_revision": PROTOCOL_REVISION,
        "source_landing_page": NASA_LANDING_PAGE,
        "source_resource_page": NASA_RESOURCE_PAGE,
        "license_on_source_page": "License not specified",
        "raw_redistributed": False,
        "source_path": str(source),
        "source_kind": "extracted_directory" if source.is_dir() else "zip_archive",
        "source_origin": source_origin,
        "source_files_sha256": _source_hashes(source),
        "official_archive_sha256": None,
        "rows": int(data.unit.size),
        "columns": 26,
        "units": int(np.unique(data.unit).size),
        "discovery_units": list(DISCOVERY_UNITS),
        "centers_original": definition.centers_original.tolist(),
        "id_regime": definition.id_regime,
        "new_regime": definition.new_regime,
        "unknown_regime": definition.unknown_regime,
        "regime_counts": np.bincount(regimes, minlength=6).astype(int).tolist(),
        "seeds": list(SEEDS),
        "role_sizes": ROLE_SIZES,
        "normalization": {
            **normalization_check,
            "mean": normalization_mean.tolist(),
            "std": normalization_std.tolist(),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_prepared(npz_path: Path) -> Tuple[CMapssFD002, RegimeDefinition]:
    blob = np.load(npz_path)
    if "protocol_revision" not in blob.files:
        raise RuntimeError(
            "Prepared data predates the frozen normalization revision; rerun "
            "experiments.cmapss_prepare_data."
        )
    revision = str(blob["protocol_revision"].item())
    if revision != PROTOCOL_REVISION:
        raise RuntimeError(
            f"Prepared data revision is {revision!r}, expected {PROTOCOL_REVISION!r}."
        )
    data = CMapssFD002(
        unit=blob["unit"],
        cycle=blob["cycle"],
        settings=blob["settings"],
        features=blob["features"],
        target=blob["target"],
    )
    definition = RegimeDefinition(
        mean=blob["regime_mean"],
        std=blob["regime_std"],
        centers_standardized=blob["centers_standardized"],
        centers_original=blob["centers_original"],
        id_regime=int(blob["id_regime"]),
        new_regime=int(blob["new_regime"]),
        unknown_regime=int(blob["unknown_regime"]),
    )
    return data, definition
