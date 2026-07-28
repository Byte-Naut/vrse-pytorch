"""Three-segment online data streams for Stage-2 (Plan2 §2, results/STAGE2_PROTOCOL.md).

Each stream is: ID pre-test (64, predict-only) -> new-region online (256,
predict-then-update) -> ID return-test (64, predict-only). The two OOD
streams (shift, extrapolation) and the ID-only stream (used solely for LR
selection) are never chained together — each is built fresh from the same
frozen data distribution.

The fixed ID test set used for both the pre-test and return-test segments is
the SAME set across the whole stream (protocol requirement), and is also
reused as-is for the ID-only stream's outer segments.
"""

import numpy as np
import torch

from src.config import DataConfig
from src.dataset import Dataset1D, true_function, _sample_uniform

N_ID_PROBE = 64
N_ONLINE = 256


def _make_xy(x: np.ndarray, noise_std: float, rng: np.random.Generator, noisy: bool):
    y_clean = true_function(x)
    y = y_clean + rng.normal(0.0, noise_std, size=x.shape) if noisy else y_clean
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return x_t, y_t


class Stream:
    """Holds the three segments as (x, y) tensor pairs plus segment metadata."""

    def __init__(self, id_pre, online, id_return, name: str):
        self.id_pre = id_pre
        self.online = online
        self.id_return = id_return
        self.name = name


def _fixed_id_test_set(data_cfg: DataConfig, rng: np.random.Generator, noise_std: float):
    x = _sample_uniform(data_cfg.train_ranges, N_ID_PROBE, rng)
    return _make_xy(x, noise_std, rng, noisy=True)


def build_shift_stream(data_cfg: DataConfig, seed: int) -> Stream:
    rng = np.random.default_rng(seed)
    id_pre = _fixed_id_test_set(data_cfg, rng, data_cfg.noise_std)
    x_online = rng.uniform(data_cfg.shift_range[0], data_cfg.shift_range[1], size=N_ONLINE)
    online = _make_xy(x_online, data_cfg.noise_std, rng, noisy=True)
    id_return = _fixed_id_test_set(data_cfg, rng, data_cfg.noise_std)
    return Stream(id_pre, online, id_return, name="shift")


def build_extrapolation_stream(data_cfg: DataConfig, seed: int) -> Stream:
    rng = np.random.default_rng(seed)
    id_pre = _fixed_id_test_set(data_cfg, rng, data_cfg.noise_std)
    x_online = rng.uniform(data_cfg.extrapolation_range[0], data_cfg.extrapolation_range[1], size=N_ONLINE)
    online = _make_xy(x_online, data_cfg.noise_std, rng, noisy=True)
    id_return = _fixed_id_test_set(data_cfg, rng, data_cfg.noise_std)
    return Stream(id_pre, online, id_return, name="extrapolation")


def build_id_only_stream(data_cfg: DataConfig, seed: int) -> Stream:
    """train -> train -> train, used ONLY for LR selection (protocol §4)."""
    rng = np.random.default_rng(seed)
    id_pre = _fixed_id_test_set(data_cfg, rng, data_cfg.noise_std)
    x_online = _sample_uniform(data_cfg.train_ranges, N_ONLINE, rng)
    online = _make_xy(x_online, data_cfg.noise_std, rng, noisy=True)
    id_return = _fixed_id_test_set(data_cfg, rng, data_cfg.noise_std)
    return Stream(id_pre, online, id_return, name="id_only")
