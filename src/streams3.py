"""Five-phase online streams for Stage-3 (Plan3.md §5, results/STAGE3_PROTOCOL.md).

Reuses Stage-2's data distribution and its five paired seeds
(`src/streams.py`'s pattern: one continued per-(stream, seed) RNG, "fixed ID
set" meaning "held out from online updates", not "bit-identical x values").

Each stream: ID pre-test (64) -> shadow-train (128) -> ID anchor (64) ->
promotion-validation (64) -> post-decision (64) -> ID return-test (64).

Three variants, all branching from the same frozen offline checkpoint:
  1. stable_shift:          x in [-6,-3], true_function throughout.
  2. stable_extrapolation:  x in [3,5],   true_function throughout.
  3. unstable_extrapolation: x in [3,5]; shadow-train uses true_function
     (B(x)+delta(x)), but ID-anchor/promotion-validation/post-decision flip
     the sign to B(x)-delta(x) -- a rule that looked regular during shadow
     training but has since reversed.
"""

import numpy as np
import torch

from src.config import DataConfig
from src.dataset import backbone, delta, true_function, _sample_uniform

N_ID_PRE = 64
N_SHADOW_TRAIN = 128
N_ID_ANCHOR = 64
N_PROMOTION_VAL = 64
N_POST_DECISION = 64
N_ID_RETURN = 64


def _flipped_function(x: np.ndarray) -> np.ndarray:
    return backbone(x) - delta(x)


def _make_xy(x: np.ndarray, y_clean_fn, noise_std: float, rng: np.random.Generator):
    y_clean = y_clean_fn(x)
    y = y_clean + rng.normal(0.0, noise_std, size=x.shape)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return x_t, y_t


def _fixed_id_probe(data_cfg: DataConfig, rng: np.random.Generator, n: int):
    x = _sample_uniform(data_cfg.train_ranges, n, rng)
    return _make_xy(x, true_function, data_cfg.noise_std, rng)


class Stream3:
    def __init__(self, id_pre, shadow_train, id_anchor, promotion_val, post_decision, id_return, name: str):
        self.id_pre = id_pre
        self.shadow_train = shadow_train
        self.id_anchor = id_anchor
        self.promotion_val = promotion_val
        self.post_decision = post_decision
        self.id_return = id_return
        self.name = name


def _build_stream(data_cfg: DataConfig, seed: int, region: tuple, shadow_fn, val_fn, name: str) -> Stream3:
    rng = np.random.default_rng(seed)

    id_pre = _fixed_id_probe(data_cfg, rng, N_ID_PRE)

    x_shadow = rng.uniform(region[0], region[1], size=N_SHADOW_TRAIN)
    shadow_train = _make_xy(x_shadow, shadow_fn, data_cfg.noise_std, rng)

    id_anchor = _fixed_id_probe(data_cfg, rng, N_ID_ANCHOR)

    x_val = rng.uniform(region[0], region[1], size=N_PROMOTION_VAL)
    promotion_val = _make_xy(x_val, val_fn, data_cfg.noise_std, rng)

    x_post = rng.uniform(region[0], region[1], size=N_POST_DECISION)
    post_decision = _make_xy(x_post, val_fn, data_cfg.noise_std, rng)

    id_return = _fixed_id_probe(data_cfg, rng, N_ID_RETURN)

    return Stream3(id_pre, shadow_train, id_anchor, promotion_val, post_decision, id_return, name=name)


def build_stable_shift_stream(data_cfg: DataConfig, seed: int) -> Stream3:
    return _build_stream(data_cfg, seed, data_cfg.shift_range, true_function, true_function, name="stable_shift")


def build_stable_extrapolation_stream(data_cfg: DataConfig, seed: int) -> Stream3:
    return _build_stream(
        data_cfg, seed, data_cfg.extrapolation_range, true_function, true_function, name="stable_extrapolation"
    )


def build_unstable_extrapolation_stream(data_cfg: DataConfig, seed: int) -> Stream3:
    return _build_stream(
        data_cfg, seed, data_cfg.extrapolation_range, true_function, _flipped_function, name="unstable_extrapolation"
    )
