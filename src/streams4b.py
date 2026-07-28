"""Stage-4B streams: Stage-3 phases plus an adjacent unseen safety probe.

The first six phases are inherited unchanged from Stage-3.  The final probe
is generated from an independent sub-seed and is never visible to shadow
training, promotion validation, or promoted-region construction.
"""

from dataclasses import dataclass

import numpy as np

from src.config import DataConfig, Stage4BConfig
from src.dataset import true_function
from src.streams3 import (
    Stream3,
    _make_xy,
    build_stable_extrapolation_stream,
    build_stable_shift_stream,
    build_unstable_extrapolation_stream,
)


@dataclass
class Stream4B:
    id_pre: tuple
    shadow_train: tuple
    id_anchor: tuple
    promotion_val: tuple
    post_decision: tuple
    id_return: tuple
    second_unknown: tuple
    name: str
    second_unknown_range: tuple
    second_unknown_seed: int
    protected_id_ranges: tuple


SECOND_UNKNOWN_RANGES = {
    "stable_shift": (-7.0, -6.0),
    "stable_extrapolation": (5.0, 6.0),
    "unstable_extrapolation": (5.0, 6.0),
}

_STREAM_OFFSETS = {
    "stable_shift": 11,
    "stable_extrapolation": 23,
    "unstable_extrapolation": 37,
}


def _extend(base: Stream3, data_cfg: DataConfig, cfg: Stage4BConfig, seed: int) -> Stream4B:
    region = SECOND_UNKNOWN_RANGES[base.name]
    second_seed = 400_000 + 1_000 * seed + _STREAM_OFFSETS[base.name]
    rng = np.random.default_rng(second_seed)
    x = rng.uniform(region[0], region[1], size=cfg.second_unknown_n)
    second_unknown = _make_xy(x, true_function, data_cfg.noise_std, rng)
    return Stream4B(
        id_pre=base.id_pre,
        shadow_train=base.shadow_train,
        id_anchor=base.id_anchor,
        promotion_val=base.promotion_val,
        post_decision=base.post_decision,
        id_return=base.id_return,
        second_unknown=second_unknown,
        name=base.name,
        second_unknown_range=region,
        second_unknown_seed=second_seed,
        protected_id_ranges=tuple(tuple(r) for r in data_cfg.train_ranges),
    )


def build_stage4b_stable_shift(data_cfg: DataConfig, cfg: Stage4BConfig, seed: int) -> Stream4B:
    return _extend(build_stable_shift_stream(data_cfg, seed), data_cfg, cfg, seed)


def build_stage4b_stable_extrapolation(data_cfg: DataConfig, cfg: Stage4BConfig, seed: int) -> Stream4B:
    return _extend(build_stable_extrapolation_stream(data_cfg, seed), data_cfg, cfg, seed)


def build_stage4b_unstable_extrapolation(data_cfg: DataConfig, cfg: Stage4BConfig, seed: int) -> Stream4B:
    return _extend(build_unstable_extrapolation_stream(data_cfg, seed), data_cfg, cfg, seed)


STREAM_BUILDERS_4B = {
    "stable_shift": build_stage4b_stable_shift,
    "stable_extrapolation": build_stage4b_stable_extrapolation,
    "unstable_extrapolation": build_stage4b_unstable_extrapolation,
}
