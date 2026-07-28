"""1D safe-residual regression dataset (Plan.md §7, Experiment 1).

y(x) = B(x) + delta(x), with B(x) = x as the trusted/safe backbone and
delta(x) a more complex correction that the Bayesian side-branch must learn.

Training data only covers a few disjoint intervals. We explicitly hold out:
  - a "gap" region inside the training support's convex hull (interpolation OOD)
  - an "extrapolation" region beyond the training support
  - a "shift" region representing a distribution shift far from training data
"""

import numpy as np
import torch

from src.config import DataConfig


def backbone(x: np.ndarray) -> np.ndarray:
    """Trusted safe baseline B(x) = x."""
    return x


def delta(x: np.ndarray) -> np.ndarray:
    """Complex correction the side-branch must learn: high-frequency + nonlinear."""
    return 0.6 * np.sin(3.0 * x) + 0.15 * x**2


def true_function(x: np.ndarray) -> np.ndarray:
    return backbone(x) + delta(x)


def _sample_uniform(ranges, n, rng):
    """Sample n points uniformly from a list of (lo, hi) ranges (equal weight per range)."""
    if isinstance(ranges, tuple):
        ranges = [ranges]
    n_ranges = len(ranges)
    counts = [n // n_ranges] * n_ranges
    counts[-1] += n - sum(counts)
    parts = []
    for (lo, hi), c in zip(ranges, counts):
        parts.append(rng.uniform(lo, hi, size=c))
    return np.concatenate(parts)


class Dataset1D:
    def __init__(self, cfg: DataConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _make_split(self, ranges, n, noisy: bool):
        x = _sample_uniform(ranges, n, self.rng)
        y_clean = true_function(x)
        if noisy:
            y = y_clean + self.rng.normal(0.0, self.cfg.noise_std, size=x.shape)
        else:
            y = y_clean
        return self._to_tensors(x, y)

    @staticmethod
    def _to_tensors(x, y):
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        return x_t, y_t

    def train(self):
        return self._make_split(self.cfg.train_ranges, self.cfg.n_train, noisy=True)

    def val(self):
        return self._make_split(self.cfg.train_ranges, max(self.cfg.n_train // 4, 20), noisy=True)

    def id_test(self):
        """In-distribution test: same support as training, no noise (clean eval)."""
        return self._make_split(self.cfg.train_ranges, 200, noisy=False)

    def gap_test(self):
        """Interpolation OOD: inside the convex hull of training data but never seen."""
        return self._make_split(self.cfg.gap_range, 200, noisy=False)

    def extrapolation_test(self):
        """Extrapolation OOD: beyond the training support."""
        return self._make_split(self.cfg.extrapolation_range, 200, noisy=False)

    def shift_test(self):
        """Distribution shift: far region, different regime entirely."""
        return self._make_split(self.cfg.shift_range, 200, noisy=False)

    def all_splits(self):
        return {
            "train": self.train(),
            "val": self.val(),
            "id_test": self.id_test(),
            "gap_test": self.gap_test(),
            "extrapolation_test": self.extrapolation_test(),
            "shift_test": self.shift_test(),
        }
