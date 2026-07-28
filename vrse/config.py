from dataclasses import dataclass, field


_PRESETS = {
    # Values match the validated Stage-4C protocol (src/config.py:
    # SNGPConfig, Stage4BConfig/Stage4CConfig) so this preset actually
    # encodes the ported, regression-tested algorithm rather than a
    # differently-tuned lookalike.
    "regional_regression": {
        "rff_dim": 128,
        "n_blocks": 2,
        "hidden_dim": 32,
        "sn_multiplier": 0.95,
        "prior_std": 1.0,
        "noise_std": 0.05,
        "tau_percentile": 95.0,
        "tau_confidence": 0.95,
        "promotion_rmse_ratio": 0.80,
        "promotion_q95_ratio": 1.0,
        "min_shadow_updates": 30,
    },
    # Phase-3 C-MAPSS preset.  It deliberately keeps the validated GP and
    # promotion ratios while changing only the input-layer constraint and
    # the internal support geometry required by N-D inputs.
    "regional_regression_highdim": {
        "rff_dim": 128,
        "n_blocks": 2,
        "hidden_dim": 32,
        "sn_multiplier": 0.95,
        "spectral_input": True,
        "prior_std": 1.0,
        "noise_std": 0.05,
        "tau_percentile": 95.0,
        "tau_confidence": 0.95,
        "promotion_rmse_ratio": 0.80,
        "promotion_q95_ratio": 1.0,
        "min_shadow_updates": 30,
        "support_kind": "knn_feature",
        "knn_k": 5,
        "max_support_prototypes": 512,
        "length_scale_max_points": 2048,
        "phi_epochs": 100,
        "phi_lr": 1e-3,
    },
}


@dataclass
class VRSEConfig:
    preset: str = "regional_regression"
    rff_dim: int = 128
    n_blocks: int = 2
    hidden_dim: int = 32
    sn_multiplier: float = 0.95
    prior_std: float = 1.0
    noise_std: float = 0.05
    tau_percentile: float = 95.0
    tau_confidence: float = 0.95
    promotion_rmse_ratio: float = 0.80
    promotion_q95_ratio: float = 1.0
    min_shadow_updates: int = 30
    spectral_input: bool = False
    support_kind: str = "auto"
    knn_k: int = 5
    max_support_prototypes: int = 512
    length_scale_max_points: int = 2048
    phi_epochs: int = 500
    phi_lr: float = 1e-3
    random_seed: int = 0
    version: str = "0.1.0"

    def __post_init__(self):
        # Apply preset defaults only for fields the caller left at their
        # dataclass default value -- never overwrite an explicitly supplied arg.
        # Strategy: compare each field against its dataclass default; if it
        # matches, the caller didn't supply it and the preset may fill it in.
        import dataclasses
        defaults = {f.name: f.default for f in dataclasses.fields(self)
                    if f.default is not dataclasses.MISSING}
        if self.preset in _PRESETS:
            for k, v in _PRESETS[self.preset].items():
                if getattr(self, k) == defaults.get(k):
                    object.__setattr__(self, k, v)
        if self.support_kind not in {"auto", "observed_span", "knn_feature"}:
            raise ValueError(f"Unsupported support_kind: {self.support_kind!r}")
        if self.knn_k < 1:
            raise ValueError("knn_k must be >= 1")
        if self.max_support_prototypes < self.knn_k:
            raise ValueError("max_support_prototypes must be >= knn_k")
        if self.length_scale_max_points < 2:
            raise ValueError("length_scale_max_points must be >= 2")
        if self.phi_epochs < 1 or self.phi_lr <= 0:
            raise ValueError("phi_epochs and phi_lr must be positive")
