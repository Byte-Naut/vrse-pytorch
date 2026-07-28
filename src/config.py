from dataclasses import dataclass, field


@dataclass
class DataConfig:
    train_ranges: list = field(default_factory=lambda: [(-3.0, -1.0), (1.0, 3.0)])
    gap_range: tuple = (-1.0, 1.0)
    extrapolation_range: tuple = (3.0, 5.0)
    shift_range: tuple = (-6.0, -3.0)
    n_train: int = 400
    noise_std: float = 0.05


@dataclass
class ModelConfig:
    input_dim: int = 1
    hidden_dim: int = 32
    n_layers: int = 3
    output_dim: int = 1
    gate_type: str = "snr_scalar"  # snr_scalar | snr_vector | learned | shuffled_variance | oracle | none
    lambda_init: float = 1.0
    eps: float = 1e-6
    prior_std: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 1e-3
    batch_size: int = 64
    kl_weight: float = 1e-3
    two_stage: bool = True
    stage1_epochs: int = 1000
    seed: int = 0


@dataclass
class SNGPConfig:
    """Stage-3 dual-track SNGP + shadow learner (Plan3.md, results/STAGE3_PROTOCOL.md).

    Locked design: feature extractor phi_SN feeds an RFF + closed-form
    Bayesian linear regression GP head. Exact RBF-GP is explicitly excluded
    this round (protocol §"GP head"). All values below are fixed by the
    pre-registered protocol and must not be tuned against any online stream.
    """

    input_dim: int = 1
    hidden_dim: int = 32
    n_blocks: int = 2
    sn_multiplier: float = 0.95

    rff_dim: int = 128
    rff_seed: int = 0
    length_scale_mode: str = "median"  # median-distance heuristic on phi_SN(x_train)

    noise_var: float = 0.05**2  # known observation noise sigma^2, matches DataConfig.noise_std
    prior_precision: float = 1.0  # lambda
    jitter: float = 1e-6

    tau_percentile: float = 95.0  # target coverage p0 (%), computed on independent ID calibration set
    tau_confidence: float = 0.95  # Stage-3B: confidence level for the one-sided tolerance limit on tau
    id_calib_n: int = 2000  # Stage-3B: 200 -> 2000, see STAGE3_PROTOCOL.md "Stage-3B amendment"


@dataclass
class Stage3Config:
    """Stream segmentation and experiment matrix for Stage-3 (protocol §"Data streams")."""

    n_id_pre: int = 64
    n_shadow_train: int = 128
    n_promotion_val: int = 64
    n_post_decision: int = 64
    n_id_return: int = 64

    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # Promotion rule thresholds (protocol §"Promotion rule")
    promotion_rmse_ratio: float = 0.8
    promotion_id_anchor_ratio: float = 1.05
    promotion_support_frac: float = 0.90


@dataclass
class Stage4BConfig:
    """Region-expert promotion without global posterior replacement.

    Stage-4B deliberately removes the global-ID non-inferiority test for the
    shadow GP: the shadow is a regional expert and is never served outside its
    promoted region.  Old-ID safety is therefore checked as a routing
    invariant against the frozen deployment posterior.
    """

    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])

    promotion_rmse_ratio: float = 0.80
    promotion_q95_ratio: float = 1.00

    scan_domain: tuple = (-8.0, 7.0)
    scan_points: int = 4097
    max_id_region_overlap_frac: float = 0.0
    exact_fidelity_tol: float = 1e-6

    second_unknown_n: int = 64
    promoted_route_min_frac: float = 0.95
    target_vs_ungated_rmse_ratio: float = 1.20

    min_seed_passes: int = 4
    max_unstable_false_promotions: int = 1


@dataclass
class Stage4CConfig(Stage4BConfig):
    """Stage-4C: Stage-4B with observed-span-first regional boundaries."""

    observed_span_min_width: float = 1e-6
