"""Stage-4C promotion-outcome regression for the vrse/v0.1 port.

Confirms that `vrse._algorithm.validate_promotion` / `build_observed_span_region`
(ported from `src/stage4c.py` in Phase 2) reproduce the exact promotion
decisions and region boundaries recorded in `results/stage4c_matrix_results.pkl`
(the original Stage-4C 10/10 stable-seed result, results/STAGE4C_RESULT.md).

Uses the same frozen checkpoint, streams, and seeds as
`experiments/exp_stage4c_matrix.py`; only the promotion/region-construction
function under test is swapped for the vrse port. Everything upstream
(phi_SN, initial GP head, tau, shadow incremental training) is bit-identical
to the original run, so any divergence in `passed` or region bounds is
attributable to the ported algorithm, not to a different experimental setup.
"""

import pickle

import torch
import torch.nn as nn

from src.config import Stage4CConfig
from src.sngp_feature import PhiSN
from src.sngp_service import residual_target
from src.streams4b import STREAM_BUILDERS_4B

from vrse._algorithm import validate_promotion
from vrse.config import VRSEConfig

CHECKPOINT = "results/frozen_stage3_checkpoint.pt"
PRECONDITIONS = "results/stage3_preconditions.pkl"
ORIGINAL_MATRIX = "results/stage4c_matrix_results.pkl"


class _IdentityBackbone(nn.Module):
    """B(x) = x, matching src/dataset.py::backbone."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _train_shadow(phi_sn, initial_head, batch):
    """Bit-identical to src/methods4b.py::_train_shadow."""
    head = initial_head.clone()
    x, y = batch
    for i in range(x.shape[0]):
        xi, yi = x[i : i + 1], y[i : i + 1]
        with torch.no_grad():
            z = phi_sn(xi)
        head.update_incremental(z.squeeze(0), residual_target(xi, yi).reshape(-1)[0])
    return head


def main():
    with open(PRECONDITIONS, "rb") as f:
        preconditions = pickle.load(f)
    if not preconditions["infrastructure_established"]:
        raise RuntimeError("Frozen Stage-3 infrastructure preconditions did not pass")

    with open(ORIGINAL_MATRIX, "rb") as f:
        original = pickle.load(f)

    checkpoint = torch.load(CHECKPOINT, weights_only=False)
    phi_sn = PhiSN(checkpoint["sngp_cfg"])
    phi_sn.load_state_dict(checkpoint["phi_sn_state_dict"])
    phi_sn.freeze()

    stage4c_cfg = Stage4CConfig()
    baseline = _IdentityBackbone()
    # Bare default preset -- no overrides. VRSEConfig's "regional_regression"
    # preset defaults (promotion_rmse_ratio=0.80, promotion_q95_ratio=1.0, ...)
    # are meant to already encode this exact validated protocol; if they
    # didn't, this regression would need overrides to pass, which would be
    # the bug this script exists to catch.
    vrse_cfg = VRSEConfig()
    assert vrse_cfg.promotion_rmse_ratio == stage4c_cfg.promotion_rmse_ratio, (
        f"VRSEConfig default promotion_rmse_ratio={vrse_cfg.promotion_rmse_ratio} "
        f"does not match validated Stage4CConfig={stage4c_cfg.promotion_rmse_ratio}"
    )
    assert vrse_cfg.promotion_q95_ratio == stage4c_cfg.promotion_q95_ratio, (
        f"VRSEConfig default promotion_q95_ratio={vrse_cfg.promotion_q95_ratio} "
        f"does not match validated Stage4CConfig={stage4c_cfg.promotion_q95_ratio}"
    )

    rows = []
    mismatches = []
    for stream_name, builder in STREAM_BUILDERS_4B.items():
        for seed in stage4c_cfg.seeds:
            stream = builder(checkpoint["data_cfg"], stage4c_cfg, seed)
            initial_gp_head = checkpoint["gp_head"]
            shadow_head = _train_shadow(phi_sn, initial_gp_head, stream.shadow_train)

            result, region = validate_promotion(
                phi_sn=phi_sn,
                deploy_head=initial_gp_head,
                shadow_head=shadow_head,
                tau_deploy=checkpoint["tau"],
                x_shadow_train=stream.shadow_train[0],
                x_val=stream.promotion_val[0],
                y_val=stream.promotion_val[1],
                x_id_guard=stream.id_anchor[0],
                baseline=baseline,
                cfg=vrse_cfg,
                protected_id_ranges=stream.protected_id_ranges,
                scan_domain=stage4c_cfg.scan_domain,
            )
            vrse_passed = result["passed"]

            orig_run = original["results"][stream_name]["Hard-original"][seed]["run"]
            orig_promoted = bool(orig_run["promotion_info"]["promoted"])
            orig_region = orig_run["promotion_info"]["region"]

            expected_stable = stream_name != "unstable_extrapolation"
            row = {
                "stream": stream_name,
                "seed": seed,
                "vrse_passed": vrse_passed,
                "orig_promoted": orig_promoted,
                "expected_stable": expected_stable,
            }
            rows.append(row)

            if vrse_passed != orig_promoted:
                mismatches.append((stream_name, seed, "promotion_decision_mismatch", row))
                continue
            if vrse_passed:
                orig_lo, orig_hi = orig_region["x_lo"], orig_region["x_hi"]
                if not (abs(region.x_lo - orig_lo) < 1e-4 and abs(region.x_hi - orig_hi) < 1e-4):
                    mismatches.append(
                        (
                            stream_name,
                            seed,
                            "region_bounds_mismatch",
                            {"vrse": (region.x_lo, region.x_hi), "original": (orig_lo, orig_hi)},
                        )
                    )

    n_stable_promoted_vrse = sum(1 for r in rows if r["expected_stable"] and r["vrse_passed"])
    n_stable_total = sum(1 for r in rows if r["expected_stable"])
    n_unstable_promoted_vrse = sum(1 for r in rows if not r["expected_stable"] and r["vrse_passed"])
    n_unstable_total = sum(1 for r in rows if not r["expected_stable"])

    print(f"Stable seeds promoted (vrse):   {n_stable_promoted_vrse}/{n_stable_total}")
    print(f"Unstable seeds promoted (vrse):  {n_unstable_promoted_vrse}/{n_unstable_total}")
    print(f"Mismatches vs original Stage-4C matrix: {len(mismatches)}")
    for m in mismatches:
        print("  MISMATCH:", m)

    for r in rows:
        print(r)

    ok = (
        n_stable_promoted_vrse == n_stable_total
        and n_unstable_promoted_vrse == 0
        and not mismatches
    )
    print(f"\nREGRESSION {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
