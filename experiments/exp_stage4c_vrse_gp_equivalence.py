"""Numerical equivalence check: vrse's own GP reimplementation vs src/sngp_gp.py.

`experiments/exp_stage4c_vrse_regression.py` tests vrse's *ported decision
logic* (validate_promotion / build_observed_span_region) but reuses the
*original* src/sngp_gp.GPHead object throughout -- it never exercises vrse's
own `_RFFMap` / `GPPosterior` / `GPHead` reimplementation in `vrse/_algorithm.py`.

This script closes that gap directly: seed a vrse GPHead with the exact
(W, b, Lambda, q, noise_var, prior_precision, jitter) copied from the frozen
Stage-3 checkpoint's original GPHead, then check that:
  1. `predict()` agrees with the original to near machine precision on real
     stream inputs.
  2. Running the identical 128-step `update_incremental` sequence (same
     features, same residuals) on both produces the same posterior (Lambda, q)
     and therefore the same subsequent predictions.

This does not re-validate the promotion decisions (that's the other script);
it isolates and verifies vrse's GP math is a faithful reimplementation.
"""

import torch

from src.config import Stage4CConfig
from src.sngp_feature import PhiSN
from src.sngp_service import residual_target
from src.streams4b import STREAM_BUILDERS_4B

from vrse._algorithm import GPHead as VrseGPHead, GPPosterior as VrseGPPosterior, _RFFMap as VrseRFFMap

CHECKPOINT = "results/frozen_stage3_checkpoint.pt"


def _vrse_head_from_original(orig_head) -> VrseGPHead:
    """Copy (W, b, Lambda, q, hyperparameters) from an original src.sngp_gp.GPHead
    into a freshly-constructed vrse GPHead, bypassing vrse's own training path."""
    rff_map = VrseRFFMap.__new__(VrseRFFMap)
    rff_map.W = orig_head.rff_map.W.clone()
    rff_map.b = orig_head.rff_map.b.clone()
    rff_map.rff_dim = orig_head.rff_map.rff_dim

    posterior = VrseGPPosterior.__new__(VrseGPPosterior)
    posterior.rff_dim = orig_head.posterior.rff_dim
    posterior.noise_var = orig_head.posterior.noise_var
    posterior.prior_precision = orig_head.posterior.prior_precision
    posterior.jitter = orig_head.posterior.jitter
    posterior.Lambda = orig_head.posterior.Lambda.clone()
    posterior.q = orig_head.posterior.q.clone()

    head = VrseGPHead.__new__(VrseGPHead)
    head.rff_map = rff_map
    head.posterior = posterior
    return head


def main():
    checkpoint = torch.load(CHECKPOINT, weights_only=False)
    phi_sn = PhiSN(checkpoint["sngp_cfg"])
    phi_sn.load_state_dict(checkpoint["phi_sn_state_dict"])
    phi_sn.freeze()

    orig_head = checkpoint["gp_head"]
    cfg = Stage4CConfig()

    stream = STREAM_BUILDERS_4B["stable_shift"](checkpoint["data_cfg"], cfg, seed=0)
    x_probe = stream.promotion_val[0]

    # --- Check 1: predict() agreement on real stream inputs, no updates yet ---
    vrse_head = _vrse_head_from_original(orig_head)
    with torch.no_grad():
        z = phi_sn(x_probe)
        mu_orig, u_orig = orig_head.predict(z)
        mu_vrse, u_vrse = vrse_head.predict(z)

    mu_diff = (mu_orig - mu_vrse).abs().max().item()
    u_diff = (u_orig - u_vrse).abs().max().item()
    print(f"[predict, pre-update] max|mu diff| = {mu_diff:.3e}, max|u diff| = {u_diff:.3e}")
    check1 = mu_diff < 1e-5 and u_diff < 1e-5

    # --- Check 2: identical 128-step update_incremental sequence ---
    orig_shadow = orig_head.clone()
    vrse_shadow = _vrse_head_from_original(orig_head)

    x_shadow, y_shadow = stream.shadow_train
    for i in range(x_shadow.shape[0]):
        xi, yi = x_shadow[i : i + 1], y_shadow[i : i + 1]
        with torch.no_grad():
            zi = phi_sn(xi)
        r = residual_target(xi, yi).reshape(-1)[0]
        orig_shadow.update_incremental(zi.squeeze(0), r)
        vrse_shadow.update_incremental(zi.squeeze(0), float(r))

    lambda_diff = (orig_shadow.posterior.Lambda - vrse_shadow.posterior.Lambda).abs().max().item()
    q_diff = (orig_shadow.posterior.q - vrse_shadow.posterior.q).abs().max().item()
    print(f"[update_incremental x128] max|Lambda diff| = {lambda_diff:.3e}, max|q diff| = {q_diff:.3e}")
    check2 = lambda_diff < 1e-8 and q_diff < 1e-8

    with torch.no_grad():
        z_val = phi_sn(stream.post_decision[0])
        mu_orig2, u_orig2 = orig_shadow.predict(z_val)
        mu_vrse2, u_vrse2 = vrse_shadow.predict(z_val)
    mu_diff2 = (mu_orig2 - mu_vrse2).abs().max().item()
    u_diff2 = (u_orig2 - u_vrse2).abs().max().item()
    print(f"[predict, post-update] max|mu diff| = {mu_diff2:.3e}, max|u diff| = {u_diff2:.3e}")
    check3 = mu_diff2 < 1e-5 and u_diff2 < 1e-5

    ok = check1 and check2 and check3
    print(f"\nGP EQUIVALENCE {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
