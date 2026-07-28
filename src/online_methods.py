"""The four Stage-2 online methods (Plan2 §3, results/STAGE2_PROTOCOL.md).

Each method exposes `predict_and_maybe_update(model, calib, x, y, optimizer,
update: bool) -> dict` with a uniform return payload:
    {"y_hat": tensor, "a_values": list|None, "grad_norm": float, "probe_y_hat": tensor|None}

`update=False` is used for the ID pre-test / return-test segments (predict
only, per protocol). `update=True` runs strict predict-then-update: forward
for the prediction is computed and recorded BEFORE backward/step touches any
parameter, so the recorded prediction always reflects the pre-update model.

Methods:
  1. FrozenLogvar    -- serve gated predictions, never update (safety baseline).
  2. OnlineUngated   -- serve a=1 predictions, update via the ungated loss
                        (max-plasticity baseline).
  3. OnlineLogvar    -- serve gated predictions, update via the SAME gated
                        loss (the mechanism under test).
  4. GatedPredictUngatedUpdate -- serve gated predictions (identical to
                        OnlineLogvar's served output), but the update step
                        uses the UNGATED loss on the same branch. Also
                        computes an ungated PROBE prediction purely for
                        diagnostics (never served, never affects LR
                        selection or the GO verdict) to distinguish "gate
                        signal is the problem" from "gradient starvation":
                        if the probe learns the new region well while the
                        served gated output does not, the missing piece is
                        a reopening mechanism, not learning capacity.
"""

from src.online import forward_gated, forward_ungated, global_grad_norm, trainable_params


def _step(model, optimizer, params, loss):
    for p in params:
        p.grad = None
    loss.backward()
    gnorm = global_grad_norm(params)
    optimizer.step()
    return gnorm


def frozen_logvar(model, calib, x, y, optimizer, update: bool):
    y_hat, a_values = forward_gated(model, calib, x)
    # FrozenLogvar never updates, regardless of the `update` flag — it is the
    # fixed safety baseline. `update` is accepted for interface uniformity
    # with the other three methods but intentionally ignored here.
    return {"y_hat": y_hat.detach(), "a_values": [a.detach() for a in a_values], "grad_norm": 0.0, "probe_y_hat": None}


def online_ungated(model, calib, x, y, optimizer, update: bool):
    y_hat = forward_ungated(model, x)
    y_hat_recorded = y_hat.detach()
    grad_norm = 0.0
    if update:
        import torch.nn.functional as F

        loss = F.mse_loss(y_hat, y)
        params = trainable_params(model)
        grad_norm = _step(model, optimizer, params, loss)
    return {"y_hat": y_hat_recorded, "a_values": None, "grad_norm": grad_norm, "probe_y_hat": None}


def online_logvar(model, calib, x, y, optimizer, update: bool):
    y_hat, a_values = forward_gated(model, calib, x)
    y_hat_recorded = y_hat.detach()
    a_recorded = [a.detach() for a in a_values]
    grad_norm = 0.0
    if update:
        import torch.nn.functional as F

        loss = F.mse_loss(y_hat, y)
        params = trainable_params(model)
        grad_norm = _step(model, optimizer, params, loss)
    return {"y_hat": y_hat_recorded, "a_values": a_recorded, "grad_norm": grad_norm, "probe_y_hat": None}


def gated_predict_ungated_update(model, calib, x, y, optimizer, update: bool):
    # Served prediction: gated forward (identical path to OnlineLogvar).
    y_hat_gated, a_values = forward_gated(model, calib, x)
    y_hat_recorded = y_hat_gated.detach()
    a_recorded = [a.detach() for a in a_values]

    # Diagnostic-only probe: ungated forward, no_grad, never served/used for update.
    import torch

    with torch.no_grad():
        probe_y_hat = forward_ungated(model, x)

    grad_norm = 0.0
    if update:
        import torch.nn.functional as F

        # Update uses the UNGATED loss on the SAME trainable branch params —
        # gradient starvation control: if OnlineLogvar fails to adapt, is it
        # because the gate blocks the loss from reaching M/mu_b (this method
        # would then adapt fine, since its update path bypasses the gate),
        # or because the branch genuinely can't learn the new region fast
        # enough regardless of gating?
        y_hat_ungated_for_update = forward_ungated(model, x)
        loss = F.mse_loss(y_hat_ungated_for_update, y)
        params = trainable_params(model)
        grad_norm = _step(model, optimizer, params, loss)

    return {
        "y_hat": y_hat_recorded,
        "a_values": a_recorded,
        "grad_norm": grad_norm,
        "probe_y_hat": probe_y_hat,
    }


METHODS = {
    "Frozen-logvar": frozen_logvar,
    "Online-ungated": online_ungated,
    "Online-logvar": online_logvar,
    "Gated-predict-ungated-update": gated_predict_ungated_update,
}
