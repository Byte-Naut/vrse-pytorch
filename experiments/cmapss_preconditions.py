"""Audit the C-MAPSS protocol and freeze one shared checkpoint per seed."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from experiments.cmapss_common import (
    CHECKPOINTS,
    PRECONDITIONS_JSON,
    PRECONDITIONS_MD,
    PREPARED,
    ROOT,
    write_json,
)
from benchmarks.cmapss_fd002 import (
    PROTOCOL_REVISION,
    SEEDS,
    load_prepared,
    make_cmapss_split,
    normalization_audit,
)
from benchmarks.cmapss_methods import build_seed_template, seed_preconditions
from vrse._algorithm import GPHead


def _bake_spectral_norm_for_save(model: torch.nn.Module) -> None:
    """Remove spectral-norm parametrizations in place (mean-preserving) so the
    frozen, no-longer-trained module can be torch.save'd. Parametrized modules
    raise on direct pickling; leave_parametrized=True bakes in the already
    normalized weight, which is exact since _phi_sn is frozen after fit()."""
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            torch.nn.utils.parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )


def _run_gate(command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _gp_equivalence(model, split) -> dict:
    n = min(128, split.id_fit.x.shape[0])
    with torch.no_grad():
        z = model._phi_sn(split.id_fit.x[:n])
        r = (split.id_fit.y[:n] - model._baseline(split.id_fit.x[:n])).reshape(-1)
        phi = model._pretrain_deploy_head.rff_map(z)
    source = model._pretrain_deploy_head.posterior
    batch = GPHead(
        model._pretrain_deploy_head.rff_map,
        source.noise_var,
        source.prior_precision,
        source.jitter,
    )
    incremental = GPHead(
        model._pretrain_deploy_head.rff_map,
        source.noise_var,
        source.prior_precision,
        source.jitter,
    )
    batch.posterior.fit_batch(phi, r)
    for i in range(n):
        incremental.posterior.update_incremental(phi[i], float(r[i].item()))
    lambda_diff = float((batch.posterior.Lambda - incremental.posterior.Lambda).abs().max().item())
    q_diff = float((batch.posterior.q - incremental.posterior.q).abs().max().item())
    return {
        "samples": n,
        "lambda_max_diff": lambda_diff,
        "q_max_diff": q_diff,
        "passed": lambda_diff < 1e-10 and q_diff < 1e-10,
    }


def _split_integrity(split, scale_audit: dict) -> dict:
    role_sets = {name: set(units) for name, units in split.roles.items()}
    names = list(role_sets)
    disjoint = all(
        role_sets[names[i]].isdisjoint(role_sets[names[j]])
        for i in range(len(names)) for j in range(i + 1, len(names))
    )
    batches = (
        split.id_fit, split.id_calibration, split.id_guard,
        split.shadow_observe, split.promotion_validation,
        split.post_new, split.post_unknown,
    )
    finite = all(torch.isfinite(batch.x).all() and torch.isfinite(batch.y).all()
                 for batch in batches)
    shapes = all(batch.x.ndim == 2 and batch.x.shape[1] == 24
                 and batch.y.ndim == 2 and batch.y.shape[1] == 1
                 for batch in batches)
    role_union = set().union(*role_sets.values())
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "normalization_source": split.normalization_source,
        "normalization_audit": scale_audit,
        "roles_disjoint": disjoint,
        "role_units": len(role_union),
        "features_24d_scalar_target": shapes,
        "all_finite": bool(finite),
        "passed": bool(
            disjoint and len(role_union) == 240 and shapes and finite
            and scale_audit["passed"]
        ),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# C-MAPSS FD002 Preconditions",
        "",
        f"> Verdict: **{payload['verdict']}**",
        f"> Protocol: `{payload['protocol_revision']}`",
        f"> Normalization audit: **{payload['normalization_audit']['passed']}**, "
        f"max |z| = `{payload['normalization_audit']['max_abs_z_all_rows']:.6g}`",
        "",
        "| seed | data | P1 ratio | P1 | P2 reject | P2 | P3 | spectral | GP eq |",
        "|---:|:---:|---:|:---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in payload["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['data_integrity']['passed']} | "
            f"{row['p1_baseline_vs_constant_ratio']:.4f} | "
            f"{row['p1_pass']} | {row['p2_initial_rejection']:.4f} | {row['p2_pass']} | "
            f"{row['p3_pass']} | {row['spectral_pass']} | {row['gp_equivalence']['passed']} |"
        )
    lines += [
        "",
        "Aggregate gates:",
        "",
        f"- P0: {payload['aggregate']['p0_pass']}",
        f"- P1 (>=4/5): {payload['aggregate']['p1_pass']}",
        f"- P2 (>=4/5): {payload['aggregate']['p2_pass']}",
        f"- P3 (>=4/5): {payload['aggregate']['p3_pass']}",
        "",
        "No matrix should be run unless the verdict is `READY_FOR_MATRIX`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-contract-gates", action="store_true",
        help="Debug only: skip the public contract tests and quickstart.",
    )
    args = parser.parse_args()
    if not PREPARED.exists():
        raise FileNotFoundError(f"Run cmapss_prepare_data.py first: missing {PREPARED}")

    gates = []
    if not args.skip_contract_gates:
        gates = [
            _run_gate([sys.executable, "-m", "pytest", "tests", "-q"]),
            _run_gate([sys.executable, "-m", "examples.quickstart"]),
        ]
    data, definition = load_prepared(PREPARED)
    scale_audit = normalization_audit(data)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        split = make_cmapss_split(data, definition, seed)
        template, build_info = build_seed_template(split)
        row = seed_preconditions(template, split)
        row["build"] = build_info
        row["gp_equivalence"] = _gp_equivalence(template, split)
        row["data_integrity"] = _split_integrity(split, scale_audit)
        rows.append(row)
        _bake_spectral_norm_for_save(template)
        torch.save(
            {
                "model": template,
                "seed": seed,
                "build": build_info,
                "protocol_revision": PROTOCOL_REVISION,
            },
            CHECKPOINTS / f"seed_{seed}.pt",
        )

    # A debug run that skips the frozen gates must never unlock the formal matrix.
    p0_pass = bool(gates) and all(g["passed"] for g in gates)
    p0_pass = p0_pass and all(
        row["spectral_pass"]
        and row["gp_equivalence"]["passed"]
        and row["data_integrity"]["passed"]
        for row in rows
    )
    aggregate = {
        "p0_pass": p0_pass,
        "p1_pass": sum(row["p1_pass"] for row in rows) >= 4,
        "p2_pass": sum(row["p2_pass"] for row in rows) >= 4,
        "p3_pass": sum(row["p3_pass"] for row in rows) >= 4,
    }
    if not aggregate["p0_pass"]:
        verdict = "INVALID"
    elif not aggregate["p1_pass"]:
        verdict = "STOP_TASK"
    elif not aggregate["p2_pass"]:
        verdict = "PIVOT_DETECTOR"
    elif not aggregate["p3_pass"]:
        verdict = "PIVOT_LEARNER"
    else:
        verdict = "READY_FOR_MATRIX"
    payload = {
        "protocol_revision": PROTOCOL_REVISION,
        "normalization_audit": scale_audit,
        "verdict": verdict,
        "debug_skipped_contract_gates": bool(args.skip_contract_gates),
        "aggregate": aggregate,
        "contract_gates": gates,
        "seeds": rows,
    }
    write_json(PRECONDITIONS_JSON, payload)
    PRECONDITIONS_MD.write_text(_markdown(payload), encoding="utf-8")
    print(verdict)
    print(PRECONDITIONS_MD)


if __name__ == "__main__":
    main()
