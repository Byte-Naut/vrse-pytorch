"""Mechanical Phase-3 verdict and human-readable result tables."""
from __future__ import annotations

import pickle

from experiments.phase3_common import (
    MATRIX_PKL,
    METRICS_MD,
    PRECONDITIONS_JSON,
    RESULT_MD,
    VERDICT_JSON,
    load_json,
    write_json,
)
from src.phase3_methods import EXACT_TOL
from src.phase3_cmapss import PROTOCOL_REVISION


SEEDS = (4300, 4301, 4302, 4303, 4304)
STABLE = "stable_condition"
REVERSED = "reversed_condition"
TARGET = "VRSE-KNN"


def _passes_4(flags) -> bool:
    return sum(bool(v) for v in flags) >= 4


def _precondition_terminal(pre: dict) -> str | None:
    if pre["verdict"] == "READY_FOR_MATRIX":
        return None
    return pre["verdict"]


def _metrics_table(matrix: dict) -> str:
    lines = [
        "# Phase 3 Metrics",
        "",
        "| stream | method | seed | promoted | new RMSE | ID route | unknown route |",
        "|---|---|---:|:---:|---:|---:|---:|",
    ]
    for stream, by_seed in matrix.items():
        for seed in SEEDS:
            for method, run in by_seed[seed].items():
                lines.append(
                    f"| {stream} | {method} | {seed} | {run['promoted']} | "
                    f"{run['domains']['post_new']['rmse']:.6f} | "
                    f"{run['domains']['id_guard']['route_frac']:.6f} | "
                    f"{run['domains']['post_unknown']['route_frac']:.6f} |"
                )
    return "\n".join(lines) + "\n"


def _result_markdown(result: dict) -> str:
    lines = [
        "# Phase 3 Result",
        "",
        f"> Verdict: **{result['verdict']}**",
        f"> Protocol: `{result['protocol_revision']}`",
        "",
        "## Mechanical bullets",
        "",
    ]
    for name, value in result.get("bullets", {}).items():
        lines.append(f"- {name}: **{value}**")
    lines += ["", "## Per-seed evidence", ""]
    for name, values in result.get("per_seed", {}).items():
        lines.append(f"- {name}: `{values}`")
    lines += [
        "",
        "This file is generated from raw matrix data. Do not edit it before verdict calculation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    pre = load_json(PRECONDITIONS_JSON)
    if pre.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError(
            "Precondition artifacts predate the Phase-3B normalization amendment; rerun them."
        )
    terminal = _precondition_terminal(pre)
    if terminal is not None:
        result = {
            "protocol_revision": PROTOCOL_REVISION,
            "verdict": terminal,
            "bullets": pre["aggregate"],
            "per_seed": {},
            "source": "phase3_preconditions.json",
        }
        write_json(VERDICT_JSON, result)
        RESULT_MD.write_text(_result_markdown(result), encoding="utf-8")
        print(terminal)
        return

    with MATRIX_PKL.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError("Matrix artifact belongs to an older Phase-3 protocol.")
    matrix = payload["matrix"]

    h1 = []
    stable_promotions = []
    reversed_promotions = []
    h3 = []
    h4 = []
    for seed in SEEDS:
        stable = matrix[STABLE][seed][TARGET]
        reversed_run = matrix[REVERSED][seed][TARGET]
        h1.append(
            stable["isolation_max_diff"] < EXACT_TOL
            and reversed_run["isolation_max_diff"] < EXACT_TOL
        )
        stable_promotions.append(bool(stable["promoted"]))
        reversed_promotions.append(bool(reversed_run["promoted"]))

        id_m = stable["domains"]["id_guard"]
        unknown_m = stable["domains"]["post_unknown"]
        new_m = stable["domains"]["post_new"]
        h3.append(
            bool(stable["promoted"])
            and id_m["route_frac"] == 0.0
            and id_m["prediction_max_diff_vs_frozen"] < EXACT_TOL
            and unknown_m["route_frac"] == 0.0
            and unknown_m["prediction_max_diff_vs_frozen"] < EXACT_TOL
            and new_m["route_frac"] >= 0.80
        )

        frozen_rmse = matrix[STABLE][seed]["Frozen"]["domains"]["post_new"]["rmse"]
        global_rmse = matrix[STABLE][seed]["Shadow-global"]["domains"]["post_new"]["rmse"]
        routed = new_m["routed_subset"]
        h4.append(
            bool(stable["promoted"])
            and new_m["rmse"] <= 0.90 * frozen_rmse
            and new_m["rmse"] <= 1.20 * global_rmse
            and routed["count"] > 0
            and routed["rmse"] <= 0.80 * routed["frozen_rmse"]
        )

    bullets = {
        "H1_isolation_10_of_10": all(h1),
        "H2_stable_promote_4_of_5": _passes_4(stable_promotions),
        "H2_reversed_false_promote_at_most_1_of_5": sum(reversed_promotions) <= 1,
        "H3_regional_isolation_4_of_5": _passes_4(h3),
        "H4_stable_utility_4_of_5": _passes_4(h4),
    }
    if not bullets["H1_isolation_10_of_10"] or not (
        bullets["H2_stable_promote_4_of_5"]
        and bullets["H2_reversed_false_promote_at_most_1_of_5"]
    ):
        verdict = "PIVOT_CORE"
    elif not bullets["H3_regional_isolation_4_of_5"] or not bullets["H4_stable_utility_4_of_5"]:
        verdict = "CONDITIONAL_PIVOT_SUPPORT"
    else:
        verdict = "PASS"
    result = {
        "protocol_revision": PROTOCOL_REVISION,
        "verdict": verdict,
        "bullets": bullets,
        "per_seed": {
            "H1": h1,
            "stable_promotions": stable_promotions,
            "reversed_promotions": reversed_promotions,
            "H3": h3,
            "H4": h4,
        },
    }
    write_json(VERDICT_JSON, result)
    RESULT_MD.write_text(_result_markdown(result), encoding="utf-8")
    METRICS_MD.write_text(_metrics_table(matrix), encoding="utf-8")
    print(verdict)
    print(RESULT_MD)


if __name__ == "__main__":
    main()
