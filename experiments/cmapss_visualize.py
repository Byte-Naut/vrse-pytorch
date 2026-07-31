"""Generate the three frozen C-MAPSS figures after the verdict."""
from __future__ import annotations

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from experiments.cmapss_common import CHECKPOINTS, MATRIX_PKL, PREPARED, RESULTS
from benchmarks.cmapss_fd002 import PROTOCOL_REVISION, load_prepared, make_cmapss_split
from benchmarks.cmapss_methods import METHODS


SEED = 4300


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_current_checkpoint(path):
    checkpoint = _load_checkpoint(path)
    if checkpoint.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError("Checkpoint belongs to an older C-MAPSS protocol.")
    return checkpoint


def _take(x: torch.Tensor, n: int = 1500) -> torch.Tensor:
    if x.shape[0] <= n:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, n).long()
    return x[idx]


def embedding_figure() -> None:
    data, definition = load_prepared(PREPARED)
    split = make_cmapss_split(data, definition, SEED)
    model = copy.deepcopy(_load_current_checkpoint(CHECKPOINTS / f"seed_{SEED}.pt")["model"])
    model.observe(split.shadow_observe.x, split.shadow_observe.y)
    proposal = model.evaluate(
        split.promotion_validation.x,
        split.promotion_validation.y,
        guard_x=split.id_guard.x,
    )
    model.promote(proposal)

    groups = {
        "ID": _take(split.id_guard.x),
        "NEW": _take(split.post_new.x),
        "UNKNOWN": _take(split.post_unknown.x),
    }
    with torch.no_grad():
        embeddings = {name: model._phi_sn(x).cpu().numpy() for name, x in groups.items()}
    joined = np.concatenate(list(embeddings.values()), axis=0)
    pca = PCA(n_components=2, random_state=0).fit(joined)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name, x in groups.items():
        z2 = pca.transform(embeddings[name])
        mask = model.route_mask(x).cpu().numpy()
        ax.scatter(z2[~mask, 0], z2[~mask, 1], s=8, alpha=0.35, label=f"{name} fallback")
        if mask.any():
            ax.scatter(z2[mask, 0], z2[mask, 1], s=12, alpha=0.8, marker="x",
                       label=f"{name} authorized")
    ax.set_title("C-MAPSS frozen PhiSN space and authorization mask")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "cmapss_fd002_embedding.png", dpi=180)
    plt.close(fig)


def behavior_figure(matrix: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, stream in zip(axes, ("stable_condition", "reversed_condition")):
        means = []
        stds = []
        for method in METHODS:
            values = [matrix[stream][seed][method]["domains"]["post_new"]["rmse"]
                      for seed in sorted(matrix[stream])]
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))
        ax.bar(np.arange(len(METHODS)), means, yerr=stds, capsize=3)
        ax.set_xticks(np.arange(len(METHODS)), METHODS, rotation=35, ha="right")
        ax.set_title(stream)
        ax.set_ylabel("Post-decision RMSE")
    fig.suptitle("Stable adaptation versus concept reversal")
    fig.tight_layout()
    fig.savefig(RESULTS / "cmapss_fd002_stream_behavior.png", dpi=180)
    plt.close(fig)


def safety_plasticity_figure(matrix: dict) -> None:
    stream = "stable_condition"
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for method in METHODS:
        gains = []
        interference = []
        for seed in sorted(matrix[stream]):
            run = matrix[stream][seed][method]
            frozen = matrix[stream][seed]["Frozen"]["domains"]["post_new"]["rmse"]
            gains.append(1.0 - run["domains"]["post_new"]["rmse"] / frozen)
            interference.append(max(
                run["domains"]["id_guard"]["route_frac"],
                run["domains"]["post_unknown"]["route_frac"],
            ))
        ax.scatter(np.mean(interference), np.mean(gains), s=70, label=method)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("ID / unknown maximum non-baseline route fraction (lower is safer)")
    ax.set_ylabel("New-condition RMSE gain over Frozen (higher is better)")
    ax.set_title("C-MAPSS safety-plasticity plane")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "cmapss_fd002_safety_plasticity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    with MATRIX_PKL.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("protocol_revision") != PROTOCOL_REVISION:
        raise RuntimeError("Matrix artifact belongs to an older C-MAPSS protocol.")
    matrix = payload["matrix"]
    embedding_figure()
    behavior_figure(matrix)
    safety_plasticity_figure(matrix)
    print(RESULTS / "cmapss_fd002_embedding.png")
    print(RESULTS / "cmapss_fd002_stream_behavior.png")
    print(RESULTS / "cmapss_fd002_safety_plasticity.png")


if __name__ == "__main__":
    main()
