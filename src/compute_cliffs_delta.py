"""Compute Cliff's delta from per-fold RMSE in transfer_cv.json and render heatmap.

Usage:
  python -m src.compute_cliffs_delta
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

RESULTS = pathlib.Path("results_v6")
SUB = pathlib.Path("submission_package/vtse_submission_20260601")


def cliffs_delta(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    diffs = b - a
    n = len(diffs)
    return float((int((diffs > 0).sum()) - int((diffs < 0).sum())) / n) if n else 0.0


def main():
    d = json.load(open(RESULTS / "transfer" / "transfer_cv.json"))
    fr = d["_fold_rmse"]
    methods = list(fr.keys())
    mat = np.zeros((len(methods), len(methods)))
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            mat[i, j] = 0.0 if i == j else cliffs_delta(fr[mi], fr[mj])

    out = {
        "methods": methods,
        "matrix": mat.tolist(),
        "interpretation": (
            "Entry (i,j) is Cliff's delta for method_i vs method_j on per-fold "
            "RMSE; positive means method_i has lower RMSE on a majority of "
            "paired folds. |delta|<=0.147 negligible; <=0.33 small; <=0.474 "
            "medium; >0.474 large (Romano et al. 2006)."
        ),
    }
    json.dump(out, open(RESULTS / "cliffs_delta.json", "w"), indent=2)
    print(f"wrote {RESULTS/'cliffs_delta.json'}")

    label_map = {
        "target_only": "Target-only",
        "pooled": "Pooled",
        "warm_start": "Warm-start",
        "tradaboost_r2": "TrAdaBoost.R2",
        "coral": "CORAL",
        "gated": "Gated",
    }
    labels = [label_map.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title(r"Cliff's $\delta$ on per-fold RMSE (row better than col)",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Cliff's $\delta$", fontsize=9)
    fig.tight_layout()
    out_png = SUB / "figures" / "cliffs_delta_heatmap.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i < j:
                print(f"  {mi:14s} vs {mj:14s}: delta={mat[i, j]:+.3f}")


if __name__ == "__main__":
    main()
