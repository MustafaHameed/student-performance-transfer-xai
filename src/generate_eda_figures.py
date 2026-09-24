"""Generate EDA figures for the VTSE submission.

Produces three figures, saved as PNGs in $PAPER_DIR/figures (default
paper_outputs/figures):
  eda_target_distribution.png   - target / class balance across 3 datasets
  eda_concept_coverage.png      - distribution of each of the 8 shared concepts
                                  across the 3 datasets
  eda_concept_separability.png  - Cohen's d of each concept against the at-risk
                                  label, per dataset

Run from anywhere; paths are resolved from this file's location.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UCI_DIR = ROOT / "data"
# xAPI-Edu cannot be redistributed: $XAPI_CSV, else data/xAPI-Edu-Data.csv.
XAPI_PATH = Path(os.environ.get("XAPI_CSV", UCI_DIR / "xAPI-Edu-Data.csv"))
OUT_DIR = Path(os.environ.get("PAPER_DIR", ROOT / "paper_outputs")) / "figures"

PALETTE = {
    "Portuguese": "#2E86AB",
    "Mathematics": "#A23B72",
    "xAPI-Edu": "#F18F01",
}

CONCEPTS = [
    "is_male",
    "high_absence",
    "family_eng",
    "engagement",
    "higher_intent",
    "failure_risk",
    "par_education",
    "distraction",
]


def load_uci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    out = pd.DataFrame()
    out["is_male"] = (df["sex"] == "M").astype(float)
    out["high_absence"] = (df["absences"] > 7).astype(float)
    out["family_eng"] = (
        (df["famsup"] == "yes").astype(float) + (df["schoolsup"] == "yes").astype(float)
    ) / 2.0
    out["engagement"] = (df["studytime"] - 1.0) / 3.0  # studytime in [1,4]
    out["higher_intent"] = (df["higher"] == "yes").astype(float)
    out["failure_risk"] = df["failures"].clip(0, 4) / 4.0
    out["par_education"] = ((df["Medu"] + df["Fedu"]) / 2.0) / 4.0
    out["distraction"] = (df["romantic"] == "yes").astype(float)
    out["G3"] = df["G3"].astype(float)
    out["at_risk"] = (out["G3"] < 10).astype(int)
    return out


def load_xapi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["is_male"] = (df["gender"] == "M").astype(float)
    out["high_absence"] = (df["StudentAbsenceDays"] == "Above-7").astype(float)
    out["family_eng"] = (
        (df["ParentAnsweringSurvey"] == "Yes").astype(float)
        + (df["ParentschoolSatisfaction"] == "Good").astype(float)
    ) / 2.0
    engagement_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    out["engagement"] = df[engagement_cols].mean(axis=1) / 100.0
    out["higher_intent"] = (df["StageID"] == "HighSchool").astype(float)
    out["failure_risk"] = 1.0 - (df["ParentschoolSatisfaction"] == "Good").astype(float)
    out["par_education"] = (df["ParentAnsweringSurvey"] == "Yes").astype(float)
    out["distraction"] = 1.0 - df["Discussion"] / 100.0
    out["Class"] = df["Class"].astype(str)
    out["at_risk"] = (df["Class"] == "L").astype(int)
    return out


def cohens_d(values: np.ndarray, labels: np.ndarray) -> float:
    a = values[labels == 1]
    b = values[labels == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def fig_target_distribution(por: pd.DataFrame, mat: pd.DataFrame, xapi: pd.DataFrame, raw_xapi: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, df, name in (
        (axes[0], por, "Portuguese"),
        (axes[1], mat, "Mathematics"),
    ):
        ax.hist(df["G3"], bins=np.arange(0, 21, 1), color=PALETTE[name], edgecolor="black", alpha=0.85)
        ax.axvline(10, color="red", linestyle="--", linewidth=1.4, label="at-risk threshold ($G_3<10$)")
        risk_pct = 100.0 * df["at_risk"].mean()
        ax.set_xlabel(r"$G_3$ (final grade, 0-20)")
        ax.set_ylabel("Students")
        ax.set_title(f"{name} (n={len(df)}, at-risk {risk_pct:.1f}%)", fontsize=10)
        ax.set_xlim(-0.5, 20.5)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax = axes[2]
    order = ["L", "M", "H"]
    counts = raw_xapi["Class"].value_counts().reindex(order, fill_value=0)
    bar_colors = ["#C0392B", "#F1C40F", "#27AE60"]
    bars = ax.bar(order, counts.values, color=bar_colors, edgecolor="black", alpha=0.9)
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c + 4, str(int(c)), ha="center", fontsize=9)
    risk_pct = 100.0 * (raw_xapi["Class"] == "L").mean()
    ax.set_xlabel("Performance class")
    ax.set_ylabel("Students")
    ax.set_title(f"xAPI-Edu (n={len(raw_xapi)}, at-risk {risk_pct:.1f}%)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(counts.values) * 1.18)

    fig.suptitle("Target distribution and at-risk class balance across the three datasets", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "eda_target_distribution.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_concept_coverage(por: pd.DataFrame, mat: pd.DataFrame, xapi: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0), sharey=False)
    axes = axes.flatten()

    labels = ["Por", "Math", "xAPI"]
    colors = [PALETTE["Portuguese"], PALETTE["Mathematics"], PALETTE["xAPI-Edu"]]

    for i, concept in enumerate(CONCEPTS):
        ax = axes[i]
        data = [por[concept].values, mat[concept].values, xapi[concept].values]
        bp = ax.boxplot(
            data,
            positions=[1, 2, 3],
            widths=0.6,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(concept.replace("_", "\\_"), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Distribution of each shared concept across the three datasets (normalised to $[0,1]$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "eda_concept_coverage.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_concept_separability(por: pd.DataFrame, mat: pd.DataFrame, xapi: pd.DataFrame) -> None:
    rows = []
    for name, df in (("Portuguese", por), ("Mathematics", mat), ("xAPI-Edu", xapi)):
        for concept in CONCEPTS:
            d = cohens_d(df[concept].values, df["at_risk"].values)
            rows.append({"dataset": name, "concept": concept, "d": d})
    eff = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    width = 0.27
    x = np.arange(len(CONCEPTS))
    for i, name in enumerate(["Portuguese", "Mathematics", "xAPI-Edu"]):
        vals = [
            eff[(eff["dataset"] == name) & (eff["concept"] == c)]["d"].iloc[0]
            for c in CONCEPTS
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=PALETTE[name], edgecolor="black", alpha=0.88)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(-0.2, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\\_") for c in CONCEPTS], fontsize=9, rotation=15)
    ax.set_ylabel("Cohen's $d$ (at-risk $-$ on-track)")
    ax.set_title("Per-concept separability between at-risk and on-track students", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = OUT_DIR / "eda_concept_separability.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    por = load_uci(UCI_DIR / "student-por.csv")
    mat = load_uci(UCI_DIR / "student-mat.csv")
    xapi_raw = pd.read_csv(XAPI_PATH)
    xapi = load_xapi(XAPI_PATH)

    print(f"Portuguese: n={len(por)}, at-risk={por['at_risk'].mean():.3f}")
    print(f"Mathematics: n={len(mat)}, at-risk={mat['at_risk'].mean():.3f}")
    print(f"xAPI-Edu:    n={len(xapi)}, at-risk={xapi['at_risk'].mean():.3f}")

    fig_target_distribution(por, mat, xapi, xapi_raw)
    fig_concept_coverage(por, mat, xapi)
    fig_concept_separability(por, mat, xapi)


if __name__ == "__main__":
    main()
