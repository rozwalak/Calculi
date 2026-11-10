#!/usr/bin/env python3
"""Plot ancient metagenome samples by age for Figure 1B."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "01-resources" / "Velsko2026_ancient_metagenomes_metadata.tsv"
if not INPUT.exists():
    INPUT = INPUT.with_name(INPUT.name + ".csv")
OUTPUT_ROOT = ROOT / "06-publication" / "main_figures"
OUTPUT_STEM = OUTPUT_ROOT / "Fig1B"
ALIGNMENT_COLUMN = "calcBGCecoevo/04-analysis/metagenome_assembly/alignment/megahit"

HOSTS = ["Homo sapiens", "Homo sapiens neanderthalensis"]
MAX_Y_SPREAD = 0.12
POINT_BASELINE = 0.035
AGE_CUTOFF = 10_000
YOUNG_AGE_STEP = 100
YOUNG_MAJOR_STEP = 2_000
OLD_AGE_STEP = 10_000
YOUNG_AXIS_FRACTION = 0.42


def age_to_axis(age: float, oldest_age: float) -> float:
    """Piecewise-linear age axis, giving recent ages more horizontal room."""
    if age <= AGE_CUTOFF:
        return YOUNG_AXIS_FRACTION * age / AGE_CUTOFF
    old_range = max(oldest_age - AGE_CUTOFF, 1)
    return YOUNG_AXIS_FRACTION + (1 - YOUNG_AXIS_FRACTION) * (age - AGE_CUTOFF) / old_range
ORANGE = "#FB7F23"
NEANDERTHAL_COLOR = "#887D49"
BLACK = "#171717"
WIDTH_MM = 175.533
HEIGHT_MM = 42.559
POINT_DIAMETER_MM = 1.480
POINT_EDGE_MM = 0.05


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t", encoding="latin-1")
    if ALIGNMENT_COLUMN in data.columns:
        data = data.loc[
            data[ALIGNMENT_COLUMN].astype("string").str.strip().str.lower().eq("yes")
        ].copy()
    data = data.loc[data["sample_host"].isin(HOSTS)].copy()

    # One mark represents each sample_name; repeated library rows are collapsed.
    samples = data.drop_duplicates("sample_name", keep="first").copy()
    samples = samples.sort_values(["sample_age", "sample_host", "sample_name"])

    # Matplotlib's scatter `s` is marker area in pt². This gives the requested
    # 1.480 mm marker width for both the circle and square markers.
    diameter_pt = POINT_DIAMETER_MM * 72 / 25.4
    marker_area_pt2 = diameter_pt**2
    edge_width_pt = POINT_EDGE_MM * 72 / 25.4

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5})
    fig, ax = plt.subplots(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        constrained_layout=True,
    )

    oldest_age = int(samples["sample_age"].max())
    oldest_axis = age_to_axis(oldest_age * 1.12, oldest_age)
    youngest_display_age = -1_000
    youngest_axis = age_to_axis(youngest_display_age, oldest_age)
    ax.set_xlim(oldest_axis, youngest_axis)
    major_ages = list(range(0, AGE_CUTOFF + 1, YOUNG_MAJOR_STEP))
    major_ages.extend(
        range(AGE_CUTOFF + OLD_AGE_STEP, oldest_age + 1, 2 * OLD_AGE_STEP)
    )
    minor_ages = list(range(YOUNG_AGE_STEP, AGE_CUTOFF, YOUNG_AGE_STEP))
    minor_ages.extend(
        age
        for age in range(AGE_CUTOFF + OLD_AGE_STEP, oldest_age + 1, OLD_AGE_STEP)
        if age not in major_ages
    )
    ax.set_xticks([age_to_axis(age, oldest_age) for age in major_ages])
    ax.set_xticklabels([f"{age:,}" for age in major_ages])
    ax.set_xticks([age_to_axis(age, oldest_age) for age in minor_ages], minor=True)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel(
        "Years before present (BP)",
        labelpad=4,
    )
    ax.tick_params(axis="x", which="major", labelsize=6.5, length=2.5, width=0.65, pad=2)
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.spines["bottom"].set_color(BLACK)

    # Mark the change in age-scale spacing, as in the supplied reference figure.
    ax.text(
        age_to_axis(AGE_CUTOFF, oldest_age),
        0,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=7,
        color=BLACK,
        clip_on=False,
        zorder=5,
    )

    samples["plot_age"] = samples["sample_age"].map(
        lambda age: age_to_axis(age, oldest_age)
    )

    # Show the overall age distribution as a grey density ridge along the x axis.
    # KDE is calculated in the displayed coordinates so its peaks align with the
    # piecewise-linear age scale used for the sample marks.
    density_x = np.linspace(
        age_to_axis(youngest_display_age, oldest_age),
        age_to_axis(oldest_age, oldest_age),
        800,
    )
    density_kde = gaussian_kde(samples["plot_age"].to_numpy(), bw_method=0.12)
    density = density_kde(density_x)
    normalized_density = density / density.max()
    density_y = 0.62 * normalized_density**0.75
    density_base = 0.01
    ax.fill_between(
        density_x,
        density_base,
        density_base + density_y,
        color="#AFAFAF",
        edgecolor="#858585",
        linewidth=0.35,
        alpha=0.78,
        zorder=0,
    )

    # Anchor samples just above the x axis and spread dense age regions upward.
    # Sparse and old samples stay near the baseline. A fixed seed keeps
    # singleton-age offsets reproducible when regenerating the figure.
    sample_density = density_kde(samples["plot_age"].to_numpy()) / density.max()
    samples["local_density"] = np.clip(sample_density, 0, 1)
    samples["plot_y"] = 0.5
    rng = np.random.default_rng(2026)
    for _, age_group in samples.groupby("sample_age", sort=False):
        half_spread = MAX_Y_SPREAD * np.sqrt(age_group["local_density"].iloc[0])
        if len(age_group) == 1:
            positions = [POINT_BASELINE + rng.uniform(0, half_spread)]
        else:
            positions = np.linspace(
                POINT_BASELINE, POINT_BASELINE + half_spread, len(age_group)
            )
        samples.loc[age_group.index, "plot_y"] = positions

    styles = {
        "Homo sapiens": (ORANGE, 2),
        "Homo sapiens neanderthalensis": (NEANDERTHAL_COLOR, 1),
    }
    sample_counts = samples.groupby("sample_host")["sample_name"].nunique().to_dict()
    # Draw squares first so circular marks remain legible where groups overlap.
    for host in ["Homo sapiens neanderthalensis", "Homo sapiens"]:
        color, zorder = styles[host]
        subset = samples.loc[samples["sample_host"] == host]
        ax.scatter(
            subset["plot_age"],
            subset["plot_y"],
            marker="o",
            s=marker_area_pt2,
            facecolor=color,
            edgecolor=BLACK,
            linewidth=edge_width_pt,
            zorder=zorder,
            rasterized=False,
        )

    host_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=diameter_pt,
               markerfacecolor=ORANGE, markeredgecolor=BLACK, markeredgewidth=edge_width_pt,
               label=(
                   "Anatomically modern humans "
                   f"(n = {sample_counts.get('Homo sapiens', 0)})"
               )),
        Line2D([], [], marker="o", linestyle="none", markersize=diameter_pt,
               markerfacecolor=NEANDERTHAL_COLOR, markeredgecolor=BLACK, markeredgewidth=edge_width_pt,
               label=(
                   "Neanderthals "
                   f"(n = {sample_counts.get('Homo sapiens neanderthalensis', 0)})"
               )),
    ]
    ax.legend(
        handles=host_handles,
        loc="upper left",
        frameon=False,
        fontsize=7,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.1,
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
