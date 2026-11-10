#!/usr/bin/env python3
"""Create the stacked study/source genome-count panel for Figure 1C."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "05-results" / "vcontact2_metadata.csv"
OUTPUT_DIR = ROOT / "06-publication" / "main_figures"

WIDTH_MM = 83
HEIGHT_MM = 52


def main() -> None:
    data = pd.read_csv(INPUT, usecols=["Study", "Source", "Color"])
    data = data.dropna(subset=["Study", "Source", "Color"])

    # Count records by source, then combine sources that share the same color.
    # This keeps the plotted colors faithful to the metadata palette.
    source_counts = (
        data.groupby(["Study", "Source", "Color"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts = (
        source_counts.groupby(["Study", "Color"], sort=False)["count"]
        .sum()
        .reset_index()
    )
    studies = list(data["Study"].drop_duplicates())
    totals = data.groupby("Study", sort=False).size().to_dict()

    plt.rcParams.update({"font.size": 5, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    fig.subplots_adjust(left=0.28, right=0.83, top=0.90, bottom=0.22)

    bar_height = 0.58
    for row, study in enumerate(studies):
        segments = counts.loc[counts["Study"] == study].sort_values(
            "count", ascending=False
        )
        left = 0
        for segment in segments.itertuples(index=False):
            ax.barh(
                row,
                segment.count,
                left=left,
                height=bar_height,
                color=segment.Color,
                edgecolor="white",
                linewidth=0.2,
            )
            left += segment.count
        ax.text(
            totals[study] + 55,
            row,
            f"n = {totals[study]:,}",
            ha="left",
            va="center",
            fontsize=4.8,
            color="#171717",
        )

    ax.set_yticks(np.arange(len(studies)))
    ax.set_yticklabels(studies, fontsize=5)
    ax.invert_yaxis()
    max_total = max(totals.values())
    last_tick = int(np.ceil(max_total / 1_000) * 1_000)
    ax.set_xticks(np.arange(0, last_tick + 1, 1_000))
    ax.set_xlim(0, last_tick + 1_000)
    ax.set_xlabel(
        "Nr of complete/high-quality bacteriophage genomes",
        fontsize=5.2,
        labelpad=3,
    )
    ax.tick_params(axis="x", labelsize=4.8, length=2, width=0.45, pad=2)
    ax.tick_params(axis="y", length=2, width=0.45, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.45)
    ax.spines["left"].set_color("#171717")
    ax.spines["bottom"].set_linewidth(0.45)
    ax.spines["bottom"].set_color("#171717")

    fig.text(0.015, 0.975, "C", ha="left", va="top", fontsize=8)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "Fig1C.png", dpi=600, facecolor="white")
    fig.savefig(OUTPUT_DIR / "Fig1C.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
