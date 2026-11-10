#!/usr/bin/env python3
"""Plot CRISPR spacer matches for host GCF_002952055.1 (Figure 2B)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "05-results"
    / "Calcyviruses_iphop_Jun_2025_pub_rw_calculus"
    / "Wdir"
    / "crisprparsed.csv"
)
OUTPUT_DIR = ROOT / "06-publication" / "main_figures"
OUTPUT_STEM = OUTPUT_DIR / "Fig2B"

HOST = "GCF_002952055.1"
WIDTH_MM = 57
HEIGHT_MM = 57
VIRUS_LABELS = {
    "ERS3774444_NODE_3_length_74272_cov_84.8": "VLD001.A",
    "ERS15409402_NODE_18_length_70845_cov_128.4": "B91",
    "ERS15409398_NODE_13_length_70913_cov_527.1": "B68",
    "SRS11653574_NODE_1_length_73277_cov_192.2": "PA.39",
    "SRS15300801_NODE_39_length_72738_cov_1583.9": "WAMP27A",
    "ERS11702779_NODE_6_length_75301_cov_34.8": "MID062",
    "ERS3395766_NODE_44_length_76109_cov_26.283207": "JAE008",
    "ERS26854148_NODE_1_length_74786_cov_128.6": "HPD006.B",
    "ERS11702752_NODE_1_length_73747_cov_71.8": "MID035",
}


def main() -> None:
    data = pd.read_csv(INPUT)
    required = {"Virus", "Host", "Spacer", "Pcent Id", "N mismatch"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    host_hits = data.loc[
        data["Host"].eq(HOST), ["Virus", "Spacer", "Pcent Id", "N mismatch"]
    ].copy()
    if host_hits.empty:
        raise ValueError(f"No rows found for host {HOST}")
    host_hits["Pcent Id"] = pd.to_numeric(host_hits["Pcent Id"], errors="coerce")
    host_hits["N mismatch"] = pd.to_numeric(host_hits["N mismatch"], errors="coerce")
    host_hits = host_hits.dropna(subset=["Virus", "Spacer", "Pcent Id", "N mismatch"])
    host_hits = host_hits.loc[host_hits["Virus"].isin(VIRUS_LABELS)]
    if host_hits.empty:
        raise ValueError(f"No valid spacer identities found for host {HOST}")

    max_mismatches = int(host_hits["N mismatch"].max())
    norm = Normalize(vmin=0, vmax=max_mismatches, clip=True)
    cmap = plt.get_cmap("OrRd_r")
    plt.rcParams.update({"font.size": 6, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    fig.subplots_adjust(left=0.34, right=0.98, top=0.90, bottom=0.24)

    viruses = [virus for virus in VIRUS_LABELS if host_hits["Virus"].eq(virus).any()]
    gap = 0.08
    for x, virus in enumerate(viruses):
        # Rank each virus's spacers from fewest to most mismatches.
        hits = host_hits.loc[host_hits["Virus"].eq(virus)].sort_values(
            "N mismatch", ascending=True, kind="stable"
        )
        for y, (_, hit) in enumerate(hits.iterrows()):
            ax.add_patch(
                Rectangle(
                    (y + gap / 2, x - 0.4),
                    1 - gap,
                    0.8,
                    facecolor=cmap(norm(hit["N mismatch"])),
                    edgecolor="none",
                )
            )

    counts = host_hits.groupby("Virus").size().reindex(viruses)
    max_count = int(counts.max())
    y = np.arange(len(viruses))
    ax.set_xlim(0, max_count + 0.25)
    ax.set_ylim(len(viruses) - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([VIRUS_LABELS[v] for v in viruses], fontsize=5)
    ax.set_xticks(np.arange(0, max_count + 1, 2))
    ax.tick_params(axis="x", labelsize=5, length=2, width=0.4, pad=1)
    ax.set_xlabel("Matches in CRISPR spacers in D.oralis (GCF_002952055.1)", fontsize=5, labelpad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.set_title(r"CRISPR spacers in $\mathit{D.\ oralis}$", fontsize=6, pad=4)

    colorbar_ax = fig.add_axes([0.38, 0.08, 0.48, 0.025])
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=[0, max_mismatches],
    )
    colorbar.set_label("Mismatches", fontsize=5, labelpad=2)
    colorbar.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    colorbar.outline.set_linewidth(0.35)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
