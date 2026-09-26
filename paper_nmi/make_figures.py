"""Generate manuscript figures from frozen experiment artefacts.

This script performs no model evaluation or training. It only reads existing
CSV/JSON results and writes publication-oriented PDF/PNG figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3569A8"
ORANGE = "#D9772A"
GREEN = "#3A8D5D"
RED = "#B64B4B"
GREY = "#555555"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, fc, fontsize=7.5):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1,
        edgecolor="#333333",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=1, color="#333333"))


def figure_1() -> None:
    fig = plt.figure(figsize=(7.1, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.1], hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax in (ax_a, ax_b):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax_a.set_title("a  Serial organization", loc="left", fontweight="bold")
    block_xs = [0.21, 0.42]
    block_width = 0.11
    rounded_box(ax_a, (0.01, 0.39), 0.12, 0.18, "$x_1$", "#F1F1F1")
    for i, x in enumerate(block_xs, start=1):
        rounded_box(ax_a, (x, 0.39), block_width, 0.18, f"$B_{{{i}}}$", "#DCE8F5")
    ax_a.text(0.59, 0.48, r"$\cdots$", ha="center", va="center", fontsize=12)
    rounded_box(ax_a, (0.66, 0.39), block_width, 0.18, "$B_{24}$", "#DCE8F5")
    rounded_box(ax_a, (0.87, 0.39), 0.12, 0.18, "$x_{25}$", "#F1F1F1")
    starts = [0.13, 0.32, 0.625, 0.77]
    ends = [0.21, 0.42, 0.66, 0.87]
    for st, en in zip(starts, ends):
        arrow(ax_a, (st, 0.48), (en, 0.48))
    ax_a.text(0.5, 0.18, r"$N=24,\ S=24,\ K=1$", ha="center")
    ax_a.text(0.5, 0.08, "24 sequential block transformations", ha="center", color=GREY, fontsize=8)

    ax_b.set_title("b  Stage-parallel organization", loc="left", fontweight="bold")
    rounded_box(ax_b, (0.02, 0.40), 0.12, 0.18, "$x_s$", "#F1F1F1")
    branch_y = [0.70, 0.40, 0.10]
    for j, y in enumerate(branch_y, start=1):
        rounded_box(ax_b, (0.34, y), 0.25, 0.17, f"$B_{{s,{j}}}(x_s)$", "#E5F1E8")
        arrow(ax_b, (0.14, 0.49), (0.34, y + 0.085))
        arrow(ax_b, (0.59, y + 0.085), (0.76, 0.49))
    rounded_box(ax_b, (0.76, 0.40), 0.20, 0.18, "merge\nupdates", "#F7E8D6")
    ax_b.text(0.5, 0.02, r"$N=24,\ S=8,\ K=3$", ha="center")

    ax_c.set_title("c  Parameter-matched width control (one seed)", loc="left", fontweight="bold")
    labels = ["24 serial\nblocks", "8 stages × 3\nparallel blocks", "8 serial blocks\nwith wider FFNs"]
    values = [78.9274, 76.5182, 86.0322]
    bars = ax_c.bar(np.arange(3), values, color=[BLUE, GREEN, ORANGE], width=0.62)
    ax_c.set_ylabel("Validation subword perplexity")
    ax_c.set_xticks(np.arange(3), labels)
    ax_c.set_ylim(72, 89)
    ax_c.grid(axis="y", alpha=0.22)
    for bar, val in zip(bars, values):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}", ha="center", va="bottom")
    ax_c.text(0.5, 0.96, "61.92M trainable parameters; 50M tokens", transform=ax_c.transAxes,
              ha="center", va="top", color=GREY, fontsize=8)
    save(fig, "fig1_topology_and_width_control")


def figure_2() -> None:
    runs = pd.read_csv(ROOT / "results_stage7" / "stage7b_multiseed_runs.csv")
    order = [24, 12, 8, 6, 4, 3, 2, 1]
    labels = ["24\n(1×)", "12\n(2×)", "8\n(3×)", "6\n(4×)", "4\n(6×)", "3\n(8×)", "2\n(12×)", "1\n(24×)"]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharex=True)
    metrics = [("val_ppl_50M", "Validation", BLUE), ("test_ppl", "Official test", GREEN)]
    markers = {42: "o", 123: "s", 2026: "^"}

    for ax, (col, title, color) in zip(axes, metrics):
        means, sds = [], []
        for depth in order:
            vals = runs.loc[runs["S"] == depth, col].to_numpy(float)
            means.append(vals.mean())
            sds.append(vals.std(ddof=1))
        ax.errorbar(x, means, yerr=sds, color=color, marker="o", lw=1.8, capsize=3, label="mean ± s.d.")
        for seed, marker in markers.items():
            vals = [runs.loc[(runs["S"] == depth) & (runs["seed"] == seed), col].iloc[0] for depth in order]
            ax.scatter(x, vals, marker=marker, s=20, facecolor="white", edgecolor=color, linewidth=0.9, alpha=0.9,
                       label=f"seed {seed}" if ax is axes[0] else None)
        ax.set_title(f"{'a' if ax is axes[0] else 'b'}  {title}", loc="left", fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Sequential stages, $S$ (compression)")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Subword perplexity")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_static_frontier")


def figure_3() -> None:
    tiny = json.loads((ROOT / "results_stage5" / "stage5a_summary.json").read_text())
    summary = pd.read_csv(ROOT / "results_stage7" / "stage7b_multiseed_summary.csv")
    wiki = summary.loc[summary["S"] == 12].iloc[0]
    rep = json.loads((ROOT / "results_stage7" / "stage7_branch_representation_summary.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    ax = axes[0]
    means = [tiny["mean_relative_penalty_pct"], float(wiki["val_penalty_mean"])]
    sds = [tiny["sd_relative_penalty_pct"], float(wiki["val_penalty_sd"])]
    bars = ax.bar([0, 1], means, yerr=sds, capsize=4, color=[BLUE, GREEN], width=0.58)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_xticks([0, 1], ["TinyStories", "WikiText-103"])
    ax.set_ylabel("Paired perplexity difference (%)")
    ax.set_title("a  Twofold sequential-depth reduction", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(-3.2, 1.2)
    for bar, val in zip(bars, means):
        offset = 0.12 if val >= 0 else -0.18
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.2f}%", ha="center", va=va)

    ax = axes[1]
    configs = [(4, "SP24_S6K4"), (6, "SP24_S4K6"), (8, "SP24_S3K8"), (12, "SP24_S2K12"), (24, "SP24_S1K24")]
    xs, ys, pair_counts = [], [], []
    for k, key in configs:
        stages = rep[key]["stages"]
        n_pairs = sum(int(s["num_pairs"]) for s in stages)
        weighted = sum(float(s["mean_cosine"]) * int(s["num_pairs"]) for s in stages) / n_pairs
        xs.append(k)
        ys.append(weighted)
        pair_counts.append(n_pairs)
    ax.plot(xs, ys, "-o", color=ORANGE, lw=1.8)
    label_offsets = {4: 0.026, 6: 0.026, 8: -0.055, 12: 0.026, 24: 0.026}
    for xval, yval, n in zip(xs, ys, pair_counts):
        dy = label_offsets[xval]
        ax.text(xval, yval + dy, f"{yval:.2f}\n({n} pairs)", ha="center",
                va="bottom" if dy > 0 else "top", fontsize=7)
    ax.set_xlabel("Parallel branches per stage, $K$")
    ax.set_ylabel("Mean pairwise update cosine")
    ax.set_title("b  Branch-update differentiation", loc="left", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_ylim(0.10, 0.62)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    save(fig, "fig3_cross_corpus_and_branches")


def load_milestones(model: str, seed: int) -> dict[int, float]:
    path = ROOT / "results_stage8" / f"wt103_{model}_seed{seed}_100M_milestones.json"
    raw = json.loads(path.read_text())
    return {int(item["tokens_M"]): float(item["val_ppl"]) for item in raw["milestones"]}


def figure_4() -> None:
    seeds = [42, 123, 2026]
    horizons = [5, 10, 20, 30, 40, 50, 60, 75, 90, 100]
    models = [("par24_3", "$S=8, K=3$", GREEN), ("sp24_s4k6", "$S=4, K=6$", ORANGE)]
    data = {"seq24": {seed: load_milestones("seq24", seed) for seed in seeds}}
    for model, _, _ in models:
        data[model] = {seed: load_milestones(model, seed) for seed in seeds}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    endpoint_penalties = {}
    for model, label, color in models:
        arr = np.array(
            [[100 * (data[model][seed][h] - data["seq24"][seed][h]) / data["seq24"][seed][h] for h in horizons]
             for seed in seeds]
        )
        for row in arr:
            ax.plot(horizons, row, color=color, lw=0.8, alpha=0.28)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        ax.plot(horizons, mean, "-o", color=color, lw=1.8, markersize=3.8, label=label)
        ax.fill_between(horizons, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        endpoint_penalties[model] = arr
    ax.axhline(0, color="#222222", lw=1, ls="--")
    ax.set_xlabel("Processed training tokens (millions)")
    ax.set_ylabel("Paired difference from serial $S=24$ (%)")
    ax.set_title("a  Relative quality over one 100M schedule", loc="left", fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1]
    x50, x100 = 0, 1
    offsets = {"par24_3": -0.05, "sp24_s4k6": 0.05}
    for model, label, color in models:
        arr = endpoint_penalties[model]
        idx50, idx100 = horizons.index(50), horizons.index(100)
        for row in arr:
            ax.plot([x50 + offsets[model], x100 + offsets[model]], [row[idx50], row[idx100]], color=color, lw=1.1, alpha=0.65)
            ax.scatter([x50 + offsets[model], x100 + offsets[model]], [row[idx50], row[idx100]], color=color, s=17)
        means = [arr[:, idx50].mean(), arr[:, idx100].mean()]
        ax.plot([x50 + offsets[model], x100 + offsets[model]], means, color=color, lw=3, label=label)
    ax.axhline(0, color="#222222", lw=1, ls="--")
    ax.set_xticks([0, 1], ["50M", "100M"])
    ax.set_ylabel("Paired difference (%)")
    ax.set_title("b  Paired endpoint shift", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig4_dynamic_frontier")


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    print(f"Wrote manuscript figures to {OUT}")
