"""Plot the unified results bar chart for the report.

Usage:
    python plot_results.py --out ../figures/results_baseline.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")
LABELS = {"original": "Original", "adain": "+AdaIN", "ours": "+Ours"}
COLORS = {"original": "#5B7C99", "adain": "#C98B5B", "ours": "#5B9C85"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=config.PROJECT_ROOT / "figures" / "results_baseline.png")
    args = parser.parse_args()

    summaries = {}
    for path in sorted(args.runs_root.glob("summary_*.json")):
        s = json.loads(path.read_text())
        summaries[s.get("group", path.stem)] = s

    if not summaries:
        print("no summaries found")
        return

    splits = ["photo_test"] + list(CROSS)
    names = [LABELS.get(g, g) for g in summaries]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.8 / len(summaries)
    x = list(range(len(splits)))
    for i, (group, summary) in enumerate(summaries.items()):
        offset = (i - (len(summaries) - 1) / 2) * width
        vals = [summary[split]["mean"] * 100 if split in summary else 0 for split in splits]
        errs = [summary[split]["std"] * 100 if split in summary else 0 for split in splits]
        ax.bar([xi + offset for xi in x], vals, width,
               yerr=errs, capsize=3, label=LABELS.get(group, group),
               color=COLORS.get(group, "#888888"), edgecolor="white", linewidth=0.6)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi + offset, v + e + 1.5, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=7.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Photo", "Art", "Cartoon", "Sketch"])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Evaluation domain")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=len(summaries), loc="upper right", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
