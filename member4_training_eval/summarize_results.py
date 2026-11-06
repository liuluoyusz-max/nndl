"""Collect run summaries into the unified results table (Table format of 3.7).

Usage:
    python summarize_results.py            # reads runs/summary_*.json
    python summarize_results.py --format markdown
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config

CROSS = ("art_painting_test", "cartoon_test", "sketch_test")
HEADER = ("Training method", "Photo", "Art", "Cartoon", "Sketch", "Avg")
PRETTY = {"original": "Original", "adain": "+AdaIN", "ours": "+Ours"}


def fmt(entry):
    if entry is None:
        return "--"
    return f"{entry['mean']*100:.1f}\\,$\\pm$\\,{entry['std']*100:.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--format", choices=["plain", "markdown", "latex"], default="plain")
    args = parser.parse_args()

    rows = {}
    for path in sorted(args.runs_root.glob("summary_*.json")):
        s = json.loads(path.read_text())
        key = PRETTY.get(s.get("group"), s.get("group", path.stem))
        rows[key] = s

    if not rows:
        print("no summary files found; run run_all.py first")
        return

    if args.format == "latex":
        print(r"\begin{tabular}{@{}lcccc@{}}")
        print(r"\toprule")
        print(" & ".join(HEADER) + r" \\")
        print(r"\midrule")
        for key, s in rows.items():
            cells = [key]
            for split in ("photo_test",) + CROSS:
                cells.append(fmt(s.get(split)))
            avg = s.get("cross_domain_avg")
            cells.append(f"{avg['mean']*100:.1f}" if avg else "--")
            print(" & ".join(cells) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    elif args.format == "markdown":
        print("| " + " | ".join(HEADER) + " |")
        print("|" + "---|" * len(HEADER))
        for key, s in rows.items():
            cells = [key]
            for split in ("photo_test",) + CROSS:
                e = s.get(split)
                cells.append(f"{e['mean']*100:.1f}±{e['std']*100:.1f}" if e else "--")
            avg = s.get("cross_domain_avg")
            cells.append(f"{avg['mean']*100:.1f}" if avg else "--")
            print("| " + " | ".join(cells) + " |")
    else:
        for key, s in rows.items():
            print(key)
            for split in ("photo_test",) + CROSS:
                e = s.get(split)
                print(f"  {split:20s} {e['mean']*100:6.2f} ± {e['std']*100:.2f}" if e
                      else f"  {split:20s} {'--':>9s}")
            avg = s.get("cross_domain_avg")
            if avg:
                print(f"  {'cross avg':20s} {avg['mean']*100:6.2f}")


if __name__ == "__main__":
    main()
