"""Demonstrate the full pipeline with labeled SYNTHETIC fixtures, never PACS.

All fixtures live under data/member3_smoke. They must never be used as
classification results or as the real training augmentation dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from member4_training_eval import config
from member3_structure_preserving.generate import parser, run


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--overwrite", action="store_true", help="replace an older smoke run after code changes")
    options = cli.parse_args()
    root = config.PROJECT_ROOT / "data" / "member3_smoke"
    marker = root / "SYNTHETIC_DATA_ONLY.txt"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise SystemExit(f"refusing to mix synthetic fixtures with existing data: {root}")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text("SYNTHETIC geometric fixtures. Not PACS. Not experimental evidence.\n", encoding="utf-8")
    rows = config.read_split(config.SPLIT_FILES["photo_train"])
    for index, (relative, _) in enumerate(rows[:2]):
        path = root / "PACS" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        picture = Image.new("RGB", (160, 128), (36, 52, 80))
        draw = ImageDraw.Draw(picture)
        draw.rectangle((25, 30, 130, 105), fill=(205, 155, 65))
        draw.ellipse((55 + 8 * index, 40, 115, 95), fill=(65, 170, 145), outline="white", width=3)
        draw.line((8, 112, 148, 26), fill="white", width=3)
        draw.text((8, 8), "SYNTHETIC FIXTURE", fill="white")
        picture.save(path, quality=95)
    arguments = [
        "build", "--data-root", str(root), "--limit", "2", "--content-size", "128",
        "--style-size", "128", "--threads", "2", "--with-adain", "--diagnostics", "--metrics"]
    if options.overwrite:
        arguments.append("--overwrite")
    args = parser().parse_args(arguments)
    summaries = run(args)
    print("SYNTHETIC pipeline check only; no real PACS or classifier results.")
    print(f"panels: {Path(summaries[0]['manifest']).parent / 'figures'}")


if __name__ == "__main__":
    main()
