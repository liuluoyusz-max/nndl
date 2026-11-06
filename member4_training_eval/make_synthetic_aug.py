"""Create a small synthetic augmented dataset + manifests for pipeline testing.

Mimics what members 2/3 will deliver: stylized images under
DATA_ROOT/augmented/<group>/ plus a manifest whose rows carry provenance
fields (original path, style name, alpha, [beta]).  Images here are just
copies of the originals (identity "stylization"), suitable only for testing
the training/fairness pipeline -- never for real results.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

import config

N = 40  # originals to cover for the synthetic test


def main():
    pool = json.loads(config.STYLE_POOL_MANIFEST.read_text(encoding="utf-8"))
    pairing = pool["pairing"]

    train_rows = config.read_split(config.SPLIT_FILES["photo_train"])[:N]
    for group in ("adain", "ours"):
        out_dir = config.DATA_ROOT / "augmented" / group
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_lines = []
        for rel, label in train_rows:
            style = pairing[rel]
            stem = Path(rel).stem
            out_rel = f"augmented/{group}/{stem}__{style}.jpg"
            dst = config.DATA_ROOT / out_rel
            with Image.open(config.IMAGE_ROOT / rel) as im:
                im.convert("RGB").save(dst, quality=95)
            fields = [out_rel, str(label + 1), rel, style, "0.5"]
            if group == "ours":
                fields.append("0.7")
            manifest_lines.append(" ".join(fields))
        manifest = config.DATA_ROOT / "augmented" / f"manifest_{group}_synthetic.txt"
        manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        print(f"{group}: {len(manifest_lines)} rows -> {manifest}")


if __name__ == "__main__":
    main()
