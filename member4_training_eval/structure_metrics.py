"""Structure-preservation metrics for stylized training images (Section 3.6).

For every generated image and its content image we compute:

  * LPIPS  -- perceptual distance (Zhang et al., CVPR 2018), lower = the
              stylized image resembles the content more closely.
  * Edge consistency -- how much of the content's edge structure survives.
              We use the F1-style overlap between Canny edge maps:

                  E(x)      = Canny edges of image x
                  Precision = |E(content) & E(stylized)| / |E(stylized)|
                  Recall    = |E(content) & E(stylized)| / |E(content)|
                  score    = 2 * P * R / (P + R)

              A high score means the stylized image keeps the content's
              contours; a low score means the stylization destroyed them.

The LPIPS model weights are downloaded on first use; if that is unavailable,
the script still reports the edge score and says so explicitly.

Usage:
    python structure_metrics.py --manifest <manifest> [--out metrics.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# Canny parameters; fixed for every image so scores are comparable.
CANNY_LOW = 100
CANNY_HIGH = 200


def to_tensor(path, size=224):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def edge_map(path, size=224):
    import cv2
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR))
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    return edges > 0


def edge_consistency(content_edges, styl_edges):
    """F1-style overlap of two binary edge maps."""
    inter = np.logical_and(content_edges, styl_edges).sum()
    p_den = styl_edges.sum()
    r_den = content_edges.sum()
    if p_den == 0 or r_den == 0:
        return float("nan")
    precision = inter / p_den
    recall = inter / r_den
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()

    from data import read_aug_manifest
    rows = read_aug_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]

    net = None
    if LPIPS_AVAILABLE:
        net = lpips_lib.LPIPS(net="alex").to(config.DEVICE)

    lpips_vals, edge_vals = [], []
    for i, (rel, label, meta) in enumerate(rows):
        styl_path = config.DATA_ROOT / rel
        content_rel = meta[0] if meta else None
        if content_rel is None:
            print("row without provenance, skipping:", rel)
            continue
        content_path = config.IMAGE_ROOT / content_rel

        if net is not None:
            with torch.no_grad():
                a = to_tensor(content_path).to(config.DEVICE) * 2 - 1
                b = to_tensor(styl_path).to(config.DEVICE) * 2 - 1
                lpips_vals.append(float(net(a, b).item()))

        edge_vals.append(edge_consistency(edge_map(content_path), edge_map(styl_path)))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}")

    def stats(values):
        values = [v for v in values if not np.isnan(v)]
        if not values:
            return None
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return {"mean": mean, "std": std, "n": len(values)}

    result = {
        "manifest": str(args.manifest),
        "lpips": stats(lpips_vals) if lpips_vals else None,
        "edge_consistency": stats(edge_vals),
        "lpips_available": LPIPS_AVAILABLE,
    }
    if result["lpips"] is None:
        result["note"] = ("lpips package not installed; install with "
                          "`pip install lpips` to complete the metric set")
    print(json.dumps(result, indent=2))
    out = args.out or args.manifest.with_suffix(".structure_metrics.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
