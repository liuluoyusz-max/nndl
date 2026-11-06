"""Convert a member-2 / member-3 generator output into member 4's manifest.

Members 2 and 3 write their augmented images with their own layout and a
provenance table.  This script turns that into the manifest format the
training code reads:

    <image path under DATA_ROOT> <label 1..7> <original> <style> <alpha> [beta]

and copies the images into ``DATA_ROOT/augmented/<group>/`` with the
``<stem>__<style>.jpg`` naming convention.

Accepted provenance formats (auto-detected):

  * ``pairs.txt``  -- tab separated: ``<out_rel>\\t<original>\\t<style>\\t<label>``
                      (member 2's ``build_adain_aug.py``)
  * a 4-column text file -- space separated:
                      ``<out_rel> <original> <style> <label>``

The style column may be a bare file name (``scene_de_rue.jpg``) or a style
name (``scene_de_rue``); both are normalised to the style name used in
``style_pool.json``.

Usage:
    python convert_aug_manifest.py --group adain \
        --pairs <generator output>/pairs.txt --alpha 0.5
    python convert_aug_manifest.py --group ours \
        --pairs <generator output>/pairs.txt --alpha 0.5 --beta 0.7
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import config

STYLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_pool():
    """Return (style_name -> file stem) and the set of valid style names."""
    pool = json.loads(config.STYLE_POOL_MANIFEST.read_text(encoding="utf-8"))
    names = set(pool["pool"]["styles"])
    return names, pool.get("pairing", {})


def normalise_style(raw, valid_names):
    """Map a style reference to a name in the frozen pool."""
    stem = Path(raw).stem
    if stem in valid_names:
        return stem
    raise SystemExit(
        f"style '{raw}' is not in the frozen pool {sorted(valid_names)}. "
        "Style images must come from code/style_pool/images/.")


def read_provenance(path):
    """Read the generator's provenance table -> list of (out_rel, original, style, label)."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        else:
            parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"{path} line {number}: expected 4 fields, got {parts}")
        out_rel, original, style, label = parts[0], parts[1], parts[2], parts[3]
        try:
            label = int(label)
        except ValueError:
            raise SystemExit(f"{path} line {number}: bad label {label!r}") from None
        rows.append((out_rel, original, style, label))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("adain", "ours"), required=True)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="the generator's provenance table")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="root the out_rel paths are relative to "
                             "(default: the directory holding the pairs file)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=None,
                        help="beta column for the 'ours' group")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="output manifest path "
                             "(default: DATA_ROOT/augmented/manifest_<group>.txt)")
    parser.add_argument("--copy-images", action="store_true", default=True,
                        help="copy the generated images into DATA_ROOT/augmented/<group>/")
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    args = parser.parse_args()

    if args.group == "ours" and args.beta is None:
        raise SystemExit("the 'ours' group needs --beta")

    valid_names, pairing = load_pool()
    rows = read_provenance(args.pairs)
    if not rows:
        raise SystemExit(f"no rows in {args.pairs}")

    source_root = Path(args.source_root) if args.source_root else args.pairs.parent

    out_dir = config.DATA_ROOT / "augmented" / args.group
    if args.copy_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    missing = 0
    for out_rel, original, style_raw, label in rows:
        style = normalise_style(style_raw, valid_names)

        # the pairing must agree with the frozen one, or the two augmented
        # groups would not be comparable
        expected = pairing.get(original)
        if expected is not None and expected != style:
            raise SystemExit(
                f"content->style pairing differs from the frozen pool for "
                f"{original}: expected '{expected}', got '{style}'. "
                "Regenerate using the frozen pairing.")

        src = source_root / out_rel
        if not src.is_file():
            missing += 1
            continue

        stem = Path(original).stem
        new_rel = f"augmented/{args.group}/{stem}__{style}.jpg"
        if args.copy_images:
            shutil.copyfile(src, config.DATA_ROOT / new_rel)

        fields = [new_rel, str(label), original, style, f"{args.alpha}"]
        if args.group == "ours":
            fields.append(f"{args.beta}")
        manifest_lines.append(" ".join(fields))

    if missing:
        raise SystemExit(f"{missing} generated images were not found under {source_root}")

    manifest = args.manifest or (config.DATA_ROOT / "augmented" / f"manifest_{args.group}.txt")
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"{args.group}: {len(manifest_lines)} rows -> {manifest}")
    if args.copy_images:
        print(f"images copied to {out_dir}")


if __name__ == "__main__":
    main()
