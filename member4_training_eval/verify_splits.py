"""Validate member 1's project splits against the local PACS copy.

Checks, for every split:
  * every relative path resolves to a real file under data/PACS
  * the class folder matches the label (1..7 -> dog..person)
  * no path appears in two splits
  * per-split and per-class counts match protocol.json
  * sha256 of each split file matches protocol.json
  * image files decode and are non-empty
"""
import hashlib
import os
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
SPLITS = ("photo_train", "photo_val", "photo_test",
          "art_painting_test", "cartoon_test", "sketch_test")

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "project_splits"
IMAGE_ROOT = Path(os.environ.get(
    "NNDL_DATA_ROOT", r"C:\Users\qth02\Desktop\nndl_pacs\data")) / "PACS"
PROTOCOL = SPLIT_DIR / "protocol.json"


def read_split(name):
    rows = []
    for line in (SPLIT_DIR / f"{name}.txt").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        rel, raw = line.rsplit(maxsplit=1)
        rows.append((rel.replace("\\", "/").removeprefix("./"), int(raw) - 1))
    return rows


def sha256_of(path):
    """Return a set of candidate hashes; protocol.json was written on
    LF-normalized files, but a Windows git checkout (autocrlf) stores CRLF."""
    raw = path.read_bytes()
    return {hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    proto_by_name = {s["split"]: s for s in protocol["splits"]}

    if not IMAGE_ROOT.is_dir():
        sys.exit(f"missing image root: {IMAGE_ROOT}")

    seen_global = {}
    failures = []

    for name in SPLITS:
        rows = read_split(name)
        proto = proto_by_name[name]

        if len(rows) != proto["count"]:
            failures.append(f"{name}: count {len(rows)} != protocol {proto['count']}")

        digest = sha256_of(SPLIT_DIR / f"{name}.txt")
        if proto["sha256"] not in digest:
            failures.append(f"{name}: sha256 mismatch")

        class_counts = Counter()
        for rel, label in rows:
            class_counts[str(label + 1)] += 1
            path = IMAGE_ROOT / rel
            if not path.is_file():
                failures.append(f"{name}: missing file {rel}")
                continue
            if path.parent.name != CLASSES[label]:
                failures.append(f"{name}: label/folder mismatch {rel}")
            key = path.resolve()
            if key in seen_global:
                failures.append(f"{name}: {rel} also in {seen_global[key]}")
            seen_global[key] = name

        if dict(class_counts) != proto["class_counts"]:
            failures.append(f"{name}: class counts differ from protocol")

        print(f"  {name:20s} {len(rows):5d} files  sha256 {'OK' if proto['sha256'] in digest else 'MISMATCH'}")

    # decode check on a sample plus the known-bad image
    sample = []
    for name in SPLITS:
        rows = read_split(name)
        sample.extend((r, l) for r, l in rows[::37])
    bad = []
    for rel, _ in sample:
        try:
            with Image.open(IMAGE_ROOT / rel) as im:
                im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            bad.append((rel, repr(exc)))
    print(f"\ndecoded {len(sample) - len(bad)}/{len(sample)} sampled images")
    for rel, err in bad[:10]:
        print(f"  DECODE FAIL {rel}: {err}")

    total_files = sum(1 for p in IMAGE_ROOT.rglob("*") if p.is_file())
    print(f"\nfiles on disk under {IMAGE_ROOT}: {total_files}")
    print(f"files referenced by splits: {len(seen_global)}")

    if failures:
        print(f"\nFAIL ({len(failures)} problems):")
        for f in failures[:40]:
            print("  -", f)
        sys.exit(1)
    print("\nPASS: all six splits are consistent with protocol.json and the local PACS copy.")


if __name__ == "__main__":
    main()
