"""Mechanical fairness verification across the three groups' manifests.

Checks that the +AdaIN and +Ours manifests satisfy every constraint listed
in Section 3.7 before any training run is accepted into the results:

  1. same original images: both manifests' provenance (content images) equal
     the photo_train split, AUG_PER_IMAGE stylizations per original;
  2. same style pool: style names in both manifests all come from the frozen
     style-pool manifest and the pool hash matches the recorded one;
  3. same augmentation count: identical number of stylized rows;
  4. same content-style pairing: original->style mapping identical between
     the two manifests (only the generator differs);
  5. no test-set leakage: every style image referenced by the pool is
     byte-disjoint from the local PACS copy, so that no target-domain test
     image can reach the training set through the style pool.

Usage:
    python fairness_check.py --adain manifest.txt --ours manifest.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import config
from data import read_aug_manifest


def check(adain_path, ours_path, subset=False):
    problems = []

    adain = read_aug_manifest(adain_path)
    ours = read_aug_manifest(ours_path)

    # 3. counts
    if len(adain) != len(ours):
        problems.append(f"row count differs: adain {len(adain)} vs ours {len(ours)}")
    expected = len(config.read_split(config.SPLIT_FILES["photo_train"])) * config.AUG_PER_IMAGE
    for name, rows in (("adain", adain), ("ours", ours)):
        if len(rows) != expected and not subset:
            problems.append(f"{name}: {len(rows)} rows != expected {expected}")

    # 1. same originals
    train_originals = {rel for rel, _ in config.read_split(config.SPLIT_FILES["photo_train"])}
    provenance = {}
    for name, rows in (("adain", adain), ("ours", ours)):
        origs = Counter(meta[0] if meta else "?" for _, _, meta in rows)
        covered = set(origs)
        target = train_originals if not subset else covered
        if not covered <= train_originals:
            problems.append(f"{name}: provenance references non-training originals")
        if covered != target:
            missing = target - covered
            problems.append(f"{name}: provenance originals mismatch (missing {len(missing)})")
        for orig, count in origs.items():
            if count != config.AUG_PER_IMAGE:
                problems.append(f"{name}: original {orig} used {count}x != {config.AUG_PER_IMAGE}")
        provenance[name] = origs

    # 2. same style pool
    pool_path = config.STYLE_POOL_MANIFEST
    pool = {}
    if pool_path.is_file():
        outer = json.loads(pool_path.read_text(encoding="utf-8"))
        pool = outer.get("pool", {})
    else:
        problems.append(f"style pool manifest missing: {pool_path}")
    for name, rows in (("adain", adain), ("ours", ours)):
        styles = {meta[1] for _, _, meta in rows if len(meta) > 1}
        unknown = styles - set(pool.get("styles", {}))
        if unknown:
            problems.append(f"{name}: styles not in frozen pool: {sorted(unknown)[:5]}")

    # 4. same content-style pairing
    def pairing(rows):
        return {meta[0]: meta[1] for _, _, meta in rows if len(meta) > 1}
    if pairing(adain) != pairing(ours):
        problems.append("content->style pairing differs between adain and ours")
    elif pool and subset is False:
        recorded = json.loads(pool_path.read_text(encoding="utf-8")).get("pairing", {})
        if pairing(adain) != recorded:
            problems.append("content->style pairing differs from the frozen pool pairing")

    # recorded hash (exclude pool_hash itself, as at build time)
    if pool:
        payload = {k: v for k, v in pool.items() if k != "pool_hash"}
        actual = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        if pool.get("pool_hash") != actual:
            problems.append(f"style pool hash mismatch: recorded {pool.get('pool_hash')} != {actual}")

    # 5. no test-set leakage: the style images must not be PACS images
    problems.extend(check_no_leak(pool))

    return problems


def check_no_leak(pool):
    """Return problems if any style image is a PACS image.

    Art Painting, Cartoon and Sketch are the evaluation domains.  A style image
    that is one of those images would put the test appearance into training,
    so the check compares content hashes against the whole local PACS copy.
    """
    problems = []
    if not pool:
        return problems

    style_dir = config.STYLE_POOL_IMAGES
    pacs_root = config.IMAGE_ROOT
    if not pacs_root.is_dir():
        problems.append(f"PACS image root not found: {pacs_root}")
        return problems

    pacs = {}
    for path in pacs_root.rglob("*"):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            pacs.setdefault(digest, path.as_posix())

    for name, meta in pool.get("styles", {}).items():
        filename = Path(meta.get("file", "")).name
        path = style_dir / filename
        if not path.is_file():
            problems.append(f"style image missing for '{name}': {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in pacs:
            problems.append(
                f"LEAK: style '{name}' is byte-identical to the PACS image "
                f"{pacs[digest]}; style images must come from outside PACS")
        if meta.get("sha256") and meta["sha256"] != digest:
            problems.append(f"style '{name}': sha256 does not match the recorded one")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adain", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--subset", action="store_true",
                        help="pipeline test: allow a subset of originals, "
                             "skip full-coverage and frozen-pairing checks")
    parser.add_argument("--runs", nargs="+", type=Path, default=None,
                        help="run directories (or summary_*.json files) whose "
                             "recorded config_hash must match the current one")
    args = parser.parse_args()

    problems = check(args.adain, args.ours, subset=args.subset)

    # 6. identical training configuration across the groups
    problems.extend(check_config_hashes(args.runs))

    if problems:
        print(f"FAIL ({len(problems)} problems):")
        for p in problems[:30]:
            print("  -", p)
        raise SystemExit(1)
    print("PASS: the two augmented manifests satisfy all fairness constraints.")


def check_config_hashes(runs):
    """Verify that the given runs used the current training configuration.

    For a run directory the authoritative record is ``train_summary.json``,
    which stores the full ``training_config`` block.  The fingerprint is
    recomputed from that block rather than trusting a stored ``config_hash``
    string, so a run recorded under an older hash definition is still judged
    on what it actually did, and a run whose hyper-parameters differ from the
    current ones is rejected.
    """
    problems = []
    if not runs:
        return problems
    expected = config.config_hash()
    for run in runs:
        run = Path(run)
        if run.is_dir():
            summary = run / "train_summary.json"
            if not summary.is_file():
                problems.append(f"no train_summary.json in {run}")
                continue
            candidates = [summary]
        else:
            candidates = [run]

        for candidate in candidates:
            try:
                payload = json.loads(Path(candidate).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{candidate}: cannot read ({exc})")
                continue
            if not isinstance(payload, dict):
                problems.append(f"{candidate}: not a JSON object")
                continue
            recorded_config = payload.get("training_config")
            if isinstance(recorded_config, dict):
                actual = config.config_hash_of(recorded_config)
                if actual != expected:
                    problems.append(
                        f"{candidate}: training config {actual} != current {expected}")
                continue
            # Fallback for artifacts that only carry the hash string.
            recorded_hash = payload.get("config_hash")
            if recorded_hash is None:
                problems.append(f"{candidate}: no config_hash found")
            elif recorded_hash != expected:
                problems.append(
                    f"{candidate}: config_hash {recorded_hash} != current {expected}")
    return problems


if __name__ == "__main__":
    main()
